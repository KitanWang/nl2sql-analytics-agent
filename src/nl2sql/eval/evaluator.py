"""评测执行与判分 —— 把 Agent 的输出与标准答案对比，并给出**可归因的错误分类**。

## 为什么错误分类比准确率本身更重要

只报「准确率 78%」没有信息量。面试官（以及自己）真正关心的是：
**剩下 22% 错在哪一类、每一类是谁的责任**。所以这里做四件事：

1. **结果规范化**：把 SQL 结果与 pandas 标准答案压成同一种结构再比，
   避免「列顺序不同 / 类型是 Decimal / 有额外列」这类假阴性；
2. **分维度统计**：按 A~G 七个能力组分别算准确率，一眼看出模型弱在哪；
3. **错误归因**：分成 8 类（口径错、时间错、关联错、安全拦截、执行失败、
   结果不符、无结果、解读不接地），每类都有明确的判定规则；
4. **保存原始记录**：每条样本的 SQL、结果、耗时、token 全部落盘，
   任何结论都可以回查——这也是别人无法质疑的地方。

## 判分口径

| 情况 | 判定 |
| :--- | :--- |
| 行数/键集合不一致 | 直接判错（不做宽松匹配，否则「漏了 3 个品类」会被放过） |
| 数值 | 相对误差 ≤ `tolerance`（默认 0.5%）；期望值为 0 时要求绝对值 ≤ 0.01 |
| 排序敏感题 | 顺序完全一致才算对 |
| 大小为 1 的结果 | 只比第一列（处理「SELECT 1 行 1 列」与「标量」的差异） |
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from nl2sql.agent.analytics import AgentConfig, AgentResult, AnalyticsAgent
from nl2sql.eval.cases import EvalCase
from nl2sql.semantics.glossary import GLOSSARY
from nl2sql.sqlsafe.executor import SQLRunner


# ──────────────────────────────────────────────────────────────────
# 规范化
# ──────────────────────────────────────────────────────────────────

def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, str):
        s = v.replace(",", "").strip()
        if s.endswith("%"):
            s = s[:-1]
        try:
            return float(s)
        except ValueError:
            return None
    try:
        import datetime as _dt
        if isinstance(v, (_dt.date, _dt.datetime)):
            return None
    except Exception:
        pass
    return None


def _key(v: Any) -> str:
    """把分组键归一成字符串：2025-06 / 2025-06-01 / datetime 都要能对上。"""
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    try:
        import datetime as _dt
        if isinstance(v, _dt.datetime):
            return v.strftime("%Y-%m-%d")
        if isinstance(v, _dt.date):
            return v.isoformat()
    except Exception:
        pass
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _row_key(row: Iterable[Any]) -> str:
    return "\u0001".join(_key(v) for v in row)


def values_equal(a: Any, b: Any, tol: float) -> bool:
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        if math.isnan(na) or math.isnan(nb):
            return False
        if abs(nb) < 1e-12:
            return abs(na) <= 0.01
        return abs(na - nb) / abs(nb) <= tol
    return _key(a) == _key(b)


# ──────────────────────────────────────────────────────────────────
# 判分
# ──────────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    correct: bool
    error_type: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {"correct": self.correct, "error_type": self.error_type, "detail": self.detail}


ERROR_TYPES = (
    "安全拦截",       # 护栏拒绝了生成的 SQL
    "执行失败",       # SQL 语法/语义错，跑不出来
    "无结果",         # 跑出 0 行但期望有值（或反之）
    "行数不符",       # 分组数/TOP N 数量对不上
    "数值误差",       # 结构对但数字错（口径/公式问题）
    "排序不符",       # 内容对但顺序错
    "口径错",         # 能用 wrong_pattern 判定用了错误字段/状态
    "时间错",         # 时间窗口用了 CURRENT_DATE / 写错区间
)


def judge(case: EvalCase, res: AgentResult) -> JudgeResult:
    # 0) 链路本身失败
    if not res.ok:
        if res.error_stage == "guard":
            return JudgeResult(False, "安全拦截", res.error[:200])
        return JudgeResult(False, "执行失败", (res.error or "未知错误")[:200])

    rows = res.rows
    exp = case.expected

    # 1) 空结果
    if not rows:
        if isinstance(exp, (int, float)) and abs(float(exp)) < 1e-9:
            return JudgeResult(True, "", "空结果符合期望 0")
        if isinstance(exp, list) and len(exp) == 0:
            return JudgeResult(True, "", "空结果符合期望")
        return JudgeResult(False, "无结果", "查询返回 0 行，但期望有值")

    # 2) 标量题
    if case.kind == "scalar":
        first = rows[0][0] if rows[0] else None
        if isinstance(exp, (int, float)):
            if values_equal(first, exp, case.tolerance):
                return JudgeResult(True)
            return JudgeResult(False, "数值误差", f"得到 {_key(first)}，期望 {exp}")
        if values_equal(first, exp, case.tolerance):
            return JudgeResult(True)
        return JudgeResult(False, "数值误差", f"得到 {_key(first)}，期望 {_key(exp)}")

    # 3) 序列 / 行集合题
    exp_rows = _normalize_expected(exp)
    got_rows = [_normalize_got(r, len(exp_rows[0]) if exp_rows else 1) for r in rows]

    if len(got_rows) != len(exp_rows):
        return JudgeResult(
            False, "行数不符",
            f"返回 {len(got_rows)} 行，期望 {len(exp_rows)} 行"
        )

    if case.order_sensitive:
        for i, (g, e) in enumerate(zip(got_rows, exp_rows)):
            if not _row_equal(g, e, case.tolerance):
                return JudgeResult(
                    False, "数值误差" if _keys_match(g, e, case.tolerance) else "排序不符",
                    f"第 {i + 1} 行：得到 {g}，期望 {e}"
                )
        return JudgeResult(True)

    # 顺序无关：按 key 配对
    unmatched = list(exp_rows)
    for g in got_rows:
        hit = next((e for e in unmatched if _row_equal(g, e, case.tolerance)), None)
        if hit is None:
            return JudgeResult(False, "数值误差", f"多出/不匹配的行：{g}")
        unmatched.remove(hit)
    return JudgeResult(True)


def _normalize_expected(exp: Any) -> list[list[Any]]:
    """标准答案统一成 [[k, v], ...]。"""
    if not isinstance(exp, list):
        return [[exp]]
    if not exp:
        return []
    if isinstance(exp[0], (list, tuple)):
        return [list(r) for r in exp]
    return [[x] for x in exp]


def _normalize_got(row: tuple, expected_ncols: int) -> list[Any]:
    """把 SQL 结果行裁到与期望相同的列数。

    现实情况：模型经常多带一列（比如除了销售额还 SELECT 了订单数），
    或者少一列。裁列而不是直接判错，是为了不让「多给了一个有用信息」变成扣分项；
    但如果**键列错位**，下面的比较会立刻暴露。
    """
    r = list(row)
    if len(r) > expected_ncols:
        # 优先保留「前 expected_ncols 列」，但如果最后一列是数值而第一列不是，
        # 说明模型把数值放在了后面——此时取最后 expected_ncols 列更合理。
        if expected_ncols >= 2:
            head_numeric = [_num(x) is not None for x in r[:expected_ncols]]
            tail_numeric = [_num(x) is not None for x in r[-expected_ncols:]]
            if sum(tail_numeric) > sum(head_numeric):
                return r[-expected_ncols:]
        return r[:expected_ncols]
    return r


def _row_equal(g: list[Any], e: list[Any], tol: float) -> bool:
    if len(g) != len(e):
        return False
    return all(values_equal(a, b, tol) for a, b in zip(g, e))


def _keys_match(g: list[Any], e: list[Any], tol: float) -> bool:
    """键（非数值列）是否一致——用于区分「排序错」和「值算错」。"""
    if len(g) != len(e):
        return False
    return all(
        _num(gi) is None and _num(ei) is None and values_equal(gi, ei, tol)
        or (_num(gi) is not None and _num(ei) is not None)
        for gi, ei in zip(g, e)
    )


# ──────────────────────────────────────────────────────────────────
# 口径归因
# ──────────────────────────────────────────────────────────────────

def diagnose_glossary(case: EvalCase, sql: str) -> str | None:
    """如果 SQL 里出现了「已知错误写法」，直接指出用了哪个错口径。

    这是把 glossary 的 `wrong_pattern` 变成**可执行断言**：
    评测报告能写出「6 条 B 组错误里，有 4 条是因为用了 order_amount_raw」，
    而不是笼统地说「口径不对」。
    """
    if not sql:
        return None
    for entry in GLOSSARY:
        if not entry.wrong_pattern:
            continue
        try:
            if re.search(entry.wrong_pattern, sql, re.IGNORECASE):
                return entry.term
        except re.error:
            continue
    return None


def refine_error(case: EvalCase, res: AgentResult, base: JudgeResult) -> JudgeResult:
    """在基础判分上，叠加「口径/时间」层面的归因。"""
    if base.correct:
        return base
    sql = res.sql or ""
    if re.search(r"CURRENT_DATE|CURRENT_TIMESTAMP|now\(\)", sql, re.IGNORECASE):
        return JudgeResult(False, "时间错", "使用了运行时当前日期，与数据集基准日不一致")
    term = diagnose_glossary(case, sql)
    if term:
        return JudgeResult(False, "口径错", f"命中错误口径：{term}")
    return base


# ──────────────────────────────────────────────────────────────────
# 运行
# ──────────────────────────────────────────────────────────────────

@dataclass
class RunSummary:
    profile: str
    n: int = 0
    correct: int = 0
    by_group: dict[str, dict[str, int]] = field(default_factory=dict)
    by_error: dict[str, int] = field(default_factory=dict)
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    repairs: int = 0
    records: list[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "n": self.n,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "avg_ms": round(self.avg_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "repairs_total": self.repairs,
            "repair_rate": round(self.repairs / self.n, 4) if self.n else 0.0,
            "by_group": self.by_group,
            "by_error": self.by_error,
        }


def run_profile(
    profile: str,
    cases: list[EvalCase],
    agent: AnalyticsAgent,
    out_dir: Path | None = None,
    *,
    verbose: bool = True,
    stop_after: int | None = None,
) -> RunSummary:
    s = RunSummary(profile=profile)
    t_start = time.perf_counter()
    for i, case in enumerate(cases, 1):
        if stop_after and i > stop_after:
            break
        res = agent.run(case.question)
        base = judge(case, res)
        verdict = refine_error(case, res, base)
        s.n += 1
        s.total_ms += res.total_ms
        s.prompt_tokens += res.prompt_tokens
        s.completion_tokens += res.completion_tokens
        s.repairs += res.repairs
        g = s.by_group.setdefault(case.group, {"n": 0, "correct": 0})
        g["n"] += 1
        if verdict.correct:
            s.correct += 1
            g["correct"] += 1
        else:
            s.by_error[verdict.error_type] = s.by_error.get(verdict.error_type, 0) + 1

        s.records.append({
            "profile": profile,
            "case_id": case.id,
            "group": case.group,
            "question": case.question,
            "kind": case.kind,
            "expected": case.expected,
            "got_rows": [[str(v) for v in r] for r in res.rows],
            "sql": res.sql,
            "answer": res.answer,
            "agent_ok": res.ok,
            "attempts": res.attempts,
            "repairs": res.repairs,
            "elapsed_ms": round(res.total_ms, 1),
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "error_stage": res.error_stage,
            "error": res.error,
            "verdict": verdict.to_dict(),
            "guard_violations": res.guard_violations,
            "retrieved": res.retrieved,
            "grounding_ok": next(
                (st.ok for st in res.steps if st.stage == "grounding"), None
            ),
        })
        if verbose:
            mark = "✓" if verdict.correct else "✗"
            extra = "" if verdict.correct else f"  [{verdict.error_type}] {verdict.detail[:70]}"
            print(f"  {mark} {case.id} {res.total_ms:7.0f}ms{extra}", flush=True)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"records_{profile}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in s.records),
            encoding="utf-8",
        )
        (out_dir / f"summary_{profile}.json").write_text(
            json.dumps(s.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if verbose:
        print(f"[{profile}] 准确率 {s.accuracy:.1%}（{s.correct}/{s.n}），"
              f"平均 {s.avg_ms:.0f}ms，修复 {s.repairs} 次，"
              f"耗时 {(time.perf_counter() - t_start):.0f}s")
    return s


# ──────────────────────────────────────────────────────────────────
# 报告
# ──────────────────────────────────────────────────────────────────

def render_markdown(summaries: list[RunSummary]) -> str:
    """把多次运行的对比写成一份报告。"""
    profiles = [s.profile for s in summaries]
    groups = sorted({g for s in summaries for g in s.by_group})
    lines = [
        "# 评测结果",
        "",
        "> 由 `scripts/run_eval.py` 自动生成；每行数字来自实际执行，"
        "原始记录见 `reports/preds/records_*.jsonl`。",
        "",
        "## 1. 总览",
        "",
        "| 配置 | 准确率 | 正确/总数 | 平均耗时 | 自修复次数 | prompt tok | completion tok |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.profile} | **{s.accuracy:.1%}** | {s.correct}/{s.n} | "
            f"{s.avg_ms:.0f} ms | {s.repairs} | {s.prompt_tokens:,} | {s.completion_tokens:,} |"
        )

    lines += ["", "## 2. 分能力组准确率", "",
              "| 能力组 | " + " | ".join(profiles) + " |",
              "| :--- | " + " | ".join("---:" for _ in profiles) + " |"]
    for g in groups:
        cells = []
        for s in summaries:
            d = s.by_group.get(g)
            cells.append(f"{d['correct']}/{d['n']}" if d else "-")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")

    lines += ["", "## 3. 错误类型分布（全部配置合计）", "",
              "| 错误类型 | 次数 | 占比 | 典型原因 |", "| :--- | ---: | ---: | :--- |"]
    total_err = sum(sum(s.by_error.values()) for s in summaries)
    causes = {
        "安全拦截": "生成的 SQL 未通过 AST 护栏（写操作/外部函数/笛卡尔积）",
        "执行失败": "SQL 语法或语义错误，DuckDB 无法执行",
        "无结果": "筛选条件过严或时间范围错，返回 0 行",
        "行数不符": "分组键或 TOP N 数量不对",
        "数值误差": "公式/口径错，结构与行数都对但数字不对",
        "排序不符": "内容对但排序方向或并列处理错",
        "口径错": "命中已知错误字段/状态（如 order_amount_raw）",
        "时间错": "用了 CURRENT_DATE 或错误的日期区间",
    }
    agg: dict[str, int] = {}
    for s in summaries:
        for k, v in s.by_error.items():
            agg[k] = agg.get(k, 0) + v
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} | {v / total_err:.1%} | {causes.get(k, '')} |")

    lines += ["", "## 4. 逐条的取舍说明", ""]
    lines += [
        "- 判分**不做宽松匹配**：行数不一致直接判错。宁可低估准确率，也不虚报；",
        "- 容差 0.5%（比率类 1%–2%），足以容纳浮点误差，又能抓住「用错金额字段」的 12% 偏差；",
        "- `SELECT` 多带一列不扣分（按位置裁列），但键列错位会立刻暴露；",
        "- 解读质量**不计入**准确率，单独统计「接地校验通过率」，避免两件事混在一起。",
        "",
    ]
    return "\n".join(lines)
