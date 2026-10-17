"""Agent 主流程：问题 → SQL → 校验 → 执行 → 自修复 → 解读。

## 设计原则

1. **每一步的失败都有明确归属**。整条链路会被拆成 6 个阶段，每次运行都输出
   `trace`（每阶段的输入/输出/耗时/错误），评测时按阶段归因，而不是只报一个「错了」；
2. **LLM 只负责「生成」，不负责「判断对不对」**。SQL 能不能跑由执行器说了算，
   安全不安全由护栏说了算，结果可信不可信由接地校验说了算；
3. **重试有预算**。`max_repairs` 是硬上限，防止模型进入「改错→再改错」的死循环烧 token。

## 六个阶段

```
plan? ──┐
        ├─ generate   LLM 生成 SQL
        ├─ guard      AST 安全校验（不通过 → repair）
        ├─ execute    DuckDB 只读执行（失败 → repair）
        ├─ repair     带着错误信息自我修复（最多 N 次）
        └─ answer     LLM 解读结果（可选，且做数字接地校验）
```
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from nl2sql.llm.client import LLMClient, LLMError, LLMResponse
from nl2sql.prompts import (
    build_answer_prompt,
    build_fix_prompt,
    build_sql_prompt,
)
from nl2sql.semantics.examples import EXAMPLES
from nl2sql.semantics.retriever import BM25Retriever, Example
from nl2sql.sqlsafe.executor import SQLRunner, cap_limit
from nl2sql.sqlsafe.guard import GuardResult, guard


# ──────────────────────────────────────────────────────────────────
# 轨迹记录
# ──────────────────────────────────────────────────────────────────

@dataclass
class Step:
    stage: str
    ok: bool
    elapsed_ms: float = 0.0
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage, "ok": self.ok,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "detail": self.detail, **self.payload,
        }


@dataclass
class AgentResult:
    question: str
    sql: str = ""
    ok: bool = False
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    answer: str = ""
    attempts: int = 0
    repairs: int = 0
    total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    steps: list[Step] = field(default_factory=list)
    error_stage: str = ""
    error: str = ""
    retrieved: list[dict] = field(default_factory=list)
    guard_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from nl2sql.sqlsafe.executor import _jsonable
        return {
            "question": self.question,
            "sql": self.sql,
            "ok": self.ok,
            "columns": self.columns,
            "rows": [[_jsonable(v) for v in r] for r in self.rows],
            "row_count": self.row_count,
            "answer": self.answer,
            "attempts": self.attempts,
            "repairs": self.repairs,
            "total_ms": round(self.total_ms, 1),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error_stage": self.error_stage,
            "error": self.error,
            "retrieved": self.retrieved,
            "guard_violations": self.guard_violations,
            "steps": [s.to_dict() for s in self.steps],
        }


# ──────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    # 消融实验的开关都放这里，`scripts/run_eval.py` 直接按 profile 覆盖
    use_glossary: bool = True
    use_fewshot: bool = True
    fewshot_k: int = 3
    use_fewshot_retrieval: bool = True     # False = 固定取前 k 条（静态少样本）
    max_repairs: int = 2
    answer_question: bool = True
    require_limit: bool = True
    max_rows: int = 500
    timeout_s: float = 8.0
    enforce_grounding: bool = True          # 解读结果里的数字必须出现在结果里
    temperature: float = 0.0


# ──────────────────────────────────────────────────────────────────
# 结果接地校验
# ──────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize_number(text: str, suffix: str = "") -> tuple[float, float] | None:
    """把 "1,234.56"、"1.2万" 归一成 (数值, 显示精度)。

    `suffix` 是数字紧跟的单位词（万/亿/%）——它不在正则匹配范围内，
    需要调用方从原文里额外取出来，否则「409.05 万元」会被当成 409.05。

    返回精度是必须的：模型说「409.05 万元」时，真实值是 4,090,463.94，
    换算成万元后是 409.046 万，四舍五入到两位才是 409.05。
    如果按「误差 ≤ 1e-6」判等，这条正确的表述会被误判成幻觉——
    这是一个真实踩到的误报，反过来会让接地校验失去可信度。
    """
    t = text.strip().replace(",", "")
    mult = 1.0
    if suffix == "%":
        return None
    if suffix == "万":
        mult = 10_000.0
    elif suffix == "亿":
        mult = 100_000_000.0
    try:
        val = float(t)
    except ValueError:
        return None
    # 显示精度：小数点后位数决定可容忍的误差
    decimals = len(t.split(".")[1]) if "." in t else 0
    # 显示精度 × 单位换算：说「409.05 万」时真实值允许 ±5000 的误差
    tol = 0.5 * (10 ** -decimals) * mult
    return val * mult, tol


def check_grounding(answer: str, rows: list[tuple], columns: list[str]) -> tuple[bool, list[str]]:
    """检查「结果解读」里的关键数字是否真的来自查询结果。

    ## 为什么需要这一步

    这是实测发现的问题：模型在解读时会「顺手」算出结果里没有的数字
    （比如结果只给了销售额，它自己除一下报了个客单价；或者把 2025 写成 2024）。
    这类错误比 SQL 写错更危险——SQL 错还能被评测集抓到，解读错**没人会去查**。

    做法：把答案里所有数字抠出来，与「结果集中所有数字」及「允许的派生值
    （相邻列相除、求和）」比对。允许一定误差（四舍五入）。
    """
    # DuckDB 的 SUM/AVG 返回 Decimal，COUNT 返回 int，都必须算数值
    def _num(v: Any) -> float | None:
        if isinstance(v, bool) or v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            import decimal as _dec
            if isinstance(v, _dec.Decimal):
                return float(v)
        except Exception:
            pass
        return None

    # 结果集里的所有数值 + 常见派生值
    values: set[float] = set()
    row_sums: list[float] = []
    for r in rows:
        nums = [n for n in (_num(v) for v in r) if n is not None]
        values.update(nums)
        if nums:
            row_sums.append(sum(nums))
    values.update(row_sums)
    # 列求和（模型常会说「合计 xx」）
    ncols = max((len(r) for r in rows), default=0)
    for ci in range(ncols):
        col = [n for n in (_num(r[ci]) if ci < len(r) else None for r in rows) if n is not None]
        if col:
            values.add(sum(col))

    def matches(x: float, tol: float) -> bool:
        if abs(x) < 1e-9:
            return True
        # 允许值与显示精度两者取宽：四舍五入到两位的数字，误差可能是半个最小单位
        for v in values:
            eps = max(tol, abs(v) * 1e-6, 0.005)
            if abs(x - v) <= eps:
                return True
            # 百分比/比率（0.1234 → 12.34%）
            if abs(x / 100 - v) <= max(tol / 100, 1e-4):
                return True
        return False

    # 只校验「像指标的数字」：长度 ≥ 3 或有小数，避免把「前 3 名」「2025 年」全算进去
    ungrounded: list[str] = []
    for m in _NUM_RE.finditer(answer):
        raw = m.group(0)
        # 数字与单位之间可能有空格（「12 万元」很常见），先跳过空白再取单位
        rest = answer[m.end():m.end() + 3].lstrip()
        suffix = rest[:1]
        parsed = _normalize_number(raw, suffix)
        if parsed is None:
            continue
        x, tol = parsed
        if abs(x) < 10 and "." not in raw:
            continue          # 排名、月份、个数之类的小整数不校验
        if 1900 <= x <= 2100 and "." not in raw:
            continue          # 年份不校验（结果里未必有年份列）
        if not matches(x, tol):
            ungrounded.append(raw)
    return (not ungrounded), ungrounded


# ──────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────

class AnalyticsAgent:
    def __init__(
        self,
        llm: LLMClient | None,
        runner: SQLRunner,
        cfg: AgentConfig | None = None,
        *,
        examples: tuple[Example, ...] = EXAMPLES,
    ):
        self.llm = llm
        self.runner = runner
        self.cfg = cfg or AgentConfig()
        self.retriever = BM25Retriever().fit(list(examples))

    # ── 检索 ──────────────────────────────────────────

    def _select_examples(self, question: str) -> list[Example]:
        if not self.cfg.use_fewshot:
            return []
        if self.cfg.use_fewshot_retrieval:
            hits = self.retriever.search(question, k=self.cfg.fewshot_k)
            return [e for e, _ in hits]
        return list(EXAMPLES)[: self.cfg.fewshot_k]

    # ── 主流程 ────────────────────────────────────────

    def run(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        res = AgentResult(question=question)

        # ── 阶段 1：生成 ──
        if self.llm is None:
            raise ValueError("未提供 LLM 客户端")
        examples = self._select_examples(question)
        res.retrieved = [{"question": e.question, "score": round(self.retriever.score(question, EXAMPLES.index(e)), 3)}
                         for e in examples]
        pb = build_sql_prompt(
            question,
            examples=tuple(e.as_prompt_pair() for e in examples),
            include_glossary=self.cfg.use_glossary,
        )
        t = time.perf_counter()
        try:
            r = self.llm.chat(pb.messages, temperature=self.cfg.temperature)
        except LLMError as e:
            res.error_stage, res.error = "generate", str(e)
            res.steps.append(Step("generate", False, (time.perf_counter() - t) * 1000, str(e)))
            res.total_ms = (time.perf_counter() - t0) * 1000
            return res
        res.prompt_tokens += r.prompt_tokens
        res.completion_tokens += r.completion_tokens
        res.attempts += 1
        sql = _extract_sql(r.text)
        res.steps.append(Step("generate", True, (time.perf_counter() - t) * 1000,
                              f"mode={pb.meta['mode']} examples={len(examples)}",
                              {"sql": sql, "tokens_out": r.completion_tokens}))

        # ── 阶段 2~4：校验 → 执行 → 修复 ──
        last_error = ""
        for attempt in range(self.cfg.max_repairs + 1):
            t = time.perf_counter()
            g: GuardResult = guard(sql, require_limit=self.cfg.require_limit)
            res.guard_violations = [f"[{v.rule}] {v.detail}" for v in g.violations]
            if not g.ok:
                res.steps.append(Step("guard", False, (time.perf_counter() - t) * 1000,
                                      g.reason[:500], {"sql": sql}))
                last_error = f"SQL 未通过安全校验：{g.reason}"
                res.error_stage, res.error = "guard", last_error
            else:
                res.sql = cap_limit(g.sql, self.cfg.max_rows)
                res.steps.append(Step("guard", True, (time.perf_counter() - t) * 1000,
                                      f"tables={g.tables_used}", {"sql": res.sql}))
                t = time.perf_counter()
                ex = self.runner.execute(res.sql)
                if ex.ok:
                    res.ok = True
                    res.columns, res.rows, res.row_count = ex.columns, ex.rows, ex.row_count
                    res.error_stage, res.error = "", ""
                    res.steps.append(Step("execute", True, ex.elapsed_ms,
                                          f"rows={ex.row_count} truncated={ex.truncated}",
                                          {"sql": res.sql}))
                    break
                last_error = f"SQL 执行失败：{ex.error}"
                res.error_stage, res.error = "execute", last_error
                res.steps.append(Step("execute", False, ex.elapsed_ms, last_error,
                                      {"sql": res.sql, "error_kind": ex.error_kind}))

            # ── 修复 ──
            if attempt >= self.cfg.max_repairs or self.llm is None:
                break
            t = time.perf_counter()
            fb = build_fix_prompt(question, sql, last_error, attempt=attempt + 1)
            try:
                fr = self.llm.chat(fb.messages, temperature=self.cfg.temperature)
            except LLMError as e:
                res.steps.append(Step("repair", False, (time.perf_counter() - t) * 1000, str(e)))
                break
            res.prompt_tokens += fr.prompt_tokens
            res.completion_tokens += fr.completion_tokens
            res.attempts += 1
            res.repairs += 1
            sql = _extract_sql(fr.text)
            res.steps.append(Step("repair", True, (time.perf_counter() - t) * 1000,
                                  f"attempt={attempt + 1}", {"sql": sql}))

        # ── 阶段 5：解读 ──
        if res.ok and self.cfg.answer_question and self.llm is not None:
            t = time.perf_counter()
            table = self.runner.render_for_llm(
                _exec_like(res), max_rows=30
            )
            ab = build_answer_prompt(question, res.sql, table)
            try:
                ar = self.llm.chat(ab.messages, temperature=self.cfg.temperature)
                res.prompt_tokens += ar.prompt_tokens
                res.completion_tokens += ar.completion_tokens
                res.attempts += 1
                answer = _strip_preamble(ar.text)
                if self.cfg.enforce_grounding:
                    grounded, bad = check_grounding(answer, res.rows, res.columns)
                    if not grounded:
                        # 不重试、不纠正：直接把不可核实的数字标出来。
                        # 重试一次的成本比标出来高，而且模型第二次大概率还是编。
                        res.steps.append(Step(
                            "grounding", False, (time.perf_counter() - t) * 1000,
                            f"解读中出现结果集里没有的数字：{bad}",
                            {"ungrounded": bad},
                        ))
                    else:
                        res.steps.append(Step("grounding", True, 0.0, "数字全部可核实"))
                res.answer = answer
                res.steps.append(Step("answer", True, (time.perf_counter() - t) * 1000,
                                      f"{len(answer)} 字"))
            except LLMError as e:
                res.steps.append(Step("answer", False, (time.perf_counter() - t) * 1000, str(e)))

        res.total_ms = (time.perf_counter() - t0) * 1000
        return res


# ──────────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────────

_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.+?)```", re.IGNORECASE | re.DOTALL)


def _extract_sql(text: str) -> str:
    """从模型输出里取出 SQL。优先取 ```sql 代码块，否则取第一段以 SELECT/WITH 开头的内容。"""
    m = _SQL_FENCE.search(text or "")
    if m:
        return m.group(1).strip().rstrip(";").strip()
    t = (text or "").strip()
    t = re.sub(r"^```\w*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"(?is)\b(select|with)\b.*", t)
    return (m.group(0) if m else t).strip().rstrip(";").strip()


def _strip_preamble(text: str) -> str:
    """去掉解读前的客套话（「好的，根据查询结果……」）。"""
    t = (text or "").strip()
    t = re.sub(r"^\s*(好的|收到|明白)[，,。.！!]?\s*", "", t)
    t = re.sub(r"^\s*(根据|依据)(上述|以上|查询)?(结果|数据)[，,：:]\s*", "", t)
    return t.strip()


def _exec_like(res: AgentResult):
    """给 `render_for_llm` 构造一个轻量对象，避免把 ExecutionResult 传来传去。"""
    from nl2sql.sqlsafe.executor import ExecutionResult

    return ExecutionResult(
        ok=res.ok, columns=res.columns, rows=res.rows,
        row_count=res.row_count, truncated=False, sql=res.sql,
    )
