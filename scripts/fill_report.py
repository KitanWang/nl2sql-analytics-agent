#!/usr/bin/env python
"""把评测结果填进 README.md 与 reports/REPORT.md 的占位符。

    venv/bin/python scripts/fill_report.py

为什么不用手工填：报告里的数字必须**每次都从真实结果生成**，
手工填一次之后就有「报告和代码不同步」的风险。这个脚本让两处文档
永远跟着 `reports/eval_summary.json` 走。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports" / "eval_summary.json"
README = ROOT / "README.md"
REPORT = ROOT / "reports" / "REPORT.md"

ABLATION_MARK = "<!-- ABLATION_TABLE_PLACEHOLDER -->"
RESULTS_MARK = "<!-- RESULTS_TABLE_PLACEHOLDER -->"

PROFILE_LABEL = {
    "baseline": "裸 DDL",
    "schema": "schema + 口径注释",
    "glossary": "+ 业务口径注入",
    "fewshot": "+ 检索式少样本",
    "full": "+ 自修复 + 解读",
}


def load() -> list[dict]:
    if not SUMMARY.exists():
        print(f"缺少 {SUMMARY}，先跑 scripts/run_eval.py --all", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def main_table(rows: list[dict]) -> str:
    lines = [
        "| 配置 | 说明 | 准确率 | 正确/总数 | 平均耗时 | 自修复 | prompt tok |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    base = next((r["accuracy"] for r in rows if r["profile"] == "baseline"), None)
    for r in rows:
        label = PROFILE_LABEL.get(r["profile"], r["profile"])
        delta = ""
        if base is not None and r["profile"] != "baseline":
            delta = f"（{'+' if r['accuracy'] > base else ''}{(r['accuracy'] - base) * 100:.0f} pp）"
        lines.append(
            f"| `{r['profile']}` | {label} | **{r['accuracy']:.1%}**{delta} | "
            f"{r['correct']}/{r['n']} | {r['avg_ms']:.0f} ms | {r['repairs_total']} | "
            f"{r['prompt_tokens']:,} |"
        )
    return "\n".join(lines)


def group_table(rows: list[dict]) -> str:
    groups = sorted({g for r in rows for g in r["by_group"]})
    profs = [r["profile"] for r in rows]
    out = ["| 能力组 | " + " | ".join(f"`{p}`" for p in profs) + " |",
           "| :--- | " + " | ".join("---:" for _ in profs) + " |"]
    for g in groups:
        cells = []
        for r in rows:
            d = r["by_group"].get(g)
            if not d:
                cells.append("-")
                continue
            mark = "✓" if d["correct"] == d["n"] else ""
            cells.append(f"{d['correct']}/{d['n']}{mark}")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def error_table(rows: list[dict]) -> str:
    agg: dict[str, int] = {}
    for r in rows:
        for k, v in r["by_error"].items():
            agg[k] = agg.get(k, 0) + v
    total = sum(agg.values()) or 1
    out = ["| 错误类型 | 次数 | 占比 |", "| :--- | ---: | ---: |"]
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        out.append(f"| {k} | {v} | {v / total:.1%} |")
    return "\n".join(out)


def build_block(rows: list[dict]) -> str:
    return "\n".join([
        main_table(rows),
        "",
        "分能力组（✓ 表示该组全对）：",
        "",
        group_table(rows),
        "",
        "错误类型分布（五档合计）：",
        "",
        error_table(rows),
        "",
        "> 说明：`baseline` 用的是**裸 DDL**（只有表名列名类型，不含任何业务口径），",
        "> 它代表「把库结构直接丢给模型」的常见做法。",
        "> 从它到下一档的差距，就是「把业务口径写清楚」这件事的价值。",
    ])


def replace_block(text: str, mark: str, block: str) -> str:
    if mark not in text:
        return text
    return text.replace(mark, block)


def main() -> int:
    rows = load()
    block = build_block(rows)

    for path, mark in ((README, RESULTS_MARK), (REPORT, ABLATION_MARK)):
        if not path.exists():
            print(f"跳过（文件不存在）：{path}", file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8")
        if mark not in text:
            print(f"跳过（占位符不存在）：{path}", file=sys.stderr)
            continue
        path.write_text(replace_block(text, mark, block), encoding="utf-8")
        print(f"已更新 {path.relative_to(ROOT)}")

    # 同时刷新 RESULTS.md 的最终版本
    print("\n最终结果：")
    for r in rows:
        print(f"  {r['profile']:10s} {r['accuracy']:6.1%}  ({r['correct']}/{r['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
