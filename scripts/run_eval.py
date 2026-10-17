#!/usr/bin/env python
"""跑消融评测。

    uv run python scripts/run_eval.py --profile baseline        # 单配置
    uv run python scripts/run_eval.py --all                     # 全部配置（顺序跑）
    uv run python scripts/run_eval.py --all --limit 10          # 快速冒烟

## 四个配置分别证明什么

| profile | 口径 | 少样本 | 自修复 | 用来回答 |
| :--- | :--- | :--- | :--- | :--- |
| `baseline` | ✗ | ✗ | ✗ | 「不注入上下文能到多少？」 |
| `glossary` | ✓ | ✗ | ✗ | 「业务口径值多少分？」 |
| `fewshot` | ✓ | ✓ 检索 | ✗ | 「示例又多值多少分？」 |
| `full` | ✓ | ✓ 检索 | ✓ 2 次 | 「工程护栏 + 自修复能救回多少？」 |

所有配置共用同一个模型、temperature=0、同一份评测集——差异只来自上面这几列。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl2sql.agent.analytics import AgentConfig, AnalyticsAgent  # noqa: E402
from nl2sql.eval.cases import build_cases, dump_cases, load_cases  # noqa: E402
from nl2sql.eval.evaluator import render_markdown, run_profile  # noqa: E402
from nl2sql.llm.client import LLMClient, LLMConfig  # noqa: E402
from nl2sql.sqlsafe.executor import SQLRunner  # noqa: E402

DB = ROOT / "data" / "warehouse.duckdb"
PROCESSED = ROOT / "data" / "processed"
CASES_JSON = ROOT / "data" / "eval" / "cases.json"
REPORTS = ROOT / "reports"
PREDS = REPORTS / "preds"

PROFILES: dict[str, AgentConfig] = {
    "baseline": AgentConfig(
        use_glossary=False, use_fewshot=False, max_repairs=0, answer_question=False,
    ),
    "glossary": AgentConfig(
        use_glossary=True, use_fewshot=False, max_repairs=0, answer_question=False,
    ),
    "fewshot": AgentConfig(
        use_glossary=True, use_fewshot=True, use_fewshot_retrieval=True,
        fewshot_k=3, max_repairs=0, answer_question=False,
    ),
    "full": AgentConfig(
        use_glossary=True, use_fewshot=True, use_fewshot_retrieval=True,
        fewshot_k=3, max_repairs=2, answer_question=True,
    ),
}


def get_cases(rebuild: bool = False):
    if rebuild or not CASES_JSON.exists():
        cases = build_cases(PROCESSED)
        dump_cases(cases, CASES_JSON)
        print(f"重建评测集：{len(cases)} 条 → {CASES_JSON.relative_to(ROOT)}")
        return cases
    return load_cases(CASES_JSON)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="full", help=f"配置名：{', '.join(PROFILES)}")
    ap.add_argument("--all", action="store_true", help="依次跑全部配置")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--rebuild-cases", action="store_true", help="用当前数据重建评测集")
    ap.add_argument("--llm-profile", default=None, help="LLM 配置 profile（configs/llm.yaml）")
    args = ap.parse_args()

    if not DB.exists():
        print("仓库库不存在，先跑 scripts/build_warehouse.py", file=sys.stderr)
        return 2

    cases = get_cases(args.rebuild_cases)
    if args.limit:
        cases = cases[: args.limit]

    llm = LLMClient(LLMConfig.from_file(ROOT / "configs", profile=args.llm_profile))
    print(f"模型：{llm.cfg.model}；评测集 {len(cases)} 条")

    names = list(PROFILES) if args.all else [args.profile]
    summaries = []
    runner = SQLRunner(DB)
    try:
        for name in names:
            if name not in PROFILES:
                print(f"未知 profile：{name}", file=sys.stderr)
                return 2
            print(f"\n=== {name} ===")
            agent = AnalyticsAgent(llm, runner, PROFILES[name])
            summaries.append(run_profile(name, cases, agent, out_dir=PREDS))
    finally:
        runner.close()
        llm.close()

    # 合并已有结果，保留历史配置的对比（--all 才是完整报告）
    existing = {}
    for s in summaries:
        existing[s.profile] = s
    order = [n for n in PROFILES if n in existing]
    md = render_markdown([existing[n] for n in order])
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "RESULTS.md").write_text(md, encoding="utf-8")
    (REPORTS / "eval_summary.json").write_text(
        json.dumps([existing[n].to_dict() for n in order], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告已写入 {REPORTS / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
