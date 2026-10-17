"""命令行入口：`nl2sql ask "..."` / `nl2sql sql "..."` / `nl2sql schema`。

存在的意义：让这个项目**不打开 Web 也能用**，也方便脚本化调用与复现。
设计上刻意不做 CLI 参数解析的过度抽象——只包住最常用的三条。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _build(args) -> int:
    from nl2sql.data.pipeline import clean, render_quality_report, write_parquet
    from nl2sql.data.synth import SynthConfig, generate, inject_dirty
    import duckdb
    from nl2sql.schema import TABLES, VIEWS_SQL, ddl

    print("合成 + 清洗 …", file=sys.stderr)
    raw = inject_dirty(generate(SynthConfig()))
    result = clean(raw)
    paths = write_parquet(result.data, ROOT / "data" / "processed")
    db = ROOT / "data" / "warehouse.duckdb"
    if db.exists():
        db.unlink()
    con = duckdb.connect(str(db))
    table_file = {"sales_orders": "orders", "order_items": "items",
                  "dim_calendar": "calendar"}
    for t in TABLES:
        con.execute(ddl(t))
        f = table_file.get(t.name, t.name)
        cols = ", ".join(f'"{c.name}"' for c in t.columns)
        con.execute(f"INSERT INTO {t.name} ({cols}) "
                    f"SELECT {cols} FROM read_parquet('{paths[f]}')")
    for sql in VIEWS_SQL:
        con.execute(sql)
    con.close()
    (ROOT / "reports" / "data_quality.md").write_text(
        render_quality_report(result), encoding="utf-8")
    print(f"完成：{db}（内容指纹 {result.stats['content_hash']}）", file=sys.stderr)
    return 0


def _ask(args) -> int:
    from nl2sql.agent.analytics import AgentConfig, AnalyticsAgent
    from nl2sql.llm.client import LLMClient, LLMConfig
    from nl2sql.sqlsafe.executor import SQLRunner

    db = ROOT / "data" / "warehouse.duckdb"
    if not db.exists():
        print("数据库不存在，先跑 `nl2sql build`", file=sys.stderr)
        return 2
    llm = LLMClient(LLMConfig.from_file(ROOT / "configs", profile=args.profile))
    runner = SQLRunner(db)
    try:
        agent = AnalyticsAgent(llm, runner, AgentConfig(answer_question=not args.no_answer))
        res = agent.run(args.question)
        if args.json:
            print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
            return 0 if res.ok else 1
        print(f"\nSQL:\n{res.sql or '（未生成）'}\n")
        if res.ok:
            print(runner.render_for_llm(
                type("E", (), {"ok": True, "columns": res.columns, "rows": res.rows,
                               "row_count": res.row_count, "truncated": False})()))
            if res.answer:
                print(f"\n解读：{res.answer}")
        else:
            print(f"失败（{res.error_stage}）：{res.error}")
        return 0 if res.ok else 1
    finally:
        runner.close()
        llm.close()


def _sql(args) -> int:
    from nl2sql.sqlsafe.executor import SQLRunner
    from nl2sql.sqlsafe.guard import guard

    g = guard(args.sql)
    if not g.ok:
        print(f"⛔ 已拦截：\n{g.reason}", file=sys.stderr)
        return 1
    runner = SQLRunner(ROOT / "data" / "warehouse.duckdb")
    try:
        res = runner.execute(g.sql)
        if not res.ok:
            print(f"执行失败：{res.error}", file=sys.stderr)
            return 1
        print(runner.render_for_llm(res))
        return 0
    finally:
        runner.close()


def _schema(args) -> int:
    from nl2sql.schema import to_prompt_schema
    from nl2sql.semantics.glossary import glossary_prompt

    print(to_prompt_schema())
    print("\n## 业务口径\n")
    print(glossary_prompt())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nl2sql", description="自然语言数据分析 Agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask", help="自然语言提问")
    p.add_argument("question")
    p.add_argument("--profile", default=None, help="LLM 配置 profile")
    p.add_argument("--no-answer", action="store_true", help="只出 SQL 与结果，不生成解读")
    p.add_argument("--json", action="store_true", help="输出完整 JSON（含 trace）")
    p.set_defaults(func=_ask)

    p = sub.add_parser("sql", help="直接执行 SQL（同样过护栏）")
    p.add_argument("sql")
    p.set_defaults(func=_sql)

    sub.add_parser("schema", help="打印表结构与业务口径").set_defaults(func=_schema)
    sub.add_parser("build", help="重建数仓").set_defaults(func=_build)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
