#!/usr/bin/env python
"""一键构建：合成 → 清洗 → 落 Parquet → 建 DuckDB → 出数据质量报告。

    uv run python scripts/build_warehouse.py            # 全量重建
    uv run python scripts/build_warehouse.py --check    # 只校验已有库的口径自洽性

产物：
    data/processed/*.parquet        清洗后的表（不入库，可重建）
    data/warehouse.duckdb           DuckDB 库（含 v_order_fact 视图）
    reports/data_quality.md         数据质量报告（入库）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nl2sql.data.pipeline import (  # noqa: E402
    clean,
    render_quality_report,
    write_parquet,
)
from nl2sql.data.synth import SynthConfig, generate, inject_dirty  # noqa: E402
from nl2sql.schema import TABLES, VIEWS_SQL, ddl  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "warehouse.duckdb"
REPORT = ROOT / "reports" / "data_quality.md"
STATS = ROOT / "data" / "build_stats.json"

# 表名 → parquet 文件名
TABLE_FILE = {
    "customers": "customers",
    "products": "products",
    "sales_orders": "orders",
    "order_items": "items",
    "payments": "payments",
    "refunds": "refunds",
    "dim_calendar": "calendar",
}


def build() -> dict:
    print("[1/4] 合成原始数据 …")
    raw = inject_dirty(generate(SynthConfig()))

    print("[2/4] pandas 清洗 …")
    result = clean(raw)
    paths = write_parquet(result.data, PROCESSED)
    print(f"      parquet → {PROCESSED}")

    print("[3/4] 建 DuckDB …")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    for table in TABLES:
        fname = TABLE_FILE[table.name]
        con.execute(ddl(table))
        # 显式列出数仓字段：Parquet 里还带着审计列（金额校验的中间结果），不应入库
        cols = ", ".join(f'"{c.name}"' for c in table.columns)
        con.execute(
            f"INSERT INTO {table.name} ({cols}) "
            f"SELECT {cols} FROM read_parquet('{paths[fname]}')"
        )
        n = con.execute(f"SELECT COUNT(*) FROM {table.name}").fetchone()[0]
        print(f"      {table.name:<15} {n:>7,} 行")
    for sql in VIEWS_SQL:
        con.execute(sql)
    print("      视图 v_order_fact 已创建")
    con.close()

    print("[4/4] 写数据质量报告 …")
    report = render_quality_report(result)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    stats = {
        "cleaning": result.stats,
        "report": str(REPORT.relative_to(ROOT)),
        "db": str(DB_PATH.relative_to(ROOT)),
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"完成。内容指纹 {result.stats['content_hash']}")
    return stats


def check() -> int:
    """口径自洽性校验：视图算出来的销售额，必须等于 pandas 独立算出来的值。

    这一步是给评测集兜底的——如果两边不一致，说明要么视图写错了，
    要么评测集的 ground truth 算错了，后面的所有准确率都不可信。
    """
    if not DB_PATH.exists():
        print(f"库不存在：{DB_PATH}，先跑一次 build", file=sys.stderr)
        return 2
    con = duckdb.connect(str(DB_PATH), read_only=True)
    fails = 0

    def expect(name: str, got, want, tol: float = 1e-6) -> None:
        nonlocal fails
        ok = (got is not None and want is not None
              and abs(float(got) - float(want)) <= tol * max(1.0, abs(float(want))))
        print(f"  {'OK  ' if ok else 'FAIL'} {name}: view={got} expect={want}")
        if not ok:
            fails += 1

    view_sum = con.execute(
        "SELECT ROUND(SUM(order_amount), 2) FROM v_order_fact "
        "WHERE status IN ('paid','shipped','completed')"
    ).fetchone()[0]

    import pandas as pd

    orders = pd.read_parquet(PROCESSED / "orders.parquet")
    items = pd.read_parquet(PROCESSED / "items.parquet")
    items_amount = items.groupby("order_id")["item_amount"].sum()
    orders["items_amount"] = orders["order_id"].map(items_amount).fillna(0.0)
    orders["amount"] = (orders["items_amount"] + orders["shipping_fee"]
                        - orders["discount_amount"]).round(2)
    valid_ids = set(con.execute(
        "SELECT order_id FROM sales_orders WHERE status IN ('paid','shipped','completed')"
    ).df()["order_id"])
    orders_in_db = con.execute("SELECT order_id FROM sales_orders").df()["order_id"]
    pandas_valid = orders[orders["order_id"].isin(valid_ids)]
    n_db = len(orders_in_db)
    expect("订单总数", n_db, len(orders), tol=1e-9)

    pandas_sum = round(float(pandas_valid["amount"].sum()), 2)
    expect("有效订单销售额", view_sum, pandas_sum, tol=1e-9)

    n_view = con.execute(
        "SELECT COUNT(*) FROM v_order_fact WHERE status IN ('paid','shipped','completed')"
    ).fetchone()[0]
    expect("有效订单数", n_view, len(pandas_valid), tol=1e-9)
    # 孤儿订单（没有明细的订单）在视图里会被 JOIN 掉，必须显式暴露出来
    orphans = int((~orders["order_id"].isin(
        con.execute("SELECT order_id FROM v_order_fact").df()["order_id"]
    )).sum())
    print(f"  提示 没有明细因而未进 v_order_fact 的订单：{orphans} 条（占总订单 {orphans / n_db:.2%}）")
    con.close()
    print("校验" + ("通过" if fails == 0 else f"失败：{fails} 项"))
    return 0 if fails == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="只做口径自洽性校验")
    args = ap.parse_args()
    if args.check:
        return check()
    build()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
