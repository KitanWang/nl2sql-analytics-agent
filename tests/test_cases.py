"""评测集本身的测试。

评测集是「标准答案」——它错了，后面所有准确率都不可信。这里断言三件事：

1. **结构完整**：50 条、7 个能力组、id 唯一、每条都有题面与期望；
2. **答案可复现**：用 DuckDB 独立跑一遍标准 SQL，结果与 pandas 算出的期望一致
   （两条完全不同的实现路径互相验证，这是数据口径最重要的一道保险）；
3. **难度分布合理**：不能全是简单题，也不能全超纲。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nl2sql.eval.cases import build_cases, build_ground_truth, load_cases
from nl2sql.sqlsafe.executor import SQLRunner


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DB = ROOT / "data" / "warehouse.duckdb"


@pytest.fixture(scope="module")
def cases():
    if not PROCESSED.exists():
        pytest.skip("需要先跑 scripts/build_warehouse.py")
    return build_cases(PROCESSED)


# ──────────────────────────────────────────────────────────────────
# 结构
# ──────────────────────────────────────────────────────────────────

def test_case_count(cases) -> None:
    assert len(cases) == 50, f"评测集应为 50 条，实际 {len(cases)}"


def test_ids_unique(cases) -> None:
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_groups_cover_seven_areas(cases) -> None:
    groups = {c.group for c in cases}
    assert len(groups) == 7, f"应有 7 个能力组，实际：{sorted(groups)}"
    for g in groups:
        n = sum(1 for c in cases if c.group == g)
        assert n >= 4, f"能力组 {g} 只有 {n} 条，样本量不足"


def test_every_case_has_question_and_expectation(cases) -> None:
    for c in cases:
        assert c.question.strip(), f"{c.id} 题面为空"
        assert c.expected is not None, f"{c.id} 缺少期望值"
        assert c.kind in ("scalar", "series", "rows"), f"{c.id} kind 非法"


def test_kinds_are_consistent(cases) -> None:
    for c in cases:
        if c.kind == "scalar":
            assert isinstance(c.expected, (int, float)), f"{c.id} 标量题期望不是数字"
        else:
            assert isinstance(c.expected, list) and c.expected, f"{c.id} 序列题期望不是非空列表"


def test_tolerance_is_sane(cases) -> None:
    for c in cases:
        assert 0 < c.tolerance <= 0.05, f"{c.id} 容差 {c.tolerance} 不合理"


def test_questions_are_self_contained(cases) -> None:
    """题面不能有「如上」「前面说的」这种依赖上下文的说法。"""
    bad = ("如上", "前面", "刚才", "上一个问题")
    for c in cases:
        for b in bad:
            assert b not in c.question, f"{c.id} 题面依赖上下文：{c.question}"


def test_dump_and_load_roundtrip(cases, tmp_path) -> None:
    from nl2sql.eval.cases import dump_cases
    p = tmp_path / "cases.json"
    dump_cases(cases, p)
    loaded = load_cases(p)
    assert len(loaded) == len(cases)
    for a, b in zip(cases, loaded):
        assert a.id == b.id
        assert a.question == b.question
        assert a.expected == b.expected


# ──────────────────────────────────────────────────────────────────
# 答案可复现：DuckDB 侧独立验证
# ──────────────────────────────────────────────────────────────────

# 每道题对应的「标准 SQL」（独立于 pandas 实现）。
# 只覆盖能稳定用一条 SQL 表达的题；其余题靠 pandas 版本自身的一致性测试保障。
CANONICAL_SQL: dict[str, str] = {
    "A01": "SELECT COUNT(*) FROM v_order_fact WHERE status IN ('paid','shipped','completed')",
    "A02": "SELECT ROUND(SUM(order_amount),2) FROM v_order_fact WHERE status IN ('paid','shipped','completed') "
           "AND order_date >= DATE '2025-06-01' AND order_date < DATE '2025-07-01'",
    "A03": "SELECT ROUND(SUM(order_amount),2) FROM v_order_fact WHERE status IN ('paid','shipped','completed') "
           "AND ship_region = '华东'",
    "A08": "SELECT COUNT(*) FROM customers",
    "B02": "SELECT ROUND(SUM(order_amount - refund_amount),2) FROM v_order_fact "
           "WHERE status IN ('paid','shipped','completed') "
           "AND order_date >= DATE '2025-06-01' AND order_date < DATE '2025-07-01'",
    "B06": "SELECT COUNT(DISTINCT order_id) FROM v_order_fact "
           "WHERE status IN ('paid','shipped','completed') "
           "AND order_date >= DATE '2025-06-01' AND order_date < DATE '2025-07-01'",
    "B07": "SELECT ROUND(SUM(refund_amount),2) FROM v_order_fact "
           "WHERE status IN ('paid','shipped','completed') "
           "AND order_date >= DATE '2025-06-01' AND order_date < DATE '2025-07-01'",
    "B10": "SELECT ROUND(SUM(order_amount),2) FROM v_order_fact",
    "C01": "SELECT channel, ROUND(SUM(order_amount),2) AS amt FROM v_order_fact "
           "WHERE status IN ('paid','shipped','completed') GROUP BY 1 ORDER BY amt DESC",
    "C02": "SELECT ship_region, ROUND(SUM(order_amount),2) AS amt FROM v_order_fact "
           "WHERE status IN ('paid','shipped','completed') GROUP BY 1 ORDER BY amt DESC",
    "D04": "SELECT ROUND(SUM(order_amount),2) FROM v_order_fact "
           "WHERE status IN ('paid','shipped','completed') "
           "AND order_date >= DATE '2025-07-01' AND order_date < DATE '2025-10-01'",
    "D05": "SELECT ROUND(SUM(order_amount),2) FROM v_order_fact "
           "WHERE status IN ('paid','shipped','completed') "
           "AND order_date >= DATE '2025-07-01' AND order_date < DATE '2026-01-01'",
    "G04": "SELECT COALESCE(ROUND(SUM(order_amount),2), 0) FROM v_order_fact "
           "WHERE status IN ('paid','shipped','completed') "
           "AND order_date >= DATE '2026-01-01' AND order_date < DATE '2026-02-01'",
}


@pytest.fixture(scope="module")
def runner():
    if not DB.exists():
        pytest.skip("需要先跑 scripts/build_warehouse.py")
    r = SQLRunner(DB)
    yield r
    r.close()


@pytest.mark.parametrize("case_id", sorted(CANONICAL_SQL))
def test_expected_matches_duckdb(cases, runner, case_id: str) -> None:
    """核心断言：pandas 算的期望值 == DuckDB 视图算出来的值。

    这两条路径实现完全不同（一个用 DataFrame 逐列算，一个用 SQL 视图 JOIN），
    能对上才说明口径没写歪。对不上就是**数据层或评测集的 bug**，
    必须停下来查清，不能靠调容差糊弄过去。
    """
    c = next(x for x in cases if x.id == case_id)
    res = runner.execute(CANONICAL_SQL[case_id])
    assert res.ok, f"{case_id} 标准 SQL 执行失败：{res.error}"
    got = res.rows

    if c.kind == "scalar":
        expected = float(c.expected)
        actual = float(got[0][0])
        if abs(expected) < 1e-9:
            assert abs(actual) <= 0.01
        else:
            assert abs(actual - expected) / abs(expected) <= c.tolerance, (
                f"{case_id}: DuckDB={actual} pandas={expected}"
            )
    else:
        assert len(got) == len(c.expected), (
            f"{case_id}: DuckDB 返回 {len(got)} 行，pandas 得到 {len(c.expected)} 行"
        )
        for (gk, gv), (ek, ev) in zip(got, c.expected):
            assert str(gk) == str(ek), f"{case_id}: 键不一致 {gk} vs {ek}"
            assert abs(float(gv) - float(ev)) / max(abs(float(ev)), 1e-9) <= c.tolerance, (
                f"{case_id}: {ek} DuckDB={gv} pandas={ev}"
            )


# ──────────────────────────────────────────────────────────────────
# 事实表与数仓视图的一致性
# ──────────────────────────────────────────────────────────────────

def test_ground_truth_matches_view_row_count(runner) -> None:
    f = build_ground_truth(PROCESSED)
    n_pandas = int(f["valid"].sum())
    n_view = runner.execute(
        "SELECT COUNT(*) FROM v_order_fact WHERE status IN ('paid','shipped','completed')"
    ).rows[0][0]
    assert int(n_view) == n_pandas


def test_ground_truth_matches_view_amount(runner) -> None:
    f = build_ground_truth(PROCESSED)
    total_pandas = round(float(f.loc[f["valid"], "order_amount"].sum()), 2)
    total_view = float(runner.execute(
        "SELECT ROUND(SUM(order_amount),2) FROM v_order_fact "
        "WHERE status IN ('paid','shipped','completed')"
    ).rows[0][0])
    assert abs(total_view - total_pandas) <= 0.01


def test_refund_and_payment_amounts_agree(runner) -> None:
    """退款/回款总额：pandas 事实表 vs DuckDB 视图必须一致。

    注意作用域：**只比较在视图里出现的订单**。
    极少数「已取消且没有明细」的订单不在 `v_order_fact` 里（视图是 INNER JOIN 明细），
    它们的回款不会体现在视图汇总中——这是已知且有意的设计（见 data_quality.md 第 10 条）。
    如果不限定作用域，测试会因为这 9117.25 元报错，看起来像口径 bug，其实不是。
    """
    f = build_ground_truth(PROCESSED)
    # 必须显式放宽 max_rows：执行器默认只回 500 行（这是线上保护），
    # 用来取全量 id 时会被静默截断，导致作用域比对失真。
    view_ids = {row[0] for row in runner.execute(
        "SELECT order_id FROM v_order_fact", max_rows=100_000).rows}
    f_view = f[f["order_id"].isin(view_ids)]
    for col, sql in [
        ("refund_amount", "SELECT ROUND(SUM(refund_amount),2) FROM v_order_fact"),
        ("pay_amount", "SELECT ROUND(SUM(pay_amount),2) FROM v_order_fact"),
    ]:
        a = round(float(f_view[col].sum()), 2)
        b = float(runner.execute(sql).rows[0][0])
        assert abs(a - b) <= 0.01, f"{col} 不一致：pandas={a} duckdb={b}"


def test_ground_truth_row_count_matches_view_when_scoped(runner) -> None:
    """同一作用域下，事实表行数必须与视图行数完全相同（没有 JOIN 放大或丢失）。"""
    f = build_ground_truth(PROCESSED)
    # 必须显式放宽 max_rows：执行器默认只回 500 行（这是线上保护），
    # 用来取全量 id 时会被静默截断，导致作用域比对失真。
    view_ids = {row[0] for row in runner.execute(
        "SELECT order_id FROM v_order_fact", max_rows=100_000).rows}
    assert int(f["order_id"].isin(view_ids).sum()) == len(view_ids)


def test_dataset_is_not_degenerate(cases) -> None:
    """数据集不能是「全是 0」或「全是同一个值」——那样评测没有区分度。"""
    scalars = [float(c.expected) for c in cases if c.kind == "scalar"]
    assert len(set(scalars)) >= len(scalars) * 0.8, "标量答案重复度过高"
    assert any(x > 1e6 for x in scalars), "缺少大额指标题"
    assert any(x < 1.0 for x in scalars), "缺少比率类小题"
