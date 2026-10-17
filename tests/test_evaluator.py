"""评测判分与检索的测试 —— 「评测本身对不对」比「模型多准」更重要。

一个判分器写错了，整份报告的数字就都是错的。所以这里重点测三件事：
1. **假阴性**：结果正确时必须判对（不能因为列顺序/Decimal/多一列就误判）；
2. **假阳性**：结果错误时必须判错（不能放过漏行、口径错）；
3. **错误分类**：判错时要给出正确的错误类型，否则归因报告会失真。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from nl2sql.agent.analytics import AgentResult, Step
from nl2sql.eval.cases import EvalCase
from nl2sql.eval.evaluator import (
    ERROR_TYPES,
    _crop_candidates,
    _normalize_got,
    diagnose_glossary,
    judge,
    refine_error,
    values_equal,
)
from nl2sql.semantics.examples import EXAMPLES
from nl2sql.semantics.retriever import BM25Retriever, tokenize


def case(kind="scalar", expected=100.0, order_sensitive=False, tol=0.005) -> EvalCase:
    return EvalCase(
        id="T01", group="测试", question="测试问题", kind=kind,
        expected=expected, order_sensitive=order_sensitive, tolerance=tol,
    )


def result(rows, ok=True, sql="SELECT 1", stage="", error="") -> AgentResult:
    r = AgentResult(question="q", ok=ok, rows=rows, sql=sql,
                    error_stage=stage, error=error)
    if rows:
        r.columns = [f"c{i}" for i in range(len(rows[0]))]
        r.row_count = len(rows)
    return r


# ──────────────────────────────────────────────────────────────────
# 数值比较
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("got,want,tol,expected", [
    (100.0, 100.0, 0.005, True),
    (Decimal("100.00"), 100.0, 0.005, True),
    (100.4, 100.0, 0.005, True),        # 0.4% < 0.5%
    (101.0, 100.0, 0.005, False),       # 1% > 0.5%
    (0.0, 0.0, 0.005, True),
    (0.005, 0.0, 0.005, True),          # 期望 0 时用绝对阈值
    (0.02, 0.0, 0.005, False),
    ("1,234.56", 1234.56, 0.005, True),  # 字符串数字
    (None, None, 0.005, True),
    (None, 0.0, 0.005, True),     # SUM 空结果 NULL vs COALESCE 0
    (0.0, None, 0.005, True),
])
def test_values_equal(got, want, tol, expected) -> None:
    assert values_equal(got, want, tol) is expected


# ──────────────────────────────────────────────────────────────────
# 标量判分
# ──────────────────────────────────────────────────────────────────

def test_scalar_correct() -> None:
    v = judge(case(), result([(Decimal("100.0"),)]))
    assert v.correct


def test_scalar_wrong_value() -> None:
    v = judge(case(), result([(Decimal("120.0"),)]))
    assert not v.correct
    assert v.error_type == "数值误差"


def test_scalar_zero_expected() -> None:
    c = case(expected=0.0)
    assert judge(c, result([(0,)], sql="SELECT COUNT(*) FROM x")).correct


def test_execution_failure_is_categorized() -> None:
    v = judge(case(), result([], ok=False, stage="execute", error="Binder Error"))
    assert not v.correct and v.error_type == "执行失败"


def test_guard_block_is_categorized() -> None:
    v = judge(case(), result([], ok=False, stage="guard", error="DROP"))
    assert not v.correct and v.error_type == "安全拦截"


def test_empty_result_when_nonzero_expected() -> None:
    v = judge(case(expected=500.0), result([], ok=True))
    assert not v.correct and v.error_type == "无结果"


# ──────────────────────────────────────────────────────────────────
# 序列/分组判分
# ──────────────────────────────────────────────────────────────────

SERIES_CASE = case(
    kind="series",
    expected=[["华东", 100.0], ["华南", 80.0]],
    order_sensitive=True,
)


def test_series_exact_match() -> None:
    v = judge(SERIES_CASE, result([("华东", Decimal("100.0")), ("华南", Decimal("80.0"))]))
    assert v.correct


def test_series_wrong_row_count() -> None:
    v = judge(SERIES_CASE, result([("华东", Decimal("100.0"))]))
    assert not v.correct
    assert v.error_type == "行数不符"


def test_series_wrong_order() -> None:
    v = judge(SERIES_CASE, result([("华南", Decimal("80.0")), ("华东", Decimal("100.0"))]))
    assert not v.correct
    assert v.error_type == "排序不符"


def test_series_wrong_value() -> None:
    v = judge(SERIES_CASE, result([("华东", Decimal("80.0")), ("华南", Decimal("80.0"))]))
    assert not v.correct
    assert v.error_type == "数值误差"


def test_series_extra_column_is_tolerated() -> None:
    """多带一列（例如顺手给了订单数）不该被扣分——裁列处理。"""
    v = judge(SERIES_CASE, result([
        ("华东", Decimal("100.0"), 12),
        ("华南", Decimal("80.0"), 9),
    ]))
    assert v.correct


def test_series_value_in_trailing_column() -> None:
    """模型把数值放在最后一列（多带了一个名称列）时也要能对上。"""
    v = judge(SERIES_CASE, result([
        ("华东", "East", Decimal("100.0")),
        ("华南", "South", Decimal("80.0")),
    ]))
    assert v.correct


def test_series_order_insensitive() -> None:
    c = case(kind="series", expected=[["华南", 80.0], ["华东", 100.0]], order_sensitive=False)
    assert judge(c, result([("华东", Decimal("100.0")), ("华南", Decimal("80.0"))])).correct


def test_normalize_got_candidates_cover_tail() -> None:
    """多列时既要能取前 N 列，也要能取末 N 列——两种模型习惯都会出现。"""
    assert ["East", Decimal("1")] in _crop_candidates(("华东", "East", Decimal("1")), 2)
    assert _normalize_got(("华东", "East", Decimal("1")), 2) == ["华东", "East"]


# ──────────────────────────────────────────────────────────────────
# 错误归因
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sql,term", [
    ("SELECT SUM(order_amount_raw) FROM sales_orders", "订单金额 / 销售额 / GMV / 营收"),
    ("SELECT MIN(signup_date) FROM customers", "新客 / 首单客户"),
    # CURRENT_DATE 同时命中「时间范围」口径；refine_error 会优先判「时间错」
    ("SELECT * FROM v_order_fact WHERE order_date > CURRENT_DATE", "时间范围"),
])
def test_diagnose_glossary(sql: str, term) -> None:
    c = EvalCase(id="x", group="g", question="q", kind="scalar", expected=1.0)
    assert diagnose_glossary(c, sql) == term


def test_refine_error_marks_time_bug() -> None:
    c = EvalCase(id="x", group="g", question="q", kind="scalar", expected=1.0)
    res = result([(999.0,)], sql="SELECT SUM(order_amount) FROM v_order_fact "
                                "WHERE order_date > CURRENT_DATE")
    v = refine_error(c, res, judge(c, res))
    assert v.error_type == "时间错"


def test_refine_error_marks_wrong_field() -> None:
    c = EvalCase(id="x", group="g", question="q", kind="scalar", expected=1.0)
    res = result([(999.0,)], sql="SELECT SUM(order_amount_raw) FROM sales_orders")
    v = refine_error(c, res, judge(c, res))
    assert v.error_type == "口径错"
    assert "order_amount_raw" in v.detail or "订单金额" in v.detail


def test_error_types_cover_all_branches() -> None:
    """错误类型必须是固定枚举，报告里的分组统计才不会漂移。"""
    assert set(ERROR_TYPES) >= {"安全拦截", "执行失败", "无结果", "行数不符",
                                "数值误差", "排序不符", "口径错", "时间错"}


# ──────────────────────────────────────────────────────────────────
# 检索
# ──────────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_chinese_bigram(self) -> None:
        toks = tokenize("销售额")
        assert "销售" in toks and "售额" in toks
        assert "销" in toks          # 单字也保留

    def test_mixed(self) -> None:
        toks = tokenize("2025 年 GDP 增长")
        assert "2025" in toks and "gd p" not in toks
        assert "年gdp" not in toks

    def test_empty(self) -> None:
        assert tokenize("") == []


class TestRetriever:
    @pytest.fixture(scope="class")
    def retriever(self):
        return BM25Retriever().fit(list(EXAMPLES))

    def test_retrieves_channel_question(self, retriever) -> None:
        hits = retriever.search("各渠道的销售额", k=3)
        assert hits
        joined = " ".join(e.question for e, _ in hits)
        assert "渠道" in joined

    def test_retrieves_ranking_question(self, retriever) -> None:
        hits = retriever.search("销售额最高的 10 个客户", k=3)
        assert any("客户" in e.question or "排名" in " ".join(e.tags) for e, _ in hits)

    def test_retrieves_mom_question(self, retriever) -> None:
        hits = retriever.search("环比增长率", k=3)
        assert any("环比" in e.question or "同比" in e.question for e, _ in hits)

    def test_retrieves_new_customer_question(self, retriever) -> None:
        hits = retriever.search("本月新增的首单客户数", k=3)
        assert any("首单" in e.question or "新客" in e.question for e, _ in hits)

    def test_deterministic(self, retriever) -> None:
        a = [e.question for e, _ in retriever.search("退款率", k=3)]
        b = [e.question for e, _ in retriever.search("退款率", k=3)]
        assert a == b, "检索必须确定性，否则评测不可复现"

    def test_k_respected(self, retriever) -> None:
        assert len(retriever.search("销售额", k=2)) <= 2
        assert len(retriever.search("销售额", k=5)) <= 5

    def test_relevant_beats_irrelevant(self, retriever) -> None:
        """相关性打分要能拉开差距，否则检索等于随机取。"""
        hits = retriever.search("各品类的毛利率", k=len(EXAMPLES))
        top = hits[0][1]
        bottom = hits[-1][1]
        assert top > bottom
        assert "毛利率" in hits[0][0].question
