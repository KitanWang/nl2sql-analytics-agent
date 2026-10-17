"""数据流水线测试 —— 每条清洗规则都有对应的「脏数据 → 干净结果」断言。

数据层的测试最容易被写成「跑一遍不报错就算过」。这里反过来：
**先构造一条脏数据，再断言它被特定的规则处理掉**。这样规则被改坏时测试会红，
而不是悄悄地把脏数据放进数仓。
"""

from __future__ import annotations

import pandas as pd
import pytest

from nl2sql.data.pipeline import (
    clean,
    content_hash,
    dedupe_orders,
    drop_invalid_items,
    fix_future_orders,
    normalize_name,
    normalize_payment_units,
    cap_refunds,
    recompute_and_validate_amounts,
)
from nl2sql.data.synth import SynthConfig, generate, inject_dirty


# ──────────────────────────────────────────────────────────────────
# 客户名标准化
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("深圳宏昌精密五金有限公司", "深圳宏昌精密五金有限公司"),
    ("深圳宏昌精密五金（有限公司）", "深圳宏昌精密五金有限公司"),
    ("深圳宏昌精密五金(有限公司)", "深圳宏昌精密五金有限公司"),
    ("  深圳宏昌精密五金有限公司  ", "深圳宏昌精密五金有限公司"),
    ("深圳宏昌精密五金 有限公司", "深圳宏昌精密五金有限公司"),
    ("深圳宏昌精密五金有限公司　", "深圳宏昌精密五金有限公司"),   # 全角空格
    ("ＡＢＣ科技有限公司", "ABC科技有限公司"),                      # 全角字母
])
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_normalize_name_keeps_real_branch_marker() -> None:
    """分公司后缀是真实信息，不能被当成噪音去掉。"""
    s = "苏州嘉禾智能装备（第2分公司）"
    assert "第2分公司" in normalize_name(s)


# ──────────────────────────────────────────────────────────────────
# 订单去重
# ──────────────────────────────────────────────────────────────────

def test_dedupe_keeps_latest_updated_at() -> None:
    df = pd.DataFrame({
        "order_id": ["A", "A", "B"],
        "status": ["paid", "cancelled", "paid"],
        "created_at": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
        "updated_at": pd.to_datetime(["2025-01-01", "2025-01-05", "2025-01-03"]),
    })
    out, info = dedupe_orders(df)
    assert len(out) == 2
    assert info["before"] == 3 and info["after"] == 2
    # A 保留 updated_at 更晚的 cancelled，而不是随意留第一条
    assert out.set_index("order_id").loc["A", "status"] == "cancelled"


def test_dedupe_stable_when_timestamps_tie() -> None:
    df = pd.DataFrame({
        "order_id": ["A", "A"],
        "status": ["paid", "cancelled"],
        "created_at": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "updated_at": pd.to_datetime(["2025-01-01", "2025-01-01"]),
    })
    out, info = dedupe_orders(df)
    assert len(out) == 1
    assert info["exact_tie"] == 1      # 无法判定的并列要被显式记录


# ──────────────────────────────────────────────────────────────────
# 明细红冲
# ──────────────────────────────────────────────────────────────────

def test_drop_invalid_items() -> None:
    df = pd.DataFrame({
        "order_item_id": ["a", "b", "c", "d"],
        "order_id": ["o", "o", "o", "o"],
        "quantity": [5, 0, -3, 1],
        "item_amount": [100.0, 0.0, -60.0, 20.0],
    })
    out, info = drop_invalid_items(df)
    assert len(out) == 2
    assert info["removed"] == 2
    assert (out["quantity"] > 0).all()


# ──────────────────────────────────────────────────────────────────
# 未来单
# ──────────────────────────────────────────────────────────────────

def test_future_orders_are_cancelled_not_deleted() -> None:
    from datetime import date
    df = pd.DataFrame({
        "order_id": ["a", "b", "c"],
        "order_date": pd.to_datetime([date(2025, 6, 1), date(2026, 3, 15), date(2025, 7, 1)]).date,
        "status": ["paid", "paid", "completed"],
    })
    out, info = fix_future_orders(df, cutoff=date(2025, 12, 31))
    assert len(out) == 3, "未来单必须保留行（审计可查），不能删除"
    assert info["flagged"] == 1
    assert out.loc[out["order_id"] == "b", "status"].iloc[0] == "cancelled"


# ──────────────────────────────────────────────────────────────────
# 回款单位
# ──────────────────────────────────────────────────────────────────

def test_payment_unit_conversion_via_order_cross_check() -> None:
    """低于 5000 万阈值的「分」金额也必须被识别出来——这是实际踩过的漏检。"""
    pay = pd.DataFrame({
        "payment_id": ["p1", "p2", "p3"],
        "order_id": ["o1", "o2", "o3"],
        "amount": [44022.10, 4_402_210.00, 100.00],   # 元 / 分（440 万元的订单）/ 元
        "pay_status": ["success"] * 3,
    })
    order_amounts = {"o1": 44022.10, "o2": 44022.10, "o3": 100.00}
    out, info = normalize_payment_units(pay, order_amounts)
    assert out.loc[0, "amount"] == pytest.approx(44022.10)     # 不动
    assert out.loc[1, "amount"] == pytest.approx(44022.10)     # 从分换算
    assert out.loc[2, "amount"] == pytest.approx(100.00)       # 不动
    assert info["converted_rows"] == 1


def test_payment_unit_no_reference_uses_threshold() -> None:
    pay = pd.DataFrame({"payment_id": ["p1"], "order_id": ["o1"],
                        "amount": [70_000_000.0], "pay_status": ["success"]})
    out, info = normalize_payment_units(pay, None)
    assert out.loc[0, "amount"] == pytest.approx(700_000.0)
    assert info["converted_by_threshold"] == 1


# ──────────────────────────────────────────────────────────────────
# 退款封顶
# ──────────────────────────────────────────────────────────────────

def test_cap_refunds() -> None:
    orders = pd.DataFrame({"order_id": ["o1", "o2"],
                           "order_amount_expected": [100.0, 200.0]})
    refunds = pd.DataFrame({"refund_id": ["r1", "r2"],
                            "order_id": ["o1", "o2"],
                            "amount": [500.0, 80.0]})
    out, info = cap_refunds(refunds, orders)
    assert out.loc[0, "amount"] == 100.0
    assert out.loc[1, "amount"] == 80.0
    assert info["capped"] == 1


# ──────────────────────────────────────────────────────────────────
# 金额重算
# ──────────────────────────────────────────────────────────────────

def test_amount_is_recomputed_from_items_not_raw() -> None:
    orders = pd.DataFrame({
        "order_id": ["o1"],
        "shipping_fee": [15.0],
        "discount_amount": [5.0],
        "order_amount_raw": [999.99],       # 源系统错误值
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o1"],
        "item_amount": [50.0, 50.0],
    })
    out, stats = recompute_and_validate_amounts(orders, items)
    assert out.loc[0, "order_amount_expected"] == pytest.approx(110.0)   # 100+15-5
    assert out.loc[0, "amount_deviation"] == pytest.approx(999.99 - 110.0)
    assert stats["mismatch_rows"] == 1


# ──────────────────────────────────────────────────────────────────
# 端到端：清洗可复现
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_raw():
    cfg = SynthConfig(n_customers=80, n_products=60, n_orders=600)
    return inject_dirty(generate(cfg))


def test_cleaning_is_deterministic(small_raw) -> None:
    """同样的原始数据跑两次，内容指纹必须一致。

    这条测试是「评测结论可复现」的前提：如果清洗结果每次都不一样，
    ground truth 也会变，那就没法比较不同 prompt 的效果。
    """
    h1 = clean(small_raw).stats["content_hash"]
    h2 = clean(small_raw).stats["content_hash"]
    assert h1 == h2


def test_synthetic_generation_is_deterministic() -> None:
    cfg = SynthConfig(n_customers=50, n_products=40, n_orders=200)
    a = generate(cfg)
    b = generate(cfg)
    assert content_hash(clean(inject_dirty(a)).data) == content_hash(clean(inject_dirty(b)).data)


def test_clean_removes_internal_test_customers(small_raw) -> None:
    out = clean(small_raw)
    names = out.data.customers["customer_name"].astype(str)
    assert not names.str.contains("内部测试").any()


def test_clean_has_no_orphans(small_raw) -> None:
    out = clean(small_raw)
    order_ids = set(out.data.orders["order_id"])
    assert set(out.data.items["order_id"]).issubset(order_ids)
    if len(out.data.payments):
        assert set(out.data.payments["order_id"]).issubset(order_ids)
    customer_ids = set(out.data.customers["customer_id"])
    assert set(out.data.orders["customer_id"]).issubset(customer_ids)


def test_clean_has_no_negative_order_amount(small_raw) -> None:
    out = clean(small_raw)
    assert out.data.orders["order_amount_expected"].min() >= -1e-9


def test_clean_keeps_no_empty_valid_orders(small_raw) -> None:
    """有效状态却没有任何明细的订单必须被剔除，否则「视图汇总 ≠ 全表汇总」。"""
    out = clean(small_raw)
    with_items = set(out.data.items["order_id"])
    o = out.data.orders
    valid = o["status"].isin(["paid", "shipped", "completed"])
    assert o.loc[valid & ~o["order_id"].isin(with_items)].empty


def test_report_lists_all_rules(small_raw) -> None:
    out = clean(small_raw)
    rules = out.report.to_frame()["rule"].tolist()
    for expected in ("客户名标准化", "订单号去重", "剔除数量≤0明细", "退款金额封顶",
                     "剔除孤儿记录", "剔除无明细的有效订单", "负金额订单归零"):
        assert expected in rules, f"缺少清洗规则记录：{expected}"
