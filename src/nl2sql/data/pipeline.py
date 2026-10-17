"""数据清洗流水线：pandas 版本。

设计要点：**清洗规则与「脏数据形态」一一对应，且每条规则都能被单独测试**。
`synth.inject_dirty()` 里注入 8 类脏数据，这里逐条清理，并在
`reports/data_quality.md` 里留下「每条规则淘汰了多少行」的可核对记录。

为什么不直接建库：

- 清洗必须在**可复现、可断言**的层面完成（pandas + 单测），而不是散落在 SQL 里；
- DuckDB 只负责「建表 + 执行」，不做业务清洗，职责清晰；
- 评测集的 ground truth 也基于同一份清洗后的 DataFrame 计算，保证账实一致。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nl2sql.data.synth import Dataset

DATA_CUTOFF = date(2025, 12, 31)

# 回款金额「单位是分」的判定阈值：单笔回款 > 5000 万且是整数分，几乎不可能是元
FEN_SUSPECT_CENTS = 5_000_000

# 内部测试单：客户名里带这些词的，整单剔除（否则会污染客户数与销售额）
INTERNAL_TEST_MARKERS = ("内部测试", "测试公司", "demo", "DEMO")


@dataclass
class CleaningReport:
    """清洗统计。每条规则一行，用于生成 data_quality.md 与单元测试断言。"""
    steps: list[dict] = field(default_factory=list)

    def add(self, rule: str, reason: str, before: int, after: int, note: str = "") -> None:
        self.steps.append({
            "rule": rule,
            "reason": reason,
            "rows_before": before,
            "rows_after": after,
            "delta": after - before,
            "note": note,
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


@dataclass
class CleanResult:
    data: Dataset
    report: CleaningReport
    stats: dict


# ──────────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────────

def normalize_name(s: str) -> str:
    """客户名标准化：全半角统一 + 去括号差异 + 去首尾空格 + 折叠内部空白。

    现实中同一个客户在不同系统里可能写成：
        「深圳宏昌精密五金有限公司」/「深圳宏昌精密五金（有限公司）」/「深圳宏昌精密五金有限公司 」
    这四条在 JOIN 时是四个客户，会让「客户数」凭空多出来——这是很典型的隐性数据缺陷。
    """
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKC", s)      # 全角 → 半角
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\((有限公司|股份有限公司|分公司)\)", r"\1", s)
    s = re.sub(r"\s+", "", s)                 # 中文公司名里的空格基本都是录入噪音
    return s.strip()


def dedupe_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """订单号重复：按 updated_at 取最新一条。

    现实来源：上游同步任务重跑且没做幂等。直接 GROUP BY 取任意一条会随机丢状态，
    必须按时间戳取最新——这条规则如果写错，后续所有「有效订单」统计都会漂移。
    """
    before = len(orders)
    # 先算有没有同号但 updated_at 相同的（此时无法判定，保留第一条并记录）
    work = orders.sort_values(["order_id", "updated_at", "created_at"], ascending=[True, False, False])
    tie = work.duplicated(subset=["order_id", "updated_at"], keep="first").sum()
    out = work.drop_duplicates(subset=["order_id"], keep="first").reset_index(drop=True)
    return out, {"before": before, "after": len(out), "exact_tie": int(tie)}


def normalize_payment_units(
    payments: pd.DataFrame, order_amounts: dict[str, float] | None = None
) -> tuple[pd.DataFrame, dict]:
    """回款金额单位混用：识别「分」并换算成「元」。

    ## 为什么不能用单一阈值

    第一版只用了「金额 > 5000 万 → 认为单位是分」。这个规则**会漏**：
    一笔 4.4 万元的订单，如果金额被写成「分」（4,402,210），
    它低于 5000 万阈值，会被当成 440 万元正常回款 —— 回款率直接变成 100 倍。
    这是本项目里实际踩到的缺陷，最终导致「整体回款率 = 1.32」这种不可能的数。

    正确做法是**用订单金额做交叉校验**：
      候选解释 ① 单位是元     → 回款/订单 ≈ 0.x
      候选解释 ② 单位是分     → 回款/订单 ≈ 100×
    哪个解释更接近业务常识（0 < ratio ≤ 1.5）就用哪个；
    两个都不合理时（订单还没生成）退化为阈值法。
    """
    p = payments.copy()
    before_max = float(p["amount"].max()) if len(p) else 0.0
    converted = 0

    if order_amounts is not None:
        ref = p["order_id"].map(order_amounts)
        # 解释 ① 的偏差：|回款/订单 - 1|
        r1 = (p["amount"] / ref).abs()
        # 解释 ② 的偏差：换算成元之后再比
        r2 = ((p["amount"] / 100) / ref).abs()
        # 只对「按元解释明显不合理、按分解释明显更合理」且换算后不超 1.5 倍的行动手
        use_fen = (
            ((r1 > 1.5) | (p["amount"] > FEN_SUSPECT_CENTS))
            & (r2 <= 1.5)
            & ref.notna()
        )
        converted = int(use_fen.sum())
        p.loc[use_fen, "amount"] = (p.loc[use_fen, "amount"] / 100).round(2)

    # 兜底：订单金额缺失时用绝对阈值
    still_big = p["amount"] > FEN_SUSPECT_CENTS
    n_fallback = int(still_big.sum())
    if n_fallback:
        p.loc[still_big, "amount"] = (p.loc[still_big, "amount"] / 100).round(2)
    converted += n_fallback

    return p, {
        "converted_rows": converted,
        "converted_by_order_check": converted - n_fallback,
        "converted_by_threshold": n_fallback,
        "max_before": before_max,
        "max_after": float(p["amount"].max()) if len(p) else 0.0,
    }


def drop_invalid_items(items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """剔除数量 ≤ 0 的明细（红冲行）。

    注意：**不能**用 `quantity > 0` 一刀切而不留痕。
    红冲行在财务口径里是有意义的（负数冲销），但在「销售额/客单价」口径下会把
    订单金额算成负数。本项目统一把它们剔除，并在报告里说明这个选择。
    """
    before = len(items)
    bad = items["quantity"] <= 0
    out = items[~bad].reset_index(drop=True)
    return out, {"before": before, "after": len(out), "removed": int(bad.sum())}


def fix_future_orders(orders: pd.DataFrame, cutoff: date = DATA_CUTOFF) -> tuple[pd.DataFrame, dict]:
    """未来单：下单日期晚于数据截止日的，状态置为 cancelled 并保留（不删除）。

    为什么不删：真实业务里「预售单」是存在的，直接删除会改变历史；
    但它没有实际成交，把它算进销售额是错的。所以保留行、置为 cancelled，
    这样「有效订单」口径自然排除它，而数据审计时能查得到。
    """
    mask = orders["order_date"] > cutoff
    out = orders.copy()
    out.loc[mask, "status"] = "cancelled"
    return out, {"flagged": int(mask.sum()), "dates": sorted(
        {str(d) for d in out.loc[mask, "order_date"].unique()}
    )}


def recompute_and_validate_amounts(
    orders: pd.DataFrame, items: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """用明细汇总重算订单金额，并统计 order_amount_raw 的偏差分布。

    这是项目里最关键的「数据资产」决策：
    **订单金额一律以明细汇总为准**，源系统的 order_amount_raw 只用于监控偏差。
    如果只依赖 raw 字段，模型生成的 SQL 无论怎么写都会得到错的销售额——
    这类问题不是 prompt 能救的，必须在数据层解决。
    """
    agg = items.groupby("order_id")["item_amount"].sum()
    out = orders.copy()
    out["items_amount"] = out["order_id"].map(agg).fillna(0.0).round(2)
    out["order_amount_expected"] = (
        out["items_amount"] + out["shipping_fee"] - out["discount_amount"]
    ).round(2)
    out["amount_deviation"] = (
        out["order_amount_raw"] - out["order_amount_expected"]
    ).round(2)
    denom = out["order_amount_expected"].replace(0, np.nan)
    out["amount_deviation_pct"] = (out["amount_deviation"] / denom).round(4)
    dev = out["amount_deviation_pct"].dropna()
    abs_dev = dev.abs()

    # 金额很小的订单（运费优惠就能主导合计）会让相对偏差爆到百万倍，
    # 直接取 mean / max 会得到没有信息量的数。这里用两种互补口径：
    #   - 百分比口径：以「偏差的金额绝对值」加权，等价于「所有订单上偏差总额 / 订单总额」；
    #   - 分布口径：只看 |偏差| ≥ 1% 的订单，它们是真正需要修的那批。
    valid_dev = out["amount_deviation_pct"].notna() & (out["order_amount_expected"] > 100)
    money_weighted = (
        out.loc[valid_dev, "amount_deviation"].abs().sum()
        / out.loc[valid_dev, "order_amount_expected"].abs().sum()
    )
    material = abs_dev[abs_dev >= 0.01]
    stats = {
        "rows": int(len(out)),
        "mismatch_rows": int((abs_dev > 1e-6).sum()),
        "mismatch_rate": round(float((abs_dev > 1e-6).mean()), 4) if len(dev) else 0.0,
        "material_rows": int(len(material)),
        "material_rate": round(float(len(material) / len(dev)), 4) if len(dev) else 0.0,
        "material_median_dev_pct": round(float(material.median()), 4) if len(material) else 0.0,
        "money_weighted_dev_pct": round(float(money_weighted), 4),
        "p95_abs_dev_pct": round(float(abs_dev.quantile(0.95)), 4) if len(dev) else 0.0,
        "amount_at_stake": round(float(out["amount_deviation"].abs().sum()), 2),
    }
    return out, stats


def drop_orphans(
    orders: pd.DataFrame, items: pd.DataFrame, payments: pd.DataFrame,
    refunds: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """剔除孤儿记录（外键指向不存在的实体）。

    合成数据本身不会产生孤儿，但清洗后的剔除（如内部测试客户）会制造孤儿，
    所以这一步必须放在所有剔除动作之后，否则会出现「明细指向已删订单」的悬空引用。
    """
    valid_customers = set(customers["customer_id"])
    valid_products = set(products["product_id"])

    before_o, before_i, before_p, before_r = len(orders), len(items), len(payments), len(refunds)
    # 先算最终留下的订单，再用它过滤明细/回款/退款。
    # 曾经的写法是先用旧 orders 算 valid_orders 再过滤订单，导致
    # 「订单被剔除但明细留下」——44 条悬空明细，视图 JOIN 时才消失，
    # 非常隐蔽。测试 test_clean_has_no_orphans 就是为了钉住这一点。
    orders2 = orders[orders["customer_id"].isin(valid_customers)].reset_index(drop=True)
    valid_orders = set(orders2["order_id"])
    items2 = items[items["order_id"].isin(valid_orders)
                   & items["product_id"].isin(valid_products)].reset_index(drop=True)
    payments2 = payments[payments["order_id"].isin(valid_orders)].reset_index(drop=True)
    refunds2 = refunds[refunds["order_id"].isin(valid_orders)].reset_index(drop=True)
    return orders2, items2, payments2, refunds2, {
        "orders_removed": before_o - len(orders2),
        "items_removed": before_i - len(items2),
        "payments_removed": before_p - len(payments2),
        "refunds_removed": before_r - len(refunds2),
    }


def cap_refunds(
    refunds: pd.DataFrame, orders: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """退款金额 > 订单金额：按订单金额封顶，并记录被截断的行数。

    现实来源：退款按含税价、订单按不含税价，或一张订单分两次退款但系统重复计入。
    处理方式选择「封顶」而不是「删除」：退款事实存在，但金额不可信，封顶是保守估计。
    """
    r = refunds.copy()
    amt = r["order_id"].map(orders.set_index("order_id")["order_amount_expected"])
    over = r["amount"] > amt
    n = int(over.sum())
    r.loc[over, "amount"] = amt[over].round(2)
    return r, {"capped": n, "total_before": round(float(refunds["amount"].sum()), 2),
               "total_after": round(float(r["amount"].sum()), 2)}


# ──────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────

def clean(raw: Dataset) -> CleanResult:
    rep = CleaningReport()

    # ── 1. 客户名标准化 ──
    customers = raw.customers.copy()
    before = len(customers)
    customers["customer_name_raw"] = customers["customer_name"]
    customers["customer_name"] = customers["customer_name"].map(normalize_name)
    dup_names = int(customers["customer_name"].duplicated().sum())
    rep.add("客户名标准化", "全半角/括号/空格不一致导致同一客户被算成多个",
            before, len(customers), note=f"标准化后仍有 {dup_names} 个重名（不同公司同名，保留）")

    # ── 2. 剔除内部测试客户 ──
    before = len(customers)
    test_mask = customers["customer_name_raw"].astype(str).str.contains(
        "|".join(re.escape(m) for m in INTERNAL_TEST_MARKERS), case=False, regex=True
    )
    customers = customers[~test_mask].reset_index(drop=True)
    rep.add("剔除内部测试客户", "内购/测试单会污染客户数与销售额", before, len(customers),
            note=f"命中关键词：{'/'.join(INTERNAL_TEST_MARKERS)}")

    # ── 3. 订单去重 ──
    orders, info = dedupe_orders(raw.orders)
    rep.add("订单号去重", "上游同步任务重跑导致同号多条，需按 updated_at 取最新",
            info["before"], info["after"], note=f"updated_at 完全相同的并列行 {info['exact_tie']} 条")

    # ── 4. 未来单处理 ──
    orders, fut = fix_future_orders(orders)
    rep.add("未来单置为取消", "预售日期填错年份，保留行但排除出有效订单口径",
            len(orders), len(orders), note=f"标记 {fut['flagged']} 条，异常日期 {fut['dates']}")

    # ── 5. 明细红冲行 ──
    items, info = drop_invalid_items(raw.items)
    rep.add("剔除数量≤0明细", "红冲/冲销行会让订单金额变负", info["before"], info["after"])

    # ⚠️ 顺序很重要：必须在剔明细之后重算金额，否则金额会少算被剔掉的行
    orders, amount_stats = recompute_and_validate_amounts(orders, items)
    rep.add("订单金额以明细汇总为准", "源系统 order_amount_raw 与明细不符（旧价目表）",
            len(orders), len(orders),
            note=f"偏差订单 {amount_stats['mismatch_rows']} 条"
                 f"（{amount_stats['mismatch_rate']:.1%}），"
                 f"其中偏差≥1% 的有 {amount_stats['material_rows']} 条"
                 f"（中位偏差 {amount_stats['material_median_dev_pct']:+.1%}）")

    # ── 6. 回款单位换算 ──
    # 用「订单金额」交叉校验单位解释，比单一阈值可靠（见函数 docstring 说明的漏检案例）
    order_amount_map = (
        orders.set_index("order_id")["order_amount_expected"].to_dict()
    )
    payments, pay_unit = normalize_payment_units(raw.payments, order_amount_map)
    rep.add("回款金额单位归一", "部分老接口以「分」为单位", len(payments), len(payments),
            note=f"换算 {pay_unit['converted_rows']} 条，最大值 {pay_unit['max_before']:.0f} → "
                 f"{pay_unit['max_after']:.0f} 元")

    # ── 7. 退款封顶 ──
    refunds, refund_info = cap_refunds(raw.refunds, orders)
    rep.add("退款金额封顶", "退款金额超过订单金额（含税/不含税口径不一致）",
            len(refunds), len(refunds),
            note=f"封顶 {refund_info['capped']} 条，总额 "
                 f"{refund_info['total_before']:.0f} → {refund_info['total_after']:.0f} 元")

    # ── 8. 孤儿记录 ──
    valid_products = raw.products
    orders, items, payments, refunds, orphan = drop_orphans(
        orders, items, payments, refunds, customers, valid_products
    )
    rep.add("剔除孤儿记录", "外键指向已剔除实体（删测试客户/红冲明细后的悬空引用）",
            orphan["orders_removed"] + orphan["items_removed"], 0,
            note=f"订单 {orphan['orders_removed']}、明细 {orphan['items_removed']}、"
                 f"回款 {orphan['payments_removed']}、退款 {orphan['refunds_removed']}")

    # ── 8.5 负金额订单 ──
    # 优惠大于成交额（合成器的折扣上限只约束了 50%，明细全被红冲剔除后仍可能出现负数）。
    # 业务上「这张订单没花客户的钱」，金额应为 0；把优惠压到等于成交额，保留订单本身。
    neg = orders["order_amount_expected"] < 0
    if int(neg.sum()):
        orders.loc[neg, "discount_amount"] = (
            orders.loc[neg, "items_amount"] + orders.loc[neg, "shipping_fee"]
        ).round(2)
        orders.loc[neg, "order_amount_expected"] = 0.0
        orders.loc[neg, "amount_deviation"] = (
            orders.loc[neg, "order_amount_raw"] - orders.loc[neg, "order_amount_expected"]
        ).round(2)
    rep.add("负金额订单归零", "整单优惠大于成交额（优惠券系统本应拦截）", len(orders), len(orders),
            note=f"修正 {int(neg.sum())} 条，把优惠压到等于成交额")

    # ── 9. 无明细订单 ──
    # 红冲剔除后，少数订单的明细被全部剔光，剩下一张「空订单」。
    # 它在业务上没有成交量，但**金额也不是 0**（还有运费/优惠）——这类行如果留在表里，
    # 任何 JOIN order_items 的统计都会漏掉它，导致「视图汇总 ≠ 全表汇总」。
    # 处理方式：有效状态下直接剔除；取消状态下保留（金额口径下它本来就被排除）。
    # 这样 v_order_fact 的「有效销售额」与「sales_orders 全表汇总」结果完全一致。
    items_per_order = items.groupby("order_id").size()
    no_items = ~orders["order_id"].isin(items_per_order.index)
    keep_cancelled = no_items & orders["status"].isin(["cancelled", "refunded", "pending"])
    drop_mask = no_items & ~keep_cancelled
    dropped_status = orders.loc[drop_mask, "status"].value_counts().to_dict()
    dropped_ids = set(orders.loc[drop_mask, "order_id"])
    n_before = len(orders)
    orders = orders[~drop_mask].reset_index(drop=True)
    rep.add("剔除无明细的有效订单", "红冲明细被剔除后留下的空订单，会让「视图汇总 ≠ 全表汇总」",
            n_before, len(orders),
            note=f"剔除 {int(drop_mask.sum())} 条（状态分布 {dropped_status}），"
                 f"另有 {int(keep_cancelled.sum())} 条非有效状态空订单保留")

    # 订单被剔除后，回款/退款里的对应记录会变成新的孤儿——必须再清一次。
    # 这是第二次调用 drop_orphans 的原因，也是「一次清洗不够」的典型例子。
    if dropped_ids:
        payments = payments[~payments["order_id"].isin(dropped_ids)].reset_index(drop=True)
        refunds = refunds[~refunds["order_id"].isin(dropped_ids)].reset_index(drop=True)

    # ── 9. 类型收敛 ──
    for col in ("order_date", "signup_date", "launch_date", "date_key"):
        src = {"order_date": orders, "signup_date": customers,
               "launch_date": valid_products, "date_key": raw.calendar}[col]
        if col in src.columns:
            if col in orders.columns:
                orders[col] = pd.to_datetime(orders[col]).dt.date
            if col in customers.columns:
                customers[col] = pd.to_datetime(customers[col]).dt.date
            if col in valid_products.columns:
                valid_products = valid_products.assign(
                    launch_date=pd.to_datetime(valid_products["launch_date"]).dt.date
                )
    # 审计列（items_amount / order_amount_expected / amount_deviation*）保留在
    # Dataset 里：数据质量报告与单元测试都要用；入库时按 schema 显式选列。

    for col in ("pay_date",):
        payments[col] = pd.to_datetime(payments[col]).dt.date
    if "refund_date" in refunds.columns:
        refunds["refund_date"] = pd.to_datetime(refunds["refund_date"]).dt.date

    # 审计列（xxx_raw）保留在 Parquet 里供追溯，但不进数仓——数仓只放「干净口径」
    customers = customers.drop(columns=["customer_name_raw"], errors="ignore")

    data = Dataset(
        customers=customers,
        products=valid_products,
        orders=orders,
        items=items,
        payments=payments,
        refunds=refunds,
        calendar=raw.calendar,
    )

    stats = {
        "amount_validation": amount_stats,
        "payment_units": pay_unit,
        "refund_cap": refund_info,
        "orphans": orphan,
        "future_orders": fut,
        "rows": {
            "customers": len(customers),
            "products": len(valid_products),
            "orders": len(orders),
            "order_items": len(items),
            "payments": len(payments),
            "refunds": len(refunds),
            "calendar": len(raw.calendar),
        },
        "date_range": [str(min(orders["order_date"])), str(max(orders["order_date"]))],
        "content_hash": content_hash(data),
    }
    return CleanResult(data=data, report=rep, stats=stats)


def content_hash(data: Dataset) -> str:
    """对清洗后的核心表做内容指纹，用于断言「清洗结果可复现」。"""
    h = hashlib.sha256()
    for name in ("customers", "products", "orders", "items", "payments", "refunds"):
        df = getattr(data, name)
        h.update(name.encode())
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────
# 落盘
# ──────────────────────────────────────────────────────────────────

def write_parquet(data: Dataset, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("customers", "products", "orders", "items", "payments", "refunds", "calendar"):
        df = getattr(data, name)
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p
    return paths


def render_quality_report(result: CleanResult) -> str:
    """生成 data_quality.md —— 数据质量报告，面试时可以直接展示。"""
    rep = result.report.to_frame()
    s = result.stats
    av = s["amount_validation"]

    lines = [
        "# 数据质量报告",
        "",
        "> 本文件由 `scripts/build_warehouse.py` 自动生成，数字全部来自实际执行结果。",
        "> 数据为脚本合成（见 `src/nl2sql/data/synth.py`），**脏数据是刻意注入的**，",
        "> 目的是让清洗规则有真实的对抗目标，而不是写一遍空转。",
        "",
        "## 1. 规模",
        "",
        "| 表 | 行数 |",
        "| :--- | ---: |",
    ]
    for k, v in s["rows"].items():
        lines.append(f"| {k} | {v:,} |")
    lines += [
        "",
        f"订单日期范围：**{s['date_range'][0]} ~ {s['date_range'][1]}**",
        "",
        f"清洗后内容指纹（SHA-256 前 16 位）：`{s['content_hash']}`"
        "（同参数重跑应完全一致，单元测试会断言）",
        "",
        "## 2. 清洗规则与影响行数",
        "",
        "| # | 规则 | 为什么要做 | 处理前 | 处理后 | 净变化 | 备注 |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | :--- |",
    ]
    for i, row in enumerate(rep.itertuples(index=False), 1):
        lines.append(
            f"| {i} | {row.rule} | {row.reason} | {row.rows_before:,} | "
            f"{row.rows_after:,} | {row.delta:+,} | {row.note} |"
        )
    lines += [
        "",
        "「净变化」为 0 的规则属于**就地改写**（去重、换算单位、置状态、封顶金额），"
        "它们不改变行数，但会改变数值——这正是最容易被漏掉的一类清洗。",
        "",
    ]
    lines += [
        "",
        "## 3. 关键决策：订单金额为什么不用源字段",
        "",
        "源系统的 `sales_orders.order_amount_raw` **与明细汇总对不上**，这是刻意注入的脏数据，",
        "也正是真实数仓里最常见的坑：口径字段没人维护，报出来的销售额永远是错的。",
        "",
        "| 指标 | 值 | 说明 |",
        "| :--- | ---: | :--- |",
        f"| 有偏差的订单 | {av['mismatch_rows']:,} 条（{av['mismatch_rate']:.1%}） | 任意金额差 ≥ 0.01 元 |",
        f"| 偏差 ≥ 1% 的订单 | {av['material_rows']:,} 条（{av['material_rate']:.1%}） | 真正需要修的这批 |",
        f"| 这批的中位偏差 | {av['material_median_dev_pct']:+.2%} | 源系统用了过期的价目表 |",
        f"| 金额加权偏差率 | {av['money_weighted_dev_pct']:.2%} | 偏差总额绝对值 / 订单总额，不受小单放大 |",
        f"| 受影响金额 | {av['amount_at_stake']:,.0f} 元 | 若直接用 raw 字段会算错的金额规模 |",
        "",
        "**结论**：销售额一律由 `order_items` 汇总计算（视图 `v_order_fact.order_amount` 已固化该口径）。",
        "这属于「在数据层解决、而不是靠提示词叮嘱模型」的问题——"
        "提示词能降低出错概率，但无法保证永远不出错。",
        "",
        "## 4. 其他被修正的数据缺陷",
        "",
        f"- **回款单位混用**：{s['payment_units']['converted_rows']} 条记录以「分」为单位，"
        f"最大单笔从 {s['payment_units']['max_before']:,.0f} 换算为 "
        f"{s['payment_units']['max_after']:,.0f} 元。",
        f"- **退款金额超过订单金额**：{s['refund_cap']['capped']} 条被按订单金额封顶，"
        f"退款总额 {s['refund_cap']['total_before']:,.0f} → {s['refund_cap']['total_after']:,.0f} 元。",
        f"- **未来单**：{s['future_orders']['flagged']} 条下单日期晚于数据截止日"
        f"（{', '.join(s['future_orders']['dates'])}），保留行但置为 `cancelled`。",
        f"- **孤儿记录**：剔除测试客户后产生 "
        f"{s['orphans']['orders_removed']} 条订单、{s['orphans']['items_removed']} 条明细悬空引用，已清理。",
        "",
        "## 5. 已知局限（如实说明）",
        "",
        "- 数据是合成的，**分布与真实 SaaS 业务有差距**，例如客户集中度、季节性强度；",
        "- 节假日表只标了大节，未覆盖调休；",
        "- 退款与回款的时间对齐是「下单后 N 天」的简化模型，未模拟跨月结算。",
        "",
        "这些局限不影响本项目要验证的东西（**口径注入 + 护栏 + 评测方法**），",
        "但真实落地时必须用真实数据重新校准。",
        "",
    ]
    return "\n".join(lines)
