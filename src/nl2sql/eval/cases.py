"""评测集 —— 50 条问答对，外加用 pandas 独立算出的标准答案。

## 为什么标准答案要用 pandas 独立算

如果用「SQL 跑出来的结果」当标准答案，那评测就变成了「测两条 SQL 是否等价」，
而真正要测的是「模型是否算对了业务问题」。这里用 pandas 从 Parquet 重新算一遍，
与 DuckDB 的实现路径完全不同——两边一致，才说明口径没写错。

## 题目设计

50 条按能力维度分层，**每层都有明确的考点**，评测时按维度归因：

| 组 | 数量 | 考什么 |
| :--- | ---: | :--- |
| A 基础指标 | 8 | 销售额/订单量/客户数，时间与地区筛选 |
| B 业务口径 | 10 | 有效订单、净销售额、客单价、毛利率、地区歧义、新客 |
| C 分组与排名 | 8 | GROUP BY、TOP N、并列处理 |
| D 时间序列 | 8 | 环比/同比、趋势、相对时间窗口 |
| E 多表关联 | 6 | 明细下钻、品类/品牌/行业维度 |
| F 比率与占比 | 6 | 退款率、复购率、占比、动销率 |
| G 边界与陷阱 | 4 | 空结果、否定条件、赠品、模糊时间 |

## 判分口径

`tolerance` 是相对误差（默认 0.5%）。理由：金额类问题的合理误差来自浮点与舍入，
0.5% 足以容纳，又不会放过「少算 8%」（那正是用错 order_amount_raw 的典型偏差）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from nl2sql.schema import TABLES  # noqa: F401  (仅供类型/文档引用)

EvalKind = Literal["scalar", "series", "rows"]

DATA_CUTOFF = "2025-12-31"


@dataclass
class EvalCase:
    id: str
    group: str
    question: str
    kind: EvalKind
    # 期望结果的规范化形式见 evaluator.normalize_result
    expected: Any
    order_sensitive: bool = False
    tolerance: float = 0.005
    notes: str = ""
    # 这道题考的口径（用于错误归因）
    glossary_terms: tuple[str, ...] = field(default_factory=tuple)
    # 期望用到的表（用于检查是否绕过了视图）
    expected_tables: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = {
            "id": self.id, "group": self.group, "question": self.question,
            "kind": self.kind, "expected": self.expected,
            "order_sensitive": self.order_sensitive, "tolerance": self.tolerance,
            "notes": self.notes, "glossary_terms": list(self.glossary_terms),
        }
        return d


# ──────────────────────────────────────────────────────────────────
# 用 pandas 计算标准答案
# ──────────────────────────────────────────────────────────────────

def _load(processed: Path) -> dict[str, pd.DataFrame]:
    return {
        "orders": pd.read_parquet(processed / "orders.parquet"),
        "items": pd.read_parquet(processed / "items.parquet"),
        "customers": pd.read_parquet(processed / "customers.parquet"),
        "products": pd.read_parquet(processed / "products.parquet"),
        "payments": pd.read_parquet(processed / "payments.parquet"),
        "refunds": pd.read_parquet(processed / "refunds.parquet"),
        "calendar": pd.read_parquet(processed / "calendar.parquet"),
    }


VALID_STATUS = ("paid", "shipped", "completed")


def build_ground_truth(processed: Path) -> pd.DataFrame:
    """构造一张「订单级事实表」，镜像 v_order_fact，但用 pandas 实现。

    刻意**逐列显式**实现，不复用 SQL 视图——这样两边独立，互相校验才有意义。
    """
    d = _load(processed)
    o, i, c, p = d["orders"], d["items"], d["customers"], d["products"]
    # 金额以明细汇总为准（与数仓口径一致）
    item_amt = i.groupby("order_id")["item_amount"].sum()
    cost = i.merge(p[["product_id", "cost"]], on="product_id", how="left")
    cost["line_cost"] = cost["quantity"] * cost["cost"]
    item_cost = cost.groupby("order_id")["line_cost"].sum()
    refund_amt = (
        d["refunds"][d["refunds"]["status"] == "success"]
        .groupby("order_id")["amount"].sum()
    )
    pay_amt = (
        d["payments"][d["payments"]["pay_status"] == "success"]
        .groupby("order_id")["amount"].sum()
    )

    f = o.merge(c[["customer_id", "customer_name", "industry", "region", "tier"]],
                on="customer_id", how="left", suffixes=("", "_cust"))
    f["items_amount"] = f["order_id"].map(item_amt).fillna(0.0)
    f["items_cost"] = f["order_id"].map(item_cost).fillna(0.0)
    f["order_amount"] = (f["items_amount"] + f["shipping_fee"]
                         - f["discount_amount"]).round(2)
    f["refund_amount"] = f["order_id"].map(refund_amt).fillna(0.0)
    f["pay_amount"] = f["order_id"].map(pay_amt).fillna(0.0)
    f["customer_region"] = f["region"]
    f = f.rename(columns={"region": "customer_region"})
    f["valid"] = f["status"].isin(VALID_STATUS)
    f["order_date"] = pd.to_datetime(f["order_date"])
    f["year_month"] = f["order_date"].dt.strftime("%Y-%m")
    return f


def build_cases(processed: Path) -> list[EvalCase]:
    f = build_ground_truth(processed)
    v = f[f["valid"]]
    cases: list[EvalCase] = []
    add = cases.append

    def money(x: float) -> float:
        return round(float(x), 2)

    # ══════════ A 基础指标（8）══════════
    add(EvalCase(
        "A01", "A 基础指标", "数据集里一共有多少有效订单？", "scalar",
        int(len(v)), notes="有效订单 = paid+shipped+completed",
        glossary_terms=("有效订单 / 有效销售额",),
    ))
    add(EvalCase(
        "A02", "A 基础指标", "2025 年 6 月的有效销售额是多少？", "scalar",
        money(v[v["year_month"] == "2025-06"]["order_amount"].sum()),
        notes="月度筛选 + 有效订单口径",
        glossary_terms=("订单金额 / 销售额 / GMV / 营收", "有效订单 / 有效销售额"),
    ))
    add(EvalCase(
        "A03", "A 基础指标", "华东区的有效销售额是多少？", "scalar",
        money(v[v["ship_region"] == "华东"]["order_amount"].sum()),
        notes="地区默认用收货大区",
        glossary_terms=("地区 / 大区",),
    ))
    add(EvalCase(
        "A04", "A 基础指标", "2025 年一共发了多少个订单（有效订单）？", "scalar",
        int((v["order_date"].dt.year == 2025).sum()),
        glossary_terms=("有效订单 / 有效销售额",),
    ))
    add(EvalCase(
        "A05", "A 基础指标", "2025 年有多少个不同的客户下过有效订单？", "scalar",
        int(v[v["order_date"].dt.year == 2025]["customer_id"].nunique()),
        glossary_terms=("客户数 / 活跃客户数",),
    ))
    add(EvalCase(
        "A06", "A 基础指标", "官网渠道今年（2025 年）的有效销售额是多少？", "scalar",
        money(v[(v["channel"] == "官网") & (v["order_date"].dt.year == 2025)]["order_amount"].sum()),
    ))
    add(EvalCase(
        "A07", "A 基础指标", "2025 年第一季度（1-3 月）的销售额是多少？", "scalar",
        money(v[(v["order_date"] >= "2025-01-01") & (v["order_date"] < "2025-04-01")]["order_amount"].sum()),
        notes="季度要用日期区间，不能用字符串比较",
    ))
    add(EvalCase(
        "A08", "A 基础指标", "一共有多少个客户在册（customers 表全量）？", "scalar",
        int(len(pd.read_parquet(processed / "customers.parquet"))),
        notes="明确问在册客户才用全表，不是有下单的客户数",
        glossary_terms=("客户数 / 活跃客户数",),
    ))

    # ══════════ B 业务口径（10）══════════
    add(EvalCase(
        "B01", "B 业务口径", "2025 年 6 月的销售额是多少？（不要用源系统的订单金额字段）", "scalar",
        money(v[v["year_month"] == "2025-06"]["order_amount"].sum()),
        notes="考：不能用 order_amount_raw（偏差 12%）",
        glossary_terms=("订单金额 / 销售额 / GMV / 营收",),
    ))
    add(EvalCase(
        "B02", "B 业务口径", "2025 年 6 月的净销售额是多少？", "scalar",
        money((v[v["year_month"] == "2025-06"]["order_amount"]
               - v[v["year_month"] == "2025-06"]["refund_amount"]).sum()),
        glossary_terms=("净销售额",),
    ))
    add(EvalCase(
        "B03", "B 业务口径",
        "2025 年 6 月的客单价是多少？（净销售额 / 有效订单数，净销售额 = 订单金额 − 退款）", "scalar",
        money((v[v["year_month"] == "2025-06"]["order_amount"].sum()
               - v[v["year_month"] == "2025-06"]["refund_amount"].sum())
              / max(1, v[v["year_month"] == "2025-06"]["order_id"].nunique())),
        tolerance=0.01,
        notes="客单价=净销售额/有效订单数",
        glossary_terms=("客单价（AOV）", "净销售额"),
    ))
    add(EvalCase(
        "B04", "B 业务口径", "2025 年整体毛利率是多少？（订单金额−商品成本−退款）／订单金额，0-1 小数", "scalar",
        round(float((v[v["order_date"].dt.year == 2025]["order_amount"]
                     - v[v["order_date"].dt.year == 2025]["items_cost"]
                     - v[v["order_date"].dt.year == 2025]["refund_amount"]).sum()
                    / v[v["order_date"].dt.year == 2025]["order_amount"].sum()), 4),
        tolerance=0.01,
        notes="毛利率=(订单金额-成本-退款)/订单金额",
        glossary_terms=("毛利率",),
    ))
    add(EvalCase(
        "B05", "B 业务口径", "在册客户里有几个的注册大区是华东？（按客户档案，不是按订单收货地区）", "scalar",
        int(pd.read_parquet(processed / "customers.parquet").query("region == '华东'")["customer_id"].nunique()),
        notes="明确说注册地 → 用 customers.region，不是 ship_region",
        glossary_terms=("地区 / 大区",),
    ))
    add(EvalCase(
        "B06", "B 业务口径", "2025 年 6 月的有效订单数是多少？", "scalar",
        int(v[v["year_month"] == "2025-06"]["order_id"].nunique()),
        glossary_terms=("有效订单 / 有效销售额",),
    ))
    # 退款有两个自然的月份口径：退款到账月（refunds.refund_date）与所属订单月（order_date）。
    # 两者差异极大（一笔 6 月订单可能 8 月才退），不写清楚就是评测集自己的缺陷。
    # 退款口径有个反直觉的坑：**退款订单的状态是 refunded，本身就不在「有效订单」里**。
    # 如果按 f.valid 过滤，会把退款 880 万漏掉 98%（只剩 completed 订单的小额售后退款）。
    # 所以这道题的分母必须是「6 月下单的全部订单」，而不是有效订单。
    jun_all = f[f["year_month"] == "2025-06"]
    add(EvalCase(
        "B07", "B 业务口径", "2025 年 6 月下单的所有订单，一共产生了多少退款金额？（含已退款订单）", "scalar",
        money(jun_all["refund_amount"].sum()),
        notes="按订单所属月份归属退款；只统计 refunds.status='success'。"
              "注意退款订单状态是 refunded，**不能用有效订单过滤**，否则漏掉 98%",
        glossary_terms=("退款率",),
    ))
    # 回款同样有两个月份口径：到账月（payments.pay_date）与所属订单月（order_date）。
    # 题面必须指定，否则模型选哪个都「有道理」——那是评测集的问题，不是模型的问题。
    pay = pd.read_parquet(processed / "payments.parquet")
    pay["pay_month"] = pd.to_datetime(pay["pay_date"]).dt.strftime("%Y-%m")
    jun_pay = pay[(pay["pay_status"] == "success") & (pay["pay_month"] == "2025-06")]
    add(EvalCase(
        "B08", "B 业务口径", "2025 年 6 月实际到账的回款总额是多少？（按到账日期）", "scalar",
        money(jun_pay["amount"].sum()),
        notes="按 payments.pay_date 的月份归属；只统计 pay_status='success'",
    ))
    add(EvalCase(
        "B09", "B 业务口径", "2025 年 4 月的有效销售额是多少？", "scalar",
        money(v[v["year_month"] == "2025-04"]["order_amount"].sum()),
        notes="同 A02 但换月，检查是否稳定（不是靠记忆）",
        glossary_terms=("订单金额 / 销售额 / GMV / 营收",),
    ))
    add(EvalCase(
        "B10", "B 业务口径", "全部订单（含取消和退款）的总金额是多少？", "scalar",
        money(f["order_amount"].sum()),
        tolerance=0.01,
        notes="明确说「含取消」→ 不套用有效订单口径，考的是**不被口径反噬**",
        glossary_terms=("有效订单 / 有效销售额",),
    ))

    # ══════════ C 分组与排名（8）══════════
    by_channel = (v.groupby("channel")["order_amount"].sum()
                  .sort_values(ascending=False).round(2))
    add(EvalCase(
        "C01", "C 分组与排名", "各渠道的销售额分别是多少？", "series",
        [[k, money(x)] for k, x in by_channel.items()],
        order_sensitive=True, notes="按销售额降序",
    ))
    by_region = (v.groupby("ship_region")["order_amount"].sum()
                 .sort_values(ascending=False).round(2))
    add(EvalCase(
        "C02", "C 分组与排名", "各大区的销售额排名，从高到低", "series",
        [[k, money(x)] for k, x in by_region.items()],
        order_sensitive=True,
        glossary_terms=("地区 / 大区", "销售排名 TOP N"),
    ))
    top_cust = (v.groupby("customer_name")["order_amount"].sum()
                .sort_values(ascending=False).head(5).round(2))
    add(EvalCase(
        "C03", "C 分组与排名", "销售额最高的 5 个客户是哪几个？", "series",
        [[k, money(x)] for k, x in top_cust.items()],
        order_sensitive=True,
        glossary_terms=("销售排名 TOP N",),
    ))
    by_tier = (v.groupby("tier")["order_amount"].sum().sort_values(ascending=False).round(2))
    add(EvalCase(
        "C04", "C 分组与排名", "不同客户分层（KA/SMB/长尾）的销售额", "series",
        [[k, money(x)] for k, x in by_tier.items()],
        order_sensitive=True,
    ))
    by_ind = (v.groupby("industry")["order_amount"].sum().sort_values(ascending=False).round(2))
    add(EvalCase(
        "C05", "C 分组与排名", "各行业客户的销售额，从高到低", "series",
        [[k, money(x)] for k, x in by_ind.items()],
        order_sensitive=True,
    ))
    by_rep = (v.groupby("sales_rep")["order_amount"].sum().sort_values(ascending=False).round(2))
    add(EvalCase(
        "C06", "C 分组与排名", "销售负责人业绩排名", "series",
        [[k, money(x)] for k, x in by_rep.items()],
        order_sensitive=True,
    ))
    # 品类需要下钻明细
    # 品类销售额有两种口径，必须二选一并在题面写清：
    #   ① v_order_fact 的订单金额按订单 × 品类展开（一单多品类会重复计入，偏大）
    #   ② 明细行金额 item_amount 汇总（准确，但漏掉运费与整单优惠）
    # 这两种口径的差值是 100%，不是舍入误差——所以题面必须指定，
    # 否则评测在测「猜口径」，而不是测能力。
    cat_item = (v.merge(pd.read_parquet(processed / "items.parquet")[["order_id", "product_id", "item_amount"]],
                        on="order_id")
                .merge(pd.read_parquet(processed / "products.parquet")[["product_id", "category"]],
                       on="product_id")
                .groupby("category")["item_amount"].sum().sort_values(ascending=False).round(2))
    add(EvalCase(
        "C07", "C 分组与排名", "各品类的销售额（按明细行金额汇总，不含运费与整单优惠），从高到低", "series",
        [[k, money(x)] for k, x in cat_item.items()],
        order_sensitive=True, tolerance=0.005,
        notes="明确按明细金额口径；必须下钻 order_items × products",
        expected_tables=("order_items", "products"),
    ))
    cnt_channel = dict(v.groupby("channel")["order_id"].nunique().items())
    add(EvalCase(
        "C08", "C 分组与排名", "每个渠道有多少个有效订单？", "series",
        [[k, int(cnt_channel[k])] for k in by_channel.index],
        order_sensitive=True,
    ))

    # ══════════ D 时间序列（8）══════════
    monthly = (v[v["order_date"].dt.year == 2025]
               .groupby("year_month")["order_amount"].sum().round(2))
    add(EvalCase(
        "D01", "D 时间序列", "2025 年每个月的销售额", "series",
        [[k, money(x)] for k, x in monthly.items()],
        order_sensitive=True,
    ))
    prev = (v[v["year_month"] == "2025-11"]["order_amount"].sum())
    cur = (v[v["year_month"] == "2025-12"]["order_amount"].sum())
    add(EvalCase(
        "D02", "D 时间序列", "2025 年 12 月的销售额环比增长率是多少？（与 2025 年 11 月比，0-1 小数）", "scalar",
        round(float((cur - prev) / prev), 4) if prev else 0.0,
        tolerance=0.02,
        notes="环比=与上月比",
        glossary_terms=("环比 / 同比",),
    ))
    add(EvalCase(
        "D03", "D 时间序列", "有效订单里最新的一个月是哪个月？它的销售额是多少？", "series",
        [[str(v["year_month"].max()), money(v[v["year_month"] == v["year_month"].max()]["order_amount"].sum())]],
        notes="「最新」= 有效订单的最大下单日期所在月（sales_orders 里还有 2026-03 的"
              "取消单，用全表 max 会得到一个没有销售额的月份——这个坑正是要考的）",
        glossary_terms=("时间范围",),
    ))
    add(EvalCase(
        "D04", "D 时间序列", "2025 年第三季度的销售额是多少？", "scalar",
        money(v[(v["order_date"] >= "2025-07-01") & (v["order_date"] < "2025-10-01")]["order_amount"].sum()),
    ))
    add(EvalCase(
        "D05", "D 时间序列", "2025 年下半年（7-12 月）的销售额是多少？", "scalar",
        money(v[(v["order_date"] >= "2025-07-01") & (v["order_date"] < "2026-01-01")]["order_amount"].sum()),
    ))
    q_sales = (v.assign(q=v["order_date"].dt.quarter, y=v["order_date"].dt.year)
               .query("y == 2025").groupby("q")["order_amount"].sum().round(2))
    add(EvalCase(
        "D06", "D 时间序列", "2025 年四个季度各自的销售额（用 1-4 表示季度）", "series",
        [[int(k), money(x)] for k, x in q_sales.items()],
        order_sensitive=True,
    ))
    daily = (v[(v["order_date"] >= "2025-06-01") & (v["order_date"] < "2025-07-01")]
             .groupby("order_date").size())
    add(EvalCase(
        "D07", "D 时间序列", "2025 年 6 月有多少天有有效订单？", "scalar",
        int(len(daily)),
        notes="按日期去重计数",
    ))
    yoy_base = (v[v["year_month"] == "2024-06"]["order_amount"].sum())
    add(EvalCase(
        "D08", "D 时间序列", "2025 年 6 月的销售额同比增长率是多少？（与 2024 年 6 月比，0-1 小数）", "scalar",
        round(float((v[v["year_month"] == "2025-06"]["order_amount"].sum() - yoy_base) / yoy_base), 4),
        tolerance=0.02,
        notes="同比=与去年同月比",
        glossary_terms=("环比 / 同比",),
    ))

    # ══════════ E 多表关联（6）══════════
    brand = (v.merge(pd.read_parquet(processed / "items.parquet")[["order_id", "product_id", "item_amount"]],
                     on="order_id")
             .merge(pd.read_parquet(processed / "products.parquet")[["product_id", "brand"]], on="product_id")
             .groupby("brand")["item_amount"].sum().sort_values(ascending=False).round(2))
    add(EvalCase(
        "E01", "E 多表关联", "各品牌的销售额排名（按明细行金额汇总）", "series",
        [[k, money(x)] for k, x in brand.head(10).items()],
        order_sensitive=True, tolerance=0.005,
        expected_tables=("order_items", "products"),
    ))
    douyin_brand = (v[v["channel"] == "抖音"]
                    .merge(pd.read_parquet(processed / "items.parquet")[["order_id", "product_id", "item_amount"]],
                           on="order_id")
                    .merge(pd.read_parquet(processed / "products.parquet")[["product_id", "brand"]], on="product_id")
                    .groupby("brand")["item_amount"].sum().sort_values(ascending=False).round(2))
    add(EvalCase(
        "E02", "E 多表关联", "抖音渠道各品牌的销售额（按明细行金额汇总）", "series",
        [[k, money(x)] for k, x in douyin_brand.items()],
        order_sensitive=True, tolerance=0.005,
        expected_tables=("order_items", "products"),
    ))
    # 销售 Top 商品：用行金额更准确（一单多品类时按订单金额分摊会失真）
    items = pd.read_parquet(processed / "items.parquet")
    prods = pd.read_parquet(processed / "products.parquet")
    item_with_valid = items.merge(v[["order_id"]], on="order_id")
    top_prod = (item_with_valid.groupby("product_id")["item_amount"].sum()
                .sort_values(ascending=False).head(5))
    top_prod_named = (top_prod.rename("amt").reset_index()
                      .merge(prods[["product_id", "product_name"]], on="product_id"))
    add(EvalCase(
        "E03", "E 多表关联", "销售额最高的 5 个商品及金额", "series",
        [[r.product_name, money(r.amt)] for r in top_prod_named.itertuples()],
        order_sensitive=True, tolerance=0.01,
        notes="按明细行金额汇总，不是订单金额",
        expected_tables=("order_items", "products"),
    ))
    # 客户行业 × 品类：先按明细筛出行业，再按品类汇总行金额
    oi = items.merge(v[["order_id", "industry"]], on="order_id")
    oi = oi.merge(prods[["product_id", "category"]], on="product_id")
    ind_cat = (oi[oi["industry"] == "汽车零部件"]
               .groupby("category")["item_amount"].sum().sort_values(ascending=False).round(2))
    add(EvalCase(
        "E04", "E 多表关联", "汽车零部件行业客户的各品类采购额排名（按明细金额，前三名）", "series",
        [[k, money(x)] for k, x in ind_cat.head(3).items()],
        order_sensitive=True, tolerance=0.005,
        expected_tables=("order_items", "products", "customers"),
    ))
    # 客户数 × 行业
    ind_cust = v.groupby("industry")["customer_id"].nunique().sort_values(ascending=False)
    add(EvalCase(
        "E05", "E 多表关联", "各行业有多少个客户下过单？", "series",
        [[k, int(x)] for k, x in ind_cust.items()],
        order_sensitive=True,
    ))
    # 客户维度下单笔数。注意这里**只能按客户分组**：早先误写成 (customer_name, channel)
    # 分组，得到的是「客户在某渠道的笔数」，最高只有 26——与真实答案（61）差了一倍多。
    # 这是评测集自身的 bug，不是模型错。
    cust_cnt = (v.groupby("customer_name")["order_id"].nunique()
                .sort_values(ascending=False).head(5))
    # 第 4、5 名同为 58 笔，谁排前面取决于二次排序键——这是「并列处理」的典型陷阱。
    # 判分不加 order_sensitive，否则答案正确也会因为顺序判错；
    # 想考排序的题已经在 C 组（C01/C02 等）用**无并列**的数据覆盖了。
    add(EvalCase(
        "E06", "E 多表关联", "哪 5 个客户的有效订单笔数最多？（只要客户名和笔数）", "series",
        [[k, int(x)] for k, x in cust_cnt.items()],
        order_sensitive=False,
        notes="第 4/5 名同分（并列），故不判顺序；它是「TOP N 集合正确」的考点",
        glossary_terms=("销售排名 TOP N",),
    ))

    # ══════════ F 比率与占比（6）══════════
    refund_rate = (v.groupby("industry").apply(
        lambda g: g["refund_amount"].sum() / g["order_amount"].sum(), include_groups=False)
        .sort_values(ascending=False).round(4))
    add(EvalCase(
        "F01", "F 比率与占比", "各行业的退款率是多少？（退款金额 / 销售额，0-1 小数）", "series",
        [[k, round(float(x), 4)] for k, x in refund_rate.items()],
        order_sensitive=True, tolerance=0.02,
        glossary_terms=("退款率",),
    ))
    share = (v.groupby("ship_region")["order_amount"].sum()
             / v["order_amount"].sum()).sort_values(ascending=False).round(4)
    add(EvalCase(
        "F02", "F 比率与占比", "各收货大区的销售额占比（0-1 的小数，不是百分数）", "series",
        [[k, round(float(x), 4)] for k, x in share.items()],
        order_sensitive=True, tolerance=0.02,
        glossary_terms=("地区 / 大区",),
    ))
    # 复购率：分母是「有下单的客户」。分 tier 后长尾层几乎全是单次客户，
    # 指标退化没有信息量，所以这里用整体口径。
    cnt_per_cust = v.groupby("customer_id")["order_id"].nunique()
    repeat_rate_all = float((cnt_per_cust >= 2).mean())
    add(EvalCase(
        "F03", "F 比率与占比", "复购率是多少？即下单次数达到 2 次及以上的客户占比（小数）", "scalar",
        round(repeat_rate_all, 4), tolerance=0.01,
        glossary_terms=("复购率",),
    ))
    # 动销率 = 有成交的商品数 / 在售商品数。
    # 注意分母只能取「在售」商品：全量商品里有已下架的，用它当分母会让指标失去意义。
    on_sale = pd.read_parquet(processed / "products.parquet").query("is_on_sale")
    sold_products = oi[oi["product_id"].isin(on_sale["product_id"])]["product_id"].nunique()
    add(EvalCase(
        "F04", "F 比率与占比", "在售商品的动销率是多少？（有成交的在售商品数 / 在售商品总数，小数）", "scalar",
        round(float(sold_products / len(on_sale)), 4),
        tolerance=0.02,
        glossary_terms=("动销 / 上架商品动销率",),
    ))
    paid_ratio = (v["pay_amount"].sum() / v["order_amount"].sum())
    add(EvalCase(
        "F05", "F 比率与占比", "有效订单的整体回款率是多少？（有效订单的回款金额 / 有效订单的订单金额，小数）", "scalar",
        round(float(paid_ratio), 4), tolerance=0.02,
    ))
    aov_chan = (v.groupby("channel").apply(
        lambda g: (g["order_amount"].sum() - g["refund_amount"].sum()) / g["order_id"].nunique(),
        include_groups=False).round(2))
    # 这一列与 B03 用完全相同的公式，保证「客单价」在两个粒度上口径一致
    add(EvalCase(
        "F06", "F 比率与占比", "哪个渠道的客单价最高？客单价是多少？（净销售额 / 有效订单数）", "series",
        [[str(aov_chan.idxmax()), money(aov_chan.max())]],
        tolerance=0.01,
        glossary_terms=("客单价（AOV）",),
    ))

    # ══════════ G 边界与陷阱（4）══════════
    # 与 B07 同理：退款订单状态是 refunded，**不过滤有效订单**，否则会漏掉大部分退款
    jan24 = f[(f["year_month"] == "2024-01") & (f["ship_region"] == "西北")]
    add(EvalCase(
        "G01", "G 边界与陷阱",
        "2024 年 1 月下单、收货大区为西北的订单，一共产生过多少退款金额？（含已退款订单）", "scalar",
        money(jan24["refund_amount"].sum()),
        notes=f"按订单归属月；全量订单（含 refunded 状态），共 {len(jan24)} 单",
    ))
    add(EvalCase(
        "G02", "G 边界与陷阱", "有多少笔有效订单的运费是 0？", "scalar",
        int((v["shipping_fee"] == 0).sum()),
    ))
    gift = items.merge(v[["order_id"]], on="order_id")
    add(EvalCase(
        "G03", "G 边界与陷阱", "有效订单里一共包含多少个赠品行？", "scalar",
        int(gift["is_gift"].sum()),
        notes="赠品行 unit_price=0，不参与客单价但计入动销",
    ))
    # 超界月份没有数据：SUM 返回 NULL、COALESCE 返回 0，两者都算对——
    # 关键是**不能返回别的月份的数据**，也不能报错。判分侧把 NULL 视同 0 处理。
    add(EvalCase(
        "G04", "G 边界与陷阱", "2026 年 1 月的销售额是多少？（数据截止 2025-12-31）", "scalar",
        0.0,
        notes="超界时间返回 0 或 NULL 均可，不能返回其他月份数据或报错",
        glossary_terms=("时间范围",),
    ))

    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids)), "评测集 id 重复"
    return cases


def dump_cases(cases: list[EvalCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.to_dict() for c in cases]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cases(path: Path) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in raw:
        r = dict(r)
        r["glossary_terms"] = tuple(r.get("glossary_terms") or ())
        r.setdefault("expected_tables", ())
        out.append(EvalCase(**r))
    return out
