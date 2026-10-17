"""业务口径（glossary）—— 把「人话指标」翻译成「SQL 口径」。

这是整个项目里最容易被低估的一块。

schema 注入只解决了「表里有什么」，但用户问的往往是「销售额」「客单价」「动销」这种
**公司内部有明确定义、且与直觉不一致**的词。

## 实测结论（与直觉相反，如实记录）

单看这一层，它在 schema 已经写清口径之后**没有测出稳定增益**：
50 条评测上 `schema` 与 `glossary` 两档互有胜负，prompt token 却翻了一倍。
详见 reports/REPORT.md 的消融表与「② 业务口径没有测出稳定增益」一节。

那为什么还保留它？因为它承载了两件 schema 做不到的事：

1. **跨字段的规则**：「客单价 = 净销售额 / 有效订单数」没法塞进单列注释；
2. **`wrong_pattern`**：每条口径都带一个「典型错误写法」正则，
   评测归因靠它把「算错了」细分成「用了 order_amount_raw」——
   **这是把业务知识变成可执行断言的唯一途径**，价值在可观测性而不在准确率。

每条口径都带 `wrong_pattern`：**不看口径时的典型错误写法**。
评测集的错误归因（见 reports/）会用到它，用于区分「口径缺失」与「其他错误」。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlossaryEntry:
    term: str
    definition: str
    sql_hint: str
    wrong_pattern: str = ""      # 典型错误写法的正则（评测归因用）
    examples: tuple[str, ...] = ()


GLOSSARY: tuple[GlossaryEntry, ...] = (
    GlossaryEntry(
        term="订单金额 / 销售额 / GMV / 营收",
        definition=(
            "一律使用视图 v_order_fact.order_amount，"
            "= 明细行金额合计 + 运费 − 整单优惠。**不要**用 sales_orders.order_amount_raw"
            "（源系统上报值与明细对不上，是历史遗留脏数据）。"
        ),
        sql_hint="SUM(f.order_amount)",
        wrong_pattern=r"order_amount_raw",
        examples=("上月销售额是多少", "华东区今年 GMV"),
    ),
    GlossaryEntry(
        term="有效订单 / 有效销售额",
        definition=(
            "status ∈ ('paid','shipped','completed') 的订单。"
            "统计销售额、订单量、客单价时默认只算有效订单；"
            "若用户明确要「全部订单」「含取消」，才放开这个限制。"
        ),
        sql_hint="WHERE f.status IN ('paid','shipped','completed')",
        wrong_pattern=r"NOT\s+IN\s*\(\s*'cancelled'\s*\)\s*(?!.*refunded)",
        examples=("本月有效销售额", "统计销售额"),
    ),
    GlossaryEntry(
        term="净销售额",
        definition="销售额 − 退款金额（refunds.status='success'）。即 v_order_fact.order_amount − refund_amount。",
        sql_hint="SUM(f.order_amount - f.refund_amount)",
        examples=("上月净销售额",),
    ),
    GlossaryEntry(
        term="客单价（AOV）",
        definition=(
            "净销售额 / 有效订单数（不是明细行数，也不是客户数）。"
            "赠品行（is_gift=true，unit_price=0）不参与单价计算，但已经包含在订单金额里。"
        ),
        sql_hint="SUM(order_amount - refund_amount) / COUNT(DISTINCT order_id)",
        examples=("上个月客单价多少", "哪个渠道客单价最高"),
    ),
    GlossaryEntry(
        term="毛利率",
        definition=(
            "(订单金额 − 商品成本 − 退款金额) / 订单金额。"
            "商品成本取 v_order_fact.items_cost（= Σ quantity × products.cost）。"
            "**不要**用 list_price 或 order_amount_raw 反推。"
        ),
        sql_hint="(SUM(order_amount - items_cost - refund_amount)) / SUM(order_amount)",
        examples=("各品类毛利率", "毛利率环比"),
    ),
    GlossaryEntry(
        term="客户数 / 活跃客户数",
        definition=(
            "默认指「统计周期内有有效订单的客户数」，"
            "即 COUNT(DISTINCT customer_id)，与 customers.is_active 无关；"
            "只有明确说「在册客户」才用 customers 全表计数。"
        ),
        sql_hint="COUNT(DISTINCT f.customer_id)",
        examples=("本月有多少客户下单", "新客数"),
    ),
    GlossaryEntry(
        term="新客 / 首单客户",
        definition=(
            "在该统计周期内完成**首笔有效订单**的客户。"
            "首单日期 = MIN(order_date) OVER (PARTITION BY customer_id)，"
            "注意不能用 customers.signup_date 代替（签约到首单常隔数月）。"
        ),
        sql_hint="首单日期 = MIN(order_date) per customer",
        wrong_pattern=r"signup_date",
        examples=("本月新增多少首单客户",),
    ),
    GlossaryEntry(
        term="复购率",
        definition=(
            "统计周期内有效订单数 ≥ 2 的客户数 / 该周期内有效订单客户数。"
            "分母是「有下单的客户」不是「全部客户」。"
        ),
        sql_hint="复购客户数 / 下单客户数",
        examples=("本季度复购率",),
    ),
    GlossaryEntry(
        term="地区 / 大区",
        definition=(
            "默认指**收货大区** sales_orders.ship_region（= v_order_fact.ship_region）。"
            "只有当用户说「客户注册地」「客户所在区域」时才用 customers.region"
            "（= v_order_fact.customer_region）。两者经常不一致，是最常见的口径坑。"
        ),
        sql_hint="f.ship_region（默认）/ f.customer_region（注册地）",
        examples=("华东区销售额", "各区域销售额对比"),
    ),
    GlossaryEntry(
        term="环比 / 同比",
        definition=(
            "环比：与上一个等长自然周期比（月对月、季对季、年对年）。"
            "同比：与去年同一自然周期比。"
            "用 LAG(..., 1) OVER (ORDER BY 期间) 与 LAG(..., n) OVER (...) 实现，"
            "期间用 dim_calendar.year_month / quarter 生成，不要用字符串截取日期。"
        ),
        sql_hint="LAG(metric, 1) OVER (ORDER BY period)",
        wrong_pattern=r"strftime\([^)]*\)\s*\)?\s*-\s*1",
        examples=("销售额环比增长率",),
    ),
    GlossaryEntry(
        term="时间范围",
        definition=(
            "「本月」「上月」「本季度」「今年」以**数据中最新的订单日期**为基准日（当前数据截至 2025-12-31），"
            "不要用 CURRENT_DATE（运行环境的时间与数据集不一致，会得到空结果）。"
            "「近 30 天」= 最新日期往前推 30 天（含端点）。"
        ),
        sql_hint="基准日 = (SELECT MAX(order_date) FROM sales_orders)",
        wrong_pattern=r"CURRENT_DATE|now\(\)|CURRENT_TIMESTAMP",
        examples=("上月销售趋势", "近 30 天销售额"),
    ),
    GlossaryEntry(
        term="动销 / 上架商品动销率",
        definition=(
            "统计周期内有有效订单明细的商品数 / 在售商品总数（products.is_on_sale=true）。"
            "赠品行是否计入动销：默认计入（有成交即有出库）。"
        ),
        sql_hint="COUNT(DISTINCT product_id) FILTER(有效) / COUNT(*) FILTER(is_on_sale)",
        examples=("本月商品动销率",),
    ),
    GlossaryEntry(
        term="销售排名 TOP N",
        definition=(
            "默认按净销售额降序。并列时按 order_id 或名称二次排序，"
            "保证结果稳定（否则同分记录顺序随机，评测不可复现）。"
        ),
        sql_hint="ORDER BY metric DESC, dim_key",
        examples=("销售额前 10 的客户",),
    ),
    GlossaryEntry(
        term="退款率",
        definition="退款金额合计 / 销售额（均为同一统计周期、同一筛选条件）。",
        sql_hint="SUM(refund_amount) / SUM(order_amount)",
        examples=("各品类退款率",),
    ),
)


def glossary_prompt(entries: tuple[GlossaryEntry, ...] = GLOSSARY) -> str:
    """渲染成注入提示词的文本。"""
    lines = []
    for e in entries:
        lines.append(f"- **{e.term}**：{e.definition}")
        lines.append(f"  - 推荐写法：`{e.sql_hint}`")
    return "\n".join(lines)


def by_term(name: str) -> GlossaryEntry | None:
    for e in GLOSSARY:
        if e.term == name:
            return e
    return None


def match_terms(question: str) -> list[GlossaryEntry]:
    """按关键词粗匹配命中的口径条目。

    这里刻意用最朴素的关键词包含判断，而不是语义检索——因为口径是**确定性规则**，
    模型自己挑哪些口径进上下文反而更稳（见 ablation 中 `glossary_mode=auto`）。
    这个函数只在「全部口径都注入」的重基线里做对照用。
    """
    hits = []
    for e in GLOSSARY:
        keys = [k.strip() for k in e.term.replace("（", "(").replace("）", ")").split("/")]
        keys = [k.split("(")[0].strip() for k in keys if k.strip()]
        if any(k and k in question for k in keys):
            hits.append(e)
    return hits
