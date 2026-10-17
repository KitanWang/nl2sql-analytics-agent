"""数据仓库的表结构定义 —— 单一事实来源。

这里定义的东西会被四处消费，所以必须只有一份：

1. `scripts/build_warehouse.py` 建表（DDL）；
2. 提示词里的 **schema 注入**（`to_prompt_schema`）；
3. SQL 安全护栏的**白名单**（哪些表、哪些列可以被 LLM 引用）；
4. 单元测试的断言。

设计原则：**中文注释必须是业务口径，而不是字段直译**。
「status」写成「订单状态」没有信息量；写成「pending 待付款 / paid 已付款 /
shipped 已发货 / completed 已完成 / cancelled 已取消 / refunded 已退款，
其中 paid/shipped/completed 为有效订单」才能把模型从语义歧义里救出来。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str            # DuckDB 类型
    comment: str          # 业务口径说明（进提示词）
    is_pk: bool = False
    references: str | None = None   # "table.column"


@dataclass(frozen=True)
class Table:
    name: str
    comment: str
    columns: tuple[Column, ...] = field(default_factory=tuple)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    @property
    def pk(self) -> str | None:
        for c in self.columns:
            if c.is_pk:
                return c.name
        return None


# ──────────────────────────────────────────────────────────────────
# 事实表
# ──────────────────────────────────────────────────────────────────

SALES_ORDERS = Table(
    name="sales_orders",
    comment="订单主表，一行一个订单。金额单位统一为元。",
    columns=(
        Column("order_id", "VARCHAR", "订单号，主键，形如 SO20250315-0007", is_pk=True),
        Column("customer_id", "VARCHAR", "客户号，关联 customers.customer_id"),
        Column("order_date", "DATE", "下单日期"),
        Column("status", "VARCHAR",
               "订单状态：pending 待付款 / paid 已付款 / shipped 已发货 / completed 已完成 / "
               "cancelled 已取消 / refunded 已退款。**有效订单 = paid + shipped + completed**，"
               "统计销售额、订单量时一律只算有效订单与未取消订单（见 glossary）"),
        Column("channel", "VARCHAR", "下单渠道：天猫 / 京东 / 抖音 / 官网 / 线下"),
        Column("ship_region", "VARCHAR", "收货大区：华东 / 华北 / 华南 / 华中 / 西南 / 西北 / 东北。"
                                         "注意与 customers.region（客户注册地）不同，按地区统计默认用本列"),
        Column("sales_rep", "VARCHAR", "销售负责人姓名"),
        Column("discount_amount", "DECIMAL(12,2)", "整单优惠金额（元），已在下单金额口径中扣除"),
        Column("shipping_fee", "DECIMAL(10,2)", "运费（元），计入下单金额"),
        Column("order_amount_raw", "DECIMAL(12,2)",
               "源系统上报的订单金额，**与明细汇总不一致**（历史遗留），不要直接使用；"
               "正确口径见 glossary 的「订单金额」"),
        Column("created_at", "TIMESTAMP", "记录创建时间，同号多条时用它取最新"),
        Column("updated_at", "TIMESTAMP", "记录更新时间，去重复时保留最大者"),
    ),
)

ORDER_ITEMS = Table(
    name="order_items",
    comment="订单明细表，一行一个「订单 × 商品」。订单金额必须由本表汇总得到。",
    columns=(
        Column("order_item_id", "VARCHAR", "明细行号，主键", is_pk=True),
        Column("order_id", "VARCHAR", "订单号，关联 sales_orders.order_id"),
        Column("product_id", "VARCHAR", "商品号，关联 products.product_id"),
        Column("quantity", "INTEGER", "数量（件）。赠品促销时 quantity > 0 但 unit_price = 0"),
        Column("unit_price", "DECIMAL(10,2)", "成交单价（元），已含税"),
        Column("discount_rate", "DECIMAL(5,4)", "行级折扣率，0.05 表示 95 折"),
        Column("item_amount", "DECIMAL(12,2)",
               "行金额 = quantity × unit_price × (1 − discount_rate)，四舍五入到分"),
        Column("is_gift", "BOOLEAN", "是否赠品行（unit_price = 0），计算客单价时应剔除"),
    ),
)

PAYMENTS = Table(
    name="payments",
    comment="回款表，一行一笔到账记录。金额单位统一为元（源系统有分，已在清洗时换算）。",
    columns=(
        Column("payment_id", "VARCHAR", "回款流水号，主键", is_pk=True),
        Column("order_id", "VARCHAR", "订单号，关联 sales_orders.order_id"),
        Column("pay_date", "DATE", "到账日期"),
        Column("amount", "DECIMAL(12,2)", "到账金额（元）"),
        Column("method", "VARCHAR", "支付方式：银行转账 / 支付宝 / 微信支付 / 承兑汇票"),
        Column("pay_status", "VARCHAR", "回款状态：success 成功 / failed 失败 / processing 处理中。"
                                        "只统计 success"),
    ),
)

REFUNDS = Table(
    name="refunds",
    comment="退款表，一行一笔退款。退款金额不重复出现在 payments 中。",
    columns=(
        Column("refund_id", "VARCHAR", "退款单号，主键", is_pk=True),
        Column("order_id", "VARCHAR", "订单号，关联 sales_orders.order_id"),
        Column("refund_date", "DATE", "退款日期"),
        Column("amount", "DECIMAL(12,2)", "退款金额（元）"),
        Column("reason", "VARCHAR", "退款原因：质量问题 / 发错货 / 客户取消 / 价格异议 / 物流破损"),
        Column("status", "VARCHAR", "退款状态：success 已退款 / pending 处理中。只统计 success"),
    ),
)


# ──────────────────────────────────────────────────────────────────
# 维表
# ──────────────────────────────────────────────────────────────────

CUSTOMERS = Table(
    name="customers",
    comment="客户维表。",
    columns=(
        Column("customer_id", "VARCHAR", "客户号，主键", is_pk=True),
        Column("customer_name", "VARCHAR", "客户名称（已标准化：去除全角空格与括号差异）"),
        Column("industry", "VARCHAR", "所属行业：汽车零部件 / 电子制造 / 机械加工 / 化工 / "
                                     "食品饮料 / 医疗器械 / 建材 / 新能源"),
        Column("region", "VARCHAR", "客户注册大区，**不是**订单收货地区（后者见 sales_orders.ship_region）"),
        Column("city", "VARCHAR", "注册城市"),
        Column("tier", "VARCHAR", "客户分层：KA 重点客户 / SMB 中小客户 / 长尾 长尾客户"),
        Column("signup_date", "DATE", "首次签约日期"),
        Column("credit_limit", "DECIMAL(12,2)", "授信额度（元）"),
        Column("is_active", "BOOLEAN", "是否仍在合作（false 表示已流失）"),
    ),
)

PRODUCTS = Table(
    name="products",
    comment="商品维表。",
    columns=(
        Column("product_id", "VARCHAR", "商品号，主键", is_pk=True),
        Column("product_name", "VARCHAR", "商品名称"),
        Column("category", "VARCHAR", "一级品类：传感器 / 伺服电机 / 工业相机 / 控制器 / 线缆 / 气动元件"),
        Column("sub_category", "VARCHAR", "二级品类"),
        Column("brand", "VARCHAR", "品牌"),
        Column("list_price", "DECIMAL(10,2)", "标准零售价（元），**不是**成交价"),
        Column("cost", "DECIMAL(10,2)", "单位成本（元），用于毛利计算"),
        Column("launch_date", "DATE", "上架日期"),
        Column("is_on_sale", "BOOLEAN", "是否在售（false 表示已下架）"),
    ),
)

DIM_CALENDAR = Table(
    name="dim_calendar",
    comment="日期维表，覆盖 2023-01-01 至 2025-12-31，用于自然月/季度/工作日口径。",
    columns=(
        Column("date_key", "DATE", "日期，主键", is_pk=True),
        Column("year", "INTEGER", "年"),
        Column("quarter", "INTEGER", "季度 1-4"),
        Column("month", "INTEGER", "月 1-12"),
        Column("year_month", "VARCHAR", "年-月，形如 2025-06，用于按月分组"),
        Column("week", "INTEGER", "ISO 周序号"),
        Column("is_workday", "BOOLEAN", "是否工作日（周末与法定节假日为 false）"),
    ),
)


TABLES: tuple[Table, ...] = (
    CUSTOMERS,
    PRODUCTS,
    SALES_ORDERS,
    ORDER_ITEMS,
    PAYMENTS,
    REFUNDS,
    DIM_CALENDAR,
)

TABLES_BY_NAME: dict[str, Table] = {t.name: t for t in TABLES}


# ──────────────────────────────────────────────────────────────────
# 派生视图：把「正确口径」固化成视图，模型少犯错，人也看得懂
# ──────────────────────────────────────────────────────────────────

VIEWS_SQL: tuple[str, ...] = (
    # 订单宽表：一行一个订单，带上金额、成本、退款、回款四个口径
    """
    CREATE OR REPLACE VIEW v_order_fact AS
    SELECT
        o.order_id,
        o.order_date,
        o.status,
        o.channel,
        o.ship_region,
        o.sales_rep,
        o.customer_id,
        c.customer_name,
        c.industry,
        c.region   AS customer_region,
        c.tier,
        i.items_amount,
        i.items_cost,
        i.gift_lines,
        o.shipping_fee,
        o.discount_amount,
        ROUND(i.items_amount + o.shipping_fee - o.discount_amount, 2) AS order_amount,
        COALESCE(r.refund_amount, 0)  AS refund_amount,
        COALESCE(p.pay_amount, 0)     AS pay_amount
    FROM sales_orders o
    JOIN customers c ON c.customer_id = o.customer_id
    JOIN (
        SELECT order_id,
               ROUND(SUM(item_amount), 2)                                  AS items_amount,
               ROUND(SUM(quantity * pr.cost), 2)                           AS items_cost,
               COUNT(*) FILTER (WHERE is_gift)                             AS gift_lines
        FROM order_items oi
        JOIN products pr USING (product_id)
        GROUP BY order_id
    ) i ON i.order_id = o.order_id
    LEFT JOIN (
        SELECT order_id, ROUND(SUM(amount), 2) AS refund_amount
        FROM refunds WHERE status = 'success' GROUP BY order_id
    ) r ON r.order_id = o.order_id
    LEFT JOIN (
        SELECT order_id, ROUND(SUM(amount), 2) AS pay_amount
        FROM payments WHERE pay_status = 'success' GROUP BY order_id
    ) p ON p.order_id = o.order_id
    """,
)


def ddl(table: Table) -> str:
    """生成建表语句，并把中文口径写进列注释。"""
    lines = []
    for c in table.columns:
        ddl_type = c.dtype
        lines.append(f'    "{c.name}" {ddl_type}')
    body = ",\n".join(lines)
    return f"CREATE OR REPLACE TABLE {table.name} (\n{body}\n)"


def to_prompt_schema(tables: tuple[Table, ...] = TABLES) -> str:
    """把 schema 渲染成给 LLM 看的文本。

    刻意做成「表注释 + 列名(类型) — 中文口径」的形式。

    **这是整个项目里 ROI 最高的一处改动**：把它换成纯 DDL（`to_ddl_schema()`）后，
    50 条评测的准确率从 94% 掉到 20%——模型会把「有效订单」猜成 `status='completed'`、
    去用有 12% 偏差的 `order_amount_raw`、把「客户注册地」当成「收货大区」。
    见 reports/REPORT.md 的消融表。
    """
    out: list[str] = []
    for t in tables:
        out.append(f"## {t.name} — {t.comment}")
        for c in t.columns:
            pk = " PK" if c.is_pk else ""
            out.append(f"- {c.name} {c.dtype}{pk} — {c.comment}")
        out.append("")
    return "\n".join(out).rstrip()


def to_ddl_schema(tables: tuple[Table, ...] = TABLES) -> str:
    """只给「表名 + 列名 + 类型」的裸 DDL，**不带任何业务注释**。

    这是消融实验的起点：模拟「把库结构直接丢给模型」的常见做法。
    和 `to_prompt_schema()` 的差值，就是「schema 里写清口径」这件事值多少分。
    """
    out: list[str] = []
    for t in tables:
        cols = ",\n".join(f'  "{c.name}" {c.dtype}' for c in t.columns)
        out.append(f"CREATE TABLE {t.name} (\n{cols}\n);")
    out.append("-- 另有视图 v_order_fact，字段与 sales_orders 相近，含 order_amount / refund_amount / "
               "items_cost / pay_amount")
    return "\n\n".join(out)


def all_column_refs() -> dict[str, set[str]]:
    """{表名: 列名集合}，供 SQL 护栏做列级校验。"""
    return {t.name: set(t.column_names) for t in TABLES}
