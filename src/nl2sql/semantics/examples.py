"""少样本示例库 —— 检索器的候选池，也是「怎么问、怎么写」的对照表。

每条示例都是**在本仓库的 DuckDB 上真实跑通过的 SQL**（`tests/test_example_bank.py` 会逐条执行校验）。
示例按「能力维度」打标签，检索时用标签做加分（见 `retriever.BM25Retriever`）。

规模说明：**20 条**。这不是拍脑袋定的数——`experiments/few_shot_size.py`
测试过 5/10/20/40 条四种池子大小，20 条之后增益饱和（40 条时提示词长度翻倍，
口径类错误反而略微上升）。
"""

from __future__ import annotations

from nl2sql.semantics.retriever import Example

EXAMPLES: tuple[Example, ...] = (
    Example(
        question="2025 年 6 月的有效销售额是多少？",
        sql="""SELECT ROUND(SUM(order_amount), 2) AS sales_amount
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
  AND order_date >= DATE '2025-06-01'
  AND order_date < DATE '2025-07-01'""",
        why="口径类：用视图、显式有效状态、月份左闭右开",
        tags=("销售额", "时间窗口", "口径"),
    ),
    Example(
        question="上个月华东区的订单量",
        sql="""SELECT COUNT(DISTINCT order_id) AS order_cnt
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
  AND ship_region = '华东'
  AND date_trunc('month', order_date) = date_trunc('month', (SELECT MAX(order_date) FROM sales_orders) - INTERVAL 1 MONTH)""",
        why="相对时间以数据最大日期为基准，取上月；地区默认用收货大区",
        tags=("订单量", "环比", "地区", "时间窗口"),
    ),
    Example(
        question="各渠道的销售额和客单价",
        sql="""SELECT channel,
       ROUND(SUM(order_amount), 2) AS sales_amount,
       ROUND(SUM(order_amount - refund_amount) / NULLIF(COUNT(DISTINCT order_id), 0), 2) AS aov
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
GROUP BY channel
ORDER BY sales_amount DESC, channel""",
        why="分组聚合；客单价=净销售额/订单数；NULLIF 防除零",
        tags=("销售额", "客单价", "渠道", "分组"),
    ),
    Example(
        question="销售额最高的 10 个客户",
        sql="""SELECT customer_name,
       ROUND(SUM(order_amount), 2) AS sales_amount,
       COUNT(DISTINCT order_id) AS order_cnt
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
GROUP BY customer_name
ORDER BY sales_amount DESC, customer_name
LIMIT 10""",
        why="排名类：二次排序保证并列时结果稳定",
        tags=("销售额", "排名", "客户"),
    ),
    Example(
        question="各品类的毛利率",
        sql="""SELECT p.category,
       ROUND((SUM(f.order_amount - f.items_cost) - SUM(f.refund_amount)) / NULLIF(SUM(f.order_amount), 0), 4) AS gross_margin
FROM v_order_fact f
JOIN order_items oi ON oi.order_id = f.order_id
JOIN products p ON p.product_id = oi.product_id
WHERE f.status IN ('paid', 'shipped', 'completed')
GROUP BY p.category
ORDER BY gross_margin DESC, p.category""",
        why="毛利率口径：(订单金额−成本−退款)/订单金额；品类要下钻明细",
        tags=("毛利率", "品类", "口径", "分组"),
    ),
    Example(
        question="2025 年每个月销售额的环比增长率",
        sql="""WITH m AS (
    SELECT date_trunc('month', order_date) AS mth,
           ROUND(SUM(order_amount - refund_amount), 2) AS net_sales
    FROM v_order_fact
    WHERE status IN ('paid', 'shipped', 'completed')
      AND order_date >= DATE '2025-01-01'
      AND order_date <  DATE '2026-01-01'
    GROUP BY 1
)
SELECT strftime(mth, '%Y-%m') AS year_month,
       net_sales,
       ROUND((net_sales - LAG(net_sales) OVER (ORDER BY mth))
             / NULLIF(LAG(net_sales) OVER (ORDER BY mth), 0), 4) AS mom_growth
FROM m
ORDER BY mth""",
        why="环比用 LAG 窗口函数，不要用日期字符串减一",
        tags=("环比", "时间序列", "口径"),
    ),
    Example(
        question="近 30 天每天的订单数趋势",
        sql="""SELECT order_date,
       COUNT(DISTINCT order_id) AS order_cnt
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
  AND order_date > (SELECT MAX(order_date) - INTERVAL 30 DAY FROM sales_orders)
GROUP BY order_date
ORDER BY order_date""",
        why="趋势类：按日期分组；近 N 天以数据最大日期为基准",
        tags=("订单量", "趋势", "时间窗口"),
    ),
    Example(
        question="不同客户分层的复购率",
        sql="""WITH per_cust AS (
    SELECT tier, customer_id, COUNT(DISTINCT order_id) AS n_orders
    FROM v_order_fact
    WHERE status IN ('paid', 'shipped', 'completed')
    GROUP BY tier, customer_id
)
SELECT tier,
       ROUND(AVG(CASE WHEN n_orders >= 2 THEN 1.0 ELSE 0.0 END), 4) AS repeat_rate,
       COUNT(*) AS active_customers
FROM per_cust
GROUP BY tier
ORDER BY repeat_rate DESC""",
        why="复购率分母是「有下单的客户」；用 CTE 先算每客户订单数",
        tags=("复购率", "客户", "口径", "分组"),
    ),
    Example(
        question="各行业客户的退款率",
        sql="""SELECT industry,
       ROUND(SUM(refund_amount) / NULLIF(SUM(order_amount), 0), 4) AS refund_rate,
       ROUND(SUM(refund_amount), 2) AS refund_amount
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
GROUP BY industry
ORDER BY refund_rate DESC, industry""",
        why="退款率=退款金额/销售额，同周期同条件",
        tags=("退款率", "行业", "分组"),
    ),
    Example(
        question="2025 年第二季度各销售负责人的业绩排名",
        sql="""SELECT sales_rep,
       ROUND(SUM(order_amount), 2) AS sales_amount,
       COUNT(DISTINCT order_id) AS order_cnt
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
  AND order_date >= DATE '2025-04-01'
  AND order_date <  DATE '2025-07-01'
GROUP BY sales_rep
ORDER BY sales_amount DESC, sales_rep""",
        why="季度用显式日期区间；排名带二次排序",
        tags=("销售额", "排名", "时间窗口"),
    ),
    Example(
        question="在售商品的动销率",
        sql="""WITH sold AS (
    SELECT COUNT(DISTINCT oi.product_id) AS n_sold
    FROM v_order_fact f
    JOIN order_items oi ON oi.order_id = f.order_id
    WHERE f.status IN ('paid', 'shipped', 'completed')
)
SELECT (SELECT n_sold FROM sold) AS sold_products,
       COUNT(*) AS on_sale_products,
       ROUND((SELECT n_sold FROM sold) * 1.0 / NULLIF(COUNT(*), 0), 4) AS sell_through_rate
FROM products
WHERE is_on_sale""",
        why="动销率=有成交商品数/在售商品数",
        tags=("动销率", "商品", "口径"),
    ),
    Example(
        question="各收货大区的销售额占比",
        sql="""SELECT ship_region,
       ROUND(SUM(order_amount), 2) AS sales_amount,
       ROUND(SUM(order_amount) / NULLIF(SUM(SUM(order_amount)) OVER (), 0), 4) AS share
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
GROUP BY ship_region
ORDER BY sales_amount DESC""",
        why="占比用窗口函数除总计；避免再跑一次子查询",
        tags=("销售额", "占比", "地区"),
    ),
    Example(
        question="本月新增的首单客户数",
        sql="""WITH first_order AS (
    SELECT customer_id, MIN(order_date) AS first_date
    FROM v_order_fact
    WHERE status IN ('paid', 'shipped', 'completed')
    GROUP BY customer_id
)
SELECT COUNT(*) AS new_customers
FROM first_order
WHERE date_trunc('month', first_date) = date_trunc('month', (SELECT MAX(order_date) FROM sales_orders))""",
        why="首单客户用 MIN(order_date)，不是 signup_date",
        tags=("客户", "新客", "口径"),
    ),
    Example(
        question="销售额前 5 的商品及其所属品类",
        sql="""SELECT p.product_name,
       p.category,
       ROUND(SUM(oi.item_amount), 2) AS sales_amount,
       SUM(oi.quantity) AS units
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
JOIN v_order_fact f ON f.order_id = oi.order_id
WHERE f.status IN ('paid', 'shipped', 'completed')
GROUP BY p.product_name, p.category
ORDER BY sales_amount DESC, p.product_name
LIMIT 5""",
        why="商品维度要连明细；别名限定列名避免歧义",
        tags=("销售额", "排名", "商品"),
    ),
    Example(
        question="2025 年各月的新客数和老客数",
        sql="""WITH first_order AS (
    SELECT customer_id, MIN(order_date) AS first_date
    FROM v_order_fact
    WHERE status IN ('paid', 'shipped', 'completed')
    GROUP BY customer_id
),
orders AS (
    SELECT date_trunc('month', f.order_date) AS mth,
           f.customer_id,
           CASE WHEN date_trunc('month', fo.first_date) = date_trunc('month', f.order_date)
                THEN 'new' ELSE 'returning' END AS cust_type
    FROM v_order_fact f
    JOIN first_order fo ON fo.customer_id = f.customer_id
    WHERE f.status IN ('paid', 'shipped', 'completed')
      AND f.order_date >= DATE '2025-01-01'
      AND f.order_date <  DATE '2026-01-01'
)
SELECT strftime(mth, '%Y-%m') AS year_month,
       COUNT(DISTINCT CASE WHEN cust_type = 'new' THEN customer_id END) AS new_customers,
       COUNT(DISTINCT CASE WHEN cust_type = 'returning' THEN customer_id END) AS returning_customers
FROM orders
GROUP BY mth
ORDER BY mth""",
        why="新老客划分基于首单月份；条件计数用 CASE WHEN",
        tags=("客户", "新客", "时间序列"),
    ),
    Example(
        question="回款率最低的 5 个客户",
        sql="""SELECT customer_name,
       ROUND(SUM(pay_amount) / NULLIF(SUM(order_amount), 0), 4) AS collection_rate,
       ROUND(SUM(order_amount - pay_amount), 2) AS outstanding
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
GROUP BY customer_name
ORDER BY collection_rate ASC, customer_name
LIMIT 5""",
        why="回款率=回款/订单金额；升序取最低要显式 ASC",
        tags=("回款", "排名", "客户"),
    ),
    Example(
        question="各品牌在抖音渠道的销售额",
        sql="""SELECT p.brand,
       ROUND(SUM(oi.item_amount), 2) AS sales_amount
FROM order_items oi
JOIN products p ON p.product_id = oi.product_id
JOIN v_order_fact f ON f.order_id = oi.order_id
WHERE f.status IN ('paid', 'shipped', 'completed')
  AND f.channel = '抖音'
GROUP BY p.brand
ORDER BY sales_amount DESC, p.brand""",
        why="单一渠道筛选 + 品牌维度",
        tags=("销售额", "渠道", "品牌", "分组"),
    ),
    Example(
        question="工作日和非工作日的平均日销售额对比",
        sql="""SELECT c.is_workday,
       ROUND(SUM(f.order_amount) / NULLIF(COUNT(DISTINCT f.order_date), 0), 2) AS avg_daily_sales
FROM v_order_fact f
JOIN dim_calendar c ON c.date_key = f.order_date
WHERE f.status IN ('paid', 'shipped', 'completed')
GROUP BY c.is_workday
ORDER BY c.is_workday DESC""",
        why="用日期维表拿工作日标记；日均要除以天数不是订单数",
        tags=("销售额", "日历", "对比"),
    ),
    Example(
        question="下单客户的地区分布（按客户注册地）",
        sql="""SELECT customer_region,
       COUNT(DISTINCT customer_id) AS customer_cnt
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
GROUP BY customer_region
ORDER BY customer_cnt DESC, customer_region""",
        why="明确说注册地时用 customer_region 而不是 ship_region",
        tags=("客户", "地区", "口径"),
    ),
    Example(
        question="销售额对比上个月的同比和环比",
        sql="""WITH m AS (
    SELECT date_trunc('month', order_date) AS mth,
           ROUND(SUM(order_amount), 2) AS sales_amount
    FROM v_order_fact
    WHERE status IN ('paid', 'shipped', 'completed')
    GROUP BY 1
)
SELECT strftime(mth, '%Y-%m') AS year_month,
       sales_amount,
       ROUND((sales_amount - LAG(sales_amount) OVER (ORDER BY mth))
             / NULLIF(LAG(sales_amount) OVER (ORDER BY mth), 0), 4) AS mom,
       ROUND((sales_amount - LAG(sales_amount, 12) OVER (ORDER BY mth))
             / NULLIF(LAG(sales_amount, 12) OVER (ORDER BY mth), 0), 4) AS yoy
FROM m
ORDER BY mth""",
        why="同比是 LAG(...,12)，环比是 LAG(...,1)",
        tags=("同比", "环比", "时间序列"),
    ),
)

EXAMPLES_BY_QUESTION: dict[str, Example] = {e.question: e for e in EXAMPLES}
