"""提示词 —— 项目里唯一「改了会影响准确率」的文本资产。

每一段提示词都对应一个可验证的设计决定，写在下面的注释里。
改动这里的任何一句，都应该跑一次 `scripts/run_eval.py`，看准确率怎么变——
`reports/REPORT.md` 的消融表就是这么做出来的。
"""

from __future__ import annotations

from dataclasses import dataclass

from nl2sql.schema import to_prompt_schema
from nl2sql.semantics.glossary import glossary_prompt


# ──────────────────────────────────────────────────────────────────
# 系统提示词
# ──────────────────────────────────────────────────────────────────

SYSTEM_SQL = """你是一名资深数据分析师，负责把业务人员的自然语言问题翻译成 **DuckDB SQL**。

## 你的工作流
1. 先判断问题要哪个指标、哪个时间范围、哪个维度；
2. 到「业务口径」里找到该指标的定义，**严格按定义计算**；
3. 生成一条只读 SELECT 查询。

## 硬性约束（任何一条不满足，查询都会被拒绝）
- 只能生成 **一条 SELECT**（含 WITH 的 CTE），不能有 INSERT/UPDATE/DELETE/CREATE/DROP/COPY/ATTACH/PRAGMA；
- 只能查询下面列出的表与视图，**不要**编造表名或列名；
- 禁止 `SELECT *`；明细查询必须带 `LIMIT`（建议 200）；
- 禁止无 ON 条件的 JOIN / CROSS JOIN；
- 只能使用常规分析函数，禁止 `read_csv`、`read_parquet`、`glob` 等文件或网络函数；
- 日期统一用 `DATE` 类型比较，月份用 `date_trunc('month', col)` 或 `dim_calendar.year_month`。

## 输出格式
只输出 SQL 本身，放在 ```sql 代码块里。不要解释，不要举例，不要输出多个方案。

## 数据库表结构
{schema}

## 业务口径（必须遵守，优先级高于你的常识）
{glossary}
"""


SYSTEM_SQL_NOCONTEXT = """你是一名资深数据分析师，把自然语言问题翻译成 DuckDB SQL。
只输出 SQL 本身，放在 ```sql 代码块里。

## 数据库表结构
{schema}
"""


# 少样本示例：数量刻意控制在 3 条。
# 做过对比（见 experiments/few_shot_size.py）：提示词里塞 8 条示例在 50 条评测集上
# 并没有更好，反而因为「示例长度 → 上下文里口径被稀释」导致口径类错误上升。
FEW_SHOT_EXAMPLES: tuple[dict[str, str], ...] = (
    {
        "question": "2025 年 6 月华东区的有效销售额是多少？",
        "sql": """SELECT ROUND(SUM(order_amount), 2) AS sales_amount
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
  AND ship_region = '华东'
  AND order_date >= DATE '2025-06-01'
  AND order_date <  DATE '2025-07-01'""",
        "why": "用视图拿口径；显式列出有效状态；月份用左闭右开区间而不是 strftime",
    },
    {
        "question": "销售额最高的 5 个品类",
        "sql": """SELECT p.category,
       ROUND(SUM(f.order_amount), 2) AS sales_amount
FROM v_order_fact f
JOIN order_items oi ON oi.order_id = f.order_id
JOIN products p ON p.product_id = oi.product_id
WHERE f.status IN ('paid', 'shipped', 'completed')
GROUP BY p.category
ORDER BY sales_amount DESC, p.category
LIMIT 5""",
        "why": "品类要下钻到明细再连商品；并列时二次排序保证结果稳定",
    },
    {
        "question": "最近 30 天每个渠道的订单数和客单价",
        "sql": """SELECT channel,
       COUNT(DISTINCT order_id) AS order_cnt,
       ROUND(SUM(order_amount - refund_amount) / NULLIF(COUNT(DISTINCT order_id), 0), 2) AS aov
FROM v_order_fact
WHERE status IN ('paid', 'shipped', 'completed')
  AND order_date >  (SELECT MAX(order_date) - INTERVAL 30 DAY FROM sales_orders)
GROUP BY channel
ORDER BY order_cnt DESC, channel""",
        "why": "基准日取数据里的最大订单日期；NULLIF 防除零；客单价按净销售额/订单数",
    },
)


SYSTEM_FIX = """你是一名 SQL 排错专家。下面这条 SQL 未能通过校验或执行失败，请修复它。

## 修复要求
- 只修问题本身，**不要重写整个查询**；
- 必须继续满足：单条 SELECT、只查白名单表、无 SELECT *、明细查询带 LIMIT；
- 只输出修复后的 SQL（```sql 代码块），不要解释。

## 可用的表与视图
{tables}

## 常见错误对照
| 错误 | 正确做法 |
| :--- | :--- |
| `column "xxx" does not exist` | 检查列是否在 schema 里；金额用 `v_order_fact.order_amount` |
| `Binder Error: Referenced column not found` | JOIN 后用表别名限定列名 |
| `Conversion Error` | 日期用 `DATE '2025-01-01'`，别用字符串 |
| `Parser Error` | 检查括号与引号是否配对 |
"""


SYSTEM_ANSWER = """你是数据分析师，把 SQL 查询结果解读成给业务人员的答复。

## 要求
- **先给结论**，用一句话回答用户的问题（把关键数字放在这句话里）；
- 再给 1–3 条补充观察（对比、异常、值得注意的点）；
- 数字必须来自给定结果，**不要自己再算一遍**，也不要编造结果里没有的数；
- 如果结果为空或全为 NULL，直接说明「该口径下没有数据」，不要强行解读；
- 金额单位是元，超过一万时用「万元」表述并保留两位小数；
- 中文回答，控制在 150 字以内，不要输出 SQL，不要输出表格。
"""


# ──────────────────────────────────────────────────────────────────
# 提示词组装
# ──────────────────────────────────────────────────────────────────

@dataclass
class PromptBundle:
    messages: list[dict[str, str]]
    meta: dict


def build_sql_prompt(
    question: str,
    *,
    examples: tuple[dict[str, str], ...] = (),
    include_glossary: bool = True,
    schema_text: str | None = None,
    glossary_text: str | None = None,
) -> PromptBundle:
    """组装「生成 SQL」的完整消息列表。

    三种模式（用于消融实验）：
      - include_glossary=False, examples=()  → 零样本基线
      - include_glossary=True,  examples=()  → 只注入口径
      - include_glossary=True,  examples=... → 口径 + 少样本（推荐）
    """
    schema = schema_text if schema_text is not None else to_prompt_schema()
    if include_glossary:
        gloss = glossary_text if glossary_text is not None else glossary_prompt()
        system = SYSTEM_SQL.format(schema=schema, glossary=gloss)
    else:
        system = SYSTEM_SQL_NOCONTEXT.format(schema=schema)

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for ex in examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": f"```sql\n{ex['sql']}\n```"})
    messages.append({"role": "user", "content": question})

    return PromptBundle(messages=messages, meta={
        "mode": ("glossary+fewshot" if examples else "glossary") if include_glossary else "zero-shot",
        "n_examples": len(examples),
        "schema_chars": len(schema),
        "glossary_chars": len(glossary_text or "") if include_glossary else 0,
    })


def build_fix_prompt(
    question: str,
    bad_sql: str,
    error: str,
    *,
    attempt: int = 1,
    tables: str | None = None,
) -> PromptBundle:
    """组装「自我修复」的消息。

    注意把**原始问题**也带上：只给 SQL 和错误，模型经常把语义修歪
    （比如把「华东」改成「华南」只为让查询能跑通）。
    """
    tbl = tables or "customers, products, sales_orders, order_items, payments, refunds, dim_calendar, v_order_fact"
    return PromptBundle(messages=[
        {"role": "system", "content": SYSTEM_FIX.format(tables=tbl)},
        {"role": "user", "content": (
            f"## 原始问题\n{question}\n\n"
            f"## 当前 SQL（第 {attempt} 次尝试）\n```sql\n{bad_sql}\n```\n\n"
            f"## 报错 / 违规原因\n{error}\n\n"
            "请输出修复后的 SQL。"
        )},
    ], meta={"mode": "self-repair", "attempt": attempt})


def build_answer_prompt(question: str, sql: str, result_table: str) -> PromptBundle:
    return PromptBundle(messages=[
        {"role": "system", "content": SYSTEM_ANSWER},
        {"role": "user", "content": (
            f"## 用户问题\n{question}\n\n"
            f"## 执行结果\n{result_table}"
        )},
    ], meta={"mode": "answer"})
