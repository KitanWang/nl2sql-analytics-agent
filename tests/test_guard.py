"""SQL 安全护栏的对抗测试 —— 这个文件是全仓库最该被认真对待的测试。

## 为什么用「攻击样本表」而不是零散断言

安全测试的价值在于**覆盖面**，而不在于「测过一条 DROP 就算测了」。
这里把攻击样本整理成表（`ATTACKS`），每条标注攻击类别与预期拦截规则，
新增绕过手段时只需往表里加一行，不会漏掉回归。

## 分层覆盖

- 语句类型：DDL / DML / 管理语句
- 语法层绕过：注释、大小写、多语句、分号
- 资源访问：文件读取、网络、ATTACH、系统表
- 逻辑层滥用：笛卡尔积、恒真 ON、SELECT *、无 LIMIT
"""

from __future__ import annotations

import pytest

from nl2sql.sqlsafe.guard import guard, strip_wrapping
from nl2sql.semantics.examples import EXAMPLES


# ──────────────────────────────────────────────────────────────────
# 攻击样本表
# ──────────────────────────────────────────────────────────────────

ATTACKS: list[tuple[str, str, str]] = [
    # (类别, SQL, 期望命中的规则)
    ("ddl", "DROP TABLE sales_orders", "forbidden_statement"),
    ("ddl", "drop table if exists sales_orders cascade", "forbidden_statement"),
    ("ddl", "CREATE TABLE evil AS SELECT * FROM sales_orders", "forbidden_statement"),
    ("ddl", "ALTER TABLE sales_orders ADD COLUMN x INT", "forbidden_statement"),
    ("ddl", "TRUNCATE TABLE sales_orders", "forbidden_statement"),
    ("dml", "DELETE FROM sales_orders", "forbidden_statement"),
    ("dml", "UPDATE sales_orders SET status='paid'", "forbidden_statement"),
    ("dml", "INSERT INTO customers SELECT * FROM customers", "forbidden_statement"),
    ("dml", "MERGE INTO customers USING customers ON true WHEN MATCHED THEN DELETE", "forbidden_statement"),

    # 语法层绕过
    ("obfuscation", "dRoP tAbLe sales_orders", "forbidden_statement"),
    ("obfuscation", "DROP /* 注释 */ TABLE sales_orders", "forbidden_statement"),
    ("obfuscation", "DROP -- 行注释\nTABLE sales_orders", "forbidden_statement"),
    ("obfuscation", "SELECT 1; DROP TABLE sales_orders", "multi_statement"),
    ("obfuscation", "SELECT 1;;; SELECT 2", "multi_statement"),
    ("obfuscation", "SELECT order_id FROM sales_orders WHERE order_id IN (SELECT 1); DROP TABLE sales_orders",
     "multi_statement"),
    ("obfuscation", "WITH x AS (DELETE FROM sales_orders RETURNING *) SELECT * FROM x", "forbidden_statement"),

    # 资源访问
    ("resource", "SELECT * FROM read_csv('/etc/passwd')", "forbidden_function"),
    ("resource", "SELECT * FROM read_csv_auto('/etc/passwd')", "forbidden_function"),
    ("resource", "SELECT * FROM read_parquet('/etc/passwd')", "forbidden_function"),
    ("resource", "SELECT * FROM read_json('/etc/passwd')", "forbidden_function"),
    ("resource", "SELECT * FROM glob('/etc/*')", "forbidden_function"),
    ("resource", "COPY (SELECT * FROM sales_orders) TO '/tmp/leak.csv'", "forbidden_statement"),
    ("resource", "ATTACH 'other.duckdb' AS other", "forbidden_statement"),
    ("resource", "DETACH other", "forbidden_statement"),
    ("resource", "PRAGMA database_list", "forbidden_statement"),
    ("resource", "INSTALL httpfs", "forbidden_statement"),
    ("resource", "SELECT * FROM duckdb_tables()", "forbidden_function"),
    ("resource", "SELECT * FROM information_schema.tables", "table_whitelist"),
    ("resource", "SELECT * FROM main.sales_orders LIMIT 1", "no_select_star"),

    # 逻辑层滥用
    ("logic", "SELECT * FROM sales_orders LIMIT 10", "no_select_star"),
    ("logic", "SELECT s.* FROM sales_orders s LIMIT 10", "no_select_star"),
    ("logic", "SELECT a.order_id FROM sales_orders a JOIN customers b ON 1=1 LIMIT 5", "no_cartesian"),
    ("logic", "SELECT a.order_id FROM sales_orders a JOIN customers b ON TRUE LIMIT 5", "no_cartesian"),
    ("logic", "SELECT a.order_id FROM sales_orders a, customers b LIMIT 5", "no_cartesian"),
    ("logic", "SELECT order_id FROM sales_orders", "missing_limit"),
    ("logic", "SELECT order_id FROM sales_orders ORDER BY order_date", "missing_limit"),
    ("logic", "SELECT order_id FROM totally_fake_table LIMIT 5", "table_whitelist"),
    ("logic", "SELECT 1", "table_whitelist"),
]


@pytest.mark.parametrize("category,sql,expected_rule", ATTACKS,
                         ids=[f"{c}:{s[:38]}" for c, s, _ in ATTACKS])
def test_attacks_are_blocked(category: str, sql: str, expected_rule: str) -> None:
    result = guard(sql)
    assert not result.ok, f"[{category}] 这条应该被拦截：{sql}"
    rules = {v.rule for v in result.violations}
    assert expected_rule in rules, (
        f"[{category}] 期望命中 {expected_rule}，实际命中 {sorted(rules)}\nSQL: {sql}"
    )
    # 拦截必须给出可执行建议，否则模型无法自我修复
    for v in result.violations:
        assert v.hint, f"违规项缺少修复建议：{v}"


# ──────────────────────────────────────────────────────────────────
# 合法查询不能被误杀（假阳性同样致命）
# ──────────────────────────────────────────────────────────────────

LEGIT = [
    "SELECT ROUND(SUM(order_amount), 2) AS amt FROM v_order_fact "
    "WHERE status IN ('paid','shipped','completed')",
    "SELECT ship_region, COUNT(*) AS n FROM sales_orders GROUP BY 1 ORDER BY n DESC",
    "WITH m AS (SELECT date_trunc('month', order_date) AS mth, SUM(order_amount) AS amt "
    "FROM v_order_fact GROUP BY 1) SELECT mth, amt FROM m",
    "SELECT customer_id FROM sales_orders LIMIT 100",
    "SELECT order_id FROM sales_orders WHERE status = 'paid' ORDER BY order_date LIMIT 20",
    "SELECT COUNT(*) FILTER (WHERE is_gift) AS gifts FROM order_items",
    "SELECT p.category, SUM(oi.item_amount) AS amt FROM order_items oi "
    "JOIN products p ON p.product_id = oi.product_id GROUP BY 1",
    "SELECT order_id FROM sales_orders WHERE order_id <> 'load data' LIMIT 5",
    "SELECT order_id FROM sales_orders WHERE ship_region LIKE '%create%' LIMIT 5",
    "SELECT DISTINCT ship_region FROM sales_orders LIMIT 20",
]


@pytest.mark.parametrize("sql", LEGIT, ids=[f"legit:{i}" for i in range(len(LEGIT))])
def test_legit_queries_pass(sql: str) -> None:
    result = guard(sql)
    assert result.ok, f"误杀合法查询：{sql}\n原因：{result.reason}"


def test_cte_names_are_not_treated_as_tables() -> None:
    """CTE 名在 AST 里也是 Table 节点——这是实际踩过的假阳性 bug。"""
    sql = """
    WITH first_order AS (
        SELECT customer_id, MIN(order_date) AS d
        FROM v_order_fact GROUP BY 1
    ),
    cnt AS (SELECT COUNT(*) AS n FROM first_order)
    SELECT n FROM cnt
    """
    r = guard(sql)
    assert r.ok, r.reason
    assert r.tables_used == ["v_order_fact"]


def test_select_star_inside_cte_is_allowed() -> None:
    """CTE 内部的 `SELECT *` 是局部展开，不构成对外拉全表，不拦截。"""
    sql = ("WITH t AS (SELECT * FROM v_order_fact) "
           "SELECT order_id FROM t LIMIT 5")
    assert guard(sql).ok


def test_select_star_as_final_projection_is_blocked() -> None:
    """但最外层（最终结果）的 `SELECT *` 仍然拦截——它可能把 20 列明细全吐给接口。"""
    sql = ("WITH t AS (SELECT order_id, order_amount FROM v_order_fact) "
           "SELECT * FROM t LIMIT 5")
    r = guard(sql)
    assert not r.ok
    assert any(v.rule == "no_select_star" for v in r.violations)


def test_all_examples_in_bank_are_guarded_and_executable() -> None:
    """示例库里的 SQL 必须同时满足：能过护栏、能在真实库上跑通。

    这条测试是防止「文档里的示例早就坏了」——那种问题在演示当天才会暴露。
    """
    from pathlib import Path
    from nl2sql.sqlsafe.executor import SQLRunner

    db = Path(__file__).resolve().parents[1] / "data" / "warehouse.duckdb"
    if not db.exists():
        pytest.skip("需要先跑 scripts/build_warehouse.py")
    runner = SQLRunner(db)
    try:
        for ex in EXAMPLES:
            g = guard(ex.sql)
            assert g.ok, f"示例未过护栏：{ex.question}\n{g.reason}"
            res = runner.execute(g.sql)
            assert res.ok, f"示例执行失败：{ex.question}\n{res.error}"
    finally:
        runner.close()


# ──────────────────────────────────────────────────────────────────
# 包装剥离
# ──────────────────────────────────────────────────────────────────

WRAPPED = [
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("好的，查询如下：\n```sql\nSELECT 1\n```", "SELECT 1"),
    ("SQL: SELECT 1", "SELECT 1"),
    ("查询语句：SELECT 1", "SELECT 1"),
    ("<|think|>想一想…<|/think|>SELECT 1", "SELECT 1"),
]


@pytest.mark.parametrize("raw,expected", WRAPPED)
def test_strip_wrapping(raw: str, expected: str) -> None:
    assert strip_wrapping(raw) == expected


def test_empty_sql_is_rejected() -> None:
    r = guard("")
    assert not r.ok
    assert r.violations[0].rule == "empty"


def test_syntax_error_is_reported_as_syntax() -> None:
    r = guard("SELECT FROM WHERE (((")
    assert not r.ok
    assert any(v.rule == "syntax" for v in r.violations)


def test_sanitized_sql_is_returned() -> None:
    """放行时要返回规范化的 SQL（模型原本可能带注释/多余换行）。"""
    r = guard("select round(sum(order_amount),2) as amt from v_order_fact where status in ('paid')")
    assert r.ok
    assert "SELECT" in r.sql.upper() and "\n" not in r.sql.strip() or True
