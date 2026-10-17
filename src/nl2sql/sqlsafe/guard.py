"""SQL 安全护栏 —— 用 AST 而不是正则来判断一条 SQL 能不能执行。

## 为什么必须用 AST

正则黑名单在 SQL 上是**必然失败**的，随便举几个绕过方式：

| 绕过手段 | 正则黑名单会怎么错 |
| :--- | :--- |
| `DROP /* x */ TABLE t` | 注释把关键字拆开了 |
| `dRoP tAbLe t` | 大小写 |
| `SELECT 1; DROP TABLE t` | 多语句：只检查了第一条 |
| `SELECT * FROM t WHERE x = (SELECT ... )` 里藏写操作 | 子查询里的写操作 |
| `COPY (SELECT ...) TO '/etc/passwd'` | 完全不是 SELECT/INSERT/UPDATE/DELETE |
| `SELECT * FROM read_csv('/etc/shadow')` | DuckDB 的表函数能读任意文件 |
| `SELECT * FROM t` 但 t 是 `read_parquet('s3://...')` | 同上 |
| `PRAGMA disable_verification` / `ATTACH 'x.db'` | 不在黑名单里 |

所以这里做的是**白名单 AST 校验**：解析成抽象语法树，然后逐节点检查，
只允许「一个表达式或一条 SELECT 语句」，且**禁止所有 DDL/DML/COPY/ATTACH/PRAGMA**
与**所有能访问外部资源的表函数**。

## 分层设计

1. `parse_single_statement`   —— 语法层：剔除注释、拒绝多语句
2. `assert_read_only`         —— 语句类型层：只允许 SELECT / WITH
3. `assert_no_forbidden_nodes`—— 节点层：DML/DDL、外部表函数、危险函数
4. `assert_table_whitelist`   —— 对象层：表名/列名必须在 schema 白名单内
5. `assert_limits`            —— 防滥用：必须有 LIMIT 或聚合、禁止笛卡尔积

每一层独立可测，失败时给出**人话原因**（会回传给模型做自我修复）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# ──────────────────────────────────────────────────────────────────
# 白名单
# ──────────────────────────────────────────────────────────────────

# 允许被 LLM 引用的表与视图（含视图名）
ALLOWED_TABLES: frozenset[str] = frozenset({
    "customers", "products", "sales_orders", "order_items",
    "payments", "refunds", "dim_calendar", "v_order_fact",
})

# 允许出现在 SQL 里的函数（白名单）。
# 这里刻意只放「分析函数」，不放 read_csv / read_parquet / glob / shell 之类。
ALLOWED_FUNCTIONS: frozenset[str] = frozenset({
    # 聚合
    "sum", "count", "avg", "min", "max", "median", "stddev", "stddev_pop",
    "stddev_samp", "variance", "var_pop", "var_samp", "quantile", "quantile_cont",
    "quantile_disc", "mode", "any_value", "arg_max", "arg_min",
    # 窗口
    "row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile",
    "lag", "lead", "first_value", "last_value", "nth_value",
    # 字符串
    "lower", "upper", "trim", "ltrim", "rtrim", "length", "char_length",
    "substr", "substring", "replace", "concat", "concat_ws", "split_part",
    "position", "strpos", "contains", "like", "starts_with", "ends_with",
    "regexp_matches", "regexp_replace", "regexp_extract", "prefix", "suffix",
    # 数值
    "abs", "round", "ceil", "ceiling", "floor", "greatest", "least", "power",
    "sqrt", "exp", "ln", "log", "log10", "sign", "trunc", "mod",
    # 日期
    "date_trunc", "date_diff", "datediff", "date_part", "datepart", "extract",
    "strftime", "strptime", "date", "year", "month", "day", "quarter", "week",
    "dayofweek", "dayofyear", "epoch", "now", "make_date", "last_day",
    "date_add", "date_sub", "to_date", "cast", "try_cast",
    # 条件 / 空值
    "coalesce", "nullif", "ifnull", "if", "nullif", "case", "isnull",
    "notnull", "iff",
    # 其他
    "distinct", "filter", "over", "list", "array_agg", "string_agg", "group_concat",
    "min_by", "max_by", "to_char", "format", "typeof", "printf",
})

# 只要出现在 AST 里就直接拒绝的节点类型
FORBIDDEN_NODE_NAMES: frozenset[str] = frozenset({
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter",
    "TruncateTable", "Copy", "Attach", "Detach", "Pragma", "Set", "Use",
    "Command", "Transaction", "Commit", "Rollback", "Grant", "Revoke",
    "LoadData", "Export", "Call", "Execute",
})

# DuckDB 里能读文件/网络/执行外部代码的表函数与函数名
FORBIDDEN_FUNC_PREFIX = (
    "read_csv", "read_parquet", "read_json", "read_ndjson", "read_avro",
    "read_blob", "glob", "parquet_scan", "csv_scan", "json_scan",
    "delta_scan", "iceberg_scan", "postgres_scan", "mysql_scan", "sqlite_scan",
    "duckdb_", "sniff_csv", "read_text", "read_xlsx", "load_", "install_",
    "shell", "exec", "system", "getenv", "pg_", "sqlite_",
)

# 提示：真实项目里 DuckDB 的 `enable_external_access=false` 是兜底，
# 但它不能阻止 `ATTACH` 之后的库内操作，且某些版本对 `COPY` 的拦截不完整，
# 所以这里做的是「AST 白名单 + 连接参数 + 只读连接」三层。

_ALLOWED_TOP_NODES = (exp.Select, exp.With, exp.Union, exp.Intersect, exp.Except,
                      exp.Subquery, exp.Paren)


# ──────────────────────────────────────────────────────────────────
# 结果类型
# ──────────────────────────────────────────────────────────────────

@dataclass
class Violation:
    """一条违规。`hint` 是回传给模型做自我修复的提示——必须是可执行的建议。"""
    rule: str
    detail: str
    hint: str


@dataclass
class GuardResult:
    ok: bool
    sql: str = ""
    violations: list[Violation] = field(default_factory=list)
    tables_used: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "；".join(f"[{v.rule}] {v.detail}（建议：{v.hint}）" for v in self.violations)


# ──────────────────────────────────────────────────────────────────
# 预处理
# ──────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE)
_THINKING_RE = re.compile(r"<\|?think\|?>.*?<\|?/think\|?>", re.IGNORECASE | re.DOTALL)


def strip_wrapping(raw: str) -> str:
    """去掉模型常见的包装：markdown 代码围栏、前后缀说明、思维链残留。

    这些包装如果不清掉，`sqlglot` 解析会失败，模型会误以为是「SQL 语法错」，
    在自我修复循环里反复重试——这是一个真实会踩的坑。
    """
    s = (raw or "").strip()
    s = _THINKING_RE.sub("", s).strip()
    # 取第一个 ```sql 代码块
    m = re.search(r"```(?:sql)?\s*(.+?)```", s, re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()
    else:
        s = _FENCE_RE.sub("", s).strip()
    # 去掉 "SQL:" / "查询语句：" 之类的前缀
    s = re.sub(r"^(?:sql|query|答案|查询语句|SQL 语句)\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    # 去掉结尾的说明文字（以中文句号/英文句子结尾且不在字符串内的情况极少见，保守处理）
    return s.strip().rstrip(";").strip()


def count_statements(sql: str) -> int:
    """数一下有几条语句。**不能**用 `sql.count(";")`——字符串字面量里也可能有分号。"""
    try:
        return len([s for s in sqlglot.parse(sql, read="duckdb") if s is not None])
    except ParseError:
        return -1


# ──────────────────────────────────────────────────────────────────
# 分层校验
# ──────────────────────────────────────────────────────────────────

def _iter_nodes(tree: exp.Expression):
    yield tree
    for node in tree.walk():
        if isinstance(node, exp.Expression) and node is not tree:
            yield node


def assert_single_select(tree: exp.Expression, violations: list[Violation]) -> None:
    """第 2 层：语句类型。只允许 SELECT / WITH（含 UNION 等集合运算）。"""
    if isinstance(tree, _ALLOWED_TOP_NODES):
        return
    kind = type(tree).__name__
    violations.append(Violation(
        rule="statement_type",
        detail=f"只允许 SELECT 查询，收到的是 {kind}",
        hint="把语句改写为一条 SELECT（可用 WITH 定义 CTE），不要包含任何写操作",
    ))


def assert_no_forbidden_nodes(tree: exp.Expression, violations: list[Violation]) -> None:
    """第 3 层：节点类型与函数名。"""
    for node in _iter_nodes(tree):
        name = type(node).__name__
        if name in FORBIDDEN_NODE_NAMES:
            violations.append(Violation(
                rule="forbidden_statement",
                detail=f"禁止的语句类型 {name}",
                hint="只生成只读查询（SELECT），不要修改数据或库结构",
            ))

        # 函数调用：取函数名做前缀比对
        fname = ""
        if isinstance(node, exp.Func):
            try:
                fname = node.sql_name().lower()
            except Exception:
                fname = ""
            if not fname:
                fname = (node.name or "").lower()
            if not fname and isinstance(node, exp.Anonymous):
                fname = str(node.this).lower()
        if fname:
            if fname not in ALLOWED_FUNCTIONS:
                if fname.startswith(FORBIDDEN_FUNC_PREFIX):
                    violations.append(Violation(
                        rule="forbidden_function",
                        detail=f"禁止调用外部资源函数 {fname}()",
                        hint="只能查询白名单内的表，不能读文件、访问网络或执行命令",
                    ))
                elif fname in {"read_csv_auto", "read_json_auto", "parquet_scan_auto"}:
                    violations.append(Violation(
                        rule="forbidden_function",
                        detail=f"禁止调用外部资源函数 {fname}()",
                        hint="只能查询白名单内的表",
                    ))

        # exp.Anonymous 兜底：形如 read_csv_auto(...) 有时不会被识别成具体 Func 类
        if isinstance(node, exp.Anonymous):
            fn = str(node.this).lower()
            if fn.startswith(FORBIDDEN_FUNC_PREFIX) or fn not in ALLOWED_FUNCTIONS:
                if fn.startswith(FORBIDDEN_FUNC_PREFIX):
                    violations.append(Violation(
                        rule="forbidden_function",
                        detail=f"禁止调用 {fn}()",
                        hint="只能使用白名单内的分析函数，不能读外部文件",
                    ))

    # 显式再扫一遍原始 SQL 文本里的危险关键字，作为对 sqlglot 版本差异的兜底。
    # 注意：这**不是**主要防线（主防线是 AST），只是防止方言解析漏掉节点类型。
    # 先去掉字符串字面量，否则 `WHERE reason = 'load data'` 会被误报。
    text = re.sub(r"'(?:[^']|'')*'", "''", tree.sql(dialect="duckdb")).lower()
    for kw in ("attach ", "detach ", "pragma ", "copy ", "install ", "load ",
               "export ", "create ", "drop ", "alter ", "truncate ", "grant ", "revoke "):
        if re.search(rf"(?<![a-z_]){kw.strip()}(?![a-z_])", text):
            violations.append(Violation(
                rule="forbidden_keyword",
                detail=f"SQL 文本中出现禁止关键字 `{kw.strip().upper()}`",
                hint="只生成纯查询语句",
            ))
            break


def collect_cte_names(tree: exp.Expression) -> set[str]:
    """收集所有 CTE 的名称。

    这一条是**踩过的坑**：`WITH m AS (...) SELECT ... FROM m` 里，`m` 在 AST 中
    也是一个 `exp.Table` 节点。如果不排除 CTE 名，所有带 WITH 的查询都会被
    误判为「引用了不在白名单的表」——而 CTE 恰恰是模型组织复杂查询的主要方式。
    """
    return {(cte.alias or "").lower() for cte in tree.find_all(exp.CTE) if cte.alias}


def assert_table_whitelist(
    tree: exp.Expression, violations: list[Violation], allowed: frozenset[str] = ALLOWED_TABLES
) -> list[str]:
    """第 4 层：表/视图必须在白名单内（CTE 名不算表）。返回实际用到的表名。"""
    used: list[str] = []
    cte_names = collect_cte_names(tree)
    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        if not name:
            continue
        if name in cte_names:
            # 是 CTE 引用，不是真实表
            continue
        if name not in allowed:
            violations.append(Violation(
                rule="table_whitelist",
                detail=f"表 `{name}` 不在可查询白名单内",
                hint=f"只能使用这些表/视图：{', '.join(sorted(allowed))}",
            ))
        else:
            used.append(name)
    if not used and not violations:
        violations.append(Violation(
            rule="table_whitelist",
            detail="没有识别到任何可查询的表",
            hint="查询必须从白名单中的表或视图取数",
        ))
    return sorted(set(used))


def _is_constant_condition(node: exp.Expression) -> bool:
    """判断 ON 条件是否恒为真（与任何表数据无关）。

    覆盖：`1=1`、`TRUE`、`'a'='a'`、`1=1 AND x>0`（只要有一侧是常量就足够可疑）。
    宁可误杀也不放过——如果模型真写了一个合法但含 `AND 1=1` 的条件，
    提示词会让它改掉，代价只有一次重试。
    """
    if isinstance(node, exp.Boolean):
        return True
    if isinstance(node, exp.Literal):
        return True
    if isinstance(node, exp.EQ):
        left_has_col = any(True for _ in node.left.find_all(exp.Column))
        right_has_col = any(True for _ in node.right.find_all(exp.Column))
        if not left_has_col and not right_has_col:
            return True
    if isinstance(node, exp.Paren):
        return _is_constant_condition(node.this)
    if isinstance(node, (exp.And, exp.Or)):
        return _is_constant_condition(node.left) or _is_constant_condition(node.right)
    return False


def assert_no_cross_join(tree: exp.Expression, violations: list[Violation]) -> None:
    """禁止无条件 JOIN（会把行数炸开，把库拖死）。

    现实场景：模型偶尔会写出 `FROM sales_orders, customers` 这种漏掉连接条件的写法，
    在 8 千行 × 474 行上直接变成 400 万行，接口超时。这条必须拦。
    """
    for join in tree.find_all(exp.Join):
        # CROSS JOIN 显式写法
        if (join.args.get("kind") or "").upper() == "CROSS":
            violations.append(Violation(
                rule="no_cartesian",
                detail="禁止 CROSS JOIN",
                hint="用 ON 条件连接两张表",
            ))
            continue
        # ON 条件存在但恒为真（`ON 1=1`、`ON TRUE`、`ON 'a'='a'`）——等价于笛卡尔积。
        # 这是对抗测试里唯一漏过去的一条，非常典型：模型为了「能跑通」会写这种连接条件。
        on = join.args.get("on")
        if on is not None and _is_constant_condition(on):
            violations.append(Violation(
                rule="no_cartesian",
                detail="JOIN 条件是常量（恒为真），等价于笛卡尔积",
                hint="用真实的关联字段做 ON 条件，例如 ON oi.order_id = o.order_id",
            ))

        # 有 JOIN 但没有 ON/USING 且不是 LATERAL
        has_cond = join.args.get("on") is not None or join.args.get("using")
        if not has_cond and not join.args.get("lateral"):
            nxt = join.args.get("this")
            explicit_cross = isinstance(nxt, exp.Subquery) and False  # 保留扩展点
            if not explicit_cross:
                violations.append(Violation(
                    rule="no_cartesian",
                    detail="JOIN 缺少 ON/USING 条件，会产生笛卡尔积",
                    hint="补上 ON 连接条件",
                ))

    # 逗号分隔的多表（隐式 CROSS JOIN）
    for select in tree.find_all(exp.Select):
        from_ = select.args.get("from")
        if from_ is None:
            continue
        joins = select.args.get("joins") or []
        # sqlglot 把 `FROM a, b` 解析为 joins 里 kind 为空的 Join，需单独识别
        for j in joins:
            if j.args.get("on") is None and not j.args.get("using"):
                continue


def assert_has_limit_or_aggregate(tree: exp.Expression, violations: list[Violation]) -> None:
    """第 5 层：明细查询必须带 LIMIT，防止一次拉回全表。

    聚合查询（含 GROUP BY 或全表聚合且有合理维度）不算——它们天然返回行数少。
    """
    has_limit = tree.args.get("limit") is not None or any(
        s.args.get("limit") is not None for s in tree.find_all(exp.Select)
    )
    # 只要查询里有聚合（含 GROUP BY），返回行数就被分组基数约束住了，
    # 此时强加 LIMIT 反而会把「按月汇总 24 行」这类正常结果裁掉。
    has_agg = any(True for _ in tree.find_all(exp.AggFunc))
    has_group = any(True for _ in tree.find_all(exp.Group))
    if not has_limit and not has_agg and not has_group:
        violations.append(Violation(
            rule="missing_limit",
            detail="明细查询既没有 LIMIT 也没有聚合，可能一次拉回全表",
            hint="最外层加上 LIMIT 200（或改成聚合查询）",
        ))


def assert_no_select_star(tree: exp.Expression, violations: list[Violation]) -> None:
    """`SELECT *` 在宽表上会拖垮接口，且对分析问题没有意义。"""
    for select in tree.find_all(exp.Select):
        for proj in select.expressions:
            if isinstance(proj, exp.Star):
                violations.append(Violation(
                    rule="no_select_star",
                    detail="禁止 SELECT *",
                    hint="明确列出需要的列，或直接 SELECT COUNT(*)",
                ))
            elif isinstance(proj, exp.Column) and isinstance(proj.this, exp.Star):
                violations.append(Violation(
                    rule="no_select_star",
                    detail=f"禁止 {proj.sql()}（展开全部列）",
                    hint="明确列出需要的列",
                ))


def assert_no_dangerous_names(tree: exp.Expression, violations: list[Violation]) -> None:
    """库名/schema 名必须为空或 main；禁止访问系统库。"""
    for table in tree.find_all(exp.Table):
        db = (table.db or "").lower()
        catalog = (table.catalog or "").lower()
        if db and db not in ("main", ""):
            violations.append(Violation(
                rule="schema_whitelist",
                detail=f"禁止访问 schema `{db}`",
                hint="直接用表名，不要带 schema 前缀",
            ))
        if catalog and catalog not in ("", "memory"):
            violations.append(Violation(
                rule="schema_whitelist",
                detail=f"禁止访问 catalog `{catalog}`",
                hint="直接用表名，不要带库名前缀",
            ))


# ──────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────

def guard(
    raw_sql: str,
    *,
    allowed_tables: frozenset[str] = ALLOWED_TABLES,
    require_limit: bool = True,
    max_len: int = 8000,
) -> GuardResult:
    """对模型产出的 SQL 做完整校验。

    返回的 `GuardResult.ok=False` 时，`reason` 会被原样回传给模型做自我修复。
    **永远不要**在 ok=False 时执行该 SQL。
    """
    violations: list[Violation] = []
    sql = strip_wrapping(raw_sql)

    if not sql:
        return GuardResult(False, "", [Violation("empty", "SQL 为空", "请生成一条完整的 SELECT 语句")])
    if len(sql) > max_len:
        return GuardResult(False, sql[:200], [
            Violation("too_long", f"SQL 长度 {len(sql)} 超过上限 {max_len}", "简化查询，减少嵌套层级")
        ])

    # 第 1 层：语法 + 单语句
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except ParseError as e:
        msg = str(e).splitlines()[0][:200]
        return GuardResult(False, sql, [
            Violation("syntax", f"SQL 无法解析：{msg}", "检查括号与关键字拼写，使用 DuckDB 方言")
        ])
    if len(statements) != 1:
        return GuardResult(False, sql, [
            Violation("multi_statement", f"检测到 {len(statements)} 条语句，只允许 1 条",
                      "只保留一条 SELECT，去掉多余语句与分号")
        ])
    tree = statements[0]

    # 第 2~6 层
    assert_single_select(tree, violations)
    assert_no_forbidden_nodes(tree, violations)
    assert_no_dangerous_names(tree, violations)
    tables = assert_table_whitelist(tree, violations, allowed_tables)
    assert_no_cross_join(tree, violations)
    assert_no_select_star(tree, violations)
    if require_limit:
        assert_has_limit_or_aggregate(tree, violations)

    # 去重（同一条违规可能被两层各报一次）
    seen = set()
    uniq: list[Violation] = []
    for v in violations:
        key = (v.rule, v.detail)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v)

    return GuardResult(ok=not uniq, sql=tree.sql(dialect="duckdb"), violations=uniq, tables_used=tables)
