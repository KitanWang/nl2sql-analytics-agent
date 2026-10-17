"""FastAPI 服务：把 Agent 包成 HTTP 接口 + 一个单页前端。

## 接口

| 方法 | 路径 | 说明 |
| :--- | :--- | :--- |
| GET | `/health` | 健康检查（含库表行数、模型名） |
| GET | `/schema` | 返回表结构 + 业务口径（前端用来自动补全提示） |
| POST | `/ask` | 自然语言 → SQL → 结果 → 解读，返回完整 trace |
| POST | `/sql` | 直接执行一条 SQL（**同样过护栏**，演示用） |
| GET | `/stats` | 数据集概览，前端首屏展示 |

## 安全

- `/ask` 生成的 SQL 一律过 AST 护栏，不安全直接返回拒绝原因，不执行；
- `/sql` 是「给面试官演示护栏」的入口：故意让它能接收危险 SQL，
  但同样会被拦截——这比在文档里写「我们有护栏」有说服力得多；
- 所有执行都走只读连接 + 超时 + 行数上限。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nl2sql.agent.analytics import AgentConfig, AnalyticsAgent
from nl2sql.llm.client import LLMClient, LLMConfig
from nl2sql.schema import TABLES, to_prompt_schema
from nl2sql.semantics.glossary import GLOSSARY, glossary_prompt
from nl2sql.sqlsafe.executor import SQLRunner, _jsonable
from nl2sql.sqlsafe.guard import ALLOWED_TABLES, guard

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).resolve().parent / "static"

DB_PATH = Path(os.environ.get("NL2SQL_DB", ROOT / "data" / "warehouse.duckdb"))
CONFIG_DIR = Path(os.environ.get("NL2SQL_CONFIG", ROOT / "configs"))
LLM_PROFILE = os.environ.get("NL2SQL_LLM_PROFILE") or None


# ──────────────────────────────────────────────────────────────────
# 惰性初始化（导入模块时不连库、不读 key）
# ──────────────────────────────────────────────────────────────────

class _State:
    runner: SQLRunner | None = None
    llm: LLMClient | None = None
    agent: AnalyticsAgent | None = None
    error: str = ""


_state = _State()


def get_runner() -> SQLRunner:
    if _state.runner is None:
        if not DB_PATH.exists():
            raise HTTPException(503, f"数据库不存在：{DB_PATH}（先跑 scripts/build_warehouse.py）")
        _state.runner = SQLRunner(DB_PATH)
    return _state.runner


def get_agent() -> AnalyticsAgent:
    if _state.agent is None:
        try:
            _state.llm = LLMClient(LLMConfig.from_file(CONFIG_DIR, profile=LLM_PROFILE))
            _state.agent = AnalyticsAgent(_state.llm, get_runner(), AgentConfig())
        except Exception as e:  # noqa: BLE001
            _state.error = str(e)
            raise HTTPException(
                503, f"LLM 未就绪：{e}。请在 configs/local.yaml 配置 api_key"
            ) from e
    return _state.agent


app = FastAPI(title="自然语言数据分析 Agent", version="0.1.0")

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# ──────────────────────────────────────────────────────────────────
# 模型
# ──────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    with_answer: bool = True


class SQLRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=8000)


# ──────────────────────────────────────────────────────────────────
# 路由
# ──────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    page = STATIC / "index.html"
    if not page.exists():
        raise HTTPException(404, "前端未构建")
    return FileResponse(page)


@app.get("/health")
def health() -> dict[str, Any]:
    info: dict[str, Any] = {"ok": True, "db": str(DB_PATH), "db_exists": DB_PATH.exists()}
    try:
        info["rows"] = get_runner().execute(
            "SELECT (SELECT COUNT(*) FROM sales_orders) AS orders,"
            " (SELECT COUNT(*) FROM customers) AS customers"
        ).rows[0]
    except Exception as e:  # noqa: BLE001
        info["ok"] = False
        info["error"] = str(e)
    info["llm"] = {"profile": LLM_PROFILE or "default", "ready": _state.agent is not None}
    if _state.llm is not None:
        info["llm"]["model"] = _state.llm.cfg.model
    if _state.error:
        info["llm"]["error"] = _state.error
    return info


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "tables": [
            {
                "name": t.name,
                "comment": t.comment,
                "columns": [
                    {"name": c.name, "dtype": c.dtype, "comment": c.comment}
                    for c in t.columns
                ],
            }
            for t in TABLES
        ],
        "allowed_tables": sorted(ALLOWED_TABLES),
        "glossary": [
            {"term": e.term, "definition": e.definition, "sql_hint": e.sql_hint}
            for e in GLOSSARY
        ],
        "prompt_schema": to_prompt_schema(),
        "prompt_glossary": glossary_prompt(),
    }


@app.get("/stats")
def stats() -> dict[str, Any]:
    r = get_runner()
    q = lambda s: r.execute(s)  # noqa: E731
    out: dict[str, Any] = {}
    cases = [
        ("订单总数", "SELECT COUNT(*) FROM sales_orders"),
        ("有效订单", "SELECT COUNT(*) FROM sales_orders "
                     "WHERE status IN ('paid','shipped','completed')"),
        ("客户数", "SELECT COUNT(*) FROM customers"),
        ("商品数", "SELECT COUNT(*) FROM products"),
        ("有效销售额（元）",
         "SELECT ROUND(SUM(order_amount), 2) FROM v_order_fact "
         "WHERE status IN ('paid','shipped','completed')"),
        ("日期范围",
         "SELECT MIN(order_date) || ' ~ ' || MAX(order_date) FROM sales_orders"),
    ]
    for label, sql in cases:
        res = q(sql)
        if res.ok and res.rows:
            out[label] = _jsonable(res.rows[0][0])
    by_channel = q("SELECT channel, ROUND(SUM(order_amount), 2) AS amt FROM v_order_fact "
                   "WHERE status IN ('paid','shipped','completed') "
                   "GROUP BY 1 ORDER BY amt DESC")
    out["channel_sales"] = [
        {"channel": row[0], "amount": _jsonable(row[1])} for row in by_channel.rows
    ]
    return out


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    agent = get_agent()
    if not req.with_answer:
        agent.cfg.answer_question = False
    else:
        agent.cfg.answer_question = True
    result = agent.run(req.question.strip())
    payload = result.to_dict()
    payload["guard_rules"] = {
        "allowed_tables": sorted(ALLOWED_TABLES),
        "checks": [
            "单条 SELECT（拒绝多语句与 DML/DDL）",
            "禁止 COPY / ATTACH / PRAGMA / INSTALL / LOAD",
            "禁止 read_csv / read_parquet / glob 等外部资源函数",
            "表名必须在白名单内（CTE 名除外）",
            "禁止 SELECT *、禁止无 ON 条件或恒真 ON 条件的 JOIN",
            "明细查询必须有 LIMIT（聚合查询豁免）",
            "只读连接 + 8s 超时 + 500 行上限 + 关闭外部访问",
        ],
    }
    return payload


@app.post("/sql")
def run_sql(req: SQLRequest) -> JSONResponse:
    """直接执行 SQL —— 同样过护栏。用来现场演示「危险 SQL 会被拦」。"""
    g = guard(req.sql)
    if not g.ok:
        return JSONResponse(status_code=200, content={
            "ok": False,
            "blocked": True,
            "reason": g.reason,
            "violations": [{"rule": v.rule, "detail": v.detail, "hint": v.hint}
                           for v in g.violations],
        })
    res = get_runner().execute(g.sql)
    return JSONResponse(status_code=200, content={
        "ok": res.ok,
        "blocked": False,
        "sanitized_sql": g.sql,
        "tables_used": g.tables_used,
        **res.to_dict(),
    })


@app.exception_handler(Exception)
async def unhandled(_request, exc: Exception) -> JSONResponse:  # noqa: ANN001
    return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})
