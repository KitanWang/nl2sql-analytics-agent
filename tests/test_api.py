"""API 层测试。

原则：**不 mock 掉被测逻辑本身**。
- `/sql` 走真实护栏 + 真实 DuckDB（这是最需要保住的行为）；
- `/ask` 需要 LLM，用 monkeypatch 换掉 `get_agent`，避免测试依赖网络与 key。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from nl2sql.api import server
from nl2sql.agent.analytics import AgentResult, Step


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    if not server.DB_PATH.exists():
        pytest.skip("需要先跑 scripts/build_warehouse.py")
    with TestClient(server.app) as c:
        yield c


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"]
    assert d["rows"][0] > 0


def test_stats(client) -> None:
    d = client.get("/stats").json()
    assert "订单总数" in d
    assert d["有效销售额（元）"] > 0
    assert len(d["channel_sales"]) >= 3


def test_schema_endpoint(client) -> None:
    d = client.get("/schema").json()
    names = {t["name"] for t in d["tables"]}
    assert {"sales_orders", "customers", "v_order_fact"}.issubset(names | {"v_order_fact"})
    assert len(d["glossary"]) >= 10
    assert "业务口径" in d["prompt_glossary"] or "客单价" in d["prompt_glossary"]
    assert "sales_orders" in d["prompt_schema"]


def test_index_page(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "自然语言数据分析" in r.text


# ──────────────────────────────────────────────────────────────────
# /sql —— 护栏行为
# ──────────────────────────────────────────────────────────────────

def test_sql_endpoint_allows_safe_query(client) -> None:
    r = client.post("/sql", json={"sql":
        "SELECT ship_region, ROUND(SUM(order_amount),2) AS amt FROM v_order_fact "
        "WHERE status IN ('paid','shipped','completed') GROUP BY 1 ORDER BY amt DESC"})
    d = r.json()
    assert d["ok"] and not d["blocked"]
    assert d["row_count"] == 7
    assert d["tables_used"] == ["v_order_fact"]


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales_orders",
    "SELECT 1; DROP TABLE sales_orders",
    "SELECT * FROM read_csv('/etc/passwd')",
    "COPY (SELECT 1) TO '/tmp/x.csv'",
    "ATTACH 'x.db' AS x",
    "SELECT a.order_id FROM sales_orders a JOIN customers b ON 1=1 LIMIT 5",
])
def test_sql_endpoint_blocks(client, sql: str) -> None:
    d = client.post("/sql", json={"sql": sql}).json()
    assert d["blocked"] is True
    assert d["ok"] is False
    assert d["reason"]
    assert d["violations"]


def test_sql_endpoint_reports_runtime_error(client) -> None:
    d = client.post("/sql", json={"sql":
        "SELECT no_such_col FROM sales_orders LIMIT 5"}).json()
    assert d["blocked"] is False
    assert d["ok"] is False
    assert d["error"]


def test_sql_endpoint_empty_input_rejected(client) -> None:
    r = client.post("/sql", json={"sql": ""})
    assert r.status_code == 422


# ──────────────────────────────────────────────────────────────────
# /ask —— 用假 Agent 验证接口契约
# ──────────────────────────────────────────────────────────────────

class _FakeAgent:
    def __init__(self, result: AgentResult):
        self.result = result
        self.cfg = type("C", (), {"answer_question": True})()

    def run(self, question: str) -> AgentResult:
        return self.result


def test_ask_returns_full_trace(client, monkeypatch) -> None:
    res = AgentResult(
        question="q", sql="SELECT 1", ok=True,
        columns=["amt"], rows=[(123.0,)], row_count=1,
        answer="销售额 123 元", attempts=1, total_ms=42.0,
        steps=[Step("generate", True, 10.0), Step("guard", True, 1.0),
               Step("execute", True, 5.0), Step("grounding", True, 0.0)],
    )
    monkeypatch.setattr(server, "get_agent", lambda: _FakeAgent(res))
    d = client.post("/ask", json={"question": "销售额多少"}).json()
    assert d["ok"] and d["answer"]
    assert d["rows"] == [[123.0]]
    assert [s["stage"] for s in d["steps"]] == ["generate", "guard", "execute", "grounding"]
    assert "guard_rules" in d
    assert d["guard_rules"]["allowed_tables"]


def test_ask_rejects_too_long_question(client) -> None:
    r = client.post("/ask", json={"question": "x" * 600})
    assert r.status_code == 422


def test_ask_rejects_empty_question(client) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_without_llm_returns_503(client, monkeypatch) -> None:
    def boom():
        raise server.HTTPException(503, "LLM 未就绪")
    monkeypatch.setattr(server, "get_agent", boom)
    r = client.post("/ask", json={"question": "销售额"})
    assert r.status_code == 503
