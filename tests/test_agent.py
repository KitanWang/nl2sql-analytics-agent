"""Agent 编排、结果接地校验、SQL 提取的测试。

LLM 用「假客户端」替换：这些测试要验证的是**编排逻辑与判定规则**，
不是模型能力。模型能力的测试在 `scripts/run_eval.py`（真实调用、有成本），
不放进单元测试，避免 CI 里跑一次就烧掉一堆 token。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from nl2sql.agent.analytics import (
    AgentConfig,
    AgentResult,
    AnalyticsAgent,
    _extract_sql,
    _strip_preamble,
    check_grounding,
)
from nl2sql.llm.client import LLMResponse, extract_json
from nl2sql.sqlsafe.executor import SQLRunner


# ──────────────────────────────────────────────────────────────────
# 假 LLM
# ──────────────────────────────────────────────────────────────────

@dataclass
class FakeLLM:
    """按顺序吐出预设回复的假客户端，记录每次调用的消息。"""
    replies: list[str]
    cfg: object = None
    calls: list[list[dict]] | None = None

    def __post_init__(self) -> None:
        self.calls = []

    def chat(self, messages, *, temperature=None, max_tokens=None, on_token=None):
        self.calls.append(messages)
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return LLMResponse(text=text, model="fake", prompt_tokens=10, completion_tokens=5,
                           elapsed_ms=1.0)

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def runner():
    db = Path(__file__).resolve().parents[1] / "data" / "warehouse.duckdb"
    if not db.exists():
        pytest.skip("需要先跑 scripts/build_warehouse.py")
    r = SQLRunner(db)
    yield r
    r.close()


# ──────────────────────────────────────────────────────────────────
# SQL 提取
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("```\nSELECT 1\n```", "SELECT 1"),
    ("这里是查询：\n```sql\nSELECT a FROM b\n```\n希望有帮助", "SELECT a FROM b"),
    ("SELECT a FROM b", "SELECT a FROM b"),
    ("好的，代码如下\nWITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
    ("```sql\nSELECT 1;\n```", "SELECT 1"),
])
def test_extract_sql(raw: str, expected: str) -> None:
    assert _extract_sql(raw) == expected


def test_strip_preamble() -> None:
    assert _strip_preamble("好的，根据查询结果：销售额为 100 元") == "销售额为 100 元"
    assert _strip_preamble("华东区销售额最高") == "华东区销售额最高"


# ──────────────────────────────────────────────────────────────────
# 结果接地校验
# ──────────────────────────────────────────────────────────────────

class TestGrounding:
    def test_exact_number_passes(self) -> None:
        ok, bad = check_grounding("销售额为 4,090,463.94 元", [(4090463.94,)], ["v"])
        assert ok and not bad

    def test_wan_unit_passes(self) -> None:
        """「409.05 万元」是对 4,090,463.94 的正确四舍五入，不能算幻觉。"""
        ok, bad = check_grounding("销售额约 409.05 万元", [(4090463.94,)], ["v"])
        assert ok, bad

    def test_wan_unit_with_space_passes(self) -> None:
        ok, _ = check_grounding("销售额约 409.05 万元。", [(4090463.94,)], ["v"])
        assert ok

    def test_fabricated_number_fails(self) -> None:
        """结果里只有销售额，模型自己编了个客单价——必须被抓出来。"""
        ok, bad = check_grounding("销售额 409 万元，客单价 3,500 元", [(4090463.94,)], ["v"])
        assert not ok
        assert any("3500" in b or "3,500" in b for b in bad)

    def test_column_sum_passes(self) -> None:
        rows = [("华东", 120000.0), ("华北", 80000.0)]
        ok, _ = check_grounding("合计 20 万元", rows, ["region", "amt"])
        assert ok

    def test_small_integers_ignored(self) -> None:
        """排名、月份这类小整数不该被当成指标校验（否则全是误报）。"""
        ok, _ = check_grounding("销售额前三的是华东，2025 年 6 月", [("华东", 100.0)], ["r", "v"])
        assert ok

    def test_empty_rows_passes(self) -> None:
        ok, _ = check_grounding("该口径下没有数据。", [], [])
        assert ok


# ──────────────────────────────────────────────────────────────────
# 编排流程
# ──────────────────────────────────────────────────────────────────

GOOD_SQL = """```sql
SELECT ROUND(SUM(order_amount), 2) AS amt
FROM v_order_fact
WHERE status IN ('paid','shipped','completed')
```"""

BROKEN_SQL = """```sql
SELECT ROUND(SUM(order_amout), 2) AS amt FROM v_order_fact
```"""

UNSAFE_SQL = """```sql
DROP TABLE sales_orders
```"""


def test_happy_path(runner) -> None:
    llm = FakeLLM([GOOD_SQL])
    agent = AnalyticsAgent(llm, runner, AgentConfig(answer_question=False))
    res = agent.run("有效销售额是多少")
    assert res.ok
    assert res.attempts == 1 and res.repairs == 0
    assert res.rows and float(res.rows[0][0]) > 0
    stages = [s.stage for s in res.steps]
    assert stages == ["generate", "guard", "execute"]


def test_execution_error_triggers_repair(runner) -> None:
    """SQL 跑不通 → 带着错误信息去修复 → 第二次成功。"""
    llm = FakeLLM([BROKEN_SQL, GOOD_SQL])
    agent = AnalyticsAgent(llm, runner, AgentConfig(answer_question=False, max_repairs=2))
    res = agent.run("有效销售额是多少")
    assert res.ok
    assert res.repairs == 1
    assert res.attempts == 2
    stages = [s.stage for s in res.steps]
    assert stages == ["generate", "guard", "execute", "repair", "guard", "execute"]
    # 修复提示里必须包含原始问题与报错，否则模型会把语义修歪
    fix_msgs = llm.calls[1]
    joined = " ".join(m["content"] for m in fix_msgs)
    assert "有效销售额是多少" in joined
    assert "order_amout" in joined


def test_unsafe_sql_is_blocked_and_repaired(runner) -> None:
    llm = FakeLLM([UNSAFE_SQL, GOOD_SQL])
    agent = AnalyticsAgent(llm, runner, AgentConfig(answer_question=False, max_repairs=2))
    res = agent.run("有效销售额是多少")
    assert res.ok, "修复后应当成功"
    assert res.repairs == 1
    assert any(s.stage == "guard" and not s.ok for s in res.steps)


def test_repair_budget_is_enforced(runner) -> None:
    """模型一直修不好时必须停下，不能无限重试烧 token。"""
    llm = FakeLLM([UNSAFE_SQL])
    agent = AnalyticsAgent(llm, runner, AgentConfig(answer_question=False, max_repairs=2))
    res = agent.run("有效销售额是多少")
    assert not res.ok
    assert res.repairs == 2
    assert res.attempts == 3          # 1 次生成 + 2 次修复
    assert res.error_stage == "guard"


def test_zero_repair_fails_fast(runner) -> None:
    llm = FakeLLM([BROKEN_SQL])
    agent = AnalyticsAgent(llm, runner, AgentConfig(answer_question=False, max_repairs=0))
    res = agent.run("有效销售额是多少")
    assert not res.ok
    assert res.repairs == 0
    assert res.error_stage == "execute"


def test_answer_stage_runs(runner) -> None:
    llm = FakeLLM([GOOD_SQL, "有效销售额约为 3.89 亿元。"])
    agent = AnalyticsAgent(llm, runner, AgentConfig(answer_question=True, max_repairs=0))
    res = agent.run("有效销售额是多少")
    assert res.answer
    assert any(s.stage == "answer" for s in res.steps)
    # 解读提示里应当带上结果表，而不是让模型自己回忆
    answer_prompt = " ".join(m["content"] for m in llm.calls[-1])
    assert "385" in answer_prompt or "|" in answer_prompt


def test_retrieval_selects_relevant_examples(runner) -> None:
    llm = FakeLLM([GOOD_SQL])
    agent = AnalyticsAgent(llm, runner, AgentConfig(answer_question=False, fewshot_k=3))
    res = agent.run("各渠道的销售额和客单价")
    assert 1 <= len(res.retrieved) <= 3
    # 与渠道/客单价相关的示例应当被召回
    joined = " ".join(r["question"] for r in res.retrieved)
    assert "渠道" in joined or "客单价" in joined


def test_fewshot_disabled_by_config(runner) -> None:
    llm = FakeLLM([GOOD_SQL])
    agent = AnalyticsAgent(llm, runner, AgentConfig(answer_question=False, use_fewshot=False))
    res = agent.run("各渠道的销售额")
    assert res.retrieved == []
    # 没有示例时，消息数应当是 system + user 两条
    assert len(llm.calls[0]) == 2


def test_glossary_toggle_changes_system_prompt(runner) -> None:
    on = FakeLLM([GOOD_SQL])
    AnalyticsAgent(on, runner, AgentConfig(answer_question=False, use_glossary=True)).run("q")
    off = FakeLLM([GOOD_SQL])
    AnalyticsAgent(off, runner, AgentConfig(answer_question=False, use_glossary=False)).run("q")
    assert len(on.calls[0][0]["content"]) > len(off.calls[0][0]["content"])
    assert "业务口径" in on.calls[0][0]["content"]
    assert "业务口径" not in off.calls[0][0]["content"]


def test_result_serialization(runner) -> None:
    llm = FakeLLM([GOOD_SQL])
    agent = AnalyticsAgent(llm, runner, AgentConfig(answer_question=False))
    d = agent.run("有效销售额是多少").to_dict()
    assert set(d) >= {"question", "sql", "ok", "rows", "columns", "steps", "retrieved"}
    assert isinstance(d["rows"], list)


# ──────────────────────────────────────────────────────────────────
# JSON 提取（结构化输出）
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('好的，结果如下：\n```json\n{"a": 1}\n```', {"a": 1}),
    ('结果：{"a": 1} 完毕', {"a": 1}),
    ('{"a": {"b": [1, 2]}}', {"a": {"b": [1, 2]}}),
    ('{"a": "含 } 和 { 的字符串"}', {"a": "含 } 和 { 的字符串"}),
])
def test_extract_json(raw: str, expected: dict) -> None:
    assert extract_json(raw) == expected


def test_extract_json_raises_on_garbage() -> None:
    with pytest.raises(ValueError):
        extract_json("完全没有 JSON")
