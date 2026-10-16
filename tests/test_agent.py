"""
Tests for the tool-using agent (generation/agent.py) — no LLM calls: the
model is replaced by a scripted fake that issues tool calls, so the loop,
the tools and the verification of the final answer are tested end to end.

Run: pytest tests/test_agent.py -v
"""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from generation.agent import MetricsAgent, agent_answer, calculate, needs_agent
from generation.verifier import Verifier
from metrics.extract import SCHEMA
from metrics.store import MetricStore


# ---------------------------------------------------------------- calculate

def test_calculate_arithmetic():
    assert calculate("(63973 - 60583) / 60583 * 100") == pytest.approx(5.5956, abs=1e-4)
    assert calculate("58,539.1 - 15,299.88") == pytest.approx(43239.22)


@pytest.mark.parametrize("expr", ["__import__('os').system('dir')", "open('x')", "a + 1", "[1, 2]"])
def test_calculate_rejects_anything_but_arithmetic(expr):
    with pytest.raises(ValueError):
        calculate(expr)


# ---------------------------------------------------------------- routing

@pytest.mark.parametrize("q, companies, expected", [
    ("What was TCS revenue growth year-on-year in Q3FY25?", ["TCS"], True),
    ("Compare ITC and Maruti revenue", ["ITC", "Maruti Suzuki"], True),
    ("What was TCS net profit margin in Q3FY25?", ["TCS"], True),
    ("Which had higher revenue, ITC or Maruti?", ["ITC", "Maruti Suzuki"], True),
    ("What was TCS revenue from operations in Q3FY25?", ["TCS"], False),
    ("Who audited TCS Q3FY25 financial results?", ["TCS"], False),
])
def test_needs_agent(q, companies, expected):
    assert needs_agent(q, companies) is expected


# ---------------------------------------------------------------- loop

def _store(tmp_path):
    db = tmp_path / "m.sqlite"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    for period, value in (("Q3FY25", 63973.0), ("Q3FY24", 60583.0)):
        con.execute("INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("TCS", "revenue_from_operations", "consolidated", period, "quarter", value,
                     "crore", "INR", value, "tcs.pdf", 7, "Revenue from operations", "Three months ended", "Q3FY25"))
    con.commit()
    return MetricStore(str(db))


def _tool_call(i, name, args):
    return SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)),
                           model_dump=lambda: {"id": f"c{i}", "type": "function",
                                               "function": {"name": name, "arguments": json.dumps(args)}})


class FakeClient:
    """Scripted model: look up both quarters, calculate, then answer."""

    def __init__(self, final_answer):
        self.steps = [
            [_tool_call(1, "lookup_metric", {"company": "TCS", "metric": "revenue_from_operations", "period": "Q3FY25"}),
             _tool_call(2, "lookup_metric", {"company": "TCS", "metric": "revenue_from_operations", "period": "Q3FY24"})],
            [_tool_call(3, "calculate", {"expression": "(63973 - 60583) / 60583 * 100"})],
            None,
        ]
        self.final_answer = final_answer
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        calls = self.steps.pop(0) if self.steps else None     # out of script: repeat the answer
        msg = SimpleNamespace(tool_calls=calls, content=None if calls else self.final_answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


Q = "What was TCS's consolidated revenue growth year-on-year in Q3FY25?"


def test_agent_runs_tools_and_verifies_a_correct_answer(tmp_path):
    answer = "TCS consolidated revenue from operations grew 5.60% YoY, from ₹60,583 crore to ₹63,973 crore."
    agent = MetricsAgent(retriever=None, store=_store(tmp_path), client=FakeClient(answer))
    r = agent_answer(agent, Verifier(), Q)
    assert [t["tool"] for t in r["tool_calls"]] == ["lookup_metric", "lookup_metric", "calculate"]
    assert r["is_grounded"], r["verification"]["unverified_claims"]
    assert {c["period"] for c in r["citations"]} == {"Q3FY25", "Q3FY24"}


def test_agent_answer_with_a_number_no_tool_produced_is_flagged(tmp_path):
    answer = "TCS consolidated revenue from operations grew 6.10% YoY, from ₹60,583 crore to ₹63,973 crore."
    agent = MetricsAgent(retriever=None, store=_store(tmp_path), client=FakeClient(answer))
    r = agent_answer(agent, Verifier(), Q)
    assert r["revised"]                      # it was given its one retry...
    assert not r["is_grounded"]              # ...and still flagged, since the retry didn't fix it
    assert any("6.1" in c for c in r["verification"]["unverified_claims"])


class FakeClientThatSelfCorrects(FakeClient):
    """First final answer states a difference computed in its head; after the
    feedback it calls calculate and answers again."""

    def __init__(self):
        super().__init__("Revenue rose by ₹3,390 crore, a 5.60% YoY increase, from ₹60,583 crore to ₹63,973 crore.")
        self.steps.insert(3, [_tool_call(4, "calculate", {"expression": "63973 - 60583"})])
        self.steps.append(None)
        self.answers = iter([self.final_answer, self.final_answer])

    def _create(self, **kwargs):
        calls = self.steps.pop(0)
        msg = SimpleNamespace(tool_calls=calls, content=None if calls else next(self.answers))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_agent_gets_one_retry_for_numbers_no_tool_produced(tmp_path):
    agent = MetricsAgent(retriever=None, store=_store(tmp_path), client=FakeClientThatSelfCorrects())
    r = agent_answer(agent, Verifier(), Q)
    assert r["revised"]
    assert [t["tool"] for t in r["tool_calls"]][-1] == "calculate"
    assert r["is_grounded"], r["verification"]["unverified_claims"]
