"""
Tests for retrieval/query_filters.py — company + period detection.

The eval set doubles as a test: every question has a hand-labelled company
and period, so detection must reproduce them from the question text alone.

Run: pytest tests/test_query_filters.py -v
"""

import json
from pathlib import Path

import pytest

from retrieval.query_filters import detect_companies, detect_period, resolve_filters

AVAILABLE = {
    "Reliance Industries": {"FY2025"}, "TCS": {"Q3FY25"}, "Infosys": {"Q1FY27"},
    "HDFC Bank": {"Q1FY27"}, "ITC": {"Q1FY27"}, "Bharti Airtel": {"Q1FY27"},
    "Sun Pharma": {"Q1FY27"}, "Larsen & Toubro": {"Q1FY27"}, "Maruti Suzuki": {"Q1FY27"},
    "Tata Steel": {"Q1FY27"}, "NTPC": {"Q1FY27"}, "Bajaj Finance": {"Q1FY27"}, "UltraTech Cement": {"Q1FY27"},
}
QUESTIONS = json.loads(
    (Path(__file__).resolve().parents[1] / "eval" / "eval_questions.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_eval_question_resolves_to_its_labelled_filters(q):
    r = resolve_filters(q["question"], AVAILABLE)
    companies = sorted(q.get("companies", [q["company"]]))
    got = sorted(r["filters"], key=lambda f: f["company"])
    assert got == [{"company": c, "period": q["period"]} for c in companies] or \
        (got == [{"company": c} for c in companies] and detect_period(q["question"]) is None), r


@pytest.mark.parametrize("text, period", [
    ("TCS revenue in Q3FY25", "Q3FY25"),
    ("TCS revenue in Q3 FY 2025", "Q3FY25"),
    ("profit for the quarter ended June 30, 2026", "Q1FY27"),
    ("profit for the quarter ended 30 June 2026", "Q1FY27"),
    ("profit for the quarter ended 30.06.2026", "Q1FY27"),
    ("revenue for the quarter ended December 31, 2024", "Q3FY25"),
    ("nine month revenue ending December 2024", "Q3FY25"),
    ("results for the quarter ended March 31, 2026", "Q4FY26"),
    ("EBITDA in FY25", "FY2025"),
    ("EBITDA in FY 2024-25", "FY2025"),
    ("dividend for the year ended March 31, 2025", "FY2025"),
    ("What is the CIN of Infosys?", None),
])
def test_period_formats(text, period):
    assert detect_period(text) == period


@pytest.mark.parametrize("text, companies", [
    ("What was Tata Consultancy Services' revenue?", ["TCS"]),
    ("L&T order inflow", ["Larsen & Toubro"]),
    ("Airtel's ARPU", ["Bharti Airtel"]),
    ("Compare ITC and Maruti margins", ["ITC", "Maruti Suzuki"]),
    ("Which company has the highest revenue?", []),
    ("Explain the credit policy", []),          # "credit" must not match ITC
    ("Tata Steel vs Tata Consultancy revenue", ["TCS", "Tata Steel"]),     # no "tata" collision
    ("UltraTech's EBITDA", ["UltraTech Cement"]),
])
def test_company_aliases(text, companies):
    assert detect_companies(text) == companies


def test_period_not_indexed_is_reported_not_silently_ignored():
    r = resolve_filters("TCS revenue for the quarter ended June 30, 2026", AVAILABLE)
    assert r["filters"] == []
    assert "TCS: Q1FY27 is not indexed (available: Q3FY25)" in r["missing"]


def test_explicit_request_filters_win_over_detection():
    r = resolve_filters("TCS revenue in Q3FY25", AVAILABLE, company="Infosys", period="Q1FY27")
    assert r["filters"] == [{"company": "Infosys", "period": "Q1FY27"}]


def test_comparison_question_gets_one_filter_per_company():
    r = resolve_filters("Compare ITC and Maruti Suzuki revenue for the quarter ended June 30, 2026", AVAILABLE)
    assert r["filters"] == [{"company": "ITC", "period": "Q1FY27"},
                            {"company": "Maruti Suzuki", "period": "Q1FY27"}]
