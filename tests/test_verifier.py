"""
Tests for the cell-level grounding verifier (generation/verifier.py).

Each wrong-answer case mirrors an error the eval actually produced: a real
number copied from an earlier-period column, from a different row, or from
the standalone table when the question asked for consolidated.

Run: pytest tests/test_verifier.py -v
"""

from generation.verifier import Verifier

TABLE = {
    "company": "TCS",
    "text": (
        "TCS | Q3FY25 | consolidated quarterly results | page 7\n"
        "Audited Consolidated Interim Statement of Financial Results\n"
        "| | Three months ended - December 31, 2024 | Three months ended - September 30, 2024 "
        "| Nine months ended - December 31, 2024 |\n"
        "|---|---|---|---|\n"
        "| Revenue from operations | 63,973 | 64,259 | 1,90,845 |\n"
        "| TOTAL INCOME | 65,216 | 64,988 | 1,93,779 |\n"
        "| PROFIT BEFORE TAX | 16,666 | 16,032 | 48,929 |\n"
    ),
}
STANDALONE = {
    "company": "TCS",
    "text": (
        "TCS | Q3FY25 | standalone quarterly results | page 13\n"
        "| | Three months ended - December 31, 2024 | Three months ended - September 30, 2024 |\n"
        "|---|---|---|\n"
        "| Revenue from operations | 53,883 | 53,990 |\n"
    ),
}


def verify(answer, query, chunks=(TABLE, STANDALONE)):
    return Verifier().verify(answer, list(chunks), query=query)


def test_correct_figure_from_the_right_cell_is_grounded():
    r = verify("- Revenue from operations (Q3 FY25): ₹ 63,973 crore【TCS | Q3FY25 | Page 7】",
               "What was TCS consolidated revenue from operations in Q3FY25?")
    assert r["is_grounded"], r["unverified_claims"]


def test_prior_quarter_value_is_flagged():
    r = verify("Revenue from operations for Q3FY25 was ₹ 64,259 crore.",
               "What was TCS revenue from operations in Q3FY25?")
    assert not r["is_grounded"]
    assert any("earlier-period" in c for c in r["unverified_claims"])


def test_prior_quarter_value_is_fine_when_framed_as_a_comparison():
    r = verify("Revenue from operations was ₹ 63,973 crore, compared to ₹ 64,259 crore in the previous quarter.",
               "What was TCS revenue from operations in Q3FY25?")
    assert r["is_grounded"], r["unverified_claims"]


def test_nine_month_question_needs_the_nine_month_column():
    r = verify("Nine-month revenue from operations was ₹ 63,973 crore.",
               "What was TCS nine month revenue from operations ending December 2024?")
    assert not r["is_grounded"]


def test_answer_mislabelling_another_row_is_flagged():
    # the model called PBT "total income" — its own wording can't vouch for the row
    r = verify("Total income (profit before tax) for the quarter: ₹ 16,666 crore.",
               "What was TCS total income in Q3FY25?")
    assert not r["is_grounded"]


def test_standalone_figure_for_a_consolidated_question_is_flagged():
    r = verify("Consolidated revenue from operations was ₹ 53,883 crore.",
               "What was TCS consolidated revenue from operations in Q3FY25?")
    assert not r["is_grounded"]
    assert any("standalone table" in c for c in r["unverified_claims"])


def test_invented_number_is_flagged():
    r = verify("Revenue from operations was ₹ 68,451 crore.",
               "What was TCS revenue from operations in Q3FY25?")
    assert not r["is_grounded"]
    assert any("not in sources" in c for c in r["unverified_claims"])


def test_dates_citations_and_identifiers_are_not_claims():
    r = verify("The CIN is L22210MH1995PLC084781 and the board met on January 9, 2025 "
               "(nine months ended 31 December 2024)【TCS | Q3FY25 | Page 7】.",
               "What is the CIN of Tata Consultancy Services?")
    assert r["is_grounded"]
    assert r["verified_claims"] == [] and r["unverified_claims"] == []


def test_not_available_answer_is_grounded():
    r = verify("This information is not available in the provided filings.", "What was X?")
    assert r["is_grounded"]
