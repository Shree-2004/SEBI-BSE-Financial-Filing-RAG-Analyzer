"""
Tests for the metrics table: header/period parsing, cell parsing, row ->
metric mapping (src/metrics/extract.py) and the voting lookup
(src/metrics/store.py).

Run: pytest tests/test_metrics.py -v
"""

import sqlite3

import pytest

from metrics.extract import SCHEMA, _column_period, _metric_for, _parse_number, extract_blocks
from metrics.store import MetricStore


@pytest.mark.parametrize("header, period, ptype", [
    ("Three months ended - December 31, 2024", "Q3FY25", "quarter"),
    ("Three months ended - September 30, 2024", "Q2FY25", "quarter"),
    ("Nine months ended - December 31, 2024", "9MFY25", "nine_months"),
    ("Year ended - March 31, 2024", "FY2024", "year"),
    ("Quarter ended - 30.06.2026 - Unaudited", "Q1FY27", "quarter"),
    ("Corresponding 3 Months ended 30.06.2025", "Q1FY26", "quarter"),      # ITC wording
    ("Yearended 31.03.2026 (Audited)", "FY2026", "year"),                  # OCR run-together (NTPC p10)
    ("Preceding 3 Months ended 31 .03.2026'", "Q4FY26", "quarter"),        # OCR gap in the date
    ("Q1 FY'26", "Q1FY26", "quarter"),                                      # L&T key parameters
    ("FY 2024-25", "FY2025", "year"),
    ("2024-25", "FY2025", "year"),                                          # Reliance annual report
])
def test_column_period(header, period, ptype):
    assert _column_period(header)[:2] == (period, ptype)


def test_column_without_year_has_no_period():
    # Infosys: "Quarter ended June 30," — filled from a donor header later
    assert _column_period("Quarter ended June 30,")[0] is None


@pytest.mark.parametrize("cell, value", [
    ("63,973", 63973), ("1,90,845", 190845), ("(1,480)", -1480), ("19.19", 19.19),
    ("21731 .84", 21731.84),      # OCR gap before the decimal point
    ("12,380 64", None),          # two merged cells — skipped, not guessed
    ("-- 2,603.72", 2603.72),     # table-rule dashes around a value (UltraTech)
    ("3,000.02 -", 3000.02),
    ("- -- 2,599.28", 2599.28),   # several rule dashes
    ("9.9%", None), ("-", None), ("", None),
])
def test_parse_number(cell, value):
    assert _parse_number(cell) == value


@pytest.mark.parametrize("label, metric", [
    ("Revenue from operati ons", "revenue_from_operations"),
    ("Total revenue from operations (I)", "revenue_from_operations"),
    ("TOTAL INCOME", "total_income"),
    ("PROFIT BEFORE TAX", "profit_before_tax"),
    ("Profit from ordinary activities before tax and minority interest (7)-(8)-(9)",
     "profit_before_tax_and_minority"),
    ("PROFIT FOR THE PERIOD", "net_profit"),
    ("Net Profit/ (Loss) for the period (before tax and Exceptional Items)", None),
    ("Earnings per equity share:- Basic and diluted (~)", "eps_basic"),
    ("Diluted (INR)", None),
    ("Depreciati on and amortisation expense", "depreciation_amortisation"),
    ("Total comprehensive income for the period", None),
    ("Tax expense/(credit) - Exceptional (Refer Note 4 and 5)", None),
    ("Income (a) Revenue from operations", "revenue_from_operations"),   # section heading glued on
    ("Tot al income", "total_income"),                                    # OCR split word
    ("7 Profit aftertax (5-6)", "net_profit"),                            # OCR run-together words
    ("Net profit margin (%)", None),
    ("Profit before regulatory deferral account balances (3-4)", None),
])
def test_metric_mapping(label, metric):
    assert _metric_for(label) == metric


def _blocks():
    header = ("| Particulars | Three months ended - December 31, 2024 | Three months ended - September 30, 2024 |\n"
              "|---|---|---|\n")
    return [
        {"block_type": "prose", "text": "Audited Consolidated Interim Statement (₹ crore)",
         "company": "TCS", "period": "Q3FY25", "source_file": "tcs.pdf", "page_number": 7},
        {"block_type": "table", "company": "TCS", "period": "Q3FY25", "source_file": "tcs.pdf",
         "page_number": 7, "text": header +
         "| Revenue from operations | 63,973 | 64,259 |\n"
         "| Profit for the period attributable to: | | |\n"
         "| Shareholders of the Company | 12,380 | 11,909 |\n"},
    ]


def test_extract_reads_every_period_column_with_unit_and_basis():
    rows = {(r["metric"], r["period"]): r for r in extract_blocks(_blocks())}
    rev = rows[("revenue_from_operations", "Q3FY25")]
    assert (rev["value"], rev["unit"], rev["basis"], rev["value_crore"]) == (63973, "crore", "consolidated", 63973)
    assert rows[("revenue_from_operations", "Q2FY25")]["value"] == 64259          # prior quarter too
    assert rows[("profit_attributable_to_owners", "Q3FY25")]["value"] == 12380   # sub-row of "attributable to:"


def _table(header, rows, page=10, company="X"):
    text = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    text += "".join("| " + " | ".join(r) + " |\n" for r in rows)
    return [{"block_type": "prose", "text": "Consolidated results (Rs. in crore)", "company": company,
             "period": "Q1FY27", "source_file": "x.pdf", "page_number": page},
            {"block_type": "table", "text": text, "company": company, "period": "Q1FY27",
             "source_file": "x.pdf", "page_number": page}]


def test_spanning_title_does_not_give_every_column_its_date():
    t = "Statement of Consolidated Results for the Three Months Ended 30/06/2026"
    rows = extract_blocks(_table(
        [f"{t} - Particulars", f"{t} - Three Months Ended - 30/06/2026", f"{t} - 30/06/2025",
         f"{t} - Year Ended - 31/03/2026"],
        [["Revenue from Operations", "24,648.20", "21,275.45", "88,511.53"]]))
    got = {r["period"]: r["value"] for r in rows}
    assert got == {"Q1FY27": 24648.20, "Q1FY26": 21275.45, "FY2026": 88511.53}


def test_label_wrapped_onto_two_rows_is_rejoined():
    rows = extract_blocks(_table(
        ["Particulars", "Quarter ended 30.06.2026", "Quarter ended 30.06.2025"],
        [["Other operating income", "141.99", "117.48"],
         ["operations", "23,165.45", "19,372.67"],
         ["Total revenue from", "", ""]]))
    got = {r["period"]: r["value"] for r in rows if r["metric"] == "revenue_from_operations"}
    assert got == {"Q1FY27": 23165.45, "Q1FY26": 19372.67}


def test_duplicated_column_period_is_dropped_not_guessed():
    rows = extract_blocks(_table(
        ["Particulars", "Quarter ended 30.06.2026", "Quarter ended 30.06.2026"],
        [["Revenue from operations", "100.00", "90.00"]]))
    assert rows == []


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "m.sqlite"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    def add(value, page, basis="consolidated", metric="revenue_from_operations", label="Revenue"):
        con.execute("INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("X", metric, basis, "Q1FY27", "quarter", value, "crore", "INR", value,
                     "x.pdf", page, label, "Quarter ended", "Q1FY27"))
    add(67941.74, 11)
    add(67942, 9)                                   # rounded summary copy of the same figure
    add(500, 3, metric="ebitda", label="EBITDA O2C")
    add(700, 4, metric="ebitda", label="EBITDA Retail")
    con.commit()
    return MetricStore(str(db))


def test_rounded_copies_agree_and_the_precise_value_wins(store):
    r = store.lookup("X", "revenue_from_operations", "Q1FY27")
    assert r["value"] == 67941.74 and r["sources_agreeing"] == 2


def test_split_sources_are_ambiguous_not_guessed(store):
    r = store.lookup("X", "ebitda", "Q1FY27")
    assert r["ambiguous"] and r["value"] is None and len(r["candidates"]) == 2


def test_unknown_figure_is_none(store):
    assert store.lookup("X", "net_profit", "Q1FY27") is None


# ---------------------------------------------------------------- embedder / collection pairing

def test_query_embedder_and_collection_are_paired(monkeypatch, tmp_path):
    from indexing import embed, onnx_models

    monkeypatch.delenv("EMBED_BACKEND", raising=False)
    monkeypatch.setattr(onnx_models, "EMBEDDER_DIR", tmp_path / "missing")
    assert embed.default_backend() == "torch" and embed.default_collection() == "filing_chunks"

    monkeypatch.setattr(onnx_models, "EMBEDDER_DIR", tmp_path)          # models prepared
    monkeypatch.setenv("EMBED_ONNX_VARIANT", "int8")
    assert embed.default_backend() == "onnx" and embed.default_collection() == "filing_chunks_int8"
    monkeypatch.setenv("EMBED_ONNX_VARIANT", "fp32")
    assert embed.default_collection() == "filing_chunks"


def test_unreadable_year_column_is_not_turned_into_a_second_quarter():
    from metrics.extract import _repair_periods
    periods = [_column_period(h) for h in
               ["Quarter ended 30.06.2026", "Quarter ended 31.03.2026", "Full 31.03.2026"]]
    assert [p[0] for p in _repair_periods(periods)] == ["Q1FY27", "Q4FY26", None]
