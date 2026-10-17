"""
Tests for row-group splitting of large tables in chunking/chunker.py.

Large financial statements (~1,500 tokens) used to be kept as a single
chunk, but bge-base and the CrossEncoder both truncate at 512 tokens, so
the bottom rows (EPS, dividends, share capital) were never embedded or
reranked. Splitting must keep every piece self-describing: caption and
column headers repeated, no row cut in half, no row lost.

Run: pytest tests/test_table_split.py -v
"""

from chunking.chunker import TABLE_MAX_TOKENS, _count_tokens, _split_table_rows

CAPTION = "Audited Consolidated Interim Statement of Financial Results"
HEADER = "| | Three months ended - December 31, 2024 | Nine months ended - December 31, 2024 |"
SEP = "|---|---|---|"


def _make_table(n_rows: int) -> str:
    rows = [f"| Line item number {i} with a descriptive label | {i},111 | {i},999 |" for i in range(n_rows)]
    return "\n".join([CAPTION, HEADER, SEP, *rows])


def test_small_table_stays_whole():
    table = _make_table(3)
    assert _split_table_rows(table) == [table]


def test_large_table_pieces_repeat_caption_and_header():
    pieces = _split_table_rows(_make_table(120))
    assert len(pieces) > 1
    for p in pieces:
        lines = p.splitlines()
        assert lines[:3] == [CAPTION, HEADER, SEP]


def test_large_table_pieces_fit_token_budget():
    for p in _split_table_rows(_make_table(120)):
        assert _count_tokens(p) <= TABLE_MAX_TOKENS


def test_large_table_split_loses_and_duplicates_no_rows():
    table = _make_table(120)
    body = table.splitlines()[3:]
    pieces = _split_table_rows(table)
    rebuilt = [row for p in pieces for row in p.splitlines()[3:]]
    assert rebuilt == body


def test_non_markdown_text_is_not_split():
    text = "plain text without a separator row " * 200
    assert _split_table_rows(text) == [text]


def test_ocr_decimal_comma_fix_leaves_indian_grouping_alone():
    from ingestion.pdf_parser import _fix_ocr_decimal_commas as fix

    assert fix("| 25108,30 | 62005,98 |") == "| 25108.30 | 62005.98 |"
    assert fix("| 1,90,845 | 63,973 | 12,34,567 |") == "| 1,90,845 | 63,973 | 12,34,567 |"
    assert fix("| 19059.72 |") == "| 19059.72 |"


def _block(page, text, block_type="prose"):
    return {
        "text": text, "block_type": block_type, "company": "ITC", "doc_type": "quarterly_result",
        "period": "Q1FY27", "source_file": "itc.pdf", "page_number": page,
    }


def test_context_header_names_company_statement_and_page():
    from chunking.chunker import chunk_blocks

    chunks = chunk_blocks([
        _block(3, "Statement of Unaudited Standalone Financial Results"),
        _block(4, "| Revenue | 1 |\n|---|---|", "table"),          # continuation page
        _block(6, "Statement of Unaudited Consolidated Financial Results"),
    ])
    headers = [c.text.splitlines()[0] for c in chunks]
    assert headers == [
        "ITC | Q1FY27 | standalone quarterly results | page 3",
        "ITC | Q1FY27 | standalone quarterly results | page 4",
        "ITC | Q1FY27 | consolidated quarterly results | page 6",
    ]


def test_untitled_page_with_minority_interest_is_consolidated():
    from chunking.chunker import chunk_blocks

    chunks = chunk_blocks([
        _block(8, "Statement of Standalone Financial Results"),
        _block(9, "UN P ED ED D D G\n| Net profit before minority interest | 20382.69 |\n|---|---|", "table"),
    ])
    assert "| consolidated quarterly results | page 9" in chunks[-1].text.splitlines()[0]


def test_ocr_dotted_thousands_are_repaired_but_dates_and_decimals_are_not():
    from ingestion.pdf_parser import _fix_ocr_number_dots as fix

    assert fix("| 60.794.29 | 2.318.35 |") == "| 60,794.29 | 2,318.35 |"      # Tata Steel
    assert fix("| 18.430.12 | 69.850.79 |") == "| 18,430.12 | 69,850.79 |"    # Bajaj Finance
    assert fix("revenue 1.39,720.22 crore") == "revenue 1,39,720.22 crore"
    assert fix("| 1,39,720.22 | 36,896.55 | 19.19 |") == "| 1,39,720.22 | 36,896.55 | 19.19 |"
    assert fix("quarter ended 30.06.2026") == "quarter ended 30.06.2026"
