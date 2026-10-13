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
