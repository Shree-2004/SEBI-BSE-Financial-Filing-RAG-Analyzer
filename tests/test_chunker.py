"""
Regression test for a real bug found and fixed during ingestion of Infosys's
Q1FY27 filing: `_split_by_tokens()` in chunking/chunker.py degenerated into
single-token-advance increments once its sliding window neared the end of a
long section, emitting dozens of near-duplicate trailing chunks per section.

Confirmed impact before the fix: one page of the Infosys filing alone
produced 64 chunks (mostly duplicate one-word-shorter slices of the same
trailing sentence) instead of 4, and ~24% of the entire corpus across all
three indexed companies was affected — real content was getting crowded out
of retrieval's top-k by a swarm of near-identical junk.

Run: pytest tests/test_chunker.py -v
"""

from chunking.chunker import _split_by_tokens, CHUNK_TARGET_TOKENS


def _make_long_text_with_short_tail() -> str:
    """Reproduces the shape that triggered the bug: a long body (forcing
    token-based splitting) followed by a short trailing sentence — the
    exact pattern of Infosys's disclosure footer that got shredded."""
    filler = "This is filler content representing a long financial disclosure section. " * 40
    tail = (
        "EPS is not annualized for the quarter ended June 30, 2026, "
        "quarter ended March 31, 2026 and quarter ended June 30, 2025.\n"
        "# Excludes non-controlling interest"
    )
    return filler + tail


def test_split_by_tokens_does_not_explode_into_duplicate_chunks():
    text = _make_long_text_with_short_tail()
    chunks = _split_by_tokens(text)

    # Before the fix, the tail alone produced up to ~overlap (60) degenerate
    # chunks. A correctly functioning sliding window over this text should
    # produce a small, bounded number of chunks, not dozens.
    assert len(chunks) < 10, (
        f"expected a small number of chunks, got {len(chunks)} — "
        "this is the signature of the sliding-window duplication bug"
    )


def test_split_by_tokens_no_near_duplicate_suffix_chunks():
    """The degenerate bug produces consecutive chunks that are near-identical
    suffixes of each other (each one word shorter than the last). Detect
    that pattern directly rather than only checking the count."""
    text = _make_long_text_with_short_tail()
    chunks = _split_by_tokens(text)

    for i in range(len(chunks) - 1):
        a, b = chunks[i], chunks[i + 1]
        # A near-duplicate suffix pair: the shorter one is (almost) a
        # trailing substring of the longer one, and they differ in length
        # by only a few characters — the hallmark of a 1-token creep.
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if len(longer) - len(shorter) <= 15 and shorter and longer.endswith(shorter[-20:]):
            assert False, (
                f"found near-duplicate suffix chunks (lengths {len(a)}, {len(b)}) — "
                "the sliding window is creeping forward one token at a time again"
            )


def test_split_by_tokens_preserves_trailing_content():
    """The fix must not lose content at the end of a long section — only
    stop the *degenerate re-emission* of it."""
    text = _make_long_text_with_short_tail()
    chunks = _split_by_tokens(text)

    assert any("Excludes non-controlling interest" in c for c in chunks), (
        "trailing content was dropped, not just de-duplicated"
    )


def test_split_by_tokens_short_text_returns_single_chunk():
    """Sanity check: text under the target token count should never be
    split at all."""
    text = "A short section that doesn't need splitting."
    chunks = _split_by_tokens(text)
    assert chunks == [text]
