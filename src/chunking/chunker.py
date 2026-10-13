"""
chunker.py

Takes the raw FilingBlock list from pdf_parser.py and produces
embedding-ready chunks with the following rules:

TABLES  → never cut mid-row. Large tables are split into row groups,
          each repeating the caption + column headers, so every piece is
          a self-describing table — "| Revenue | 45" alone means nothing.

PROSE   → split by section heading first (MD&A, Risk Factors, Notes to
          Accounts, etc.). If a section is still too long, split by token
          count with overlap so context isn't lost at chunk boundaries.

Every chunk gets a unique ID + all metadata so Qdrant payload filtering
works later (e.g. "only search Q2FY25 Reliance documents").

Usage:
    from chunker import chunk_blocks
    from ingestion.pdf_parser import parse_filing

    blocks = parse_filing("reliance_q2fy25.pdf", "Reliance Industries",
                          "quarterly_result", "Q2FY25")
    chunks = chunk_blocks(blocks)
    # chunks is a list of dicts, ready to embed + upsert into Qdrant
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Target token count per prose chunk. Kept deliberately modest so the
# CrossEncoder reranker later isn't flooded with giant context windows.
CHUNK_TARGET_TOKENS = 400

# Overlap between consecutive prose chunks so sentences at a boundary
# appear in both chunks — reduces "answer split across two chunks" misses.
CHUNK_OVERLAP_TOKENS = 60

# Tables above this are split into row groups (header repeated in each).
# Kept under the 512-token input limit of both bge-base and the reranker,
# with headroom for the cl100k vs WordPiece tokenizer mismatch.
TABLE_MAX_TOKENS = 400

# Section heading patterns common in Indian financial filings (annual
# reports, quarterly results). When we detect these, we split at the
# heading boundary regardless of token count — so MD&A doesn't bleed
# into Notes to Accounts.
SECTION_HEADING_PATTERNS = [
    r"^management[\s\']*s?\s+discussion\s+and\s+analysis",
    r"^director[\s\']*s?\s+report",
    r"^notes?\s+to\s+(the\s+)?financial\s+statements",
    r"^independent\s+auditor[\s\']*s?\s+report",
    r"^risk\s+factors?",
    r"^corporate\s+governance",
    r"^related[\s-]party\s+transactions?",
    r"^segment[\s-]wise\s+(performance|results?)",
    r"^standalone\s+financial\s+statements?",
    r"^consolidated\s+financial\s+statements?",
    r"^balance\s+sheet",
    r"^(statement\s+of\s+)?profit\s+(and|&)\s+loss",
    r"^cash\s+flow\s+statement",
    r"^(statement\s+of\s+)?changes?\s+in\s+equity",
    r"^chairman[\s\']*s?\s+(message|statement|letter)",
    r"^(key\s+)?financial\s+(highlights?|summary)",
    r"^operational\s+(highlights?|performance)",
    r"^business\s+overview",
    r"^\d+[\.\)]\s+[A-Z]",  # numbered sections common in Notes: "1. Basis of preparation"
]

_HEADING_RE = re.compile(
    "|".join(f"({p})" for p in SECTION_HEADING_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    block_type: str        # "prose" | "table"
    company: str
    doc_type: str
    period: str
    source_file: str
    page_number: int
    section_hint: str      # detected heading label, empty string if none
    token_count: int
    table_index: int | None
    extraction_method: str
    chunk_index: int       # position within the original block (for ordering)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "block_type": self.block_type,
            "company": self.company,
            "doc_type": self.doc_type,
            "period": self.period,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "section_hint": self.section_hint,
            "token_count": self.token_count,
            "table_index": self.table_index,
            "extraction_method": self.extraction_method,
            "chunk_index": self.chunk_index,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(text: str, company: str, page: int, chunk_index: int) -> str:
    """Deterministic chunk ID — same input always produces same ID,
    so re-running the pipeline doesn't create duplicates in Qdrant."""
    raw = f"{company}|{page}|{chunk_index}|{text[:120]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _detect_section(text: str) -> str:
    """Return the first section heading found in text, or empty string."""
    m = _HEADING_RE.search(text)
    if m:
        # return the matched heading, cleaned up, max 80 chars
        return m.group(0).strip()[:80]
    return ""


def _split_by_section(text: str) -> list[str]:
    """
    Split prose text at section headings. Each split keeps the heading
    attached to the content that follows it (not left dangling at the
    end of the previous chunk).
    """
    parts = _HEADING_RE.split(text)
    if len(parts) <= 1:
        return [text]  # no headings found — treat as single section

    # _HEADING_RE.split returns [before, g1, g2...gN, between, g1...] pattern
    # because of multiple capture groups. We need to reassemble cleanly.
    # Simpler: use re.split on line-start boundaries.
    sections = re.split(
        r"(?=(?:" + "|".join(SECTION_HEADING_PATTERNS) + r"))",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return [s.strip() for s in sections if s is not None and s.strip()]


def _split_by_tokens(
    text: str,
    target: int = CHUNK_TARGET_TOKENS,
    overlap: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Split text into token-limited chunks with overlap.
    Splits on sentence boundaries where possible (looks for ". " or "\n")
    so chunks don't end mid-sentence.
    """
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= target:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + target, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _TOKENIZER.decode(chunk_tokens)

        # Try to end at a clean sentence boundary within the last 300 chars —
        # but only accept it if the window still advances by more than
        # `overlap` tokens. Without this guard, a separator found near the
        # start of the window barely shrinks `end`, so the next `start`
        # (end - overlap) can land behind or right next to the current
        # `start`, degenerating into single-token-advance chunks.
        if end < len(tokens):
            for sep in [". \n", ".\n", ". ", "\n\n", "\n"]:
                last_sep = chunk_text.rfind(sep, max(0, len(chunk_text) - 300))
                if last_sep != -1:
                    candidate_text = chunk_text[: last_sep + len(sep)].strip()
                    candidate_end = start + len(_TOKENIZER.encode(candidate_text))
                    if candidate_end - start > overlap:
                        chunk_text = candidate_text
                        end = candidate_end
                    break

        chunks.append(chunk_text.strip())

        # Reached the end of the text — nothing left to slide the window
        # over, so stop instead of recomputing `start` (which, once `end`
        # is pinned at len(tokens), would otherwise creep forward by a
        # single token per iteration and emit dozens of near-duplicate
        # trailing chunks).
        if end >= len(tokens):
            break

        # next window starts overlap tokens before end
        start = max(end - overlap, start + 1)

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Core chunking logic
# ---------------------------------------------------------------------------

def _chunk_prose_block(block: dict, global_chunk_index: int) -> list[Chunk]:
    """
    Chunk a single prose block:
    1. Split by section headings
    2. For each section, split by token count if still too long
    """
    chunks: list[Chunk] = []
    idx = global_chunk_index

    sections = _split_by_section(block["text"])
    for section_text in sections:
        section_hint = _detect_section(section_text)
        token_chunks = _split_by_tokens(section_text)

        for tc in token_chunks:
            if not tc.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=_make_id(tc, block["company"], block["page_number"], idx),
                    text=tc,
                    block_type="prose",
                    company=block["company"],
                    doc_type=block["doc_type"],
                    period=block["period"],
                    source_file=block["source_file"],
                    page_number=block["page_number"],
                    section_hint=section_hint,
                    token_count=_count_tokens(tc),
                    table_index=None,
                    extraction_method=block.get("extraction_method", "text_layer"),
                    chunk_index=idx,
                )
            )
            idx += 1

    return chunks


def _split_table_rows(text: str, max_tokens: int = TABLE_MAX_TOKENS) -> list[str]:
    """
    Split a markdown table into row groups, repeating everything above the
    first body row (caption + header row + separator) at the top of every
    group — so each piece is still a self-describing table, never a bare
    "| Revenue | 45 |" fragment.

    Needed because both the embedder (bge-base) and the CrossEncoder
    truncate input at 512 tokens: a full P&L statement runs ~1,500 tokens,
    so as a single chunk its bottom rows (EPS, dividends, share capital)
    were never embedded or reranked at all.
    """
    if _count_tokens(text) <= max_tokens:
        return [text]

    lines = text.splitlines()
    sep_idx = next(
        (i for i, l in enumerate(lines) if re.fullmatch(r"\|[-|\s:]+\|", l.strip())),
        None,
    )
    if sep_idx is None:
        return [text]   # not a markdown table — keep whole rather than guess

    head = "\n".join(lines[: sep_idx + 1])
    budget = max_tokens - _count_tokens(head)
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for row in lines[sep_idx + 1 :]:
        row_tokens = _count_tokens(row) + 1
        if current and current_tokens + row_tokens > budget:
            groups.append(head + "\n" + "\n".join(current))
            current, current_tokens = [], 0
        current.append(row)
        current_tokens += row_tokens
    if current:
        groups.append(head + "\n" + "\n".join(current))
    return groups


def _chunk_table_block(block: dict, global_chunk_index: int) -> list[Chunk]:
    """
    Tables are never cut mid-row. Small tables stay as one chunk; large
    ones are split into row groups that each carry the caption + column
    headers (see _split_table_rows).
    """
    chunks = []
    for i, text in enumerate(_split_table_rows(block["text"])):
        idx = global_chunk_index + i
        chunks.append(
            Chunk(
                chunk_id=_make_id(text, block["company"], block["page_number"], idx),
                text=text,
                block_type="table",
                company=block["company"],
                doc_type=block["doc_type"],
                period=block["period"],
                source_file=block["source_file"],
                page_number=block["page_number"],
                section_hint=_detect_section(text),
                token_count=_count_tokens(text),
                table_index=block.get("table_index"),
                extraction_method=block.get("extraction_method", "text_layer"),
                chunk_index=idx,
            )
        )
    return chunks


def chunk_blocks(blocks: list[dict]) -> list[Chunk]:
    """
    Main entry point. Takes raw blocks from pdf_parser (as dicts or
    FilingBlock objects) and returns a flat list of Chunk objects,
    ordered by page then chunk position.

    Args:
        blocks: list of block dicts (from FilingBlock.to_dict()) or
                FilingBlock objects directly.

    Returns:
        list[Chunk] ready to embed and upsert into Qdrant.
    """
    # Normalize: accept both FilingBlock objects and plain dicts
    raw = []
    for b in blocks:
        raw.append(b.to_dict() if hasattr(b, "to_dict") else b)

    all_chunks: list[Chunk] = []
    idx = 0
    basis = _page_basis(raw)

    for block in raw:
        if block["block_type"] == "table":
            new_chunks = _chunk_table_block(block, idx)
        else:
            new_chunks = _chunk_prose_block(block, idx)

        header = _context_header(block, basis.get((block["source_file"], block["page_number"]), ""))
        for c in new_chunks:
            c.text = f"{header}\n{c.text}"
            c.token_count = _count_tokens(c.text)
            c.chunk_id = _make_id(c.text, c.company, c.page_number, c.chunk_index)

        all_chunks.extend(new_chunks)
        idx += len(new_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# Contextual chunk headers
# ---------------------------------------------------------------------------

_DOC_LABELS = {"quarterly_result": "quarterly results", "annual_report": "annual report"}


def _page_basis(blocks: list[dict]) -> dict[tuple[str, int], str]:
    """
    Decide, per page, whether it belongs to the standalone or consolidated
    statements, from the page's own text (headings + table captions).

    Filings print near-identical standalone and consolidated tables on
    neighbouring pages; without this, a row-group chunk from one is
    indistinguishable from the other, and retrieval picked the wrong
    sibling (all 10 misses on the six Q1FY27 companies were this).
    A page that names neither is consolidated if it has minority /
    non-controlling-interest lines (only a group with subsidiaries has
    those — this catches HDFC Bank's consolidated page, whose title OCR'd
    as "UN P ED ED D D G"); otherwise it's a statement continued from the
    previous page and inherits that page's basis.
    """
    pages: dict[tuple[str, int], list[str]] = {}
    for b in blocks:
        pages.setdefault((b["source_file"], b["page_number"]), []).append(b["text"].lower())

    basis: dict[tuple[str, int], str] = {}
    prev: dict[str, str] = {}
    for key in sorted(pages):
        text = " ".join(pages[key])
        has_c, has_s = "consolidated" in text, "standalone" in text
        if has_c and has_s:
            value = "standalone and consolidated"
        elif has_c:
            value = "consolidated"
        elif has_s:
            value = "standalone"
        elif "minority interest" in text or "non-controlling interest" in text:
            value = "consolidated"
        else:
            value = prev.get(key[0], "")
        basis[key] = prev[key[0]] = value
    return basis


def _context_header(block: dict, basis: str) -> str:
    """One line naming where a chunk comes from, prepended before embedding
    so dense + BM25 retrieval (and the LLM) can tell sibling tables apart,
    e.g. "ITC | Q1FY27 | standalone quarterly results | page 3"."""
    doc = _DOC_LABELS.get(block["doc_type"], block["doc_type"])
    kind = f"{basis} {doc}" if basis else doc
    return f"{block['company']} | {block['period']} | {kind} | page {block['page_number']}"


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------

def chunk_jsonl(
    input_path: str,
    output_path: str,
) -> int:
    """
    Read blocks from a JSONL file (pdf_parser output), chunk them,
    and write resulting chunks to another JSONL file.
    Returns the number of chunks written.
    """
    blocks = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                blocks.append(json.loads(line))

    chunks = chunk_blocks(blocks)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    return len(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk parsed filing blocks into embedding-ready units.")
    parser.add_argument("input", help="JSONL file from pdf_parser (data/processed/chunks.jsonl)")
    parser.add_argument("--output", default="data/processed/chunked.jsonl")
    args = parser.parse_args()

    n = chunk_jsonl(args.input, args.output)
    print(f"✓ {n} chunks written to {args.output}")

    # Quick stats
    with open(args.output, encoding="utf-8") as f:
        all_c = [json.loads(l) for l in f if l.strip()]

    prose = [c for c in all_c if c["block_type"] == "prose"]
    tables = [c for c in all_c if c["block_type"] == "table"]
    avg_tokens = sum(c["token_count"] for c in all_c) / max(len(all_c), 1)

    print(f"  Prose chunks : {len(prose)}")
    print(f"  Table chunks : {len(tables)}")
    print(f"  Avg tokens   : {avg_tokens:.0f}")