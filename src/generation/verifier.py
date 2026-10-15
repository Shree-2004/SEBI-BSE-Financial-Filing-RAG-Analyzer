"""
verifier.py

Post-generation grounding check: is every figure in the answer backed by
the right cell of the retrieved sources?

Verifier (current) checks each number in the answer against the source
cell it could have come from:
- row:      the cell's row label matches what the answer says the number is
- column:   the cell is in the period asked about (latest quarter vs nine
            months vs year), unless the answer frames it as a comparison
- basis:    consolidated claims come from consolidated tables (and vice versa)
- question: at least one figure comes from the row the QUESTION asks about,
            so an answer can't vouch for a figure just by mislabelling it

Why: on the eval, wrong answers almost never contain invented numbers.
The LLM copies a REAL number from the wrong row, column or statement, so
"does this number appear in the context?" passes nearly all of them.
KeywordVerifier (the original substring check) is kept as the baseline.

Measured with eval/verifier_eval.py:
- mutation test (correct answers with a wrong figure swapped in, not used
  for tuning): 85% caught (113/133) vs 0% for KeywordVerifier
- real wrong answers from the current pipeline: 3/6 caught (vs 0/6), with
  1 false alarm in 54 correct answers

Not caught by design: answers that omit part of the answer, errors already
present in the parsed source table, and non-numeric answers.

Usage:
    from generation.verifier import Verifier
    result = Verifier().verify(answer, chunks, query=question)
    result["is_grounded"], result["confidence"], result["unverified_claims"]
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Patterns for extracting verifiable claims
# ---------------------------------------------------------------------------

# Financial figures — numbers with currency or % symbols
# matches: Rs 19,101 crore | ₹4.3% | 23.5% | 1,234.56
FIGURE_PATTERN = re.compile(
    r"(?:Rs\.?|₹)?\s*[\d,]+(?:\.\d+)?\s*(?:crore|lakh|billion|million|%)?",
    re.IGNORECASE,
)

# Period references — Q1/Q2/Q3/Q4 + FY
PERIOD_PATTERN = re.compile(r"Q[1-4]FY\d{2,4}|FY\d{2,4}", re.IGNORECASE)

# Key financial terms that should appear in source if claimed
FINANCIAL_TERMS = [
    "net profit", "revenue", "ebitda", "margin", "nim", "net interest margin",
    "dividend", "eps", "earnings per share", "debt", "equity", "cash flow",
    "operating profit", "gross profit", "total income", "net income",
    "return on equity", "roe", "return on assets", "roa", "book value",
    "market cap", "shareholding", "promoter", "related party",
]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    is_grounded: bool
    confidence: float           # 0.0 = fully ungrounded, 1.0 = fully grounded
    verified_claims: list[str]
    unverified_claims: list[str]
    warning: str | None         # human-readable warning if ungrounded


class KeywordVerifier:
    """Original keyword/substring grounding check, kept as the baseline
    for eval/verifier_eval.py. Superseded by Verifier below."""

    def verify(
        self,
        answer: str,
        chunks: list[dict],
        confidence_threshold: float = 0.5,
    ) -> dict:
        """
        Check whether the generated answer is grounded in the source chunks.

        Args:
            answer               : generated answer text from AnswerGenerator
            chunks               : the same chunks passed to AnswerGenerator
            confidence_threshold : below this, mark as ungrounded

        Returns:
            dict with keys:
                is_grounded       : bool
                confidence        : float 0-1
                verified_claims   : claims found in source chunks
                unverified_claims : claims NOT found in source chunks
                warning           : warning string or None
        """
        # combine all chunk text into one searchable corpus
        corpus = " ".join(c["text"].lower() for c in chunks)

        # special case: if model said "not available", that's honest + grounded
        not_available_phrases = [
            "not available in the provided",
            "not found in the provided",
            "cannot find this information",
            "no information available",
        ]
        if any(phrase in answer.lower() for phrase in not_available_phrases):
            return VerificationResult(
                is_grounded=True,
                confidence=1.0,
                verified_claims=["Model correctly reported information not found"],
                unverified_claims=[],
                warning=None,
            ).__dict__

        verified = []
        unverified = []

        # --- Check 1: financial figures ---
        figures = FIGURE_PATTERN.findall(answer)
        for fig in figures:
            fig_clean = fig.strip()
            if len(fig_clean) < 2:
                continue
            # normalize: remove spaces and commas for comparison
            fig_norm = re.sub(r"[\s,]", "", fig_clean.lower())
            corpus_norm = re.sub(r"[\s,]", "", corpus)

            if fig_norm in corpus_norm:
                verified.append(f"figure: {fig_clean}")
            else:
                unverified.append(f"figure: {fig_clean}")

        # --- Check 2: financial terms ---
        answer_lower = answer.lower()
        for term in FINANCIAL_TERMS:
            if term in answer_lower:
                if term in corpus:
                    verified.append(f"term: {term}")
                else:
                    unverified.append(f"term: {term}")

        # --- Check 3: period references ---
        periods = PERIOD_PATTERN.findall(answer)
        for period in periods:
            if period.lower() in corpus:
                verified.append(f"period: {period}")
            else:
                unverified.append(f"period: {period}")

        # --- Confidence score ---
        total = len(verified) + len(unverified)
        if total == 0:
            # no verifiable claims found — give benefit of the doubt
            confidence = 0.7
        else:
            confidence = len(verified) / total

        is_grounded = confidence >= confidence_threshold

        warning = None
        if not is_grounded:
            warning = (
                f"⚠ Answer may contain ungrounded claims. "
                f"Only {len(verified)}/{total} verifiable claims found in source chunks. "
                f"Unverified: {', '.join(unverified[:3])}"
                + (" and more." if len(unverified) > 3 else ".")
            )

        return VerificationResult(
            is_grounded=is_grounded,
            confidence=round(confidence, 3),
            verified_claims=verified,
            unverified_claims=unverified,
            warning=warning,
        ).__dict__


# ---------------------------------------------------------------------------
# Cell-level verifier
# ---------------------------------------------------------------------------
#
# Measured on 139 labelled eval answers (eval/verifier_eval.py), the keyword
# verifier above flagged only 10 of 42 wrong answers. The reason: a wrong
# answer almost never contains an invented number — the LLM copies a REAL
# number from the wrong row, the wrong column (prior quarter), or the wrong
# statement (standalone vs consolidated). "Is this number somewhere in the
# context?" is true for nearly all of them.
#
# So each number in the answer is checked against the specific source cell
# it could have come from: row label, column (period), and statement basis.

_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
           r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_STRIP_PATTERNS = [
    re.compile(r"\[[^\]]*\]|【[^】]*】"),                                      # citations
    re.compile(_MONTHS + r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}", re.I),        # June 30, 2026
    re.compile(r"\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTHS + r",?\s+\d{4}", re.I),  # 30 June 2026
    re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"),                         # 30.06.2026
    re.compile(r"\bQ[1-4]\s*FY\s*'?\d{2,4}\b", re.I),                           # Q1FY27, Q3 FY 25
    re.compile(r"\bFY\s*'?\d{2,4}(?:\s*-\s*\d{2,4})?\b", re.I),                 # FY2025, FY 2024-25
    re.compile(r"\b(?:pages?|pp?\.)\s*\d+(?:\s*(?:,|and|&|-)\s*\d+)*", re.I),  # page 5, pages 27 and 146
    re.compile(r"\bexcerpt\s+\d+\b", re.I),
    re.compile(r"\b(?=\w*[A-Za-z])(?=\w*\d)\w{6,}\b"),                           # CIN-style IDs
    re.compile(r"[×x*]\s*100\b(?!\s*[,.]\d)|\b100\s*[×x*]"),                     # formula "× 100"
    re.compile(r"\b(?:19|20)\d{2}\b(?![,.]\d)"),                                # bare years
]
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z]{3,}")
_STOPWORDS = set("""
the and for from with was were are this that its per each into than over under
period quarter quarters ended ending three months month nine year years total
crore crores lakh million billion inr rupees company group limited ltd figure
figures amount reported value values stood came based financial results result
statement statements consolidated standalone unaudited audited reviewed what
""".split())
_COMPARISON_WORDS = ("previous", "prior", "compared", "versus", " vs", "last year", "year-ago",
                     "year ago", "preceding", "corresponding", "sequential", "earlier",
                     "up from", "down from", "increase from", "decrease from")


_UNICODE_FIXES = str.maketrans({"\u2011": "-", "\u2010": "-", "\u2013": "-", "\u2014": "-",
                                "\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u2007": " "})


def _normalise(text: str) -> str:
    return text.translate(_UNICODE_FIXES)


def _num_values(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _content_words(text: str, extra_stop=frozenset()) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and w not in extra_stop}


def _fuzzy_overlap(row_words: set[str], claim_words: set[str]) -> int:
    """Shared words, tolerating OCR typos ("reverue" ~ "revenue")."""
    from difflib import SequenceMatcher
    n = 0
    for r in row_words:
        if r in claim_words or any(len(r) >= 5 and SequenceMatcher(None, r, c).ratio() >= 0.8
                                   for c in claim_words):
            n += 1
    return n


def _row_matches(row_label: str, claim_words: set[str]) -> bool:
    row_words = _content_words(row_label)
    if not row_words:
        return True
    shared = _fuzzy_overlap(row_words, claim_words)
    # most of the row label is what the claim is about, or a strong 2-word overlap
    return shared / len(row_words) >= 0.5 or shared >= 2


# Row qualifiers that change what a figure is: "profit BEFORE tax" is not
# "profit", "OTHER income" is not "income", "total COMPREHENSIVE income" is
# not "profit". A row with one of these only answers a question that asks
# for it.
_QUALIFIERS = {"before", "comprehensive", "other", "deferred"}


def _answers_question(row_label: str, question_words: set[str]) -> bool:
    row_words = _content_words(row_label)
    if not row_words:
        return False
    if (row_words & _QUALIFIERS) - question_words:
        return False
    # either the row label is mostly what was asked ("Manufacturing" for
    # "manufacturing segment revenue"), or the question is mostly the row
    return _row_matches(row_label, question_words) or         _fuzzy_overlap(question_words, row_words) / len(question_words) >= 0.5


def _period_kind(text: str) -> str:
    t = text.lower()
    if re.search(r"nine[\s-]*months?|9[\s-]*months", t):
        return "nine"
    # "\bannual\b": "(not annualised)" next to a quarterly EPS is not a year claim
    if re.search(r"year ended|full[\s-]year|\bannual\b|twelve months|\bfy\s*\d{4}\s*-\s*\d{2}", t):
        return "year"
    if re.search(r"three months|\b3\s*months|quarter|\bq[1-4]\b", t):   # ITC: "3 Months ended"
        return "quarter"
    return ""


_MONTH_NUM = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _header_date(text: str) -> tuple[int | None, int | None]:
    """(month, year) from a column header ("June 30, 2026", "30.06.2026",
    "31 .03.2026"); either can be None. Shared with metrics/extract.py."""
    t = text.lower()
    m = re.search(r"(\d{1,2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{4})", t)
    if m:
        return int(m.group(2)), int(m.group(3))
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*,?\s*(?:\d{1,2}(?:st|nd|rd|th)?)?"
                  r"\s*,?\s*-?\s*(\d{4})?", t)
    if m:
        return _MONTH_NUM[m.group(1)], int(m.group(2)) if m.group(2) else None
    return None, None


def _basis(text: str) -> str:
    t = text.lower()
    c, s = "consolidated" in t, "standalone" in t
    return "both" if c and s else "consolidated" if c else "standalone" if s else ""


@dataclass
class _Cell:
    value: float
    row_label: str
    column: str        # header text of the cell's column
    column_rank: int   # 0 = leftmost column of its period kind (the latest period)
    kind: str          # "quarter" | "nine" | "year" | "" (header carries no date)
    basis: str         # from the chunk's context header
    is_table: bool


def _cells_from_chunk(chunk: dict) -> list[_Cell]:
    text = _normalise(chunk.get("text", ""))
    lines = text.splitlines()
    # context header written by the chunker: "ITC | Q1FY27 | standalone quarterly results | page 3"
    basis = _basis(lines[0]) if lines and lines[0].count(" | ") >= 3 else ""

    sep = next((i for i, l in enumerate(lines) if re.fullmatch(r"\|[-|\s:]+\|", l.strip())), None)
    if sep is not None and sep > 1:
        basis = _basis(" ".join(lines[:sep - 1])) or basis
    cells: list[_Cell] = []
    if sep is not None and sep > 0:
        header = [h.strip() for h in lines[sep - 1].strip().strip("|").split("|")]
        kinds = [_period_kind(h) if re.search(r"\d", h) else "" for h in header]
        # Recency by header DATE, not position: ITC's summary table puts the
        # standalone and consolidated current-quarter columns side by side,
        # so "leftmost quarter column = latest" mislabels the second one.
        dates = [_header_date(h) for h in header]
        latest: dict[str, tuple] = {}
        for k, (m, y) in zip(kinds, dates):
            if k and m and y:
                latest[k] = max(latest.get(k, (0, 0)), (y, m))
        seen: dict[str, int] = {}
        col_rank = []
        for k, (m, y) in zip(kinds, dates):
            if k and m and y:
                col_rank.append(0 if (y, m) == latest[k] else 1)
            else:                                  # undated column: fall back to position
                col_rank.append(seen.get(k, 0) if k else 0)
                if k:
                    seen[k] = seen.get(k, 0) + 1
        for line in lines[sep + 1:]:
            if not line.strip().startswith("|"):
                continue
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            label = " ".join(p for p in parts if re.search(r"[A-Za-z]{3,}", p))
            for j, p in enumerate(parts):
                for v in _num_values(re.sub(r"\(\s*\d+\s*\)|\b[ivx]+\)|\b\d+\)", " ", p)
                                     if re.search(r"[A-Za-z]{3,}", p) else p):
                    cells.append(_Cell(v, label, header[j] if j < len(header) else "",
                                       col_rank[j] if j < len(col_rank) else 0,
                                       kinds[j] if j < len(kinds) else "", basis, True))
        body = "\n".join(lines[:sep - 1])   # prose above the table (context header, caption)
    else:
        body = text
    # prose: each sentence/line acts as its own "row"
    if lines and lines[0].count(" | ") >= 3:
        body = body.split("\n", 1)[1] if "\n" in body else ""
    for sent in re.split(r"(?<=[.;])\s+|\n", body):
        for v in _num_values(sent):
            cells.append(_Cell(v, sent, "", 0, "", basis, False))
    return cells


class Verifier:
    """
    Cell-level grounding check. For every number in the answer, look for a
    source cell with the same value AND:
      - row:    the cell's row label shares a content word with what the
                answer (or question) says the number is
      - column: the cell sits in the period the claim is about (latest
                quarter vs nine months vs year), unless the answer line
                frames it as a comparison ("compared to the previous quarter")
      - basis:  a "consolidated" claim comes from a consolidated table and
                vice versa (uses the chunk's context header)
    The question is used as context too: answers often omit "consolidated"
    or the period that the question specified.
    """

    def verify(self, answer: str, chunks: list[dict], query: str = "",
               confidence_threshold: float = 1.0) -> dict:
        not_available = ("not available in the provided", "not found in the provided",
                         "cannot find this information", "no information available")
        if any(p in answer.lower() for p in not_available):
            return VerificationResult(True, 1.0, ["Model correctly reported information not found"],
                                      [], None).__dict__

        cells = [c for ch in chunks for c in _cells_from_chunk(ch)]
        company_words = {w for ch in chunks for w in _WORD_RE.findall(str(ch.get("company", "")).lower())}
        q_basis, q_kind = _basis(query), _period_kind(query)

        verified, unverified, matched_cells = [], [], []
        for line in re.split(r"\n|(?<=[.;])\s+", _normalise(answer)):
            clean = line
            for pat in _STRIP_PATTERNS:
                clean = pat.sub(" ", clean)
            clean = clean.replace("**", " ")
            values = _num_values(clean)
            if not values:
                continue
            claim_words = _content_words(clean + " " + query, company_words)
            line_basis = _basis(clean) or q_basis
            line_kind = _period_kind(clean) or q_kind or "quarter"
            comparison = any(w in clean.lower() for w in _COMPARISON_WORDS)

            for v in values:
                ok, reason, cell = self._check(v, cells, claim_words, line_basis, line_kind, comparison)
                label = f"{v:,.2f}".rstrip("0").rstrip(".")
                (verified if ok else unverified).append(f"{label}: {reason}")
                if cell is not None:
                    matched_cells.append(cell)

        # The answer must actually report the metric the question asks for:
        # an answer can label a figure however it likes ("Total income
        # (profit before tax): 16,666"), so its own wording can't vouch for
        # the row — the question's wording must match at least one source row.
        question_words = _content_words(query, company_words)
        if query and question_words and (verified or unverified) \
                and not any(_answers_question(c.row_label, question_words) for c in matched_cells):
            unverified.append("no figure in the answer comes from the row the question asks about")

        total = len(verified) + len(unverified)
        confidence = len(verified) / total if total else 1.0
        is_grounded = confidence >= confidence_threshold
        warning = None
        if not is_grounded:
            warning = ("⚠ Some figures could not be matched to the right source cell: "
                       + "; ".join(unverified[:3]) + (" …" if len(unverified) > 3 else ""))
        return VerificationResult(is_grounded, round(confidence, 3), verified, unverified, warning).__dict__

    @staticmethod
    def _check(value, cells, claim_words, basis, kind, comparison):
        same = [c for c in cells if abs(c.value - value) < 1e-9]
        if not same:
            return False, "not in sources", None
        reasons = []
        for c in same:
            if c.is_table and not _content_words(c.row_label):
                reasons.append("found only in an unlabelled table row")
                continue
            if not _row_matches(c.row_label, claim_words):
                reasons.append(f"found under a different row ('{c.row_label[:60]}')")
                continue
            if c.is_table and basis in ("consolidated", "standalone") \
                    and c.basis in ("consolidated", "standalone") and c.basis != basis:
                reasons.append(f"found in a {c.basis} table, answer is about {basis}")
                continue
            ck = c.kind
            if c.is_table and ck and not comparison:
                if ck != kind:
                    reasons.append(f"found in a different period column ('{c.column[:50]}')")
                    continue
                if c.column_rank > 0:
                    reasons.append(f"found in an earlier-period column ('{c.column[:50]}')")
                    continue
            where = f"row '{c.row_label[:50]}'" + (f", column '{c.column[:40]}'" if c.column else "")
            return True, ("table " if c.is_table else "text ") + where, c
        return False, reasons[0], None


# ---------------------------------------------------------------------------
# Convenience: generate + verify in one call
# ---------------------------------------------------------------------------

def generate_and_verify(
    query: str,
    chunks: list[dict],
    generator,
    verifier: Verifier | None = None,
) -> dict:
    """
    Run AnswerGenerator then Verifier in sequence and return a combined
    response dict — this is what the FastAPI endpoint calls.

    Returns:
        dict with all generator output + verification result merged in.
    """
    if verifier is None:
        verifier = Verifier()

    gen_result = generator.generate(query=query, chunks=chunks)
    ver_result = verifier.verify(
        answer=gen_result["answer"],
        chunks=chunks,
        query=query,
    )

    return {
        **gen_result,
        "verification": ver_result,
        "is_grounded": ver_result["is_grounded"],
        "grounding_confidence": ver_result["confidence"],
        "warning": ver_result["warning"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verifier = Verifier()

    # Test 1: grounded answer
    grounded_answer = (
        "Reliance Industries reported a net profit of Rs 19,101 crore in Q2FY25, "
        "up 2.8% year-on-year. [Reliance Industries | Q2FY25 | Page 12]"
    )
    grounded_chunks = [
        {
            "text": "Reliance Industries reported a consolidated net profit of Rs 19,101 crore for Q2FY25, up 2.8% year-on-year.",
            "company": "Reliance Industries",
            "period": "Q2FY25",
        }
    ]

    result = verifier.verify(grounded_answer, grounded_chunks)
    print("Test 1 (should be grounded):")
    print(f"  is_grounded: {result['is_grounded']}")
    print(f"  confidence : {result['confidence']}")
    print(f"  warning    : {result['warning']}")

    # Test 2: hallucinated answer
    hallucinated_answer = (
        "Reliance Industries reported a net profit of Rs 25,000 crore in Q2FY25."
    )
    result2 = verifier.verify(hallucinated_answer, grounded_chunks)
    print("\nTest 2 (should be ungrounded — wrong number):")
    print(f"  is_grounded: {result2['is_grounded']}")
    print(f"  confidence : {result2['confidence']}")
    print(f"  warning    : {result2['warning']}")