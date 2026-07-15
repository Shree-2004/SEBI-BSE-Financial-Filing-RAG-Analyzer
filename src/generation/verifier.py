"""
verifier.py

Post-generation grounding check — verifies that the key claims in the
generated answer actually appear in the retrieved source chunks.

This is the "production rigor" feature that makes this project stand out:
instead of silently returning a hallucinated answer, we detect when the
LLM went off-context and flag it clearly.

How it works:
- Extracts numbers and key terms from the generated answer
- Checks whether those figures/terms appear in the source chunks
- Returns a confidence score + a flag if the answer looks ungrounded
- Ungrounded answers are flagged (not silently returned) so the API
  response always tells you whether to trust the answer

This is not a perfect hallucination detector — it's a practical
grounding heuristic that catches the most common failure mode in
financial RAG: the LLM confidently stating a number that isn't in
the retrieved context.

Usage:
    from generation.verifier import Verifier

    verifier = Verifier()
    result = verifier.verify(
        answer="Reliance net profit was Rs 19,101 crore [Reliance | Q2FY25 | Page 12]",
        chunks=[...]
    )
    print(result["is_grounded"])     # True / False
    print(result["confidence"])      # 0.0 to 1.0
    print(result["unverified_claims"])  # list of claims not found in context
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


class Verifier:
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