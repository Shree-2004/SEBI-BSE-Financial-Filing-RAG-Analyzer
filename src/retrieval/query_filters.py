"""
query_filters.py

Detects which company (or companies) and which reporting period a question
is about, so retrieval can be filtered without the user picking from a
dropdown. Without a filter, the search runs across every company and
happily mixes, say, TCS's and Infosys's revenue tables.

Rule-based on purpose: the set of companies and the shape of Indian
reporting periods are small and fixed, so this is fast, free (no LLM
call), deterministic, and unit-testable.

Periods follow the Indian fiscal year (April-March), in the index's format:
    "Q3FY25", "Q3 FY 2025", "quarter ended December 31, 2024"   -> Q3FY25
    "quarter ended June 30, 2026"                                -> Q1FY27
    "FY25", "FY2025", "FY 2024-25", "year ended March 31, 2025"  -> FY2025
"""

import re

# lower-case aliases -> company name exactly as stored in the index
COMPANY_ALIASES: dict[str, list[str]] = {
    "Reliance Industries": [r"reliance", r"\bril\b"],
    "TCS": [r"\btcs\b", r"tata consultancy"],
    "Infosys": [r"infosys", r"\binfy\b"],
    "HDFC Bank": [r"hdfc"],
    "ITC": [r"\bitc\b"],
    "Bharti Airtel": [r"airtel", r"\bbharti\b"],
    "Sun Pharma": [r"sun\s*pharma"],
    "Larsen & Toubro": [r"larsen", r"\bl\s*&\s*t\b", r"\blnt\b"],
    "Maruti Suzuki": [r"maruti", r"\bmsil\b"],
}

_MONTH_NUM = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"


def _fy(year: int) -> str:
    return f"{year % 100:02d}"


def _quarter_for_month(month: int, year: int) -> str:
    """Indian FY: Apr-Jun = Q1 ... Jan-Mar = Q4; FY is named for its March."""
    q = (month - 4) % 12 // 3 + 1
    fy_end = year + 1 if month >= 4 else year
    return f"Q{q}FY{_fy(fy_end)}"


def _full_year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


def detect_companies(query: str) -> list[str]:
    q = query.lower()
    found = []
    for company, patterns in COMPANY_ALIASES.items():
        if any(re.search(p, q) for p in patterns):
            found.append(company)
    return found


def detect_period(query: str) -> str | None:
    q = query.lower().replace("‑", "-")

    # explicit quarter: Q3FY25, Q3 FY 2025, Q3-FY25
    m = re.search(r"\bq([1-4])\s*-?\s*fy\s*'?(\d{2,4})\b", q)
    if m:
        return f"Q{m.group(1)}FY{_fy(_full_year(m.group(2)))}"

    # "nine months" / "year" ended <date> belong to the filing for that date's quarter;
    # a plain full-year question ("year ended March 31, 2025") maps to the FY
    m = re.search(r"year\s+end(?:ed|ing)\s+(?:on\s+)?(?:" + _MONTH + r"\s+\d{1,2},?\s+|\d{1,2}(?:st|nd|rd|th)?\s+"
                  + _MONTH + r",?\s+)?(\d{4})", q)
    if m and "nine" not in q and "quarter" not in q:
        return f"FY{m.group(m.lastindex)}"

    # a date: "June 30, 2026", "30 June 2026", "30.06.2026", "December 2024"
    m = (re.search(_MONTH + r"\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?(\d{4})", q)
         or re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH + r",?\s+(\d{4})", q))
    if m:
        return _quarter_for_month(_MONTH_NUM[m.group(1)[:3]], int(m.group(2)))
    m = re.search(r"\b\d{1,2}[./-](\d{1,2})[./-](\d{4})\b", q)
    if m:
        return _quarter_for_month(int(m.group(1)), int(m.group(2)))

    # fiscal year: FY25, FY2025, FY 2024-25, 2024-25
    m = re.search(r"\bfy\s*'?(\d{2,4})(?:\s*-\s*(\d{2,4}))?\b", q)
    if m:
        return f"FY{_full_year(m.group(2) or m.group(1))}"
    m = re.search(r"\b(20\d{2})\s*-\s*(\d{2})\b", q)
    if m:
        return f"FY{_full_year(m.group(2))}"
    return None


def resolve_filters(
    query: str,
    available: dict[str, set[str]],
    company: str | None = None,
    period: str | None = None,
) -> dict:
    """
    Decide the retrieval filters for a question.

    Args:
        available: company -> periods actually indexed (read from the index)
        company / period: explicit choices from the request; they win over
            anything detected in the question text

    Returns dict with:
        filters:  list of filter dicts, one per company (empty = unfiltered)
        detected: what was auto-detected (for the API response)
        missing:  human-readable message if the question asks for a period
                  that isn't indexed for that company, else None
    """
    companies = [company] if company else [c for c in detect_companies(query) if c in available]
    wanted = period or detect_period(query)

    filters, missing = [], []
    for c in companies:
        f = {"company": c}
        periods = available.get(c, set())
        if wanted and periods:
            # annual reports are indexed as FY2025; quarterly as Q1FY27
            if wanted in periods:
                f["period"] = wanted
            elif not period and len(periods) == 1 and wanted.startswith("FY") \
                    and next(iter(periods)).endswith(wanted[-2:]):
                f["period"] = next(iter(periods))
            else:
                missing.append(f"{c}: {wanted} is not indexed (available: {', '.join(sorted(periods))})")
                continue
        filters.append(f)

    return {
        "filters": filters,
        "detected": {"companies": companies if not company else [], "period": wanted if not period else None},
        "missing": "; ".join(missing) or None,
    }
