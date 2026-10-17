"""
extract.py

Pulls key financial figures out of the parsed filing tables into a small
SQLite table, one row per (company, metric, basis, period, source cell):

    company | metric | basis | period | period_type | value | unit | currency
            | value_crore | source_file | page | row_label | column_header

Why: retrieval + an LLM reading a table is the weakest way to answer "what
was X's revenue" and the only way to answer "compare X and Y" or "what was
the YoY growth" — the model has to find two cells, align units (₹ crore vs
₹ million) and do arithmetic. A lookup table makes those exact. It is the
data layer for the lookup_metric / calculate tools.

Every column's period is read from its own header date, not its position:
filings order columns differently (ITC puts the year-ago quarter before
the previous quarter), and the earlier-period columns are what growth
questions need ("Quarter ended June 30, 2025" -> Q1FY26).

Run:
    set PYTHONPATH=src
    python src/metrics/extract.py data/processed/docling/*_blocks.jsonl \
        --output data/processed/metrics.sqlite
"""

import json
import re
import sqlite3
from pathlib import Path

from chunking.chunker import _page_basis
from generation.verifier import _header_date, _period_kind

# metric -> (row-label pattern, exclusion pattern). First match wins, so
# more specific metrics come first.
METRICS: list[tuple[str, str, str | None]] = [
    ("interest_earned", r"^\W*\d*\W*interest earned", None),
    ("profit_attributable_to_owners",
     r"(profit|net profit).*attributable.*(owners|shareholders|parent|equity holders)", r"comprehensive"),
    ("eps_basic", r"\bbasic\b|earnings per (equity )?share", r"^(?!.*basic).*diluted|weighted|number of"),
    ("revenue_from_operations", r"^\W*\w{0,3}\W*(total |net )?revenues? from operat", r"segment|growth|%|other operat"),
    ("total_income", r"^\W*\w{0,4}\W*total income", r"comprehensive|other"),
    ("other_income", r"^\W*\w{0,4}\W*other income", r"comprehensive"),
    ("total_expenses", r"^\W*\w{0,4}\W*total (expenses|expenditure)", None),
    ("employee_benefit_expense", r"employee(s)? (benefits? )?(expenses?|cost)", None),
    ("finance_costs", r"^\W*\w{0,3}\W*finance costs?\b", r"financial services"),
    ("depreciation_amortisation", r"depreciat", None),
    ("current_tax", r"^\W*\w{0,3}\W*current tax", None),
    ("total_tax_expense", r"^\W*\w{0,4}\W*(total )?tax expense\b", r"current|deferred|items|relating|exceptional"),
    ("profit_before_tax",
     r"profit\s*(/\s*\(loss\)\s*)?(from ordinary activities )?before (income )?tax",
     r"exceptional|depreciation|share of|items that|minority|non-controlling|regulatory"),
    ("profit_before_tax_and_minority", r"before tax and (minority|non-?controlling)", None),
    ("net_profit",
     r"(net profit|profit)\s*(/\s*\(loss\)\s*)?(for the (period|quarter|year)|after tax)",
     r"attributable|comprehensive|before minority|but before|other compre|before tax|before exceptional"
     r"|margin|%|ratio"),
    ("ebitda", r"^\W*ebitda\b", r"%|margin"),
]

_UNIT_RE = [("million", r"\bmillions?\b"), ("crore", r"\bcrores?\b|\bcr\b"), ("lakh", r"\blakhs?\b"),
            ("billion", r"\bbillions?\b")]
_TO_CRORE = {"crore": 1.0, "million": 0.1, "lakh": 0.01, "billion": 100.0}

# ---------------------------------------------------------------------------
# Header / period parsing
# ---------------------------------------------------------------------------

def _quarter_label(month: int, year: int) -> str:
    q = (month - 4) % 12 // 3 + 1
    fy = year + 1 if month >= 4 else year
    return f"Q{q}FY{fy % 100:02d}"


def _column_period(header: str) -> tuple[str | None, str | None, int | None, int | None]:
    """-> (period, period_type, month, year) for one column header."""
    h = header.lower()
    m = re.search(r"\bq([1-4])\s*fy\s*'?(\d{2,4})", h)                 # "Q1 FY'27"
    if m:
        return f"Q{m.group(1)}FY{int(m.group(2)) % 100:02d}", "quarter", None, None
    m = re.search(r"\bfy\s*(\d{4})\s*-\s*(\d{2})", h) or \
        re.search(r"(?<![\d.])(20\d{2})\s*-\s*(\d{2})(?![\d.])", h)        # "FY 2024-25", "2024-25"
    if m and (int(m.group(1)) + 1) % 100 == int(m.group(2)) % 100:
        return f"FY{2000 + int(m.group(2))}", "year", None, None
    kind = _period_kind(header)
    month, year = _header_date(header)
    if not kind or month is None or year is None:
        return None, {"quarter": "quarter", "nine": "nine_months", "year": "year"}.get(kind), month, year
    if kind == "quarter":
        return _quarter_label(month, year), "quarter", month, year
    fy = year + 1 if month >= 4 else year
    if kind == "nine":
        return f"9MFY{fy % 100:02d}", "nine_months", month, year
    return f"FY{fy}", "year", month, year


def _strip_spanning_title(header: list[str]) -> list[str]:
    """
    The statement title spanning every column is joined into each header
    cell ("Statement of ... for the Three Months Ended 30/06/2026 - Year
    Ended - 31/03/2026"), so every column would read the title's date.
    Drop a leading " - " segment that all header cells share.
    """
    firsts = {h.split(" - ", 1)[0] for h in header}
    if len(header) >= 3 and len(firsts) == 1 and all(" - " in h for h in header):
        return [h.split(" - ", 1)[1] for h in header]
    return header


def _repair_periods(periods: list[tuple], header: list[str] | None = None) -> list[tuple]:
    """
    Two header failures seen in real filings:
    - A "Quarter ended" heading spanning several columns gets attached to one
      of them by the table parser, leaving the others with a date but no
      period type (Bajaj Finance). If the table has explicit quarter columns,
      a dated, untyped column whose date is a quarter-end is a quarter too.
    - The statement's title lands in the header row of every column
      (UltraTech: "...for the Three Months Ended 30/06/2026"), so all columns
      claim the same period. Columns can't share a period, so any duplicated
      label is unreliable: reset those columns and let a same-layout donor
      header (or nothing) fill them — prior-period values must never be
      filed under the current quarter.
    """
    periods = list(periods)
    if any(p[1] == "quarter" for p in periods):
        for i, (label, kind, month, year) in enumerate(periods):
            if label is None and kind is None and month in (3, 6, 9, 12) and year:
                periods[i] = (_quarter_label(month, year), "quarter", month, year)
    # a column's identity is (period, statement basis): ITC's summary table
    # legitimately has standalone AND consolidated columns for one quarter
    def basis(i):
        h = (header[i] if header and i < len(header) else "").lower()
        return "c" if "consolidated" in h else "s" if "standalone" in h else ""
    keys = [(p[0], basis(i)) for i, p in enumerate(periods)]
    dup = {k for k in keys if k[0] and keys.count(k) > 1}
    return [(None, None, None, None) if keys[i] in dup else p for i, p in enumerate(periods)]


def _parse_number(cell: str) -> float | None:
    """One clean number per cell, '(1,234)' = negative. OCR-merged cells
    holding two numbers are skipped rather than guessed."""
    # rejoin OCR gaps around separators ("21731 .84"), but a space between two
    # numbers ("12,380 64") means two cells merged — skip it
    c = re.sub(r"\s*([.,])\s*", r"\1", cell.strip())
    # table-rule artifacts around a value ("-- 2,603.72", "3,000.02 -" in
    # UltraTech's filing): dashes separated from the number by a space.
    # A real negative is "(1,480)" or "-1,480", never "- 1,480".
    c = re.sub(r"^(?:[-–—]+\s+)+|(?:\s+[-–—]+)+$", "", c)
    m = re.fullmatch(r"(\()?[₹$]?(-?\d[\d,]*(?:\.\d+)?)(\))?%?", c)
    if not m or c.endswith("%"):
        return None
    v = float(m.group(2).replace(",", ""))
    return -v if m.group(1) and m.group(3) else v


def _unit_and_currency(texts: list[str]) -> tuple[str | None, str]:
    for t in texts:
        tl = t.lower()
        currency = "USD" if re.search(r"us\s*\$|\busd\b|\$\s*(in\s*)?million", tl) else "INR"
        for unit, pat in _UNIT_RE:
            if re.search(pat, tl):
                return unit, currency
    return None, "INR"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_OCR_SPLIT_WORDS = {"tot al": "total", "ot her": "other", "in come": "income", "pro fit": "profit"}


def _metric_for(label: str) -> str | None:
    l = label.lower()
    for bad, good in _OCR_SPLIT_WORDS.items():     # OCR'd scans split words: "Tot al income"
        l = re.sub(rf"\b{bad}\b", good, l)
    l = re.sub(r"^\W*\w{0,4}\W*income\s+(?=\(a\))", "", l)   # "Income (a) Revenue from operations"
    for name, pat, excl in METRICS:
        if re.search(pat, l) and not (excl and re.search(excl, l)):
            return name
    return None


def _split_table(text: str):
    lines = text.splitlines()
    sep = next((i for i, l in enumerate(lines) if re.fullmatch(r"\|[-|\s:]+\|", l.strip())), None)
    if sep is None or sep == 0:
        return None, [], []
    header = [h.strip() for h in lines[sep - 1].strip().strip("|").split("|")]
    rows = [[p.strip() for p in l.strip().strip("|").split("|")]
            for l in lines[sep + 1:] if l.strip().startswith("|")]
    return lines[: sep - 1], header, rows


def extract_blocks(blocks: list[dict]) -> list[dict]:
    basis_by_page = _page_basis(blocks)
    prose_by_page = {(b["source_file"], b["page_number"]): b["text"]
                     for b in blocks if b["block_type"] == "prose"}

    tables = []
    for b in blocks:
        if b["block_type"] != "table":
            continue
        caption, header, rows = _split_table(b["text"])
        if not header:
            continue
        cols = _strip_spanning_title(header)
        tables.append((b, caption, header, rows, _repair_periods([_column_period(h) for h in cols], header)))

    # per filing: headers whose period columns are all fully dated — used to
    # fill headers that lost their dates to OCR ("Quarter ended", no year)
    complete = [t[4] for t in tables
                if any(p[0] for p in t[4]) and all(p[0] or not p[1] for p in t[4])]

    def compatible(donor, periods):
        if len(donor) != len(periods):
            return False
        for d, p in zip(donor, periods):
            if p[1] and d[1] and p[1] != d[1]:
                return False
            if p[2] and d[2] and p[2] != d[2]:      # months must agree when both known
                return False
        return True

    doc_unit = _unit_and_currency([b["text"][:800] for b in blocks if b["block_type"] == "table"])[0]
    out = []
    tables_periods: dict[tuple, list] = {}
    for b, caption, header, rows, periods in tables:
        if any((p[1] or p[2]) and not p[0] for p in periods):
            donor = next((c for c in complete if compatible(c, periods)), None)
            if donor:
                periods = [p if p[0] else (donor[i] if (p[1] or p[2]) else p) for i, p in enumerate(periods)]
        elif not any(p[0] for p in periods) and rows:
            # header row lost entirely (Bharti's consolidated page kept only
            # "CIN:..."): borrow a dated header with the same column count
            width = max(len(r) for r in rows)
            donor = next((c for c in complete if len(c) == width), None)
            if donor:
                header, periods = [""] * width, donor

        key = (b["source_file"], b["page_number"])
        tables_periods.setdefault(key, []).append(periods)
        unit, currency = _unit_and_currency(caption + [" ".join(header)] + [prose_by_page.get(key, "")])
        if "$" in b["text"] and not re.search(r"[₹~]|\brs\b|\binr\b|crore", " ".join(caption + header).lower()):
            currency = "USD"     # e.g. Infosys IFRS statements: "Basic (in$ per share)"
        unit = unit or doc_unit
        page_basis = basis_by_page.get(key, "")

        section, last_metric = "", None
        for i, parts in enumerate(rows):
            label = " ".join(p for p in parts if re.search(r"[A-Za-z]{2,}", p))
            # a label wrapped onto two rows, the parser putting the numbers on
            # the continuation ("operations" | 23,165.45 ...) and the start of
            # the label on a number-less neighbour ("Total revenue from")
            if re.match(r"[a-z]", label) and any(_parse_number(p) is not None for p in parts):
                for j in (i + 1, i - 1):
                    if 0 <= j < len(rows) and not any(_parse_number(p) is not None for p in rows[j][1:]):
                        nb = " ".join(p for p in rows[j] if re.search(r"[A-Za-z]{2,}", p))
                        if nb and re.search(r"\b(from|of|before|after|for|and|to)$", nb.lower()):
                            label = nb + " " + label
                            break
            ll = label.lower()
            if "attributable" in ll:
                section = "profit" if re.search(r"profit", ll) and "comprehensive" not in ll else "other"
            metric = _metric_for(label)
            if metric is None and re.search(r"owners|shareholders|parent", ll) and (
                    section == "profit" or (last_metric == "net_profit" and "attributable" in ll)):
                metric = "profit_attributable_to_owners"
            if "comprehensive" in ll:
                last_metric = None
            if metric is None:
                continue
            last_metric = metric
            for j, cell in enumerate(parts):
                if j >= len(periods) or not periods[j][0]:
                    continue
                v = _parse_number(cell)
                if v is None:
                    continue
                col = header[j]
                cl = col.lower()
                col_currency = "USD" if re.search(r"us\s*\$|\busd\b", cl) else currency
                # "Standalone Consolidated - 2024-25" (merged header): fall back to the page
                basis = ("consolidated" if "consolidated" in cl and "standalone" not in cl else
                         "standalone" if "standalone" in cl and "consolidated" not in cl else page_basis)
                is_eps = metric == "eps_basic"
                out.append({
                    "company": b["company"], "metric": metric, "basis": basis,
                    "period": periods[j][0], "period_type": periods[j][1],
                    "value": v, "unit": "per_share" if is_eps else unit, "currency": col_currency,
                    "value_crore": None if is_eps or unit not in _TO_CRORE else round(v * _TO_CRORE[unit], 4),
                    "source_file": b["source_file"], "page": b["page_number"],
                    "row_label": label[:200], "column_header": col[:120], "filing_period": b["period"],
                })
    fallback = _extract_fallback_lines(blocks, tables_periods, basis_by_page, prose_by_page, doc_unit)
    # on a merged page the text layer is the trustworthy reading: drop the
    # parser's (row-merged) cells wherever the text layer gives the same figure
    covered = {(r["source_file"], r["page"], r["metric"], r["period"]) for r in fallback}
    out = [r for r in out if (r["source_file"], r["page"], r["metric"], r["period"]) not in covered]
    return out + fallback


_FALLBACK_LINE_RE = re.compile(
    r"^(?P<label>.*?[A-Za-z].*?)\s+(?P<nums>(?:\(?-?\d[\d,]*\.\d{2}\)?\s*){2,})$")


def _extract_fallback_lines(blocks, tables_periods, basis_by_page, prose_by_page, doc_unit) -> list[dict]:
    """
    Rows from the PDF's own text layer, for pages where the table parser
    merged rows (NTPC: "(a) Revenue from operations 43831.86 43109.28 ...").
    A line's numbers are mapped onto the page's dated columns only when the
    count matches exactly — otherwise the line is skipped, not guessed.
    """
    out = []
    for b in blocks:
        if b["block_type"] != "prose" or not b["text"].startswith("Statement rows (PDF text layer)"):
            continue
        key = (b["source_file"], b["page_number"])
        cols = max(([p for p in ps if p[0]] for ps in tables_periods.get(key, [])), key=len, default=[])
        if len(cols) < 2:
            continue
        unit, _ = _unit_and_currency([prose_by_page.get(key, "")])
        unit = unit or doc_unit
        for line in b["text"].splitlines()[1:]:
            m = _FALLBACK_LINE_RE.match(line.strip())
            if not m:
                continue
            metric = _metric_for(m.group("label"))
            nums = [_parse_number(x) for x in re.findall(r"\(?-?\d[\d,]*\.\d{2}\)?", m.group("nums"))]
            if metric is None or len(nums) != len(cols) or any(v is None for v in nums):
                continue
            is_eps = metric == "eps_basic"
            for (period, ptype, _, _), v in zip(cols, nums):
                out.append({
                    "company": b["company"], "metric": metric, "basis": basis_by_page.get(key, ""),
                    "period": period, "period_type": ptype, "value": v,
                    "unit": "per_share" if is_eps else unit, "currency": "INR",
                    "value_crore": None if is_eps or unit not in _TO_CRORE else round(v * _TO_CRORE[unit], 4),
                    "source_file": b["source_file"], "page": b["page_number"],
                    "row_label": m.group("label")[:200], "column_header": "(text layer)",
                    "filing_period": b["period"],
                })
    return out


SCHEMA = """
CREATE TABLE metrics (
    company TEXT, metric TEXT, basis TEXT, period TEXT, period_type TEXT,
    value REAL, unit TEXT, currency TEXT, value_crore REAL,
    source_file TEXT, page INTEGER, row_label TEXT, column_header TEXT, filing_period TEXT
);
CREATE INDEX idx_lookup ON metrics (company, metric, period);
"""


def build(block_files: list[str], output: str) -> int:
    rows = []
    for f in block_files:
        with open(f, encoding="utf-8") as fh:
            rows += extract_blocks([json.loads(l) for l in fh if l.strip()])
    Path(output).unlink(missing_ok=True)
    con = sqlite3.connect(output)
    con.executescript(SCHEMA)
    cols = ["company", "metric", "basis", "period", "period_type", "value", "unit", "currency",
            "value_crore", "source_file", "page", "row_label", "column_header", "filing_period"]
    con.executemany(f"INSERT INTO metrics VALUES ({','.join('?' * len(cols))})",
                    [[r[c] for c in cols] for r in rows])
    con.commit()
    con.close()
    return len(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract key metrics from parsed filing blocks into SQLite.")
    ap.add_argument("inputs", nargs="+", help="*_blocks.jsonl files from pdf_parser.py")
    ap.add_argument("--output", default="data/processed/metrics.sqlite")
    args = ap.parse_args()
    n = build(args.inputs, args.output)
    print(f"✓ {n} metric rows -> {args.output}")
