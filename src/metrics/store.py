"""
store.py

Read side of the metrics table built by extract.py — this is what the
lookup_metric tool calls.

The same figure is usually printed several times in one filing (the
results statement, an "extract" table, the press release), and OCR can
misread one copy. lookup() collects every source cell for the requested
(company, metric, period, basis) and returns the value most sources agree
on, along with how many agreed and where each came from — so a single
misread copy can't silently win, and disagreement is visible.
"""

import sqlite3
from collections import Counter

METRIC_NAMES = {
    "revenue_from_operations": "Revenue from operations",
    "total_income": "Total income",
    "other_income": "Other income",
    "total_expenses": "Total expenses",
    "employee_benefit_expense": "Employee benefit expense",
    "finance_costs": "Finance costs",
    "depreciation_amortisation": "Depreciation and amortisation",
    "profit_before_tax": "Profit before tax",
    "profit_before_tax_and_minority": "Profit before tax and minority interest",
    "current_tax": "Current tax",
    "total_tax_expense": "Total tax expense",
    "net_profit": "Net profit for the period",
    "profit_attributable_to_owners": "Profit attributable to owners of the company",
    "eps_basic": "Basic EPS",
    "interest_earned": "Interest earned (banks)",
    "ebitda": "EBITDA",
}


class MetricStore:
    def __init__(self, path: str = "data/processed/metrics.sqlite"):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.row_factory = sqlite3.Row

    def lookup(self, company: str, metric: str, period: str, basis: str | None = None) -> dict | None:
        """
        Best value for one figure, or None if it isn't in the table.
        basis: "standalone" | "consolidated" | None (prefer consolidated —
        the headline number companies report — when both exist).
        """
        rows = [dict(r) for r in self.con.execute(
            "SELECT * FROM metrics WHERE company=? AND metric=? AND period=? AND currency='INR'",
            (company, metric, period))]
        if basis:
            rows = [r for r in rows if r["basis"] in (basis, "standalone and consolidated")]
        elif any(r["basis"] == "consolidated" for r in rows):
            rows = [r for r in rows if r["basis"] == "consolidated"]
        if not rows:
            return None

        def decimals(v: float) -> int:
            s = f"{v:.4f}".rstrip("0")
            return len(s.split(".")[1]) if "." in s else 0

        # Group values that agree up to rounding (a summary table's 67,942 is
        # the statement's 67,941.74). One group must clearly outnumber the
        # rest — if sources split with no majority (e.g. several segment-level
        # "EBITDA" rows), the figure is ambiguous and returned as such, so the
        # caller falls back to reading the filing instead of trusting a guess.
        groups: list[list[dict]] = []
        for r in sorted(rows, key=lambda r: -decimals(r["value"])):
            g = next((g for g in groups
                      if abs(g[0]["value"] - r["value"]) <= max(abs(g[0]["value"]) * 1e-3, 0.5)), None)
            (g.append(r) if g else groups.append([r]))
        groups.sort(key=lambda g: -len(g))
        if len(groups) > 1 and len(groups[0]) == len(groups[1]):
            return {
                "company": company, "metric": metric, "metric_name": METRIC_NAMES.get(metric, metric),
                "period": period, "value": None, "ambiguous": True,
                "candidates": [{"value": g[0]["value"], "page": g[0]["page"], "row_label": g[0]["row_label"]}
                               for g in groups],
            }

        best = groups[0]
        value = best[0]["value"]                     # most precise member of the group
        n_agree = len(best)
        first = min(best, key=lambda r: r["page"])
        votes = Counter(r["value"] for r in rows)
        return {
            "company": company,
            "metric": metric,
            "metric_name": METRIC_NAMES.get(metric, metric),
            "period": period,
            "basis": first["basis"],
            "value": value,
            "unit": first["unit"],
            "value_crore": best[0]["value_crore"],    # same (most precise) row as value
            "sources_agreeing": n_agree,
            "sources_total": len(rows),
            "conflicting_values": sorted(v for v in votes if v != value),
            "pages": sorted({r["page"] for r in best}),
            "source_file": first["source_file"],
        }

    def comparative_periods(self) -> dict[str, dict[str, str]]:
        """company -> {period: filing that prints it}, e.g. NTPC Q4FY26 -> Q1FY27.
        Only periods that are a column of some filing. (If two filings ever
        print the same period, the last in sort order wins — today every
        period has exactly one source filing.)"""
        out: dict[str, dict[str, str]] = {}
        for r in self.con.execute(
                "SELECT DISTINCT company, period, filing_period FROM metrics "
                "WHERE currency='INR' ORDER BY company, filing_period"):
            out.setdefault(r["company"], {})[r["period"]] = r["filing_period"]
        return out

    def available(self, company: str | None = None) -> list[dict]:
        """What can be looked up: (company, metric, period, basis) combinations."""
        q = ("SELECT DISTINCT company, metric, period, period_type, basis FROM metrics "
             "WHERE currency='INR'" + (" AND company=?" if company else "") +
             " ORDER BY company, metric, period")
        return [dict(r) for r in self.con.execute(q, (company,) if company else ())]
