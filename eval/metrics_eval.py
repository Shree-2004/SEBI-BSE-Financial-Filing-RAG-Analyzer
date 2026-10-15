"""
metrics_eval.py

Accuracy of the extracted metrics table (src/metrics) against figures
already verified by hand: the single-figure eval questions plus a few
earlier-period values checked on the page images (eval/metrics_gold.json).

No LLM involved — this measures extraction alone.

Run:
    set PYTHONPATH=src
    python eval/metrics_eval.py
"""

import json
import sys

sys.path.insert(0, "src")

from metrics.store import MetricStore  # noqa: E402


def main(db: str = "data/processed/metrics.sqlite") -> None:
    with open("eval/metrics_gold.json", encoding="utf-8") as f:
        gold = json.load(f)
    store = MetricStore(db)

    correct, wrong, missing = [], [], []
    for g in gold:
        r = store.lookup(g["company"], g["metric"], g["period"], g["basis"])
        if r is None or r.get("ambiguous"):
            missing.append(g)
        elif r.get("value") is not None and abs(r["value"] - g["value"]) < 1e-6:
            correct.append(g)
        else:
            wrong.append((g, r))

    n = len(gold)
    print(f"Metrics table vs verified gold ({n} figures)\n")
    print(f"  correct : {len(correct):3} ({len(correct)/n*100:.1f}%)")
    print(f"  wrong   : {len(wrong):3} ({len(wrong)/n*100:.1f}%)")
    print(f"  missing : {len(missing):3} ({len(missing)/n*100:.1f}%)   (not found, or sources disagree -> 'ambiguous')")
    if wrong:
        print("\nWrong:")
        for g, r in wrong:
            print(f"  {g['id']:5} {g['company']} {g['metric']} {g['basis']} {g['period']}: "
                  f"got {r['value']:,} (p{r['pages']}, {r['sources_agreeing']}/{r['sources_total']} agree), "
                  f"gold {g['value']:,}")
    if missing:
        print("\nMissing:")
        for g in missing:
            print(f"  {g['id']:5} {g['company']} {g['metric']} {g['basis']} {g['period']}")




def consistency(db: str = "data/processed/metrics.sqlite") -> None:
    """
    Held-out check over EVERY extracted (company, basis, period), not just
    the hand-verified figures: accounting identities that must hold in any
    correct P&L.
        revenue_from_operations + other_income == total_income
        profit_before_tax - total_tax_expense == net_profit
    (only where all three lookups resolve; banks and summary tables that
    don't print one of the lines are skipped)
    """
    store = MetricStore(db)
    keys = {(r["company"], r["basis"], r["period"]) for r in store.available()
            if r["basis"] in ("standalone", "consolidated")}
    checks = [("revenue + other income = total income",
               ("revenue_from_operations", "other_income"), "total_income", lambda a, b: a + b),
              ("PBT - tax = net profit",
               ("profit_before_tax", "total_tax_expense"), "net_profit", lambda a, b: a - b)]
    print("\nAccounting identities across all extracted figures:")
    for name, (m1, m2), m3, f in checks:
        ok = bad = 0
        failures = []
        for company, basis, period in sorted(keys):
            vals = [store.lookup(company, m, period, basis) for m in (m1, m2, m3)]
            if any(v is None or v.get("ambiguous") for v in vals):
                continue
            a, b, c = (v["value"] for v in vals)
            if abs(f(a, b) - c) <= max(abs(c) * 0.002, 1.0):
                ok += 1
            else:
                bad += 1
                failures.append(f"{company} {basis} {period}: {f(a, b):,.2f} vs {c:,.2f}")
        print(f"  {name:40} {ok}/{ok + bad} hold")
        for x in failures[:6]:
            print(f"     x {x}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
    consistency(*sys.argv[1:2])
