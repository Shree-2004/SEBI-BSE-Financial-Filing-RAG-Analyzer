"""
fetch_nse_filing.py

Finds and downloads a company's results filing (PDF) from NSE's corporate
announcements archive — the exchange copy of the same SEBI-mandated
filing that fetch_bse_filing.py pulls from BSE.

Why a second source: since Oct 2026 BSE's announcements API answers 403
to automated requests (tested with the `bse` package 3.3.0 and 3.3.3 and
with plain browser headers), while NSE's public archive still serves the
filings. One request per company, a short pause between them.

Usage:
    # list recent board-meeting outcome / results filings (no download)
    python fetch_nse_filing.py NTPC --list --after 2026-07-15 --before 2026-08-10

    # download the first (most recent) match
    python fetch_nse_filing.py NTPC --after 2026-07-15 --before 2026-08-10 \
        --output data/raw_pdfs/ntpc/ntpc_q1fy27.pdf
"""

import argparse
import re
import sys
import time
from datetime import datetime, timedelta

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0 Safari/537.36")
API = "https://www.nseindia.com/api/corporate-announcements"
# results are filed as "Outcome of Board Meeting" or "Financial Result Updates"
RESULTS_RE = re.compile(r"outcome of board meeting|financial result", re.I)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "application/json,text/html,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
    })
    s.get("https://www.nseindia.com/", timeout=30)   # session cookie (status may be 403; cookie still set)
    return s


def find_results_filings(symbol: str, after: datetime, before: datetime) -> list[dict]:
    s = _session()
    time.sleep(2)
    r = s.get(API, params={"index": "equities", "symbol": symbol,
                           "from_date": after.strftime("%d-%m-%Y"),
                           "to_date": before.strftime("%d-%m-%Y")}, timeout=30)
    r.raise_for_status()
    rows = [a for a in r.json() if RESULTS_RE.search(a.get("desc") or "")
            and (a.get("attchmntFile") or "").lower().endswith(".pdf")]
    return [{"date": a.get("an_dt"), "desc": a.get("desc"), "url": a["attchmntFile"],
             "text": (a.get("attchmntText") or "")[:160]} for a in rows]


def download(url: str, output: str) -> int:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Referer": "https://www.nseindia.com/"})
    r = s.get(url, timeout=120)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError(f"not a PDF ({r.headers.get('content-type')})")
    from pathlib import Path
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(r.content)
    return len(r.content)


def main():
    ap = argparse.ArgumentParser(description="Download a results filing PDF from NSE.")
    ap.add_argument("symbol", help="NSE symbol, e.g. NTPC, BAJFINANCE, ULTRACEMCO")
    ap.add_argument("--after", type=lambda d: datetime.strptime(d, "%Y-%m-%d"),
                    default=datetime.now() - timedelta(days=120))
    ap.add_argument("--before", type=lambda d: datetime.strptime(d, "%Y-%m-%d"), default=datetime.now())
    ap.add_argument("--list", action="store_true", help="only list matching filings")
    ap.add_argument("--index", type=int, default=0, help="which listed filing to download (0 = newest)")
    ap.add_argument("--output", help="where to save the PDF")
    args = ap.parse_args()

    filings = find_results_filings(args.symbol, args.after, args.before)
    if not filings:
        sys.exit(f"No results filings for {args.symbol} between {args.after:%Y-%m-%d} and {args.before:%Y-%m-%d}")
    for i, f in enumerate(filings):
        print(f"[{i}] {f['date']} - {f['desc']} - {f['url'].rsplit('/', 1)[-1]}")
    if args.list:
        return
    if not args.output:
        sys.exit("--output is required to download")
    n = download(filings[args.index]["url"], args.output)
    print(f"Downloaded: {args.output} ({n:,} bytes)")


if __name__ == "__main__":
    main()
