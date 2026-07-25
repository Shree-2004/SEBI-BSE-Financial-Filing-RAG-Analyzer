"""
fetch_bse_filing.py

Automates the manual "find and download the filing PDF" step of Phase 1
(adding a new company) — previously done by hand: web-searching for the
company's investor-relations or BSE page, guessing which link is the real
quarterly-results PDF, and downloading it with trial-and-error headers.

Uses the unofficial `bse` package (BennyThadikaran/BseIndiaApi) to query
BSE's own announcements feed, filtered client-side to CATEGORYNAME=="Result"
+ SUBCATNAME=="Financial Results" (BSE's `category` query param did not
reliably filter server-side when tested — client-side filtering on the
returned rows is what actually works).

Scope: quarterly/annual RESULTS filings only. Annual reports are typically
separate, much larger documents distributed via investor-relations pages,
not this announcements feed — still source those manually.

Caveat: this hits an unofficial, undocumented BSE endpoint. It could break
if BSE changes their API. Treat failures here as "fall back to manual
sourcing," not as a sign something in the RAG pipeline itself is broken.

Usage:
    # List recent results filings for a company (no download)
    python fetch_bse_filing.py "TCS" --list

    # Download the most recent results filing
    python fetch_bse_filing.py "TCS" --output data/raw_pdfs/tcs/tcs_latest.pdf

    # Download a filing from a specific date range
    python fetch_bse_filing.py "Infosys" --after 2026-04-01 --before 2026-07-01 \
        --output data/raw_pdfs/infosys/infosys_q1fy27.pdf
"""

import argparse
import sys
from datetime import datetime, timedelta

import requests
from bse import BSE

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
_PDF_PATHS = ["AttachLive", "AttachHis"]  # try recent-filing path first, then archive


def find_results_filings(company: str, after: datetime, before: datetime) -> list[dict]:
    """Query BSE announcements for a company and return only the rows that
    are actual financial-results filings, newest first."""
    bse = BSE(download_folder=".")
    try:
        scrip_code = bse.getScripCode(company)
    except ValueError as e:
        raise SystemExit(f"Could not find a BSE scrip code for '{company}': {e}")

    results = bse.announcements(scripcode=scrip_code, from_date=after, to_date=before)
    rows = results.get("Table", [])

    filings = [
        row for row in rows
        if row.get("CATEGORYNAME", "").strip() == "Result"
        and row.get("SUBCATNAME", "").strip() == "Financial Results"
    ]
    filings.sort(key=lambda r: r.get("NEWS_DT", ""), reverse=True)
    return filings


def download_attachment(attachment_name: str, output_path: str) -> None:
    """Try the live-filing path first, fall back to the historical archive
    path — BSE moves attachments between them and there's no field in the
    announcement row indicating which one currently holds a given file."""
    headers = {"User-Agent": _UA, "Referer": "https://www.bseindia.com/"}

    for path_segment in _PDF_PATHS:
        url = f"https://www.bseindia.com/xml-data/corpfiling/{path_segment}/{attachment_name}"
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            with open(output_path, "wb") as f:
                f.write(resp.content)
            print(f"Downloaded ({path_segment}): {output_path} ({len(resp.content):,} bytes)")
            return

    raise SystemExit(
        f"Could not download attachment '{attachment_name}' from either "
        f"{' or '.join(_PDF_PATHS)} - BSE may have moved or removed it."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and download a company's financial results filing from BSE.")
    parser.add_argument("company", help="Company name or symbol, e.g. 'TCS', 'Infosys'")
    parser.add_argument("--after", type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
                         default=datetime.now() - timedelta(days=180),
                         help="Only consider filings after this date (YYYY-MM-DD). Default: 180 days ago.")
    parser.add_argument("--before", type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
                         default=datetime.now(),
                         help="Only consider filings before this date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--list", action="store_true",
                         help="List matching filings without downloading anything.")
    parser.add_argument("--output", help="Where to save the downloaded PDF (required unless --list).")
    args = parser.parse_args()

    filings = find_results_filings(args.company, args.after, args.before)

    if not filings:
        raise SystemExit(
            f"No financial-results filings found for '{args.company}' between "
            f"{args.after.date()} and {args.before.date()}. Try widening --after/--before, "
            f"or the filing may need to be sourced manually."
        )

    print(f"Found {len(filings)} results filing(s) for '{args.company}':\n")
    for i, row in enumerate(filings):
        print(f"[{i}] {row['NEWS_DT'][:10]} - {row['NEWSSUB']}")

    if args.list:
        sys.exit(0)

    if not args.output:
        raise SystemExit("--output is required when downloading (or pass --list to just browse).")

    chosen = filings[0]
    print(f"\nDownloading most recent: {chosen['NEWSSUB']} ({chosen['NEWS_DT'][:10]})")
    download_attachment(chosen["ATTACHMENTNAME"], args.output)
