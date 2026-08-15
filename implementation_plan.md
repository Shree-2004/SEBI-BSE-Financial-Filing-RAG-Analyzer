
# Project Priorities & Next Steps Roadmap

Status of the **FinFiling RAG Analyzer** as of the current session. Replaces the
original roadmap, which had gone stale — it listed 5 "supported" companies
(including HDFC Bank and Tata Motors) that were never actually ingested, a bug
fixed this session (see Phase 1 below).

---

## Phase 1: Expanding the Dataset — Infosys added, index corrected

**Indexed companies (verified live in Qdrant): `Reliance Industries`, `TCS`, `Infosys`.**
HDFC Bank and Tata Motors were removed from the API/frontend — they were listed
as "supported" without ever being ingested, which caused 404s whenever selected.

**What happened this session:**
1. Sourced and ingested Infosys's Q1 FY27 quarterly results (real filing,
   downloaded from BSE India, verified genuine before use).
2. Found and fixed a real bug in `src/chunking/chunker.py` — a sliding-window
   loop that degenerated into single-token increments near the end of long
   sections, producing dozens of near-duplicate junk chunks per section. This
   was silently bloating the index for **all** companies, not just Infosys
   (~85-90% of affected sections were duplicate junk; one page went from 64
   chunks down to 4 after the fix).
3. Re-chunked and re-embedded all three companies against the fix and wiped +
   rebuilt the Qdrant collection clean: **31,477 bloated chunks → 3,637 real
   ones.** Rebuilt the BM25 index to match.
4. Fixed a regression caught during the rebuild: wiping the Qdrant collection
   also wipes its payload indexes (`company`/`period`/`doc_type`/`block_type`),
   which are required for filtered queries — `create_indexes.py` re-run to fix.
5. Made the frontend company dropdown dynamic — it now fetches `GET /companies`
   at load instead of a hardcoded list, so this class of drift (advertising an
   unindexed company) can't happen again.

**Workflow to add another company (e.g. "ITC"), current and correct:**
1. Source the actual filing PDF. For **quarterly/annual results specifically**,
   this step is now automated — `fetch_bse_filing.py` queries BSE's
   announcements feed directly and downloads the real PDF, verified working
   for both an already-indexed company (TCS) and a fresh one (Wipro):
   ```
   python fetch_bse_filing.py "ITC" --list                    # browse candidates first
   python fetch_bse_filing.py "ITC" --output data/raw_pdfs/itc/itc_latest.pdf
   ```
   Uses the unofficial `bse` package against an undocumented BSE endpoint —
   if it breaks (BSE changes their API) or the filing isn't a results
   announcement (e.g. annual reports, which are separate documents typically
   distributed via investor-relations pages, not this feed), fall back to
   manual sourcing as before.
2. Place it in `data/raw_pdfs/<company>/<file>.pdf` (already done by step 1
   for results filings).
3. Parse: `python src/ingestion/pdf_parser.py <pdf> --company "ITC" --doc-type quarterly_result --period Q1FY27 --output data/processed/itc_chunks.jsonl`
4. Chunk: `python src/chunking/chunker.py data/processed/itc_chunks.jsonl --output data/processed/itc_chunked.jsonl`
5. Embed + upsert: `python embed_and_upsert.py --input data/processed/itc_chunked.jsonl --company "ITC"`
6. Rebuild BM25 over **all** companies' chunked files together (not just the
   new one) and overwrite `data/processed/bm25_index.pkl`.
7. Append the company to `INDEXED_COMPANIES` in
   [main.py](src/api/main.py#L62) — the frontend picks it up automatically via
   `/companies`, no frontend edit needed anymore.
8. Add a handful of eval questions grounded in real figures from the filing
   (see `eval/eval_questions.json` for the pattern) and re-run
   `eval/run_eval.py`.

**Status: not started for a 4th company** — deprioritized in favor of
reliability work, since the pipeline is already proven for 3 companies and a
4th is repetition of a known-working process, not new risk.

---

## Phase 2: Evaluation & Tuning — done

45-question eval set across all 3 companies. First fully valid run **against
the corrected, de-duplicated index** (44/45 completed — Q045 hit Groq's
daily quota on the last question of the run, not a real accuracy failure):

| Metric | Corrected index | Pre-chunker-fix baseline |
|---|---|---|
| Keyword accuracy | **95.6%** (97.7% on completed Qs) | 93.3% |
| Grounding rate | **91.1%** | 84.4% |
| Citation rate | **97.8%** | 95.6% |

Real improvement across the board, consistent with expectations — retrieval
no longer competes against thousands of duplicate junk chunks for top-k
slots. See `eval/eval_summary.txt` for the full breakdown.

---

## Phase 3: Production Deployment & Reliability Hardening

**Status (updated 2026-08-15 — this section had drifted stale for ~6 commits;
full detail and reasoning behind every line lives in
[deployment_readiness_and_efficiency.md](deployment_readiness_and_efficiency.md),
this is a summary only):**

- ✅ CORS locked down, reads `CORS_ALLOWED_ORIGINS` from env
- ✅ Rate limiting — `/query` capped at 10 req/min per IP, verified live
- ✅ Qdrant keep-alive workflow exists **and its GitHub repo secrets
  (`QDRANT_URL`, `QDRANT_API_KEY`) are now added** — confirmed via
  `gh secret list`, so this is genuinely mitigated now, not just in code
- ✅ Query caching — verified live (repeat question: 6.35s real call → ~0ms cached)
- ✅ Chunker regression test (`tests/test_chunker.py`)
- ✅ Groq → OpenAI fallback on rate-limit (`answer_generator.py`)
- ✅ Dockerfile + `railway.json` + Vercel config — deployment mechanics exist
- ✅ Cold-start handling — Docker pre-downloads model weights at build time;
  frontend polls `/health` and shows a "waking up" state
- ✅ Structured JSON logging + Sentry (opt-in via `SENTRY_DSN`)
- ✅ RAM measured (~510MB, tight against a 512MB free-tier ceiling) with a
  `USE_RERANKER=false` mitigation toggle — **but not yet verified against the
  actual built Docker image** (no Docker available in the sessions so far)
- ✅ `data/processed/embeddings.pkl` confirmed dead code, removed

**Genuinely still open:**
1. **Not actually deployed yet** — mechanics are ready, but no live Railway/
   Vercel URL exists. Needs your own accounts; not something automatable.
2. **RAM verification against the real built Docker image** — blocked on
   Docker not being available in any session so far.
3. **Broader `pytest` suite** — only the chunker test exists; nothing for
   retrieval filtering or the `/companies` ↔ `INDEXED_COMPANIES` sync.
4. **Eval set is thin (45 Qs)** for confidently advertising the accuracy
   numbers externally, though the numbers that exist are real and verified.

---

## Verification Plan

### Automated Tests
- No real `pytest` suite exists yet (`test_pipeline.py` is a manual smoke
  script, not CI-runnable) — see Phase 3 item #3 above.
- `eval/run_eval.py` is the closest thing to a regression suite currently;
  re-run it after any change to chunking, retrieval, or the indexed company
  list.

### Manual Verification
- Ask a specific financial question about each indexed company through the
  React UI and confirm the citation points to the exact PDF page — already
  spot-checked for Infosys this session (Q1FY27 revenue query, correct answer,
  correct citations).
