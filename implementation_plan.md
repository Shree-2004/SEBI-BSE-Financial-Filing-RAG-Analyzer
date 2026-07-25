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
1. Source the actual filing PDF (BSE/NSE/company investor-relations page — must
   have a real text layer, not a scanned image).
2. Place it in `data/raw_pdfs/<company>/<file>.pdf`.
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

Full detail and reasoning for all of this lives in
[deployment_readiness_and_efficiency.md](deployment_readiness_and_efficiency.md)
— this section is a status summary only.

**Done this session:**
- ✅ CORS locked down — reads `CORS_ALLOWED_ORIGINS` from env instead of a
  wildcard (`src/api/main.py`). **Note:** you'll need to set this env var to
  your real frontend URL once it's deployed, or it'll be blocked too.
- ✅ Rate limiting — `/query` capped at 10 req/min per IP (`slowapi`),
  verified live (10 requests through, 11th got a real 429).
- ✅ Qdrant keep-alive — scheduled GitHub Action
  (`.github/workflows/keep-qdrant-alive.yml`) pings the cluster every 4 hours
  so it can't silently auto-pause again (it did, mid-project, and needed a
  manual resume). **Needs `QDRANT_URL` and `QDRANT_API_KEY` added as GitHub
  repo secrets before it'll actually run** — not done yet, has to be done via
  the GitHub web UI (Settings → Secrets and variables → Actions).
- ✅ Query caching — identical questions now served instantly from an
  in-memory LRU/TTL cache instead of re-calling Groq. Verified live: repeat
  question went from 6.35s / real Groq call to ~0ms / `cached: true`, zero
  extra tokens spent.
- ✅ Chunker regression test — `tests/test_chunker.py` guards against the
  duplication bug recurring. Verified it has real teeth: run against a
  reconstruction of the original buggy logic, it produces 62 chunks on the
  test case and fails; 4/4 pass against the fix.

**Still open, in priority order:**
1. **Groq fallback to a second LLM provider** — blocked on a real billing
   decision (you said "not yet" when asked about Claude API access; `.env`
   already has an unused `OPENAI_API_KEY` as another option). Without this,
   the app still goes down entirely whenever Groq's free tier is exhausted.
2. **Dockerfile + Procfile** for Railway/Render, and Vercel/Netlify config for
   the frontend — the actual mechanics of going live. Not started.
3. **Cold-start handling** — model loading (embedder + reranker) takes real
   time; free-tier hosts spin down when idle, so the first visitor after any
   idle period eats that full load time. Needs either a paid always-on tier or
   a frontend loading state, decided before deployment.
4. Lower-priority polish: structured logging/error tracking, verifying actual
   RAM usage against free-tier limits, a broader eval set, confirming
   `data/processed/embeddings.pkl` is dead code.

---

## Verification Plan

### Automated Tests
- No real `pytest` suite exists yet (`test_pipeline.py` is a manual smoke
  script, not CI-runnable) — see Phase 3 item #2 above.
- `eval/run_eval.py` is the closest thing to a regression suite currently;
  re-run it after any change to chunking, retrieval, or the indexed company
  list.

### Manual Verification
- Ask a specific financial question about each indexed company through the
  React UI and confirm the citation points to the exact PDF page — already
  spot-checked for Infosys this session (Q1FY27 revenue query, correct answer,
  correct citations).
