
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

## Phase 2: Evaluation & Tuning — SUPERSEDED (metric was lenient, see Phase 4)

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

## Phase 4: Parser swap + strict eval (2026-10-12)

**Why:** the Phase 2 "95.6%" counted an answer correct if ANY expected keyword
appeared, and most keyword lists repeated words from the question. Re-scored
strictly, that same run was 20/45 (44.4%). Root cause: `pdfplumber.find_tables()`
put column boundaries through numbers ("63,973" became "93"), and the prose
pass then cut the table region out, leaving orphaned number fragments.

**Done:**
- Docling parser (`--parser docling`, now the default); tables carry their
  section heading, so consolidated vs standalone is explicit
- Large tables split into row groups with caption + headers repeated (bge and
  the CrossEncoder truncate at 512 tokens; bottom rows were never embedded)
- `gold` field on every question, verified against the PDFs. Two labels were
  wrong and are fixed with a `gold_note`: Q033 (O2C EBITDA is 54,988, not
  74,163) and Q003 (no figure; Rs 5.50/share)
- `eval/strict_score.py` + reworked `run_eval.py` (strict headline, context
  hit, retrieval-only mode, `--resume`, `--no-fallback`, per-minute retry)
- `build_index.py` rebuilds Qdrant + BM25 together; `QDRANT_COLLECTION`,
  `QDRANT_PATH` (local on-disk Qdrant), `BM25_INDEX_PATH`, `GROQ_MODEL` envs
- Tests: 15 passing (`tests/test_table_split.py`, `tests/test_strict_score.py`)

**Results** (same model `openai/gpt-oss-120b`, local Qdrant, `eval/runs/`):

| | pdfplumber | Docling |
|---|---|---|
| Gold in top-5 context | 25/45 (55.6%) | 35/45 (77.8%) |
| Strict accuracy | 13/45 (28.9%) | 30/45 (66.7%) |
| Misses: retrieval/parse | 20 | 10 |
| Misses: generation | 12 | 5 |

All 45 gold answers now exist somewhere in the Docling corpus; the 10
retrieval misses are ranking problems (6 of them retrieved the right page but
a different chunk of it).

**Known issues / next:**
1. **Qdrant Cloud cluster is unreachable** (TLS reset). The keep-alive
   workflow has failed on every one of its last 200 runs (back to at least
   2026-08-25). Check the Qdrant console;
   the Docling index currently exists only in local `data/qdrant_local`.
2. **Groq retired `llama-3.3-70b-versatile`**. With the default unchanged,
   production silently answers via the gpt-4o-mini fallback. Pick a new
   default (`openai/gpt-oss-120b` was used for the eval).
3. Docling shifts a few densely packed rows at the bottom of TCS's P&L
   (page 7: attributable-to rows, paid-up capital). This causes Q027/Q035.
4. Ranking: next step is contextual chunk headers + retrieval ablation.
5. Reliance's rupee glyph extracts as "J"/"H"/"C" in some fonts (cosmetic).

---

## Phase 5: Six more companies (2026-10-13)

Added Q1FY27 results for **HDFC Bank, ITC, Bharti Airtel, Sun Pharma,
Larsen & Toubro, Maruti Suzuki** — 9 companies across 7 sectors, 2,453 chunks.
`fetch_bse_filing.py` needs ticker symbols (HDFCBANK, BHARTIARTL, SUNPHARMA,
LT, MARUTI), not company names.

Parsing needed OCR for 4 of the 6:
- HDFC Bank (21/22 pages) and L&T (18/32) are scanned images -> `--ocr`
- Maruti and Bharti have an embedded text layer that is itself bad OCR
  (Maruti consolidated revenue read "<24 60R"; one standalone figure read
  524,551 instead of 524,557) -> `--force-ocr`, which renders each page to an
  image so OCR is the only text source (Docling's force_full_page_ocr still
  took table cells from the bad text layer)
- OCR decimal commas ("25108,30") are repaired before indexing

18 new eval questions (Q046-Q063), gold verified against the page images.
Result (gpt-oss-120b): **8/18 strict (44.4%)**, and all 10 misses are
retrieval: the gold value exists in the corpus every time, but the top-5
chunks come from a sibling table (standalone vs consolidated, or the wrong
page). Next step: contextual chunk headers (company / statement / page
prepended before embedding), then re-run all 63 questions.

---

## Phase 6: Contextual chunk headers (2026-10-13)

Every chunk now starts with `<company> | <period> | <standalone|consolidated>
<doc type> | page N`. Basis is decided per page from the page's text; a page
naming neither is consolidated if it has minority / non-controlling-interest
lines (HDFC Bank's consolidated title OCR'd as "UN P ED ED D D G"), otherwise
it inherits the previous page. Cloud collection + BM25 rebuilt.

| (gpt-oss-120b) | Before | After |
|---|---|---|
| Gold in top-5 context, all 63 | 43/63 | 57/63 (90.5%) |
| Strict accuracy, original 45 | 30/45 | 39/45 (86.7%) |
| Strict accuracy, new 18 | 8/18 | 15/18 (83.3%) |
| Strict accuracy, all 63 | 38/63 | 54/63 (85.7%) |

Remaining misses: 6 retrieval (Q009 Q028 Q037 Q046 Q053 Q062), 3 generation
(Q006 formula wording, Q012 partial dividend answer, Q035 Docling row shift
on TCS p7).

---

## Phase 7: Cell-level answer verifier (2026-10-14)

The old verifier checked whether answer numbers appear anywhere in the
retrieved text. Measured on 139 labelled answers it caught 10/42 wrong ones,
and 0/6 on the current pipeline. Wrong answers almost never invent numbers;
they copy a real one from the wrong row, column or statement.

New `Verifier` checks each figure against its source cell: row label,
period column (latest quarter / nine months / year, comparisons allowed),
standalone vs consolidated, and that at least one figure comes from the row
the question asks about. Old one kept as `KeywordVerifier` for comparison.
`generate_and_verify` now passes the question.

`eval/verifier_eval.py`:
- real wrong answers, current pipeline: 3/6 caught (old 0/6), 1 false alarm
  in 54 (that one is the known TCS p7 Docling row shift)
- held-out mutation test (wrong figure swapped into correct answers):

| Mutation | Old | New |
|---|---|---|
| earlier-period value | 0/44 | 32/44 |
| different row's value | 0/44 | 37/44 |
| invented value | 0/45 | 45/45 |
| **all (current index)** | **0%** | **85.7%** |
| all (docling index) | 8.1% | 87.1% |

Caveat: the rules were tuned on the labelled answers; the mutation test is
the held-out number. Not caught by design: partial answers, errors already
in the parsed table, non-numeric answers.

---

## Phase 8: Company + period auto-detection (2026-10-14)

Without a company picked, /query searched all 9 companies at once.
`src/retrieval/query_filters.py` detects them from the question
(rule-based: name/ticker/alias lists, Indian-FY date and FY parsing;
available periods read from the index at startup, not hardcoded).
Explicit request filters still win. Several companies -> one search each
(`HybridRetriever.retrieve_multi`). A period that isn't indexed -> 404
"TCS: Q1FY27 is not indexed (available: Q3FY25)".

Measured on the 63 eval questions (retrieval only):

| | No filter (old default) | Auto-detected | Hand-labelled |
|---|---|---|---|
| Gold in top-5 context | 55/63 (87.3%) | 57/63 (90.5%) | 57/63 (90.5%) |
| Questions with another company's/period's chunks mixed in | 5/63 | 0/63 | 0/63 |

All 63 questions resolve to their labelled company + period (tested in
`tests/test_query_filters.py`, 84 tests). Frontend: dropdown now reads
"Auto-detect from question" and the result shows what was searched.

---

## Phase 9: Metrics table — router session 1 of 2 (2026-10-15)

`src/metrics/extract.py` builds `data/processed/metrics.sqlite` (1,100 rows,
9 companies) from the Docling blocks; `src/metrics/store.py` is the lookup
the `lookup_metric` tool will call. SQLite instead of DuckDB: stdlib, no new
dependency (matters for the 512MB deploy target), same result at this size.

Handled: period read per column from its header date (filings order
columns differently); headers that lost dates borrow from a same-layout
table in the filing; units from caption/header/page text; US$ columns and
tables tagged USD and kept out of INR lookups; OCR-merged cells skipped;
"attributable to owners" sub-rows. Lookups group values equal up to
rounding and require a clear majority, else return `ambiguous`.

`eval/metrics_eval.py`:
- 48 hand-verified figures (41 eval questions + 7 year-ago/prior-quarter
  values checked on page images): **46 correct, 0 wrong, 2 not returned**
  (Reliance standalone EBITDA lives only in prose; TCS profit-to-owners
  cell is OCR-merged)
- held-out: accounting identities over all extracted combinations —
  revenue + other income = total income **37/37**, PBT − tax = net profit
  **23/23**

Side effect on the verifier (shared period code): ITC's "3 Months ended"
columns are now recognised, and column recency is ranked by header date
instead of position (ITC prints standalone and consolidated current
quarters side by side); "(not annualised)" no longer reads as a year claim.
Re-measured: false alarms 1/54 (unchanged), mutation catch 113/133 = 85.0%
(was 114/133).

Next (session 2): `lookup_metric` / `calculate` / `search_filing` tools,
tool-calling loop on gpt-oss-120b, comparison + growth eval questions.

---

## Phase 10: Agentic router — session 2 of 2 (2026-10-15)

`src/generation/agent.py`: gpt-oss-120b tool loop (Groq tool calling, max 8
steps) with `lookup_metric` (metrics table), `calculate` (AST arithmetic
only — names, calls, lists rejected) and `search_filing` (hybrid retrieval
fallback). Routing (`needs_agent`) is a rule: comparison / growth / ratio
wording or 2+ companies -> agent; everything else stays on RAG. The answer
is verified against the tool outputs (lookups, calculations with their
inputs, retrieved chunks) with the same cell-level Verifier.
API: `mode` = auto | rag | agent; response carries `path` and `tool_calls`.

13 new eval questions (Q064-Q076: 7 growth, 4 cross-company comparisons
incl. ₹ crore vs ₹ million, 2 margins), gold computed from hand-verified
figures. Same model:

| | Plain RAG (auto filters) | Router -> agent |
|---|---|---|
| Strict accuracy, 13 new questions | 5/13 | **13/13** |
| Answers verified grounded | — | 13/13 |
| Tool calls per question | — | median 3 |

Router sends only 1 of the original 63 questions to the agent (Q005, a
"margin ratio formula" question): still correct, via search_filing.

Verifier fix found here: formula constants ("× 100") are not claims.

Next idea (measured, not done): single-figure questions via lookup_metric
too — the table is 46/48 correct with 0 wrong, above RAG's 85.7% strict
(e.g. Q046 ITC standalone revenue: RAG answers 26,794.47, flagged by the
verifier; the table has 26,943.23). Needs a full 63-question run.

---

## Phase 11: API off PyTorch, onto ONNX Runtime — fits a 512 MB free tier (2026-10-16)

Render's free 512 MB instance crashed on the first /query. Measured
(Windows RSS, retrieval stack): models load at ~500 MB, then PyTorch's
first inference adds ~330 MB -> **~1,008 MB peak**. `USE_RERANKER=false`
could never have fixed it (828 MB without the reranker).

Steps, each measured (retrieval = gold in top-5, original 63 questions):

| Config | Retrieval | Peak RSS |
|---|---|---|
| PyTorch (before) | 57/63 | 1,008 MB |
| FastEmbed defaults (arena on, fp16 embedder) | — | 1,022-1,077 MB |
| ONNX fp32, lean session (arena off) | 57/63 (cosine 1.0000 vs torch) | 690 MB |
| int8 embedder + int8 reranker, fp32-embedded index | 56/63 (loses Q001) | 387 MB |
| int8 + reranker pool 20 | 55/63 | — (rejected) |
| **int8 + index re-embedded with int8 (`filing_chunks_int8`)** | **58/63** | **387 MB** |

Full API under real /query calls (RAG + agent): **~406 MB** (Windows), no torch.

**Verified in the real image, `docker run --memory=512m --memory-swap=512m`**
(Linux cgroup peak). The first run was OOM-killed on the first question,
and Windows RSS had hidden it. Fixes, each measured in the container:

| Fix | Effect |
|---|---|
| Rerank in batches of 2 (`RERANK_BATCH`), not all 10 pairs at once | reranker burst 619 -> 200 MiB, and faster (1.6 s -> 0.96 s); identical scores |
| `session.disable_prepacking` (no second copy of int8 weights) | -40 MiB steady; identical outputs, same latency |
| Pipeline on ONE dedicated worker thread + a lock (one question at a time) | /health answers in 0.3 s mid-question (the event loop used to block for the whole LLM call); 3 simultaneous questions 491 -> 472 MiB peak |
| `MALLOC_ARENA_MAX=2` | worse (495 MiB) — rejected |

Final: idle 379 MiB, **peak 472 MiB over 8 questions incl. 3 simultaneous
and agent questions**, no OOM, all answers verified.

- `src/indexing/onnx_models.py`: own ONNX Runtime sessions (arena off,
  2 threads, `tokenizers`); embedder int8 = per-channel MatMul-only dynamic
  quantization (cos 0.990; default whole-model int8 was 0.967); reranker =
  Xenova's int8 export. `prepare` builds them (Dockerfile build step).
- Query embedder and collection are paired automatically
  (`default_collection()`): int8 -> `filing_chunks_int8`, fp32/torch ->
  `filing_chunks`. `EMBED_ONNX_VARIANT=fp32` + `RERANK_ONNX_VARIANT=fp32`
  restores exact-parity models on a host with ~1 GB.
- requirements.txt is now the API only (no torch / sentence-transformers /
  PDF libs); ingestion deps moved to requirements-ingest.txt.
- Agent: one self-correction round — figures in the answer that no tool
  produced (arithmetic "in its head") are sent back with "recompute with
  calculate"; seen live once on ITC vs Maruti.

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
1. ~~**Not actually deployed yet**~~ **Deployed 2026-10-16** — backend on Render free (Docker, ONNX int8, 512 MB; see Phase 11), frontend on Vercel: https://sebi-bse-financial-filing-rag-analy.vercel.app. Original note: — mechanics are ready, but no live Railway/
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
- `pytest` (174 tests, no API keys or models needed) runs on every push via
  GitHub Actions (`.github/workflows/tests.yml`): chunking, table splitting,
  query filters (incl. every eval question's company/period), strict scorer,
  verifier, metrics extraction/lookup, and the agent loop with a scripted
  fake model.
- No-LLM benchmarks to re-run after pipeline changes: `eval/metrics_eval.py`,
  `eval/verifier_eval.py score` / `mutate`, `eval/run_eval.py --retrieval-only`.
- `eval/run_eval.py` (with `--router`) is the end-to-end regression run;
  it needs Groq quota (~1 full run/day on the free tier).

### Manual Verification
- Ask a specific financial question about each indexed company through the
  React UI and confirm the citation points to the exact PDF page — already
  spot-checked for Infosys this session (Q1FY27 revenue query, correct answer,
  correct citations).
