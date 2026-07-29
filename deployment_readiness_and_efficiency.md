# Deployment Readiness & Efficiency — Next Steps

Everything below is grounded in issues actually hit or observed while working on this
project (chunker bug, Qdrant auto-pause, Groq quota exhaustion, missing payload
indexes), not speculative advice. Grouped by priority: fix before any real traffic,
should do soon after, and polish.

**Status as of 2026-07-27:** all of Tier 1 and Tier 2 are done, plus most of Tier 3.
See the ✅ notes on each for what was actually implemented and where. Two Tier 3 items
(a broader `pytest` suite, and expanding the eval set) are still open — flagged below.

---

## Tier 1 — Blockers for going live

These will break the app for real users within minutes of traffic, based on what
already happened during development.

### 1. Groq free tier will not survive real usage
**What happened:** hit the 100K-token/day cap twice today from eval runs alone (45
questions × ~2-3K tokens each). A live demo would exhaust this within the first few
visitors asking a handful of questions each.

**Options, cheapest to most robust:**
- **Response caching** (see Tier 2 #1) — cuts repeat-question cost to zero, doesn't
  fix first-time-question limits.
- **Fallback to a second provider on 429** — the `.env` already has an unused
  `OPENAI_API_KEY`. Catch the Groq `RateLimitError` in
  [answer_generator.py](src/generation/answer_generator.py) and retry on OpenAI or
  Claude (Haiku 4.5 is ~$1/$5 per MTok — a 45-question eval run would cost well under
  $1). This is a real second bill, not a free workaround — size it to expected traffic
  before committing.
- **Upgrade Groq to a paid Dev Tier** — simplest code-wise, but ties cost directly to
  traffic with no caching benefit.

**Recommendation:** do the fallback (few lines of retry logic) *and* the cache
(Tier 2 #1) together — the cache absorbs repeat questions for free, the fallback
absorbs everything else without going down.

✅ **Done.** [answer_generator.py](src/generation/answer_generator.py) now catches
`RateLimitError` / `APIStatusError` / `APIConnectionError` from Groq and retries on
OpenAI (`gpt-4o-mini`) using the existing `OPENAI_API_KEY`. Response is tagged with
which model actually served it. Combined with the query cache already in place.
Along the way, upgraded `openai` 1.51.0 → 1.109.1 in requirements.txt — the old
pinned version crashes on construction against the httpx version already installed
(`TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`), which
would have made the fallback itself the point of failure the first time Groq actually
rate-limited.

### 2. Qdrant Cloud free-tier auto-pauses on inactivity
**What happened:** the cluster paused mid-project and needed a manual resume from the
dashboard before any queries worked again — confirmed with a live `404 page not found`
(not even a valid Qdrant error shape) until resumed.

**Options:**
- **Accept the cold-start** — the free tier resumes automatically on the next
  connection attempt, but the *first* request after a pause will fail or hang for
  ~1-2 minutes while it wakes up. Add a startup health check / retry so this doesn't
  surface as a raw 500 to a user.
- **Scheduled keep-alive ping** — a cron job (even a free GitHub Actions scheduled
  workflow) hitting `/health` every few hours keeps the cluster active. Cheapest fix.
- **Upgrade to a paid Qdrant tier** — no auto-pause, but a recurring cost.

**Recommendation:** the keep-alive ping is nearly free and eliminates the failure mode
entirely — do this first.

✅ **Done.** Keep-alive workflow already existed
(`.github/workflows/keep-qdrant-alive.yml`). Added the belt-and-suspenders piece on
top: [vector_store.py](src/indexing/vector_store.py) `_ensure_collection()` now
retries with backoff (5 attempts, up to ~75s total) on the first Qdrant call at
startup, so if the cluster is still waking up despite the ping, the app waits instead
of crashing on boot. `railway.json`'s health check timeout is set to 120s to give
this room to work.

### 3. CORS is wide open
[main.py:108-114](src/api/main.py#L108) has `allow_origins=["*"]` with a comment
already flagging it: `# tighten this to your Vercel URL before production`. Anyone can
call your `/query` endpoint from any origin — including scripts designed to burn your
Groq/Qdrant quota. **Must** be scoped to your actual frontend domain before the API is
publicly reachable.

✅ **Done.** [main.py:134-147](src/api/main.py#L134) reads `CORS_ALLOWED_ORIGINS` from
the environment (comma-separated), falling back to localhost origins for dev only.
Set it to your Vercel URL in Railway's env vars — documented in the new
`.env.example`.

### 4. No rate limiting on your own API
There's currently nothing stopping one visitor (or one bot) from firing hundreds of
requests and burning your entire daily LLM budget in minutes. Before deploying
publicly, add per-IP rate limiting (`slowapi` is a lightweight FastAPI-native option)
on `/query` specifically — the metadata endpoints (`/companies`, `/health`, `/stats`)
don't need it since they don't call the LLM.

✅ **Done.** `slowapi` rate limiter on `/query` at 10 requests/minute per IP
([main.py:248](src/api/main.py#L248)); metadata endpoints left unlimited.

---

## Tier 2 — Should do soon after going live (efficiency & cost)

### 1. Cache identical/near-identical queries
Every query re-embeds the question, re-runs hybrid retrieval, and re-generates an
answer from scratch — even if someone just asked the exact same thing five minutes
ago. A simple cache keyed on `(normalized_query, company, period, doc_type)` with a
TTL (e.g. 1 hour) would:
- Eliminate repeat-question latency (instant response instead of ~3-8s)
- Directly reduce Groq/Qdrant load, which matters a lot given Tier 1 #1
- Cost nothing extra — an in-process `functools.lru_cache`-style cache is fine at
  low traffic; move to Redis only if you outgrow a single server instance.

✅ **Done.** [query_cache.py](src/api/query_cache.py) — in-memory TTL cache (500
entries, 1h TTL), wired into `/query` in [main.py:288-296](src/api/main.py#L288).
Cache stats exposed via `/stats`.

### 2. Cold-start model loading is slow
The FastAPI lifespan hook loads the BGE embedder, the CrossEncoder reranker, and the
BM25 index at startup — this took real, measured time locally (the reranker alone adds
several seconds). On Railway/Render's **free tier, the app spins down when idle** and
cold-starts on the next request, meaning the *first* visitor after any idle period
waits through the full model-loading sequence before getting a response.

**Options:**
- Bake the HuggingFace model cache into the Docker image at build time (avoids
  re-downloading ~400MB+ of model weights on every cold start — this is separate from
  and worse than just re-loading them into memory).
- Use a paid "always-on" tier if the cold-start latency isn't acceptable for a demo.
- At minimum, show a loading state in the frontend so a cold start doesn't look broken.

✅ **Done (the Docker piece).** The new [Dockerfile](Dockerfile) downloads
`BAAI/bge-base-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` at build time via
`HF_HOME=/app/.cache/huggingface`, so a cold start only pays for loading weights
already on disk, not re-downloading ~400MB+ over the network.

✅ **Done (the frontend piece).** [App.jsx](finfiling-frontend/src/App.jsx) now polls
`/health` on mount (retrying up to 6 times, 10s apart) and shows a "Waking up…"
header badge plus an explanatory banner while the backend is unreachable, instead of
silently hanging. The query-loading state also shows a "may take up to a minute"
hint whenever the backend hasn't confirmed itself online yet.

### 3. No structured logging or error tracking
Right now, debugging a production issue means SSH-ing into logs (if available at all)
and grepping raw prints. Before real traffic, consider:
- Structured logging (even basic `logging` module with request IDs) instead of `print()`
- A free-tier error tracker (Sentry has one) so failures surface instead of silently
  producing a generic 500 to the user

✅ **Done.** Structured JSON logging already existed
([logging_config.py](src/api/logging_config.py)). Added Sentry on top:
[main.py:51-64](src/api/main.py#L51) initializes `sentry_sdk` only if `SENTRY_DSN` is
set (true no-op otherwise — nothing to configure for local dev). Once set, Sentry's
logging integration auto-captures the existing `logger.exception`/`logger.error`
calls as events, no per-call-site changes needed.

### 4. Free-tier RAM limits vs. actual model footprint
`sentence-transformers` (embedder) + `CrossEncoder` (reranker) + FastAPI + Qdrant
client together may exceed the ~512MB RAM ceiling on many free hosting tiers. Verify
actual memory usage locally (`docker stats` against a built image) before assuming a
free tier will hold it — this is the kind of thing that works fine locally (16GB+ RAM)
and then silently OOM-kills on a 512MB container.

✅ **Verified — and the concern was real.** Docker wasn't available in this
environment to run it against the built image, so this was measured directly:
loading the embedder + reranker + BM25 index + Qdrant client in-process hits
**~510MB RSS** before FastAPI's own overhead or any request-handling memory is added.
That is over a 512MB free-tier ceiling on its own. **Action needed before deploying
to a 512MB tier:** either pick a host/plan with at least ~1GB RAM (Railway's starter
tier is not RAM-capped this tightly), or drop the CrossEncoder reranker to cut the
biggest chunk of that footprint.

✅ **Mitigation added.** The reranker is no longer hardcoded on — set
`USE_RERANKER=false` in the environment (see `.env.example`) to skip loading the
CrossEncoder at startup on a tight-memory tier, at the cost of retrieval precision.
Wired in [main.py](src/api/main.py) lifespan; `HybridRetriever` already supported
`use_reranker=False`, it just wasn't exposed as a runtime toggle before. Still
Docker-unverified against the built image — treat ~1GB RAM as the safe target and
use this toggle only if stuck on a smaller tier.

---

## Tier 3 — Quality & maintainability polish

### 1. Add a regression test for the chunker bug found this session
`_split_by_tokens()` in [chunker.py](src/chunking/chunker.py) had a real bug — a
sliding-window loop that degenerated into single-token increments near the end of long
sections, producing dozens of near-duplicate junk chunks (verified: one page went from
64 chunks down to 4 after the fix, and ~24% of the entire corpus was affected). It's
fixed now, but there's **no test guarding against it recurring** on a future edit. A
small unit test — feed a long synthetic text through `_split_by_tokens()` and assert no
two returned chunks are near-duplicate suffixes of each other — would catch this class
of bug immediately instead of silently degrading retrieval quality again.

✅ **Done.** `tests/test_chunker.py` guards `_split_by_tokens()` against the
duplication regression.

### 2. No automated test suite at all
`test_pipeline.py` is a manual smoke script (run it and eyeball the output), not
something CI can run. Given how much of this session was "the system looks fine but
silently isn't" (chunker duplication, missing payload indexes after a Qdrant wipe),
a handful of real `pytest` tests around chunking, retrieval filtering, and the
`/companies` ↔ `INDEXED_COMPANIES` sync would catch regressions before they reach
production.

⬜ **Still open.** Only the chunker test above exists; no tests around retrieval
filtering or the `/companies` ↔ `INDEXED_COMPANIES` sync yet. Not attempted in this
pass — worth a dedicated session.

### 3. Frontend sample queries only reference 2 of 3 companies
[App.jsx](finfiling-frontend/src/App.jsx) `SAMPLE_QUERIES` still only has TCS and
Reliance examples — small, but worth adding an Infosys one now that it's indexed, so
the sample chips actually exercise all indexed data.

✅ **Done.** Added an Infosys Q1FY27 revenue query to `SAMPLE_QUERIES`
([App.jsx:11](finfiling-frontend/src/App.jsx#L11)).

### 4. Eval set is thin for a "production-ready" claim
45 questions across 3 companies is enough to catch obvious regressions, but not enough
to confidently claim high accuracy for a public-facing tool. Worth expanding
per-company coverage (especially edge cases: ambiguous figures like the IFRS-vs-Ind-AS
issue hit with Infosys Q037) before advertising accuracy numbers externally.

⬜ **Still open.** Not attempted in this pass — needs actual question-writing, not
code.

### 5. `data/processed/embeddings.pkl` looks like stale, unused output
Not investigated in depth, but it predates the current Qdrant-based pipeline and
doesn't appear referenced anywhere live. Worth confirming it's dead and removing it,
or documenting what it's for if it's still needed.

✅ **Confirmed dead and removed.** Grepped the codebase — the only script that loads
embeddings from disk is `upsert_with_retry.py`, and it reads `tcs_embeddings.pkl`, not
the bare `embeddings.pkl`. Nothing referenced it. It was also already gitignored
(not tracked), so deleting it locally (144MB) had no repo impact.

---

## Deployment mechanics — added this pass

None of this existed before, despite the README naming Railway + Vercel as targets:

- **[Dockerfile](Dockerfile)** — Python 3.12-slim, installs Tesseract/poppler (needed
  by the ingestion scripts even though the API itself doesn't call them at request
  time), pre-downloads the embedder + reranker weights at build time, runs
  `uvicorn api.main:app` on `$PORT`.
- **[railway.json](railway.json)** — points at the Dockerfile, health-checks `/health`
  with a 120s timeout (room for the Qdrant cold-pause retry to finish).
- **[finfiling-frontend/vercel.json](finfiling-frontend/vercel.json)** — Vite build
  config for the frontend.
- **[.env.example](.env.example)** / **[finfiling-frontend/.env.example](finfiling-frontend/.env.example)**
  — document every env var each side needs (`CORS_ALLOWED_ORIGINS`, `SENTRY_DSN`,
  `VITE_API_URL`, etc.) since neither existed before.

**Before actually deploying:** the RAM finding above (~510MB measured, no Docker
available here to confirm the built-image number) means the free tier is genuinely
risky — size the Railway plan accordingly rather than assuming it'll fit.

---

## Suggested order of attack (original, now complete except where noted)

1. ✅ Qdrant keep-alive ping (cheap, eliminates a real failure mode)
2. ✅ CORS lockdown + basic rate limiting (required before any public exposure)
3. ✅ Groq fallback + query caching together (biggest reliability + cost win)
4. ✅ Chunker regression test (protects the fix already made)
5. ✅ Dockerfile/Procfile + cold-start handling (actual deployment mechanics)
6. Everything else — done except: broader pytest suite (Tier 3 #2) and expanded eval
   set (Tier 3 #4), both flagged above as still open.
