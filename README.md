# FinFiling RAG — SEBI/BSE Financial Filing Analyzer

[![Tests](https://github.com/Shree-2004/SEBI-BSE-Financial-Filing-RAG-Analyzer/actions/workflows/tests.yml/badge.svg)](https://github.com/Shree-2004/SEBI-BSE-Financial-Filing-RAG-Analyzer/actions/workflows/tests.yml)


**Live demo:** https://sebi-bse-financial-filing-rag-analy.vercel.app · **API docs:** https://finfiling-rag-api.onrender.com/docs  
*(free tier: the API sleeps when idle, so the first question can take ~1 minute while it wakes up)*

FinFiling RAG is a domain-specific Retrieval-Augmented Generation (RAG) system built over real Indian financial filings (SEBI/BSE corporate disclosures). It answers natural-language questions with **exact citations** from source documents, prioritizing financial accuracy and traceability.

## Features

- **Hybrid Retrieval:** Combines Dense Retrieval (BAAI/bge-base-en-v1.5 embeddings) with Keyword Search (BM25) to accurately surface both semantic concepts and exact financial numbers/company terms.
- **Advanced Reranking:** Employs CrossEncoder to rerank the top chunks for maximum relevance before generating answers.
- **Tool-using agent for comparisons, growth and ratios:** questions like "Compare Bharti Airtel and Sun Pharma revenue growth YoY" are routed (by a plain rule, no extra LLM call) to a gpt-oss-120b tool loop with `lookup_metric` (exact figures from the metrics table), `calculate` (AST-evaluated arithmetic; the model never does maths itself) and `search_filing` (fallback). Units are aligned (₹ crore vs ₹ million) before comparing, and every number in the answer is verified against the tool outputs. On 13 comparison/growth/ratio questions: **13/13 correct vs 5/13 for plain RAG**. Plain single-figure questions ("What was UltraTech's profit attributable to owners?") also go to the agent when the metrics table holds that figure, so the answer is the reported statement cell rather than whatever top-5 retrieval surfaced (a press-release rounding, or Infosys's USD EPS). Segment, nine-month and EBITDA questions stay on RAG.
- **Metrics table (data layer for tool use):** key P&L figures (revenue, total income, PBT, tax, net profit, profit to owners, EPS, bank interest earned, EBITDA…) extracted from every parsed table into SQLite, with company, standalone/consolidated, period, unit (₹ crore / ₹ million / per share) and source page. Each column's period is read from its own header date, so prior-quarter and year-ago figures are captured for growth questions. Lookups vote across every copy of a figure in the filing and return *ambiguous* rather than guess when sources disagree. **64/66 hand-verified figures correct, 0 wrong**; accounting identities (revenue + other income = total income; PBT − tax = net profit) hold for **66/66** and **32/32** extracted company-period combinations (NTPC's PBT − tax check is skipped with a stated reason: a regulated utility's net profit adds regulatory-deferral movements) (`eval/metrics_eval.py`).
- **Company + period auto-detection:** "HDFC Bank net profit for the quarter ended June 30, 2026" is filtered to HDFC Bank / Q1FY27 without picking from a dropdown (names, tickers, aliases; dates and FY formats mapped to the Indian fiscal calendar). Questions naming several companies get one search per company; a period that isn't indexed returns a clear message instead of an answer from the wrong quarter.
- **Cell-level answer verification:** every figure in an answer is traced to the source table cell it came from (row label, period column, standalone vs consolidated) and flagged if it doesn't fit the question. In a held-out mutation test (correct answers with a wrong figure swapped in), it catches **85%** of wrong figures; the original substring check caught **0%**. See `eval/verifier_eval.py`.
- **Structured Ingestion:** Docling's layout + table-structure models rebuild financial tables with row labels and multi-level column headers intact, and tag each table with its heading (consolidated vs standalone). Large tables are split into row groups that repeat the caption and headers, so every chunk fits the 512-token embedder/reranker window. The original `pdfplumber` parser is kept behind `--parser pdfplumber`.
- **FastAPI Backend:** A fast, asynchronous API serving the complete RAG pipeline with Swagger UI. Runs in a 512 MB free tier (verified with `docker run --memory=512m`: peak 472 MiB over 8 questions, including 3 simultaneous): the embedder and reranker run on ONNX Runtime with int8 weights instead of PyTorch, and the index is embedded with the same int8 model — retrieval 58/63 vs 57/63 for the original PyTorch setup. `EMBED_ONNX_VARIANT=fp32` / `RERANK_ONNX_VARIANT=fp32` switch to exact-precision models on a bigger host.
- **Qdrant Vector Database:** Uses Qdrant Cloud to store and search high-dimensional embeddings efficiently.

## 🛠️ Tech Stack

- **Ingestion:** `docling` (layout + TableFormer, RapidOCR for scanned filings; see `requirements-ingest.txt`), `pdfplumber` (legacy parser)
- **Embeddings:** `BAAI/bge-base-en-v1.5`, served with ONNX Runtime (int8; no PyTorch in the API — peak memory ~1 GB → ~400 MB, retrieval unchanged, see below)
- **Vector store:** Qdrant Cloud (or embedded on-disk Qdrant via `QDRANT_PATH` for local eval)
- **Hybrid retrieval:** dense + `rank-bm25`, fused with Reciprocal Rank Fusion
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (ONNX int8)
- **Generation + tool calling:** `openai/gpt-oss-120b` on Groq (OpenAI `gpt-4o-mini` as rate-limit fallback)
- **Metrics store:** SQLite (`data/processed/metrics.sqlite`)
- **Backend:** FastAPI, Uvicorn, Pydantic, slowapi (rate limiting), optional Sentry
- **Frontend:** React + Vite (`finfiling-frontend/`)
- **Deployment:** Docker image (Render config) or Hugging Face Space (`deploy/`); Vercel for the frontend

**Indexed filings (13 companies, 10 sectors):** Reliance Industries (annual report FY2025), TCS (Q3FY25), Infosys, HDFC Bank, ITC, Bharti Airtel, Sun Pharma, Larsen & Toubro, Maruti Suzuki, Tata Steel, NTPC, Bajaj Finance, UltraTech Cement (Q1FY27 results).

## 📁 Project Structure

```
├── src/
│   ├── ingestion/pdf_parser.py      # Docling parser (+ --ocr / --force-ocr), legacy pdfplumber parser
│   ├── chunking/chunker.py          # table row-groups, contextual chunk headers
│   ├── indexing/                    # embeddings, Qdrant client, BM25 index
│   ├── retrieval/
│   │   ├── hybrid_retriever.py      # dense + BM25 + RRF + CrossEncoder; per-company multi-search
│   │   └── query_filters.py         # company / period auto-detection
│   ├── metrics/
│   │   ├── extract.py               # key figures from parsed tables -> SQLite
│   │   └── store.py                 # voting lookup (returns "ambiguous" instead of guessing)
│   ├── generation/
│   │   ├── answer_generator.py      # RAG answer generation (Groq)
│   │   ├── agent.py                 # tool-calling agent: lookup_metric / calculate / search_filing
│   │   └── verifier.py              # cell-level grounding verifier
│   └── api/main.py                  # FastAPI app
├── eval/                            # questions + gold, run_eval, strict scorer, verifier & metrics evals, runs/
├── tests/                           # pytest suite (no API keys needed)
├── deploy/                          # Hugging Face Space launcher + upload script
├── fetch_bse_filing.py              # download results PDFs from BSE (by ticker, e.g. HDFCBANK)
├── build_index.py                   # rebuild Qdrant collection + BM25 index from chunked files
├── embed_and_upsert.py              # add one company's chunks to an existing collection
├── finfiling-frontend/              # React frontend
├── Dockerfile, railway.json
└── requirements.txt, requirements-ingest.txt
```

## 🚀 Getting Started

### 1. Install

```bash
python -m venv venv
venv\Scripts\activate             # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt                 # API runtime (no PyTorch)
# for parsing PDFs, building indexes and the eval scripts:
pip install -r requirements-ingest.txt
docling-tools models download layout tableformer rapidocr
```

### 2. Environment variables

Create `.env` (see `.env.example`; never commit it):

```env
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=...
GROQ_API_KEY=...
OPENAI_API_KEY=...        # optional fallback when Groq is rate-limited
```

The embedder, reranker and BM25 run locally and need no keys.

### 3. Build the ONNX models and indexes

```bash
set PYTHONPATH=src
python src/indexing/onnx_models.py prepare      # downloads + int8-quantizes the embedder/reranker into ./models
```

### 3b. Indexes

The BM25 index and the metrics table are committed under `data/processed/`. To (re)build the Qdrant collection from the chunked files:

```bash
set PYTHONPATH=src
python build_index.py data/processed/docling/*_chunked.jsonl --recreate
```

Adding a company:

```bash
python fetch_bse_filing.py ITC --output data/raw_pdfs/itc/itc_q1fy27.pdf
python src/ingestion/pdf_parser.py data/raw_pdfs/itc/itc_q1fy27.pdf --company ITC --doc-type quarterly_result --period Q1FY27 --output data/processed/docling/itc_blocks.jsonl --overwrite   # add --ocr for scanned PDFs
python src/chunking/chunker.py data/processed/docling/itc_blocks.jsonl --output data/processed/docling/itc_chunked.jsonl
python build_index.py data/processed/docling/*_chunked.jsonl --recreate
python src/metrics/extract.py data/processed/docling/*_blocks.jsonl --output data/processed/metrics.sqlite
# then add the company to INDEXED_COMPANIES in src/api/main.py
```

### 4. Run

```bash
set PYTHONPATH=src
uvicorn api.main:app --reload --port 8000      # Swagger UI at http://localhost:8000/docs
cd finfiling-frontend && npm install && npm run dev
```

### 5. Tests

```bash
pip install -r requirements-test.txt
pytest          # no API keys or models needed; also runs in CI on every push
```

## 🔌 API Endpoints

- **`POST /query`**: `{"query": "...", "company"?, "period"?, "mode"?: "auto" | "rag" | "agent"}`. Company and period are auto-detected when omitted. In `auto` mode, comparison/growth/ratio questions, and single-figure questions the metrics table can answer, go to the tool-using agent; the rest go to RAG. The response includes the answer, citations, cell-level verification, `path`, `tool_calls` and the filters applied.
- **`GET /companies`**: indexed companies.
- **`GET /health`**: Qdrant connection and model status.
- **`GET /stats`**: collection statistics.

## 🧪 Evaluation

124 hand-written questions across 13 companies (the tables below were measured on the first 76). Q096-Q124 (added 2026-10-17, worded differently from the earlier set) cover unanswerable questions (period or company not indexed, figures a results filing never discloses; scored as correct only if the answer says so), cross-company comparisons across units, quarter-on-quarter and prior-period figures, new phrasings ("bottom line", "tax bill", "PAT") and auditor / board-date questions (`eval/eval_questions.json`), each with a `gold` answer verified against the source PDF (page images for OCR'd filings). Scoring is **strict**: the answer must contain the gold figure, compared by value (`eval/strict_score.py`).

All runs use the same model (`openai/gpt-oss-120b` on Groq):

| | pdfplumber | Docling | Docling + contextual chunk headers |
|---|---|---|---|
| Original 45 questions (Reliance, TCS, Infosys): strict accuracy | 13/45 (28.9%) | 30/45 (66.7%) | **39/45 (86.7%)** |
| 18 questions on the six companies added later | — | 8/18 (44.4%) | **15/18 (83.3%)** |
| All 63: gold answer in top-5 retrieved chunks | — | 43/63 (68.3%) | **57/63 (90.5%)** |
| **All 63: strict accuracy** | — | 38/63 (60.3%) | **54/63 (85.7%)** |

**After adding Tata Steel, NTPC, Bajaj Finance and UltraTech Cement (2026-10-17, retrieval only; end-to-end LLM run pending):** 95 questions. Gold in top-5 for **66/79** questions whose answer appears verbatim in a filing (the other 16 are computed growth/margin/comparison answers that retrieval alone can't contain). Original 63: 55/63 (two borderline questions now rank 6th-10th in the larger index). New companies' single-figure questions: 10/16 — on two scanned filings the reranker prefers auditor-report prose over the results table; that gap is open and tracked.

Contextual chunk headers prepend one line to every chunk before embedding (e.g. `ITC | Q1FY27 | standalone quarterly results | page 3`), so retrieval can tell a company's standalone and consolidated tables apart.

An earlier version of this README reported 95.6% "keyword accuracy". That metric counted an answer correct if *any* expected keyword appeared, and most keyword lists repeated words from the question, so wrong figures still passed. It is still printed as "lenient" for comparison, but it is not an accuracy number.

### Comparison, growth and ratio questions (tool-using agent)

13 more questions (Q064–Q076: YoY/QoQ growth, cross-company comparisons including ₹ crore vs ₹ million, margins), with gold computed from hand-verified figures:

| | Plain RAG | Router → agent |
|---|---|---|
| Strict accuracy | 5/13 | **13/13** |
| Answers verified grounded | — | 13/13 |

### Other benchmarks (no LLM needed)

| Script | Measures | Result |
|---|---|---|
| `eval/verifier_eval.py` | Does the verifier flag wrong answers? Held-out mutation test (wrong figure swapped into correct answers) | **85%** caught vs 0% for the original substring check |
| `eval/metrics_eval.py` | Metrics-table accuracy vs 66 hand-verified figures, plus accounting identities over every extracted period | 64/66 correct, **0 wrong**; identities hold 66/66 and 32/32 |
| `eval/run_eval.py --retrieval-only --auto-filters` | Retrieval with auto-detected company/period | 57/63 gold in top-5 (= hand-labelled filters) |

```bash
set PYTHONPATH=src
python eval/run_eval.py --retrieval-only        # context hit only, no LLM quota
python eval/run_eval.py --no-fallback --resume  # full run, resumable after quota stops
python eval/run_eval.py --no-fallback --router  # route like the API (agent for comparisons)
python eval/verifier_eval.py score              # + `mutate headers`
python eval/metrics_eval.py
```

Per-run outputs are in `eval/runs/`. The full history of what changed and why is in `implementation_plan.md`.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License.
