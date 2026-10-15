# FinFiling RAG — SEBI/BSE Financial Filing Analyzer


FinFiling RAG is a domain-specific Retrieval-Augmented Generation (RAG) system built over real Indian financial filings (SEBI/BSE corporate disclosures). It answers natural-language questions with **exact citations** from source documents, prioritizing financial accuracy and traceability.

## Features

- **Hybrid Retrieval:** Combines Dense Retrieval (BAAI/bge-base-en-v1.5 embeddings) with Keyword Search (BM25) to accurately surface both semantic concepts and exact financial numbers/company terms.
- **Advanced Reranking:** Employs CrossEncoder to rerank the top chunks for maximum relevance before generating answers.
- **Tool-using agent for comparisons, growth and ratios:** questions like "Compare Bharti Airtel and Sun Pharma revenue growth YoY" are routed (by a plain rule, no extra LLM call) to a gpt-oss-120b tool loop with `lookup_metric` (exact figures from the metrics table), `calculate` (AST-evaluated arithmetic; the model never does maths itself) and `search_filing` (fallback). Units are aligned (₹ crore vs ₹ million) before comparing, and every number in the answer is verified against the tool outputs. On 13 comparison/growth/ratio questions: **13/13 correct vs 5/13 for plain RAG**.
- **Metrics table (data layer for tool use):** key P&L figures (revenue, total income, PBT, tax, net profit, profit to owners, EPS, bank interest earned, EBITDA…) extracted from every parsed table into SQLite, with company, standalone/consolidated, period, unit (₹ crore / ₹ million / per share) and source page. Each column's period is read from its own header date, so prior-quarter and year-ago figures are captured for growth questions. Lookups vote across every copy of a figure in the filing and return *ambiguous* rather than guess when sources disagree. **46/48 hand-verified figures correct, 0 wrong**; accounting identities (revenue + other income = total income; PBT − tax = net profit) hold for **60/60** extracted company-period combinations (`eval/metrics_eval.py`).
- **Company + period auto-detection:** "HDFC Bank net profit for the quarter ended June 30, 2026" is filtered to HDFC Bank / Q1FY27 without picking from a dropdown (names, tickers, aliases; dates and FY formats mapped to the Indian fiscal calendar). Questions naming several companies get one search per company; a period that isn't indexed returns a clear message instead of an answer from the wrong quarter.
- **Cell-level answer verification:** every figure in an answer is traced to the source table cell it came from (row label, period column, standalone vs consolidated) and flagged if it doesn't fit the question. In a held-out mutation test (correct answers with a wrong figure swapped in), it catches **85%** of wrong figures; the original substring check caught **0%**. See `eval/verifier_eval.py`.
- **Structured Ingestion:** Docling's layout + table-structure models rebuild financial tables with row labels and multi-level column headers intact, and tag each table with its heading (consolidated vs standalone). Large tables are split into row groups that repeat the caption and headers, so every chunk fits the 512-token embedder/reranker window. The original `pdfplumber` parser is kept behind `--parser pdfplumber`.
- **FastAPI Backend:** A fast, asynchronous API serving the complete RAG pipeline with Swagger UI.
- **Qdrant Vector Database:** Uses Qdrant Cloud to store and search high-dimensional embeddings efficiently.

## 🛠️ Tech Stack

- **Ingestion:** `docling` (layout + TableFormer, RapidOCR for scanned filings; see `requirements-ingest.txt`), `pdfplumber` (legacy parser)
- **Embeddings:** `BAAI/bge-base-en-v1.5` via `sentence-transformers`
- **Vector store:** Qdrant Cloud (or embedded on-disk Qdrant via `QDRANT_PATH` for local eval)
- **Hybrid retrieval:** dense + `rank-bm25`, fused with Reciprocal Rank Fusion
- **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Generation + tool calling:** `openai/gpt-oss-120b` on Groq (OpenAI `gpt-4o-mini` as rate-limit fallback)
- **Metrics store:** SQLite (`data/processed/metrics.sqlite`)
- **Backend:** FastAPI, Uvicorn, Pydantic, slowapi (rate limiting), optional Sentry
- **Frontend:** React + Vite (`finfiling-frontend/`)
- **Deployment:** Docker image (Render config) or Hugging Face Space (`deploy/`); Vercel for the frontend

**Indexed filings (9 companies, 7 sectors):** Reliance Industries (annual report FY2025), TCS (Q3FY25), Infosys, HDFC Bank, ITC, Bharti Airtel, Sun Pharma, Larsen & Toubro, Maruti Suzuki (Q1FY27 results).

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
pip install -r requirements.txt
# only for parsing new PDFs:
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

### 3. Build the indexes

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
pytest          # no API keys or models needed
```

## 🔌 API Endpoints

- **`POST /query`**: `{"query": "...", "company"?, "period"?, "mode"?: "auto" | "rag" | "agent"}`. Company and period are auto-detected when omitted. In `auto` mode, comparison/growth/ratio questions go to the tool-using agent and the rest to RAG. The response includes the answer, citations, cell-level verification, `path`, `tool_calls` and the filters applied.
- **`GET /companies`**: indexed companies.
- **`GET /health`**: Qdrant connection and model status.
- **`GET /stats`**: collection statistics.

## 🧪 Evaluation

76 hand-written questions across 9 companies (`eval/eval_questions.json`), each with a `gold` answer verified against the source PDF (page images for OCR'd filings). Scoring is **strict**: the answer must contain the gold figure, compared by value (`eval/strict_score.py`).

All runs use the same model (`openai/gpt-oss-120b` on Groq):

| | pdfplumber | Docling | Docling + contextual chunk headers |
|---|---|---|---|
| Original 45 questions (Reliance, TCS, Infosys): strict accuracy | 13/45 (28.9%) | 30/45 (66.7%) | **39/45 (86.7%)** |
| 18 questions on the six companies added later | — | 8/18 (44.4%) | **15/18 (83.3%)** |
| All 63: gold answer in top-5 retrieved chunks | — | 43/63 (68.3%) | **57/63 (90.5%)** |
| **All 63: strict accuracy** | — | 38/63 (60.3%) | **54/63 (85.7%)** |

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
| `eval/metrics_eval.py` | Metrics-table accuracy vs 48 hand-verified figures, plus accounting identities over every extracted period | 46/48 correct, **0 wrong**; identities hold 60/60 |
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
