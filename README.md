# FinFiling RAG — SEBI/BSE Financial Filing Analyzer

FinFiling RAG is a domain-specific Retrieval-Augmented Generation (RAG) system built over real Indian financial filings (SEBI/BSE corporate disclosures). It answers natural-language questions with **exact citations** from source documents, prioritizing financial accuracy and traceability.

## Features

- **Hybrid Retrieval:** Combines Dense Retrieval (BAAI/bge-base-en-v1.5 embeddings) with Keyword Search (BM25) to accurately surface both semantic concepts and exact financial numbers/company terms.
- **Advanced Reranking:** Employs CrossEncoder to rerank the top chunks for maximum relevance before generating answers.
- **Robust Answer Generation & Verification:** Generates answers and features a post-generation verification step that checks whether the cited claims actually appear in the retrieved source chunks.
- **Structured Ingestion:** Docling's layout + table-structure models rebuild financial tables with row labels and multi-level column headers intact, and tag each table with its heading (consolidated vs standalone). Large tables are split into row groups that repeat the caption and headers, so every chunk fits the 512-token embedder/reranker window. The original `pdfplumber` parser is kept behind `--parser pdfplumber`.
- **FastAPI Backend:** A fast, asynchronous API serving the complete RAG pipeline with Swagger UI.
- **Qdrant Vector Database:** Uses Qdrant Cloud to store and search high-dimensional embeddings efficiently.

## 🛠️ Tech Stack

- **Ingestion:** `docling` (default, see `requirements-ingest.txt`), `pdfplumber` + `pytesseract` (legacy parser / OCR fallback)
- **Embeddings:** `BAAI/bge-base-en-v1.5` via `sentence-transformers`
- **Vector Store:** Qdrant Cloud (or FAISS for local dev)
- **Hybrid Retrieval:** Dense Embeddings + `rank-bm25`
- **Reranking:** `CrossEncoder`
- **Generation:** LLM API (OpenAI / Anthropic / Groq)
- **Backend:** `FastAPI`, `Uvicorn`, `Pydantic`
- **Frontend:** React (in `finfiling-frontend/`)
- **Deployment:** Railway (backend), Vercel (frontend)

## 📁 Project Structure

```
finfiling-rag/
├── data/
│   ├── raw_pdfs/                  # downloaded filings, organized by company
│   └── processed/                 # parsed + chunked JSON output
├── src/
│   ├── ingestion/                 # extraction, text + tables, OCR
│   ├── chunking/                  # structure-aware chunking
│   ├── indexing/                  # embeddings, Qdrant client, BM25 index
│   ├── retrieval/                 # hybrid retrieval, CrossEncoder reranking
│   ├── generation/                # LLM answer generation and verification
│   └── api/
│       └── main.py                # FastAPI app endpoints
├── eval/                          # Evaluation scripts and question sets
├── finfiling-frontend/            # React frontend
├── requirements.txt
└── .env
```

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3.9+ installed.

```bash
# Clone the repository and cd into it
git clone <your-repo-url>
cd "SEBIBSE Filing RAG Analyzer"

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root of your project based on the following keys. **Do not commit this file to version control.**

```env
QDRANT_URL=https://<your-cluster-url>:6333
QDRANT_API_KEY=your-qdrant-key
OPENAI_API_KEY=your-openai-key

# Optional: Add this if using Claude instead of OpenAI
# ANTHROPIC_API_KEY=your-anthropic-key
```

*Note: Models like `BAAI/bge-base-en-v1.5`, CrossEncoder, and BM25 run locally via `sentence-transformers` and do not require API keys.*

### 3. Initialize Qdrant Indexes

Before running the application or ingesting documents, you need to create the required payload indexes in your Qdrant cluster for efficient metadata filtering:

```bash
python create_indexes.py
```
This script creates `company`, `period`, `doc_type`, and `block_type` keyword indexes in the `filing_chunks` collection.

### 4. Running the Backend API

Start the FastAPI server:

```bash
# On Windows
set PYTHONPATH=src && uvicorn src.api.main:app --reload --port 8000

# On Linux/macOS
export PYTHONPATH=src
uvicorn src.api.main:app --reload --port 8000
```

Once running, visit **[http://localhost:8000/docs](http://localhost:8000/docs)** to access the auto-generated Swagger UI and test the endpoints.

## 🔌 API Endpoints

- **`POST /query`**: Main RAG endpoint. Accepts a query and optional filters (company, period, doc_type). Returns an answer, source citations, and verification metrics.
- **`GET /companies`**: Returns a list of companies currently indexed in the system.
- **`GET /health`**: Health check to verify Qdrant connection and model availability.
- **`GET /stats`**: Returns collection statistics (total chunks indexed, status, etc.).

## 🧪 Evaluation

63 hand-written questions across 9 companies (the table below covers the original 45 on Reliance, TCS and Infosys; the 18 for the six companies added later currently score 8/18 strict — see implementation_plan.md Phase 5) (`eval/eval_questions.json`), each with a `gold` answer verified against the source PDF. Scoring is **strict**: the answer must contain the gold figure, compared by value (`eval/strict_score.py`).

Same model (`openai/gpt-oss-120b` on Groq) and same questions, only the parser changed:

| | pdfplumber | Docling |
|---|---|---|
| Gold answer in top-5 retrieved chunks | 25/45 (55.6%) | **35/45 (77.8%)** |
| Strict answer accuracy | 13/45 (28.9%) | **30/45 (66.7%)** |
| Misses from retrieval/parsing | 20 | 10 |
| Misses from generation | 12 | 5 |

An earlier version of this README reported 95.6% "keyword accuracy". That metric counted an answer correct if *any* expected keyword appeared, and most keyword lists repeated words from the question, so wrong figures still passed. It is still printed as "lenient" for comparison, but it is not an accuracy number.

```bash
set PYTHONPATH=src
python eval/run_eval.py --retrieval-only        # context hit only, no LLM quota
python eval/run_eval.py --no-fallback --resume  # full run, resumable after quota stops
```

Per-run outputs live in `eval/runs/`.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License.
