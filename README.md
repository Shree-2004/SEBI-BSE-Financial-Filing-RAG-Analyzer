# FinFiling RAG — SEBI/BSE Financial Filing Analyzer

FinFiling RAG is a domain-specific Retrieval-Augmented Generation (RAG) system built over real Indian financial filings (SEBI/BSE corporate disclosures). It answers natural-language questions with **exact citations** from source documents, prioritizing financial accuracy and traceability.

## Features

- **Hybrid Retrieval:** Combines Dense Retrieval (BAAI/bge-base-en-v1.5 embeddings) with Keyword Search (BM25) to accurately surface both semantic concepts and exact financial numbers/company terms.
- **Advanced Reranking:** Employs CrossEncoder to rerank the top chunks for maximum relevance before generating answers.
- **Robust Answer Generation & Verification:** Generates answers and features a post-generation verification step that checks whether the cited claims actually appear in the retrieved source chunks.
- **Structured Ingestion:** Structure-aware chunking keeps financial tables whole while prose is split by section, using `pdfplumber` and `pytesseract` (for OCR fallback).
- **FastAPI Backend:** A fast, asynchronous API serving the complete RAG pipeline with Swagger UI.
- **Qdrant Vector Database:** Uses Qdrant Cloud to store and search high-dimensional embeddings efficiently.

## 🛠️ Tech Stack

- **Ingestion:** `pdfplumber`, `pytesseract` (OCR), `pdf2image`, `Pillow`
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

To evaluate the pipeline's retrieval precision/recall and citation accuracy, navigate to the `eval/` directory. It contains `eval_questions.json` with 30-40 hand-written Q&A pairs and expected sources for automated benchmarking.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License.
