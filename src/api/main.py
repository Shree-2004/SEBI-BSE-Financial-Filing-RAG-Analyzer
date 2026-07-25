"""
main.py

FastAPI application that wires together the full FinFiling RAG pipeline:
query → hybrid_retriever → answer_generator → verifier → response

All heavy models (embedder, reranker, BM25 index) are loaded ONCE at
startup and reused across requests — not reloaded per request, which
would be too slow for a real API.

Endpoints:
    POST /query          — main RAG query endpoint
    GET  /companies      — list of indexed companies
    GET  /health         — system health check
    GET  /stats          — collection stats
    POST /ingest/trigger — (dev only) trigger re-ingestion of a filing

Run locally:
    set PYTHONPATH=src && uvicorn src.api.main:app --reload --port 8000

Then open: http://localhost:8000/docs  (Swagger UI auto-generated)
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from generation.answer_generator import AnswerGenerator
from generation.verifier import Verifier, generate_and_verify
from indexing.bm25_index import BM25Index
from indexing.embed import Embedder
from indexing.vector_store import VectorStore
from retrieval.hybrid_retriever import HybridRetriever
from api.query_cache import QueryCache

load_dotenv()

# ---------------------------------------------------------------------------
# Rate limiting — /query calls an LLM (Groq) with a small daily token quota,
# so it needs its own per-IP limit. Metadata endpoints don't touch the LLM
# and are left unlimited.
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Known companies in the index — used for /companies endpoint + validation
# ---------------------------------------------------------------------------

# Only companies with data actually ingested into the index belong here.
# Add a company to this list *after* running the ingestion pipeline for it
# (see implementation_plan.md Phase 1) — never before, or the API will
# validate queries for a company that has zero chunks and return 404s.
INDEXED_COMPANIES = [
    "Reliance Industries",
    "TCS",
    "Infosys",
]

BM25_INDEX_PATH = "data/processed/bm25_index.pkl"

# ---------------------------------------------------------------------------
# App state — models loaded once at startup
# ---------------------------------------------------------------------------

class AppState:
    retriever: HybridRetriever = None
    generator: AnswerGenerator = None
    verifier: Verifier = None
    vector_store: VectorStore = None
    startup_time: float = None
    query_cache: QueryCache = None

state = AppState()
state.query_cache = QueryCache(max_entries=500, ttl_seconds=3600)


# ---------------------------------------------------------------------------
# Lifespan — startup + shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup. FastAPI keeps them alive across requests."""
    print("\n[INFO] Starting FinFiling RAG API...")
    t0 = time.time()

    state.retriever = HybridRetriever(bm25_index_path=BM25_INDEX_PATH)
    state.generator = AnswerGenerator()
    state.verifier = Verifier()
    state.vector_store = VectorStore()
    state.startup_time = time.time() - t0

    print(f"[OK] All models loaded in {state.startup_time:.1f}s\n")
    yield
    print("Shutting down FinFiling RAG API...")


# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FinFiling RAG API",
    description=(
        "Domain-specific RAG system over real Indian financial filings "
        "(SEBI/BSE corporate disclosures). Answers natural-language questions "
        "with exact citations from source documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allows the React frontend to call this API. Set CORS_ALLOWED_ORIGINS
# (comma-separated) in production to the deployed frontend's exact URL(s);
# a wildcard here lets any site burn your Groq/Qdrant quota by proxying
# requests through a visitor's browser.
_cors_env = os.getenv("CORS_ALLOWED_ORIGINS")
ALLOWED_ORIGINS = (
    [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
    if _cors_env
    else ["http://localhost:5173", "http://localhost:3000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
        example="What was HDFC Bank's net interest margin in Q2FY25?",
    )
    company: Optional[str] = Field(
        None,
        example="HDFC Bank",
        description="Filter results to a specific company",
    )
    period: Optional[str] = Field(
        None,
        example="Q2FY25",
        description="Filter results to a specific period e.g. Q2FY25 or FY2024",
    )
    doc_type: Optional[str] = Field(
        None,
        example="quarterly_result",
        description="Filter by document type: annual_report or quarterly_result",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of source chunks to retrieve",
    )


class CitationSchema(BaseModel):
    company: str
    period: str
    page_number: Optional[int]
    doc_type: Optional[str]
    source_file: Optional[str]
    excerpt: str


class VerificationSchema(BaseModel):
    is_grounded: bool
    confidence: float
    verified_claims: list[str]
    unverified_claims: list[str]
    warning: Optional[str]


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationSchema]
    verification: VerificationSchema
    is_grounded: bool
    grounding_confidence: float
    warning: Optional[str]
    model: str
    chunks_used: int
    filters_applied: dict
    latency_ms: float
    cached: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
@limiter.limit("10/minute")
async def query(request: Request, payload: QueryRequest):
    """
    Main RAG endpoint.

    Takes a natural-language question about an Indian listed company,
    retrieves relevant passages from the filing corpus using hybrid
    retrieval (dense + BM25 + CrossEncoder reranking), generates a
    grounded answer with source citations, and verifies the answer
    is actually supported by the retrieved context.

    Rate-limited to 10 requests/minute per IP — this endpoint calls an
    LLM against a small daily token quota, so it needs its own ceiling
    separate from the metadata endpoints below.
    """
    t0 = time.time()

    # Build metadata filter from optional request fields
    filters = {}
    if payload.company:
        if payload.company not in INDEXED_COMPANIES:
            raise HTTPException(
                status_code=400,
                detail=f"Company '{payload.company}' not in index. "
                       f"Available: {INDEXED_COMPANIES}",
            )
        filters["company"] = payload.company
    if payload.period:
        filters["period"] = payload.period
    if payload.doc_type:
        filters["doc_type"] = payload.doc_type

    # Serve identical questions from cache instead of re-embedding,
    # re-retrieving, and re-generating — saves both latency and Groq's
    # scarce daily token quota. Exact-match only (see query_cache.py).
    cache_key = state.query_cache.make_key(
        payload.query, payload.company, payload.period, payload.doc_type, payload.top_k
    )
    cached_result = state.query_cache.get(cache_key)
    if cached_result is not None:
        return cached_result.model_copy(
            update={"cached": True, "latency_ms": round((time.time() - t0) * 1000, 2)}
        )

    try:
        # Retrieve relevant chunks
        chunks = state.retriever.retrieve(
            query=payload.query,
            top_k=payload.top_k,
            filters=filters or None,
        )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found for this query. "
                       "Try removing filters or rephrasing the question.",
            )

        # Generate + verify
        result = generate_and_verify(
            query=payload.query,
            chunks=chunks,
            generator=state.generator,
            verifier=state.verifier,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    latency_ms = (time.time() - t0) * 1000

    response = QueryResponse(
        query=result["query"],
        answer=result["answer"],
        citations=[CitationSchema(**c) for c in result["citations"]],
        verification=VerificationSchema(**result["verification"]),
        is_grounded=result["is_grounded"],
        grounding_confidence=result["grounding_confidence"],
        warning=result["warning"],
        model=result["model"],
        chunks_used=result["chunks_used"],
        filters_applied=filters,
        latency_ms=round(latency_ms, 2),
        cached=False,
    )
    state.query_cache.set(cache_key, response)
    return response


@app.get("/companies", tags=["Meta"])
async def get_companies():
    """
    Returns the list of companies currently indexed in the system.
    Used by the frontend to populate the company filter dropdown.
    """
    return {
        "companies": INDEXED_COMPANIES,
        "count": len(INDEXED_COMPANIES),
    }


@app.get("/health", tags=["Meta"])
async def health():
    """
    Health check — verifies Qdrant connection and model availability.
    Used by Railway deployment health checks and monitoring.
    """
    try:
        info = state.vector_store.get_collection_info()
        qdrant_status = "ok"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"

    return {
        "status": "ok" if qdrant_status == "ok" else "degraded",
        "qdrant": qdrant_status,
        "models_loaded": state.retriever is not None,
        "startup_time_s": round(state.startup_time, 2) if state.startup_time else None,
    }


@app.get("/stats", tags=["Meta"])
async def stats():
    """
    Collection statistics — total chunks indexed, breakdown by company
    and document type. Useful for monitoring ingestion completeness.
    """
    try:
        info = state.vector_store.get_collection_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "total_chunks": info["points_count"],
        "collection_status": info["status"],
        "indexed_companies": INDEXED_COMPANIES,
        "note": "Per-company breakdown available after full ingestion",
        "query_cache": state.query_cache.stats(),
    }


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )