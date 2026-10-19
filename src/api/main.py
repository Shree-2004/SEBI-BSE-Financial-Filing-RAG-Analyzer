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

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

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
from retrieval.query_filters import resolve_filters
from generation.agent import MetricsAgent, agent_answer, needs_agent
from metrics.store import MetricStore
from api.query_cache import QueryCache
from api.logging_config import setup_logging, get_logger

load_dotenv()
setup_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Error tracking — optional. Right now a production failure is only visible
# as a JSON log line someone has to go looking for; Sentry surfaces it
# proactively instead. No-op unless SENTRY_DSN is set, so local dev and
# anyone who hasn't set up a Sentry project are unaffected.
# ---------------------------------------------------------------------------

_sentry_dsn = os.getenv("SENTRY_DSN")
if _sentry_dsn:
    import sentry_sdk

    sentry_sdk.init(
        dsn=_sentry_dsn,
        environment=os.getenv("ENVIRONMENT", "production"),
        traces_sample_rate=0.1,
    )
    logger.info("Sentry error tracking enabled")

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
    "HDFC Bank",
    "ITC",
    "Bharti Airtel",
    "Sun Pharma",
    "Larsen & Toubro",
    "Maruti Suzuki",
    "Tata Steel",
    "NTPC",
    "Bajaj Finance",
    "UltraTech Cement",
]

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "data/processed/bm25_index.pkl")

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
    available_periods: dict = None   # company -> periods actually in the index
    comparative_periods: dict = None  # company -> {prior period: filing that prints it}
    agent: MetricsAgent = None       # tool-using path for comparison / growth / ratio questions
    # One pipeline run at a time: each run's reranking burst is sized to fit a
    # 512 MB instance; three simultaneous runs measured 491 of 512 MiB.
    pipeline_lock: asyncio.Lock = None
    # ...on ONE dedicated thread: glibc keeps a malloc arena per thread that
    # does heavy allocation, so rotating through a thread pool measured
    # ~35 MiB higher peaks than a single worker.
    pipeline_executor: ThreadPoolExecutor = None

state = AppState()
state.query_cache = QueryCache(max_entries=500, ttl_seconds=3600)


# ---------------------------------------------------------------------------
# Lifespan — startup + shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup. FastAPI keeps them alive across requests."""
    logger.info("Starting FinFiling RAG API")
    t0 = time.time()

    use_reranker = os.getenv("USE_RERANKER", "true").lower() not in ("false", "0", "no")
    state.retriever = HybridRetriever(bm25_index_path=BM25_INDEX_PATH, use_reranker=use_reranker)
    state.generator = AnswerGenerator()
    state.verifier = Verifier()
    state.vector_store = VectorStore()
    # which periods exist per company, read from the index itself, so
    # auto-detected filters never point at a period that isn't ingested
    state.available_periods = {c: set() for c in INDEXED_COMPANIES}
    for chunk in (state.retriever.bm25.chunks if state.retriever.bm25 else []):
        if chunk.get("company") in state.available_periods:
            state.available_periods[chunk["company"]].add(chunk.get("period"))
    state.agent = MetricsAgent(
        state.retriever, MetricStore(os.getenv("METRICS_DB_PATH", "data/processed/metrics.sqlite"))
    )
    # prior periods printed inside an indexed filing (Q1FY27 results also
    # report Q4FY26 / Q1FY26 / FY2026) -> search that filing for them
    state.comparative_periods = state.agent.store.comparative_periods()
    state.pipeline_lock = asyncio.Lock()
    state.pipeline_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
    state.startup_time = time.time() - t0

    logger.info("All models loaded", extra={"startup_time_s": round(state.startup_time, 2)})
    yield
    logger.info("Shutting down FinFiling RAG API")


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


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Tags every request with a short ID (also returned as X-Request-ID)
    so a single log line can be correlated with a user's bug report, and
    logs one structured line per request — the thing raw print() couldn't
    give us: a queryable record of what happened, not just what was
    printed to a scrollback buffer."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    t0 = time.time()

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        },
    )
    return response


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
        example="What was HDFC Bank's standalone net profit for the quarter ended June 30, 2026?",
    )
    company: Optional[str] = Field(
        None,
        example="HDFC Bank",
        description="Filter to one company (auto-detected from the question if omitted)",
    )
    period: Optional[str] = Field(
        None,
        example="Q1FY27",
        description="Filter to a period e.g. Q1FY27 or FY2025 (auto-detected from the question if omitted)",
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
    mode: Literal["auto", "rag", "agent"] = Field(
        default="auto",
        description="auto: comparison/growth/ratio questions go to the tool-using agent, "
                    "the rest to retrieval + answer. rag / agent force one path.",
    )


class CitationSchema(BaseModel):
    company: Optional[str]
    period: Optional[str]
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
    path: str = "rag"                    # "rag" or "agent"
    tool_calls: list[dict] = []          # agent path: every tool call made


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
    request_id = getattr(request.state, "request_id", None)

    # Validate an explicitly requested company
    if payload.company:
        if payload.company not in INDEXED_COMPANIES:
            logger.warning(
                "unknown company requested",
                extra={"request_id": request_id, "company": payload.company},
            )
            raise HTTPException(
                status_code=400,
                detail=f"Company '{payload.company}' not in index. "
                       f"Available: {INDEXED_COMPANIES}",
            )
    # Company / period: explicit request fields win; otherwise detected from
    # the question text. Without a company filter the search would run
    # across all companies and mix their tables.
    resolution = resolve_filters(
        payload.query, state.available_periods, payload.company, payload.period,
        comparatives=state.comparative_periods,
    )
    if resolution["missing"] and not resolution["filters"]:
        raise HTTPException(status_code=404, detail=f"Not in the index — {resolution['missing']}.")
    filter_list = resolution["filters"]
    if payload.doc_type:
        filter_list = [{**f, "doc_type": payload.doc_type} for f in filter_list] or \
            [{"doc_type": payload.doc_type}]
    filters = {
        "filters": filter_list,
        "auto_detected": resolution["detected"],
        "note": resolution["missing"],
    }

    # Serve identical questions from cache instead of re-embedding,
    # re-retrieving, and re-generating — saves both latency and Groq's
    # scarce daily token quota. Exact-match only (see query_cache.py).
    cache_key = state.query_cache.make_key(
        payload.query + f"||{payload.mode}", payload.company, payload.period, payload.doc_type, payload.top_k
    )
    companies = [payload.company] if payload.company else resolution["detected"]["companies"]
    use_agent = payload.mode == "agent" or (
        payload.mode == "auto" and needs_agent(
            payload.query, companies, state.agent.store,
            resolution["detected"]["period"]
            or (filter_list[0].get("period") if len(filter_list) == 1 else None))
    )
    cached_result = state.query_cache.get(cache_key)
    if cached_result is not None:
        logger.info("cache hit", extra={"request_id": request_id, "filters": filters})
        return cached_result.model_copy(
            update={"cached": True, "latency_ms": round((time.time() - t0) * 1000, 2)}
        )

    def run_pipeline() -> dict:
        if use_agent:
            # comparisons / growth / ratios: lookup_metric + calculate tools,
            # verified against the tool outputs
            return agent_answer(state.agent, state.verifier, payload.query,
                                period=payload.period or resolution["detected"]["period"])
        # Retrieve relevant chunks (one search per company for comparisons)
        if len(filter_list) > 1:
            chunks = state.retriever.retrieve_multi(payload.query, filter_list, top_k=max(payload.top_k, 6))
        else:
            chunks = state.retriever.retrieve(
                query=payload.query,
                top_k=payload.top_k,
                filters=filter_list[0] if filter_list else None,
            )
        if not chunks:
            logger.warning(
                "no chunks retrieved",
                extra={"request_id": request_id, "filters": filters},
            )
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found for this query. "
                       "Try removing filters or rephrasing the question.",
            )
        # Generate + verify
        return generate_and_verify(
            query=payload.query,
            chunks=chunks,
            generator=state.generator,
            verifier=state.verifier,
        )

    try:
        # The pipeline is blocking (model inference + LLM calls of up to ~20s).
        # Run it in a worker thread so the event loop keeps answering /health
        # (the host's health check) while a question is in flight, and hold
        # the lock so concurrent questions queue instead of stacking memory.
        async with state.pipeline_lock:
            result = await asyncio.get_running_loop().run_in_executor(state.pipeline_executor, run_pipeline)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "pipeline error",
            extra={"request_id": request_id, "filters": filters},
        )
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    latency_ms = (time.time() - t0) * 1000

    logger.info(
        "query completed",
        extra={
            "request_id": request_id,
            "filters": filters,
            "is_grounded": result["is_grounded"],
            "grounding_confidence": result["grounding_confidence"],
            "chunks_used": result["chunks_used"],
            "latency_ms": round(latency_ms, 2),
        },
    )

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
        path="agent" if use_agent else "rag",
        tool_calls=result.get("tool_calls", []),
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
        logger.error("Qdrant health check failed", extra={"error": str(e)})

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
        logger.error("Qdrant stats fetch failed", extra={"error": str(e)})
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