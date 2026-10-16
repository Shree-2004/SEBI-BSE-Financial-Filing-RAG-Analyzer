"""
hybrid_retriever.py

Combines Qdrant dense retrieval + BM25 keyword retrieval using
Reciprocal Rank Fusion (RRF) — the same approach you used in DocMind
with EnsembleRetriever, but implemented directly here without LangChain.

Why RRF over score averaging?
- BM25 scores and cosine similarity scores live on completely different
  scales — you can't just average them meaningfully
- RRF only uses rank position, not raw score, so it's scale-agnostic
- Empirically beats score averaging consistently across retrieval benchmarks
- Simple to implement and explain in interviews

Then a CrossEncoder reranker re-scores the top-N fused results against
the query for a final precision boost before passing to the LLM.

Usage:
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()
    results = retriever.retrieve(
        query="What were HDFC Bank's NIM figures in Q2FY25?",
        top_k=5,
        filters={"company": "HDFC Bank", "period": "Q2FY25"},
    )
"""

import os
from pathlib import Path


from indexing.bm25_index import BM25Index
from indexing.embed import Embedder, default_backend
from indexing.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# CrossEncoder model for reranking — lightweight, strong on retrieval tasks
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_reranker():
    if default_backend() == "onnx":
        from indexing.onnx_models import OnnxReranker

        return OnnxReranker()
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)

# How many candidates to pull from each retriever before fusion + reranking
# More candidates = better recall, slower reranking
RETRIEVAL_CANDIDATES = 20

# RRF constant — 60 is the standard value from the original RRF paper
RRF_K = 60

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "data/processed/bm25_index.pkl")


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    id_key: str = "chunk_id",
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    Formula: RRF_score(doc) = sum(1 / (k + rank_i))
    where rank_i is the 1-based position of doc in list i.

    Args:
        ranked_lists : list of ranked result lists (each list is
                       already ordered best-first)
        id_key       : the key in each result dict that uniquely
                       identifies a chunk (chunk_id)
        k            : RRF constant (default 60)

    Returns:
        Single merged list sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    doc_store: dict[str, dict] = {}   # chunk_id -> chunk dict

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc.get(id_key, str(rank))  # fallback if no chunk_id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_store[doc_id] = doc

    # sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    results = []
    for doc_id in sorted_ids:
        doc = doc_store[doc_id].copy()
        doc["rrf_score"] = rrf_scores[doc_id]
        results.append(doc)

    return results


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(
        self,
        bm25_index_path: str = BM25_INDEX_PATH,
        use_reranker: bool = True,
    ):
        """
        Initialize embedder, Qdrant vector store, BM25 index, and reranker.

        Args:
            bm25_index_path : path to saved BM25 index pickle
            use_reranker    : set False to skip CrossEncoder (faster, less precise)
                              useful for dev/testing
        """
        print("Initializing HybridRetriever...")

        self.embedder = Embedder()
        self.vector_store = VectorStore()

        # BM25 index — load if exists, warn if not (need to run ingestion first)
        if Path(bm25_index_path).exists():
            self.bm25 = BM25Index.load(bm25_index_path)
        else:
            print(
                f"⚠ BM25 index not found at {bm25_index_path}. "
                "Run ingestion pipeline first to build it. "
                "Falling back to dense-only retrieval."
            )
            self.bm25 = None

        # CrossEncoder reranker
        self.use_reranker = use_reranker
        if use_reranker:
            print(f"Loading reranker: {RERANKER_MODEL} ...")
            self.reranker = _load_reranker()
            print("✓ Reranker loaded")
        else:
            self.reranker = None

        print("✓ HybridRetriever ready\n")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        candidates: int = RETRIEVAL_CANDIDATES,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline:
        1. Embed query
        2. Dense search (Qdrant)
        3. BM25 keyword search
        4. RRF fusion
        5. CrossEncoder reranking
        6. Return top_k

        Args:
            query     : natural language user query
            top_k     : final number of results to return to the LLM
            filters   : metadata filter e.g. {"company": "TCS"}
            candidates: how many to pull from each retriever before fusion

        Returns:
            List of top_k chunk dicts with retrieval scores attached.
        """

        # ---- Step 1: Embed query ----
        query_embedding = self.embedder.embed_query(query)

        # ---- Step 2: Dense retrieval from Qdrant ----
        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidates,
            filters=filters,
        )

        # ---- Step 3: BM25 retrieval ----
        if self.bm25 is not None:
            bm25_results = self.bm25.search(
                query=query,
                top_k=candidates,
                filters=filters,
            )
            ranked_lists = [dense_results, bm25_results]
        else:
            # dense-only fallback
            ranked_lists = [dense_results]

        # ---- Step 4: RRF fusion ----
        fused = _reciprocal_rank_fusion(ranked_lists)

        # ---- Step 5: CrossEncoder reranking ----
        if self.use_reranker and self.reranker is not None and fused:
            # rerank the top fused candidates (RERANK_POOL, default 2*top_k / 10)
            pool = int(os.getenv("RERANK_POOL", "0")) or max(top_k * 2, 10)
            rerank_pool = fused[:pool]
            pairs = [(query, doc["text"]) for doc in rerank_pool]
            rerank_scores = self.reranker.predict(pairs)

            for doc, score in zip(rerank_pool, rerank_scores):
                doc["rerank_score"] = float(score)

            rerank_pool.sort(key=lambda x: x["rerank_score"], reverse=True)
            return rerank_pool[:top_k]

        return fused[:top_k]

    def retrieve_multi(
        self,
        query: str,
        filter_list: list[dict],
        top_k: int = 6,
    ) -> list[dict]:
        """
        Retrieve for a question that names several companies ("compare ITC
        and Maruti margins"): one filtered search per company, then an equal
        share of the context for each — a single unfiltered search would let
        whichever company's tables rank highest crowd the others out.
        """
        per_company = max(2, top_k // max(len(filter_list), 1))
        results = []
        for f in filter_list:
            results.extend(self.retrieve(query, top_k=per_company, filters=f))
        return results

    def retrieve_for_display(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> dict:
        """
        Same as retrieve() but returns a structured dict with:
        - results      : the top_k chunks
        - query        : original query
        - filters_used : what metadata filters were applied
        - source_files : unique source files referenced

        Useful for the FastAPI response and frontend citation display.
        """
        results = self.retrieve(query, top_k=top_k, filters=filters)

        source_files = list({
            r["source_file"] for r in results if "source_file" in r
        })

        return {
            "query": query,
            "filters_used": filters or {},
            "results": results,
            "source_files": source_files,
            "result_count": len(results),
        }


# ---------------------------------------------------------------------------
# CLI — manual retrieval test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test hybrid retrieval.")
    parser.add_argument("query", help="Query to test e.g. 'Reliance net profit Q2FY25'")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--company", default=None, help="Filter by company name")
    parser.add_argument("--period", default=None, help="Filter by period e.g. Q2FY25")
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()

    filters = {}
    if args.company:
        filters["company"] = args.company
    if args.period:
        filters["period"] = args.period

    retriever = HybridRetriever(use_reranker=not args.no_reranker)
    output = retriever.retrieve_for_display(
        query=args.query,
        top_k=args.top_k,
        filters=filters or None,
    )

    print(f"\nQuery   : {output['query']}")
    print(f"Filters : {output['filters_used']}")
    print(f"Sources : {output['source_files']}")
    print(f"\nTop {output['result_count']} results:\n")

    for i, r in enumerate(output["results"], 1):
        print(f"[{i}] {r.get('company')} | {r.get('period')} | {r.get('block_type')} | page {r.get('page_number')}")
        if "rerank_score" in r:
            print(f"    Rerank score : {r['rerank_score']:.4f}")
        print(f"    RRF score    : {r.get('rrf_score', 0):.6f}")
        print(f"    {r['text'][:300]}...")
        print()