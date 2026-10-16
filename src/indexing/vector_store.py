"""
vector_store.py

Qdrant client wrapper that handles:
- Creating the collection (with correct vector dimensions)
- Upserting chunks with their embeddings + full metadata as payload
- Dense vector search with optional metadata filtering
  (e.g. "only search Reliance quarterly results")

This sits between the embedder and the hybrid retriever. The BM25
index lives separately in bm25_index.py — hybrid_retriever.py
combines both.

Usage:
    from indexing.vector_store import VectorStore

    store = VectorStore()
    store.upsert_chunks(chunks, embeddings)
    results = store.search("what was the net profit?", query_embedding, top_k=10)
"""

import os
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from tqdm import tqdm

load_dotenv()

from indexing.embed import default_collection

# explicit QDRANT_COLLECTION wins; otherwise the collection matching the embedder
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION") or default_collection()
EMBEDDING_DIM = 768          # bge-base-en-v1.5 output dim
UPSERT_BATCH_SIZE = 100      # Qdrant free tier is comfortable with this


class VectorStore:
    def __init__(self):
        # QDRANT_PATH = embedded on-disk Qdrant (no server, no cloud account).
        # Used for local eval runs so results don't depend on the free-tier
        # cluster being awake; takes precedence over QDRANT_URL when set.
        local_path = os.getenv("QDRANT_PATH")
        if local_path:
            self.client = QdrantClient(path=local_path)
            self._ensure_collection()
            return

        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url or not api_key:
            raise ValueError(
                "Set QDRANT_URL and QDRANT_API_KEY in your .env file "
                "(or QDRANT_PATH for a local on-disk index)."
            )

        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        self._ensure_collection()

    def _ensure_collection(self, max_retries: int = 5, backoff_seconds: float = 5.0):
        """
        Create the collection if it doesn't exist yet.

        Retries with backoff on the first connection — Qdrant Cloud's free
        tier auto-pauses on inactivity, and the first request after a pause
        can take up to ~1-2 minutes to wake the cluster (it returns
        connection errors or a bare 404 in the meantime, not a normal
        Qdrant error). Without this, that manifests as a raw startup crash
        instead of the app just waiting for the cluster to come back.
        """
        for attempt in range(1, max_retries + 1):
            try:
                existing = [c.name for c in self.client.get_collections().collections]
                break
            except Exception as e:
                if attempt == max_retries:
                    raise
                wait = backoff_seconds * attempt
                print(
                    f"⚠ Qdrant not reachable yet (attempt {attempt}/{max_retries}: {e}). "
                    f"Retrying in {wait:.0f}s — likely waking from free-tier auto-pause."
                )
                time.sleep(wait)
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✓ Created Qdrant collection: '{COLLECTION_NAME}'")
        else:
            print(f"✓ Collection '{COLLECTION_NAME}' already exists — reusing.")

    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """
        Upsert chunks with their embeddings into Qdrant.

        Args:
            chunks   : list of Chunk.to_dict() outputs from chunker.py
            embeddings: list of embedding vectors (same order as chunks)

        Returns:
            Number of points upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )

        points = [
            PointStruct(
                id=_chunk_id_to_int(chunk["chunk_id"]),
                vector=embedding,
                payload=chunk,       # full chunk dict stored as Qdrant payload
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Upsert in batches — avoids timeouts on large corpora
        total = 0
        for i in tqdm(range(0, len(points), UPSERT_BATCH_SIZE), desc="Upserting to Qdrant"):
            batch = points[i : i + UPSERT_BATCH_SIZE]
            self.client.upsert(collection_name=COLLECTION_NAME, points=batch)
            total += len(batch)

        print(f"✓ Upserted {total} points to '{COLLECTION_NAME}'")
        return total

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Dense vector search with optional payload filtering.

        Args:
            query_embedding : embedded query vector from Embedder.embed_query()
            top_k           : number of results to return
            filters         : optional dict for filtering, e.g.:
                              {"company": "Reliance Industries"}
                              {"company": "TCS", "doc_type": "quarterly_result"}
                              {"period": "Q2FY25"}

        Returns:
            List of dicts with keys: text, score, and all chunk metadata.
        """
        qdrant_filter = _build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                **hit.payload,
                "score": hit.score,
            }
            for hit in results
        ]

    def get_collection_info(self) -> dict:
        """Quick stats on the collection — useful for debugging."""
        info = self.client.get_collection(COLLECTION_NAME)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }

    def delete_collection(self):
        """Nuclear option — wipe the whole collection and start fresh."""
        self.client.delete_collection(COLLECTION_NAME)
        print(f"✓ Deleted collection '{COLLECTION_NAME}'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id_to_int(chunk_id: str) -> int:
    """
    Qdrant point IDs must be unsigned integers or UUIDs.
    We convert our hex chunk_id (first 16 hex chars) to an int.
    Collision probability at this scale is negligible.
    """
    return int(chunk_id[:16], 16)


def _build_filter(filters: dict) -> Filter:
    """
    Build a Qdrant Filter from a simple key-value dict.
    All conditions are ANDed together.

    Supported filter keys (map to chunk metadata fields):
        company   : e.g. "Reliance Industries", "TCS"
        doc_type  : "annual_report" | "quarterly_result"
        period    : e.g. "Q2FY25", "FY2024"
        block_type: "prose" | "table"
    """
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ]
    return Filter(must=conditions)


# ---------------------------------------------------------------------------
# CLI — connection + upsert smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from indexing.embed import Embedder

    store = VectorStore()
    info = store.get_collection_info()
    print(f"\nCollection info: {info}")

    # Optional: pass a chunked JSONL to do a real upsert test
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
        with open(jsonl_path, encoding="utf-8") as f:
            chunks = [json.loads(l) for l in f if l.strip()]

        embedder = Embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_documents(texts)

        store.upsert_chunks(chunks, embeddings)

        # Quick search test
        q = "What was the net profit last quarter?"
        q_emb = embedder.embed_query(q)
        results = store.search(q_emb, top_k=3)

        print(f"\nTop 3 results for: '{q}'")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score: {r['score']:.4f} | {r['company']} | {r['period']} | page {r['page_number']}")
            print(f"    {r['text'][:200]}...")