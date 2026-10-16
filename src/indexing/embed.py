"""
embed.py

Embeddings for chunks and queries with BAAI/bge-base-en-v1.5 (768-dim).

BGE-specific note: for retrieval, queries are prefixed with
"Represent this sentence for searching relevant passages: " but
documents/chunks are not. Handled here so callers don't have to.

Two interchangeable backends, chosen by EMBED_BACKEND:
- "onnx"  (default when the ONNX models are prepared): ONNX Runtime via
  indexing/onnx_models.py.
  The API uses this: PyTorch's first inference alone added ~330 MB of RSS
  (measured), which is what pushed the API past a 512 MB free tier.
- "torch": sentence-transformers / PyTorch, the original backend.
Same weights either way — ONNX query/document embeddings match the
PyTorch ones at cosine 1.00000 on the eval questions, so the existing
Qdrant index needs no re-embedding.

Usage:
    from indexing.embed import Embedder

    embedder = Embedder()
    doc_embeddings = embedder.embed_documents(["chunk text 1", "chunk text 2"])
    query_embedding = embedder.embed_query("what were reliance revenues Q2?")
"""

import os

import numpy as np
from tqdm import tqdm

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32  # safe for 1GB RAM environments

def default_backend() -> str:
    env = os.getenv("EMBED_BACKEND")
    if env:
        return env
    try:
        import onnxruntime  # noqa: F401
        from indexing.onnx_models import EMBEDDER_DIR
        if EMBEDDER_DIR.exists():
            return "onnx"
    except ImportError:
        pass
    return "torch"           # ONNX runtime or prepared models not available


def default_collection() -> str:
    """Qdrant collection whose vectors match the embedder in use. Queries
    must be embedded by the same model variant as the stored chunks:
    re-embedding the index with the int8 model scored 58/63 on retrieval,
    vs 56/63 when int8 queries searched the fp32-embedded index."""
    if default_backend() == "onnx" and os.getenv("EMBED_ONNX_VARIANT", "int8") == "int8":
        return "filing_chunks_int8"
    return "filing_chunks"          # fp32 vectors (PyTorch or ONNX fp32 — identical)


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu", backend: str | None = None):
        self.backend = backend or default_backend()
        print(f"Loading embedding model: {model_name} ({self.backend}) ...")
        if self.backend == "onnx":
            from indexing.onnx_models import OnnxEmbedder

            self.model = OnnxEmbedder()
            self.embedding_dim = 768
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded — embedding dim: {self.embedding_dim}")

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "onnx":
            return np.vstack([self.model.embed(texts[i:i + BATCH_SIZE])
                              for i in range(0, len(texts), BATCH_SIZE)])
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                 batch_size=BATCH_SIZE)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks (no prefix), as plain floats for Qdrant."""
        if not texts:
            return []
        all_embeddings = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
            all_embeddings.extend(self._encode(texts[i : i + BATCH_SIZE]).tolist())
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query WITH the BGE retrieval prefix."""
        return self._encode([QUERY_PREFIX + query])[0].tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Batch embed multiple queries (useful for eval runs)."""
        return self._encode([QUERY_PREFIX + q for q in queries]).tolist()


# ---------------------------------------------------------------------------
# CLI — quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    embedder = Embedder()

    sample_chunks = [
        "Reliance Industries reported a consolidated net profit of ₹19,101 crore for Q2FY25.",
        "The Board of Directors recommends a dividend of ₹9 per equity share.",
        "Risk factors include regulatory changes and commodity price volatility.",
    ]

    doc_embs = embedder.embed_documents(sample_chunks)
    query_emb = embedder.embed_query("What was Reliance's net profit in Q2?")

    print(f"\n✓ Embedded {len(doc_embs)} chunks")
    print(f"✓ Embedding dim: {len(doc_embs[0])}")
    print(f"✓ Query embedding dim: {len(query_emb)}")

    sims = [np.dot(query_emb, d) for d in doc_embs]
    best = sample_chunks[np.argmax(sims)]
    print(f"\nMost similar chunk to query:\n  '{best}'")
