"""
embed.py

Generates embeddings for chunks using BAAI/bge-base-en-v1.5.
This is the same model used in your DocMind and financial RAG projects,
so you already know its characteristics:
- 768-dim output
- Strong on retrieval tasks (trained with retrieval-specific fine-tuning)
- Works well on financial/technical text

BGE-specific note: for retrieval, queries should be prefixed with
"Represent this sentence for searching relevant passages: "
but documents/chunks should NOT have this prefix. We handle this here
so the caller doesn't have to think about it.

Usage:
    from indexing.embed import Embedder

    embedder = Embedder()
    doc_embeddings = embedder.embed_documents(["chunk text 1", "chunk text 2"])
    query_embedding = embedder.embed_query("what were reliance revenues Q2?")
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32  # safe for 1GB RAM environments


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        """
        Load the BGE embedding model.
        Device: "cpu" by default — switch to "cuda" if you have a GPU.
        Model is cached locally after first download (~400MB).
        """
        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded — embedding dim: {self.embedding_dim}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of document chunks (no prefix).
        Returns list of embedding vectors as plain Python floats
        (Qdrant-compatible format).
        """
        if not texts:
            return []

        all_embeddings = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding chunks"):
            batch = texts[i : i + BATCH_SIZE]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,   # cosine similarity works correctly
                show_progress_bar=False,
            )
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single user query WITH the BGE retrieval prefix.
        Always use this for queries, never for documents.
        """
        prefixed = QUERY_PREFIX + query
        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Batch embed multiple queries (useful for eval runs)."""
        prefixed = [QUERY_PREFIX + q for q in queries]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
        )
        return embeddings.tolist()


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

    # quick similarity check — first chunk should be most similar to query
    sims = [np.dot(query_emb, d) for d in doc_embs]
    best = sample_chunks[np.argmax(sims)]
    print(f"\nMost similar chunk to query:\n  '{best}'")