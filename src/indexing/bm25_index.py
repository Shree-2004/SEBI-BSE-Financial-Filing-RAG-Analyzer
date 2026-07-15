"""
bm25_index.py

Builds and queries a BM25 keyword index over your filing chunks.

Why BM25 alongside Qdrant?
- Dense embeddings are great for semantic similarity ("what was the profit")
- BM25 is great for exact keyword matches ("HDFC Bank Q2FY25 NIM 4.3%")
- Financial filings are full of exact numbers, company names, and ticker
  symbols that embeddings can blur together — BM25 catches these precisely
- Together (hybrid retrieval) they beat either alone consistently

The index is saved to disk as a pickle so you don't rebuild it every time
the FastAPI server starts.

Usage:
    from indexing.bm25_index import BM25Index

    # Build once after ingestion
    index = BM25Index()
    index.build(chunks)          # chunks = list of dicts from chunker.py
    index.save("data/processed/bm25_index.pkl")

    # Load and query at runtime
    index = BM25Index.load("data/processed/bm25_index.pkl")
    results = index.search("Reliance net profit Q2FY25", top_k=10)
"""

import json
import pickle
import re
import string
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """
    Simple but effective tokenizer for financial text:
    - Lowercase
    - Keep numbers intact (critical — "4.3%" and "43%" are different)
    - Split on whitespace and punctuation except . % - /
      (so "Q2FY25", "4.3%", "NIM" survive as single tokens)
    - Remove very short tokens (single chars, empty strings)
    """
    text = text.lower()
    # replace punctuation except . % - / with space
    text = re.sub(r"[^\w\s.%\-/]", " ", text)
    tokens = text.split()
    # filter tokens that are just punctuation or too short
    tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
    return tokens


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[dict]) -> None:
        """
        Build BM25 index from a list of chunk dicts (chunker.py output).

        Args:
            chunks: list of dicts with at minimum a "text" key.
                    All other keys are stored and returned in search results.
        """
        print(f"Building BM25 index over {len(chunks)} chunks...")
        self.chunks = chunks
        self._tokenized_corpus = [
            _tokenize(chunk["text"])
            for chunk in tqdm(chunks, desc="Tokenizing")
        ]
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        print("✓ BM25 index built")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Keyword search over the index.

        Args:
            query   : natural language or keyword query
            top_k   : number of results to return
            filters : optional metadata filter, same format as VectorStore.search
                      e.g. {"company": "TCS", "period": "Q2FY25"}

        Returns:
            List of dicts with chunk metadata + "bm25_score" key.
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # pair each chunk with its score and apply optional metadata filter
        scored = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            if filters and not _matches_filter(chunk, filters):
                continue
            scored.append({**chunk, "bm25_score": float(score)})

        # sort by score descending, return top_k
        scored.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored[:top_k]

    def save(self, path: str) -> None:
        """Persist the index to disk so it survives server restarts."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "tokenized_corpus": self._tokenized_corpus,
                },
                f,
            )
        print(f"✓ BM25 index saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load a previously saved index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance.chunks = data["chunks"]
        instance._tokenized_corpus = data["tokenized_corpus"]
        instance.bm25 = BM25Okapi(instance._tokenized_corpus)
        print(f"✓ BM25 index loaded from {path} ({len(instance.chunks)} chunks)")
        return instance

    @classmethod
    def build_from_jsonl(cls, jsonl_path: str, save_path: str | None = None) -> "BM25Index":
        """
        Convenience: build directly from a chunked JSONL file
        (data/processed/chunked.jsonl output from chunker.py).
        Optionally save to disk immediately.
        """
        with open(jsonl_path, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]

        instance = cls()
        instance.build(chunks)

        if save_path:
            instance.save(save_path)

        return instance


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _matches_filter(chunk: dict, filters: dict) -> bool:
    """Check if a chunk's metadata matches all filter conditions."""
    return all(chunk.get(k) == v for k, v in filters.items())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BM25 index from chunked JSONL.")
    parser.add_argument(
        "--input",
        default="data/processed/chunked.jsonl",
        help="Chunked JSONL file from chunker.py",
    )
    parser.add_argument(
        "--output",
        default="data/processed/bm25_index.pkl",
        help="Where to save the index",
    )
    parser.add_argument(
        "--test-query",
        default="Reliance Industries net profit Q2FY25",
        help="A test query to run after building",
    )
    args = parser.parse_args()

    index = BM25Index.build_from_jsonl(args.input, save_path=args.output)

    results = index.search(args.test_query, top_k=3)
    print(f"\nTop results for: '{args.test_query}'")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] BM25 score: {r['bm25_score']:.4f} | {r.get('company')} | {r.get('period')} | page {r.get('page_number')}")
        print(f"    {r['text'][:200]}...")