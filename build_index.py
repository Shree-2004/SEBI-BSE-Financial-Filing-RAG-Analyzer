"""
build_index.py

Rebuilds BOTH retrieval indexes from a set of chunked JSONL files in one
step: the dense index (Qdrant collection) and the BM25 index (pickle).
Replaces doing steps 5-6 of the "add a company" workflow by hand, where
it was easy to update one index and forget the other.

The target is controlled by the same env vars the API reads:
    QDRANT_COLLECTION   collection name (default: the one matching the embedder —
                        filing_chunks_int8 for the ONNX int8 model, else filing_chunks)
    QDRANT_PATH         if set, a local on-disk Qdrant instead of the cloud
    BM25_INDEX_PATH     where to write the BM25 pickle

Run (local index for the Docling-parsed corpus):
    set PYTHONPATH=src
    set QDRANT_PATH=data/qdrant_local
    set QDRANT_COLLECTION=filing_chunks_docling
    set BM25_INDEX_PATH=data/processed/docling/bm25_index.pkl
    python build_index.py data/processed/docling/*_chunked.jsonl --recreate
"""

import argparse
import json
import os

from dotenv import load_dotenv

load_dotenv()

from indexing.bm25_index import BM25Index
from indexing.embed import Embedder
from indexing.vector_store import COLLECTION_NAME, VectorStore


def main():
    parser = argparse.ArgumentParser(description="Rebuild dense + BM25 indexes from chunked JSONL.")
    parser.add_argument("inputs", nargs="+", help="Chunked JSONL files (all companies)")
    parser.add_argument("--recreate", action="store_true", help="Drop the collection first")
    args = parser.parse_args()

    chunks = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            chunks.extend(json.loads(line) for line in f if line.strip())
    print(f"✓ Loaded {len(chunks)} chunks from {len(args.inputs)} files")

    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk_ids across input files — would silently overwrite points.")

    store = VectorStore()
    if args.recreate:
        store.delete_collection()
        store._ensure_collection()
    for field in ["company", "period", "doc_type", "block_type"]:
        # no-op in local mode; required for filtered search on Qdrant Cloud
        store.client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema="keyword")

    embeddings = Embedder().embed_documents([c["text"] for c in chunks])
    store.upsert_chunks(chunks, embeddings)

    bm25_path = os.getenv("BM25_INDEX_PATH", "data/processed/bm25_index.pkl")
    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save(bm25_path)

    print(f"\n✓ Collection '{COLLECTION_NAME}': {store.get_collection_info()['points_count']} points")
    print(f"✓ BM25: {bm25_path}")


if __name__ == "__main__":
    main()
