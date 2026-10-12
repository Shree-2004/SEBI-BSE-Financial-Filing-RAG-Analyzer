"""
embed_and_upsert.py

Embeds a chunked JSONL file and upserts directly to Qdrant.
Use this for new PDFs after chunking.

Run:
    set PYTHONPATH=src && python embed_and_upsert.py --input data/processed/tcs_chunked.jsonl --company TCS
"""
import json
import time
import os
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv()

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "filing_chunks")
BATCH_SIZE = 50
RETRY_ATTEMPTS = 5
RETRY_DELAY = 10


def chunk_id_to_int(chunk_id: str) -> int:
    return int(chunk_id[:16], 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to chunked JSONL file")
    parser.add_argument("--company", required=True, help="Company name for confirmation")
    args = parser.parse_args()

    # Load chunks
    with open(args.input, encoding="utf-8") as f:
        chunks = [json.loads(l) for l in f if l.strip()]
    print(f"✓ Loaded {len(chunks)} chunks for {args.company}")

    # Embed
    from indexing.embed import Embedder
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_documents(texts)
    print(f"✓ Embedded {len(embeddings)} chunks")

    # Connect to Qdrant
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )

    # Build points
    points = [
        PointStruct(
            id=chunk_id_to_int(chunk["chunk_id"]),
            vector=embedding,
            payload=chunk,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    # Upsert with retry
    total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Upserting {len(points)} points in {total_batches} batches...")

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(points))
        batch = points[start:end]

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=batch)
                print(f"  Batch {batch_idx+1}/{total_batches} ✓ ({end}/{len(points)} points)")
                break
            except Exception as e:
                if attempt < RETRY_ATTEMPTS:
                    print(f"  Retry {attempt}: {e}")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  Failed after {RETRY_ATTEMPTS} attempts. Stopping.")
                    return

    info = client.get_collection(COLLECTION_NAME)
    print(f"\n✓ Done! Collection now has {info.points_count} total points")


if __name__ == "__main__":
    main()