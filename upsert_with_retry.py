"""
upsert_with_retry.py

Loads pre-computed embeddings from disk and upserts to Qdrant
with retry logic and checkpoint resuming — so if the connection
drops mid-way, it picks up from where it left off instead of
starting from zero.

Run:
    set PYTHONPATH=src && python upsert_with_retry.py
"""

import json
import pickle
import time
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv()

CHUNKS_PATH = "data/processed/tcs_chunked.jsonl"
EMBEDDINGS_PATH = "data/processed/tcs_embeddings.pkl"
CHECKPOINT_PATH = "data/processed/upsert_checkpoint.txt"
COLLECTION_NAME = "filing_chunks"
BATCH_SIZE = 50   # smaller batch = less likely to timeout
RETRY_ATTEMPTS = 5
RETRY_DELAY = 10  # seconds between retries


def load_checkpoint() -> int:
    """Return the last successfully upserted batch index."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return int(f.read().strip())
    return 0


def save_checkpoint(batch_idx: int):
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(str(batch_idx))


def chunk_id_to_int(chunk_id: str) -> int:
    return int(chunk_id[:16], 16)


def main():
    print("Loading chunks and embeddings from disk...")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(l) for l in f if l.strip()]

    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings = pickle.load(f)

    print(f"✓ Loaded {len(chunks)} chunks + {len(embeddings)} embeddings")
    assert len(chunks) == len(embeddings), "Mismatch between chunks and embeddings!"

    # Connect to Qdrant
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )

    # Build all points
    points = [
        PointStruct(
            id=chunk_id_to_int(chunk["chunk_id"]),
            vector=embedding,
            payload=chunk,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    # Resume from checkpoint
    start_batch = load_checkpoint()
    total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE

    if start_batch > 0:
        print(f"Resuming from batch {start_batch}/{total_batches}")
    else:
        print(f"Starting upsert: {len(points)} points in {total_batches} batches")

    for batch_idx in range(start_batch, total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(points))
        batch = points[start:end]

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=batch)
                save_checkpoint(batch_idx + 1)
                print(f"  Batch {batch_idx+1}/{total_batches} ✓ ({end}/{len(points)} points)")
                break
            except Exception as e:
                if attempt < RETRY_ATTEMPTS:
                    print(f"  Batch {batch_idx+1} failed (attempt {attempt}): {e}")
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"  Batch {batch_idx+1} failed after {RETRY_ATTEMPTS} attempts. Stopping.")
                    print(f"  Checkpoint saved at batch {batch_idx}. Re-run to resume.")
                    return

    # Clear checkpoint on success
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"\n✓ All {len(points)} points upserted to Qdrant successfully!")

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"✓ Collection now has {info.points_count} points")


if __name__ == "__main__":
    main()