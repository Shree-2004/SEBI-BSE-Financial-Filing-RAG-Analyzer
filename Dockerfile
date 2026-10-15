FROM python:3.12-slim

WORKDIR /app

# System deps for pdfplumber/pdf2image/pytesseract (ingestion scripts import
# cleanly even though the API itself doesn't call them at request time).
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch first: otherwise sentence-transformers pulls the default
# Linux wheel with ~3GB of CUDA libraries that a CPU host never uses —
# slower builds and a much bigger image on free tiers (Render, Spaces).
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedder + reranker weights into the image at build time.
# Without this, the first request after every cold start (Railway/Render
# free tier spins the container down when idle) pays for a ~400MB+ model
# download on top of the load itself.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Hugging Face Spaces runs the container as a non-root user (uid 1000);
# the model cache baked in above must stay writable for lock files.
RUN chmod -R 777 /app/.cache

COPY src/ ./src/
COPY data/processed/bm25_index.pkl ./data/processed/bm25_index.pkl
COPY data/processed/metrics.sqlite ./data/processed/metrics.sqlite

ENV PYTHONPATH=/app/src
ENV PORT=8000
EXPOSE 8000

# --proxy-headers: behind a host's proxy (Spaces, Railway) every request
# otherwise appears to come from the proxy IP, so all visitors would share
# one per-IP rate-limit bucket.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
