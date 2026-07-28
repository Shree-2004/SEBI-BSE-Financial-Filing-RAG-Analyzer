FROM python:3.12-slim

WORKDIR /app

# System deps for pdfplumber/pdf2image/pytesseract (ingestion scripts import
# cleanly even though the API itself doesn't call them at request time).
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
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

COPY src/ ./src/
COPY data/processed/bm25_index.pkl ./data/processed/bm25_index.pkl

ENV PYTHONPATH=/app/src
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
