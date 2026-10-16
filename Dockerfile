FROM python:3.12-slim

WORKDIR /app

# API runtime only — no PyTorch. The embedder and reranker run on ONNX
# Runtime: PyTorch's runtime alone pushed peak memory to ~1 GB, past a
# 512 MB free tier (measurements in implementation_plan.md).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ONNX models into the image (no download on cold start):
# BAAI's bge-base ONNX export, int8-quantized here, and Xenova's MiniLM
# reranker (int8 + fp32). Both precisions are kept so EMBED_ONNX_VARIANT /
# RERANK_ONNX_VARIANT=fp32 works on a bigger host; only the selected file is
# loaded into memory. `onnx` is needed for quantization only.
ENV MODELS_DIR=/app/models HF_HOME=/app/.cache/huggingface
COPY src/indexing/onnx_models.py ./src/indexing/onnx_models.py
RUN pip install --no-cache-dir onnx==1.23.2 \
    && python src/indexing/onnx_models.py prepare \
    && pip uninstall -y onnx \
    && rm -rf /app/.cache/huggingface \
    && chmod -R a+rX /app/models

COPY src/ ./src/
COPY data/processed/bm25_index.pkl ./data/processed/bm25_index.pkl
COPY data/processed/metrics.sqlite ./data/processed/metrics.sqlite

ENV PYTHONPATH=/app/src
ENV PORT=8000
EXPOSE 8000

# --proxy-headers: behind a host's proxy (Render, Spaces) every request
# otherwise appears to come from the proxy IP, so all visitors would share
# one per-IP rate-limit bucket.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
