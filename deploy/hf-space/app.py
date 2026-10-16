"""
Hugging Face Space entrypoint.

Docker Spaces are paid, so the API runs in a (free) Gradio-SDK Space
instead: the Space just executes `python app.py` and expects a web server
on port 7860. Gradio itself is never used — this starts the same FastAPI
app the Dockerfile runs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import uvicorn  # noqa: E402

from indexing.onnx_models import EMBEDDER_DIR, prepare  # noqa: E402

if not EMBEDDER_DIR.exists():      # no Docker build step here: build models on first start
    prepare()

from api.main import app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        # behind HF's proxy: use X-Forwarded-For so per-IP rate limits work
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
