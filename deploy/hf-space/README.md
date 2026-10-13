---
title: FinFiling RAG API
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# FinFiling RAG API

Backend for [FinFiling RAG](https://github.com/Shree-2004/SEBI-BSE-Financial-Filing-RAG-Analyzer):
cited question answering over SEBI/BSE financial filings (Reliance, TCS, Infosys)
with hybrid dense + BM25 retrieval, CrossEncoder reranking, and a grounding check.

- `GET /health` · `GET /companies` · `GET /stats`
- `POST /query` with `{"query": "...", "company": "TCS"}`
- Interactive docs at `/docs`

This Space is deployed from the GitHub repo with `deploy/push_hf_space.py`.
It uses the Gradio SDK only as a free Python runtime (Docker Spaces are paid):
`app.py` starts the FastAPI app with uvicorn on port 7860.
