---
title: FinFiling RAG API
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
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
