"""
answer_generator.py

Takes retrieved chunks from hybrid_retriever.py and generates a
grounded, cited answer using Groq's gpt-oss-120b model.

Key design decisions:
- Prompt explicitly instructs the model to ONLY use the provided context
  and cite the source (company, period, page) for every claim
- If the answer isn't in the context, the model says so rather than
  hallucinating — this is the core "grounded RAG" guarantee
- Structured output: returns both the answer text AND parsed citations
  so the frontend can render clickable source references

Usage:
    from generation.answer_generator import AnswerGenerator

    generator = AnswerGenerator()
    response = generator.generate(
        query="What was HDFC Bank's net interest margin in Q2FY25?",
        chunks=[...],   # from HybridRetriever.retrieve()
    )
    print(response["answer"])
    print(response["citations"])
"""

import os
from groq import Groq, RateLimitError, APIStatusError, APIConnectionError
from openai import OpenAI
from dotenv import load_dotenv

from api.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)

# Was llama-3.3-70b-versatile until Groq retired it (404 model_not_found,
# 2026-10-12) — every request was silently falling through to the OpenAI
# fallback. gpt-oss-120b is what the current eval numbers were measured on.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = "gpt-4o-mini"   # used only when Groq is rate-limited or unreachable
MAX_TOKENS = 1024
TEMPERATURE = 0.1   # low temperature = more factual, less creative


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions about Indian listed companies using ONLY the provided document excerpts from their official SEBI/BSE filings.

Rules you must follow:
1. Answer ONLY from the provided context. Do not use any outside knowledge.
2. For every factual claim, cite the source using [Company | Period | Page X] format.
3. If the answer is not found in the provided context, say exactly: "This information is not available in the provided filings."
4. Be precise with numbers — do not round or paraphrase financial figures.
5. If multiple excerpts contain conflicting data, mention the conflict explicitly.
6. Keep answers concise and structured. Use bullet points for multiple figures.
"""

def _build_user_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        company = chunk.get("company", "Unknown")
        period = chunk.get("period", "Unknown")
        page = chunk.get("page_number", "?")
        doc_type = chunk.get("doc_type", "filing")
        block_type = chunk.get("block_type", "text")

        header = f"[Excerpt {i} | {company} | {period} | {doc_type} | Page {page} | {block_type}]"
        context_blocks.append(f"{header}\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_blocks)

    return f"""Use the following excerpts from official Indian financial filings to answer the question.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER (cite sources using [Company | Period | Page X] for every claim):"""


# ---------------------------------------------------------------------------
# Citation parser
# ---------------------------------------------------------------------------

def _parse_citations(answer_text: str, chunks: list[dict]) -> list[dict]:
    cited = []
    seen = set()

    for chunk in chunks:
        company = chunk.get("company", "")
        period = chunk.get("period", "")
        page = chunk.get("page_number")
        key = (company, period, page)

        if key in seen:
            continue

        if company in answer_text and period in answer_text:
            cited.append({
                "company": company,
                "period": period,
                "page_number": page,
                "doc_type": chunk.get("doc_type"),
                "source_file": chunk.get("source_file"),
                "excerpt": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
            })
            seen.add(key)

    return cited


# ---------------------------------------------------------------------------
# AnswerGenerator
# ---------------------------------------------------------------------------

class AnswerGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY must be set in your .env file.")
        self.client = Groq(api_key=api_key)

        # Optional fallback — used only when Groq is rate-limited or down.
        # Groq's free tier is a hard 100K-token/day cap that a handful of
        # real users can exhaust; without this, every request past the cap
        # 500s until the quota resets.
        openai_key = os.getenv("OPENAI_API_KEY")
        self.fallback_client = OpenAI(api_key=openai_key) if openai_key else None

        print(f"✓ AnswerGenerator ready (model: {MODEL}, fallback: {FALLBACK_MODEL if self.fallback_client else 'none'})")

    def generate(
        self,
        query: str,
        chunks: list[dict],
        max_chunks: int = 6,
    ) -> dict:
        context_chunks = chunks[:max_chunks]
        user_prompt = _build_user_prompt(query, context_chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        model_used = MODEL
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            answer_text = response.choices[0].message.content.strip()
        except (RateLimitError, APIStatusError, APIConnectionError) as e:
            if not self.fallback_client:
                raise
            logger.warning(
                "Groq unavailable, falling back to OpenAI",
                extra={"error": str(e), "fallback_model": FALLBACK_MODEL},
            )
            response = self.fallback_client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            answer_text = response.choices[0].message.content.strip()
            model_used = FALLBACK_MODEL

        citations = _parse_citations(answer_text, context_chunks)

        return {
            "answer": answer_text,
            "citations": citations,
            "model": model_used,
            "chunks_used": len(context_chunks),
            "query": query,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generator = AnswerGenerator()

    dummy_chunks = [
        {
            "text": "Reliance Industries reported a consolidated net profit of Rs 19,101 crore for Q2FY25, up 2.8% year-on-year.",
            "company": "Reliance Industries",
            "period": "Q2FY25",
            "page_number": 12,
            "doc_type": "quarterly_result",
            "source_file": "reliance_q2fy25.pdf",
            "block_type": "prose",
        },
        {
            "text": "Revenue from operations stood at Rs 2,35,481 crore in Q2FY25 compared to Rs 2,23,672 crore in Q2FY24.",
            "company": "Reliance Industries",
            "period": "Q2FY25",
            "page_number": 13,
            "doc_type": "quarterly_result",
            "source_file": "reliance_q2fy25.pdf",
            "block_type": "table",
        },
    ]

    result = generator.generate(
        query="What was Reliance's net profit in Q2FY25?",
        chunks=dummy_chunks,
    )

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nCitations ({len(result['citations'])}):")
    for c in result["citations"]:
        print(f"  - {c['company']} | {c['period']} | Page {c['page_number']}")