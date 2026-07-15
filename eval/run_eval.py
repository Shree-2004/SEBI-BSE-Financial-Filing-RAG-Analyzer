"""
run_eval.py

Runs the 35-question eval set against the live RAG pipeline and
produces citation accuracy, answer grounding rate, and per-category
scores — these are your resume numbers.

Run:
    set PYTHONPATH=src && python eval/run_eval.py

Output:
    eval/eval_results.json  — full results per question
    eval/eval_summary.txt   — the numbers you put on your resume
"""

import json
import time
import sys
import os

sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from retrieval.hybrid_retriever import HybridRetriever
from generation.answer_generator import AnswerGenerator
from generation.verifier import Verifier, generate_and_verify

QUESTIONS_PATH = "eval/eval_questions.json"
RESULTS_PATH = "eval/eval_results.json"
SUMMARY_PATH = "eval/eval_summary.txt"


def keyword_match(answer: str, keywords: list[str]) -> bool:
    """Check if any expected keyword appears in the answer."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


def run_eval():
    print("Loading eval questions...")
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"✓ {len(questions)} questions loaded\n")
    print("Initializing pipeline...")

    retriever = HybridRetriever()
    generator = AnswerGenerator()
    verifier = Verifier()

    print("✓ Pipeline ready\n")
    print("=" * 60)

    results = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{total}] {q['id']}: {q['question'][:60]}...")

        filters = {"company": q["company"]}
        if q.get("period"):
            filters["period"] = q["period"]

        try:
            chunks = retriever.retrieve(
                query=q["question"],
                top_k=5,
                filters=filters,
            )

            result = generate_and_verify(
                query=q["question"],
                chunks=chunks,
                generator=generator,
                verifier=verifier,
            )

            answer = result["answer"]
            is_grounded = result["is_grounded"]
            confidence = result["grounding_confidence"]
            citations = result["citations"]
            not_available = "not available in the provided" in answer.lower()
            keyword_hit = keyword_match(answer, q["expected_keywords"])

            record = {
                "id": q["id"],
                "question": q["question"],
                "company": q["company"],
                "period": q["period"],
                "category": q["category"],
                "answer": answer,
                "expected_keywords": q["expected_keywords"],
                "keyword_hit": keyword_hit,
                "is_grounded": is_grounded,
                "grounding_confidence": confidence,
                "not_available": not_available,
                "citation_count": len(citations),
                "citations": citations,
            }

            status = "✓" if keyword_hit else ("⚠ N/A" if not_available else "✗")
            print(f"       {status} | grounded: {is_grounded} ({confidence:.2f}) | citations: {len(citations)}")

        except Exception as e:
            print(f"       ERROR: {e}")
            record = {
                "id": q["id"],
                "question": q["question"],
                "company": q["company"],
                "period": q["period"],
                "category": q["category"],
                "answer": f"ERROR: {e}",
                "expected_keywords": q["expected_keywords"],
                "keyword_hit": False,
                "is_grounded": False,
                "grounding_confidence": 0.0,
                "not_available": False,
                "citation_count": 0,
                "citations": [],
            }

        results.append(record)
        time.sleep(1)  # avoid rate limiting on Groq free tier

    # Save full results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Compute summary metrics
    n = len(results)
    keyword_hits = sum(1 for r in results if r["keyword_hit"])
    grounded = sum(1 for r in results if r["is_grounded"])
    not_available = sum(1 for r in results if r["not_available"])
    with_citations = sum(1 for r in results if r["citation_count"] > 0)
    avg_confidence = sum(r["grounding_confidence"] for r in results) / n
    avg_citations = sum(r["citation_count"] for r in results) / n

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "hits": 0}
        categories[cat]["total"] += 1
        if r["keyword_hit"]:
            categories[cat]["hits"] += 1

    summary = f"""
FinFiling RAG — Eval Summary
=============================
Total questions   : {n}
Keyword accuracy  : {keyword_hits}/{n} = {keyword_hits/n*100:.1f}%
Grounding rate    : {grounded}/{n} = {grounded/n*100:.1f}%
Citation rate     : {with_citations}/{n} = {with_citations/n*100:.1f}%
Not-available     : {not_available}/{n} = {not_available/n*100:.1f}%
Avg confidence    : {avg_confidence:.3f}
Avg citations/Q   : {avg_citations:.1f}

Per-category accuracy:
"""
    for cat, stats in sorted(categories.items()):
        acc = stats["hits"] / stats["total"] * 100
        summary += f"  {cat:<20} {stats['hits']}/{stats['total']} = {acc:.0f}%\n"

    summary += f"""
Resume bullet (fill in):
  - Evaluated RAG pipeline on a hand-labeled {n}-question eval set
    across {len(set(r['company'] for r in results))} Indian listed companies,
    achieving {keyword_hits/n*100:.0f}% answer accuracy and
    {grounded/n*100:.0f}% grounding rate with {with_citations/n*100:.0f}%
    citation coverage.
"""

    print("\n" + summary)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"✓ Full results saved to {RESULTS_PATH}")
    print(f"✓ Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    run_eval()