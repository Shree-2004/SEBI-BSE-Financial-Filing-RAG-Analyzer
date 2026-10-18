"""
run_eval.py

Runs the 45-question eval set against the live RAG pipeline.

Headline metric is STRICT accuracy (eval/strict_score.py): the answer must
contain the gold figure/fact. The original any-keyword metric is still
reported, labelled "lenient", so old runs stay comparable — but it is not
an accuracy number (an answer that repeats the question with the wrong
figure passes it).

Also reports context hit rate: whether the gold answer was present in the
top-5 retrieved chunks at all. Strict miss + context hit = generation
problem; strict miss + context miss = retrieval/parsing problem.

Run:
    set PYTHONPATH=src && python eval/run_eval.py
    python eval/run_eval.py --retrieval-only     # no LLM calls, context hit only
    python eval/run_eval.py --resume             # continue after a quota stop

Which index is evaluated is controlled by QDRANT_COLLECTION and
BM25_INDEX_PATH (see .env.example).

Output (paths overridable with --out / --summary):
    eval/eval_results.json  — full results per question
    eval/eval_summary.txt   — summary metrics
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from groq import RateLimitError

from retrieval.hybrid_retriever import HybridRetriever
from generation.answer_generator import AnswerGenerator
from generation.verifier import Verifier, generate_and_verify
from strict_score import is_not_available_gold, strict_match
from retrieval.query_filters import resolve_filters
from generation.agent import MetricsAgent, agent_answer, needs_agent

QUESTIONS_PATH = "eval/eval_questions.json"
RESULTS_PATH = "eval/eval_results.json"
SUMMARY_PATH = "eval/eval_summary.txt"


def keyword_match(answer: str, keywords: list[str]) -> bool:
    """Original lenient metric: ANY expected keyword appears in the answer."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


def _pct(k: int, n: int) -> str:
    return f"{k}/{n} = {k / n * 100:.1f}%" if n else "n/a"


def run_eval(args):
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only.split(","))]

    done: dict[str, dict] = {}
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {r["id"]: r for r in json.load(f) if not r["answer"].startswith("ERROR")}
        print(f"Resuming: {len(done)} questions already answered")

    from indexing.vector_store import COLLECTION_NAME
    print(f"Index: collection={COLLECTION_NAME} "
          f"bm25={os.getenv('BM25_INDEX_PATH', 'data/processed/bm25_index.pkl')}")
    retriever = HybridRetriever()
    available: dict[str, set] = {}
    for ch in (retriever.bm25.chunks if retriever.bm25 else []):
        available.setdefault(ch["company"], set()).add(ch["period"])
    from metrics.store import MetricStore
    comparatives = MetricStore().comparative_periods()
    generator = None if args.retrieval_only else AnswerGenerator()
    if generator and args.no_fallback:
        # keep every answer from the same model so runs are comparable
        generator.fallback_client = None
    verifier = Verifier()
    agent = MetricsAgent(retriever) if args.router and not args.retrieval_only else None

    results = []
    for i, q in enumerate(questions, 1):
        if q["id"] in done:
            results.append(done[q["id"]])
            continue
        print(f"[{i}/{len(questions)}] {q['id']}: {q['question'][:60]}...")

        resolution = resolve_filters(q["question"], available, comparatives=comparatives)
        na_question = is_not_available_gold(q["gold"])
        if (args.auto_filters or args.router) and resolution["missing"] and not resolution["filters"]:
            # the API answers this without retrieval or an LLM call (HTTP 404)
            answer = f"Not in the index — {resolution['missing']}."
            record = {"id": q["id"], "question": q["question"], "company": q["company"],
                      "period": q["period"], "category": q["category"], "context_hit": None,
                      "retrieved_pages": [], "path": "refused", "answer": answer}
            if not args.retrieval_only:
                record.update({"model": None, "strict_hit": strict_match(answer, q["gold"]),
                               "keyword_hit": keyword_match(answer, q["expected_keywords"]),
                               "is_grounded": True, "grounding_confidence": 1.0,
                               "not_available": True, "citation_count": 0, "citations": []})
            print(f"       [refused] {answer}")
            results.append(record)
            continue
        use_agent = bool(agent) and needs_agent(
            q["question"], resolution["detected"]["companies"], agent.store,
            resolution["detected"]["period"]
            or (resolution["filters"][0].get("period") if len(resolution["filters"]) == 1 else None))
        if use_agent:
            chunks = []                     # the agent fetches its own evidence
        elif args.no_filters:
            chunks = retriever.retrieve(query=q["question"], top_k=5, filters=None)
        elif args.auto_filters or args.router:
            # what the API does when the user doesn't pick a filter
            filter_list = resolution["filters"]
            if len(filter_list) > 1:
                chunks = retriever.retrieve_multi(q["question"], filter_list, top_k=6)
            else:
                chunks = retriever.retrieve(query=q["question"], top_k=5,
                                            filters=filter_list[0] if filter_list else None)
        else:
            filters = {"company": q["company"]}
            if q.get("period"):
                filters["period"] = q["period"]
            chunks = retriever.retrieve(query=q["question"], top_k=5, filters=filters)
        # an unanswerable question has no gold chunk to retrieve
        context_hit = None if na_question else strict_match("\n".join(c["text"] for c in chunks), q["gold"])
        record = {
            "id": q["id"],
            "question": q["question"],
            "company": q["company"],
            "period": q["period"],
            "category": q["category"],
            "context_hit": context_hit,
            "retrieved_pages": [c.get("page_number") for c in chunks],
            "path": "agent" if use_agent else "rag",
        }

        if args.retrieval_only:
            record["answer"] = ""
            results.append(record)
            print(f"       context: {'-' if context_hit is None else '✓' if context_hit else '✗'}")
            continue

        try:
            for attempt in range(4):
                try:
                    if use_agent:
                        result = agent_answer(agent, verifier, q["question"])
                        # "context" for the agent = everything its tools returned
                        if not na_question:
                            record["context_hit"] = context_hit = strict_match(result["evidence_text"], q["gold"])
                        record["tool_calls"] = result["tool_calls"]
                    else:
                        result = generate_and_verify(
                            query=q["question"], chunks=chunks, generator=generator, verifier=verifier
                        )
                    break
                except RateLimitError as e:
                    # per-minute (TPM/RPM) limits clear in seconds — wait them
                    # out; anything else (daily TPD cap) falls through to stop
                    if "per minute" not in str(e) or attempt == 3:
                        raise
                    print("       per-minute rate limit — waiting 30s")
                    time.sleep(30)
            answer = result["answer"]
            record.update({
                "answer": answer,
                "model": result.get("model"),
                "strict_hit": strict_match(answer, q["gold"]),
                "keyword_hit": keyword_match(answer, q["expected_keywords"]),
                "is_grounded": result["is_grounded"],
                "grounding_confidence": result["grounding_confidence"],
                "not_available": "not available in the provided" in answer.lower(),
                "citation_count": len(result["citations"]),
                "citations": result["citations"],
            })
            print(f"       [{record['path']}] strict: {'✓' if record['strict_hit'] else '✗'} | "
                  f"context: {'✓' if context_hit else '✗'} | grounded: {record['is_grounded']}")
        except RateLimitError as e:
            print(f"       QUOTA STOP: {e}\n       Re-run later with --resume.")
            record.update({"answer": f"ERROR: {e}"})
            results.append(record)
            break
        except Exception as e:
            print(f"       ERROR: {e}")
            record.update({"answer": f"ERROR: {e}"})

        results.append(record)
        # save after every question so a quota stop loses nothing
        _save(args.out, results, done)
        time.sleep(1)  # avoid rate limiting on Groq free tier

    # a quota stop breaks out early: keep the answers earlier runs saved for
    # questions this run never reached (they used to be dropped from --out)
    seen = {r["id"] for r in results}
    results += [r for qid, r in done.items() if qid not in seen]
    _save(args.out, results, {})

    write_summary(results, args, total_questions=len(questions))


def _save(path, results, done):
    """Results so far plus previously saved answers not re-run yet."""
    seen = {r["id"] for r in results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results + [r for qid, r in done.items() if qid not in seen], f, indent=2, ensure_ascii=False)


def write_summary(results, args, total_questions):
    from indexing.vector_store import COLLECTION_NAME
    scored = [r for r in results if r["context_hit"] is not None]   # unanswerable ones have no gold chunk
    n = len(scored)
    ctx = sum(1 for r in scored if r["context_hit"])
    lines = [
        "FinFiling RAG — Eval Summary",
        "=============================",
        f"Index             : {COLLECTION_NAME} (rerank pool {os.getenv('RERANK_POOL') or 10})",
        f"Questions         : {len(results)}/{total_questions}"
        f"  ({len(results) - n} unanswerable/refused, excluded from context hit)",
        f"Context hit @5    : {_pct(ctx, n)}   (gold answer present in retrieved chunks)",
    ]

    answered = [r for r in results if "strict_hit" in r]
    if answered:
        a = len(answered)
        strict = sum(1 for r in answered if r["strict_hit"])
        lines += [
            f"Answered          : {a}  (models: {', '.join(sorted({str(r.get('model')) for r in answered}))})",
            "",
            f"STRICT ACCURACY   : {_pct(strict, a)}",
            f"Lenient keyword   : {_pct(sum(1 for r in answered if r['keyword_hit']), a)}   (legacy any-keyword metric, not accuracy)",
            f"Grounding rate    : {_pct(sum(1 for r in answered if r['is_grounded']), a)}",
            f"Citation rate     : {_pct(sum(1 for r in answered if r['citation_count'] > 0), a)}",
            f"Not-available     : {_pct(sum(1 for r in answered if r['not_available']), a)}",
            "",
            "Failure split (strict misses):",
            f"  retrieval/parse : {sum(1 for r in answered if not r['strict_hit'] and r['context_hit'] is False)}   (gold not in retrieved context)",
            f"  generation      : {sum(1 for r in answered if not r['strict_hit'] and r['context_hit'])}   (gold retrieved, answer still wrong)",
            "",
            "Per-category strict accuracy:",
        ]
        cats: dict[str, list[bool]] = {}
        for r in answered:
            cats.setdefault(r["category"], []).append(r["strict_hit"])
        for cat, hits in sorted(cats.items()):
            lines.append(f"  {cat:<20} {sum(hits)}/{len(hits)}")
        lines += ["", "Strict misses: " + " ".join(r["id"] for r in answered if not r["strict_hit"])]
    else:
        lines += ["", "Context misses: " + " ".join(r["id"] for r in scored if not r["context_hit"])]

    summary = "\n".join(lines) + "\n"
    print("\n" + summary)
    with open(args.summary, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"✓ Results: {args.out}\n✓ Summary: {args.summary}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--retrieval-only", action="store_true", help="skip the LLM; report context hit only")
    p.add_argument("--resume", action="store_true", help="skip questions already answered in --out")
    p.add_argument("--no-fallback", action="store_true", help="disable the OpenAI fallback so all answers come from one model")
    p.add_argument("--only", help="comma-separated question ids")
    p.add_argument("--auto-filters", action="store_true",
                   help="detect company/period from the question instead of using the labelled ones")
    p.add_argument("--router", action="store_true",
                   help="route like the API: comparison/growth/ratio questions to the tool-using agent, "
                        "the rest to RAG with auto-detected filters")
    p.add_argument("--no-filters", action="store_true",
                   help="no company/period filter at all (what the API did before auto-detection)")
    p.add_argument("--out", default=RESULTS_PATH)
    p.add_argument("--summary", default=SUMMARY_PATH)
    run_eval(p.parse_args())
