"""
verifier_eval.py

Measures the grounding verifier itself: does it flag the answers that are
actually wrong?

Ground truth comes from saved eval runs — each answer is already labelled
correct/incorrect by the strict scorer (eval/strict_score.py). The chunks
each answer was generated from are rebuilt by re-running retrieval against
the same index the run used (retrieval is deterministic), cached once in
eval/runs/<run>_chunks.json, so the verifier can then be iterated on
offline without the LLM.

Metrics (over answered questions, excluding "not available" answers):
    catch rate   = wrong answers flagged ungrounded / wrong answers
    false alarms = correct answers flagged ungrounded / correct answers

Usage:
    # 1) cache chunks for a run (needs the run's index; see RUNS below)
    python eval/verifier_eval.py cache headers
    # 2) score verifier(s) on all cached runs
    python eval/verifier_eval.py score
"""

import json
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# run name -> (results file, env that points retrieval at the run's index)
RUNS = {
    "pdfplumber": (
        "eval/runs/pdfplumber_gptoss120b.json",
        {"QDRANT_PATH": "data/qdrant_local", "QDRANT_COLLECTION": "filing_chunks",
         "BM25_INDEX_PATH": "data/processed/bm25_index_local_baseline.pkl"},
    ),
    "docling": (
        "eval/runs/docling_gptoss120b.json",
        {"QDRANT_PATH": "data/qdrant_local", "QDRANT_COLLECTION": "filing_chunks_docling",
         "BM25_INDEX_PATH": "data/processed/docling/bm25_index.pkl"},
    ),
    "headers": ("eval/runs/headers_gptoss120b.json", {}),   # current cloud index
}

NOT_AVAILABLE = ("not available in the provided", "not found in the provided",
                 "cannot find this information", "no information available")


def _chunks_path(run: str) -> str:
    return f"eval/runs/{run}_chunks.json"


def cache(run: str) -> None:
    results_path, env = RUNS[run]
    os.environ.update(env)
    from dotenv import load_dotenv
    load_dotenv(".env")
    from retrieval.hybrid_retriever import HybridRetriever

    with open("eval/eval_questions.json", encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    retriever = HybridRetriever()
    out = {}
    for r in results:
        q = questions[r["id"]]
        filters = {"company": q["company"], "period": q["period"]}
        chunks = retriever.retrieve(query=q["question"], top_k=5, filters=filters)
        pages = [c.get("page_number") for c in chunks]
        if pages != r.get("retrieved_pages"):
            print(f"  ! {r['id']}: retrieved pages differ from the run ({pages} vs {r.get('retrieved_pages')})")
        out[r["id"]] = [{k: c.get(k) for k in ("text", "page_number", "block_type", "company", "period")}
                        for c in chunks]
    with open(_chunks_path(run), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"✓ cached chunks for {len(out)} questions -> {_chunks_path(run)}")


def labelled_examples(run: str) -> list[dict]:
    results_path, _ = RUNS[run]
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)
    with open(_chunks_path(run), encoding="utf-8") as f:
        chunks = json.load(f)
    with open("eval/eval_questions.json", encoding="utf-8") as f:
        questions = {q["id"]: q["question"] for q in json.load(f)}
    examples = []
    for r in results:
        answer = r.get("answer", "")
        if "strict_hit" not in r or answer.startswith("ERROR"):
            continue
        if any(p in answer.lower() for p in NOT_AVAILABLE):
            continue
        examples.append({"id": r["id"], "run": run, "answer": answer, "question": questions[r["id"]],
                         "correct": r["strict_hit"], "chunks": chunks[r["id"]]})
    return examples


def score_verifier(name: str, verify, examples: list[dict]) -> dict:
    wrong = [e for e in examples if not e["correct"]]
    right = [e for e in examples if e["correct"]]
    flagged = {e["run"] + e["id"] for e in examples
               if not verify(e)["is_grounded"]}
    caught = [e for e in wrong if e["run"] + e["id"] in flagged]
    false_alarms = [e for e in right if e["run"] + e["id"] in flagged]
    print(f"{name:14} catch rate {len(caught):3}/{len(wrong):<3} = {len(caught)/max(len(wrong),1)*100:5.1f}%   "
          f"false alarms {len(false_alarms):3}/{len(right):<3} = {len(false_alarms)/max(len(right),1)*100:5.1f}%")
    return {"caught": caught, "missed": [e for e in wrong if e not in caught], "false_alarms": false_alarms}


def score() -> None:
    from generation.verifier import KeywordVerifier, Verifier

    examples = []
    for run in RUNS:
        if os.path.exists(_chunks_path(run)):
            examples += labelled_examples(run)
    print(f"{len(examples)} labelled answers "
          f"({sum(not e['correct'] for e in examples)} wrong, {sum(e['correct'] for e in examples)} correct)\n")
    old, new = KeywordVerifier(), Verifier()
    score_verifier("keyword (old)", lambda e: old.verify(e["answer"], e["chunks"]), examples)
    res = score_verifier("cell-level", lambda e: new.verify(e["answer"], e["chunks"], query=e["question"]), examples)
    print("\nper run:")
    for run in RUNS:
        sub = [e for e in examples if e["run"] == run]
        if sub:
            score_verifier(f"  {run} old", lambda e: old.verify(e["answer"], e["chunks"]), sub)
            score_verifier(f"  {run} new", lambda e: new.verify(e["answer"], e["chunks"], query=e["question"]), sub)
    if "-v" in sys.argv:
        for kind in ("missed", "false_alarms"):
            print(f"\n--- {kind} ---")
            for e in res[kind]:
                r = new.verify(e["answer"], e["chunks"], query=e["question"])
                print(f"{e['run'][:5]} {e['id']}  unverified={r['unverified_claims'][:2]}  verified={r['verified_claims'][:2]}")


def mutate(run: str = "headers") -> None:
    """
    Held-out stress test (the verifier was tuned on the labelled answers
    above, and the current pipeline only produced a handful of wrong ones).

    Take each CORRECT answer and swap its gold figure for a realistic wrong
    one drawn from the same retrieved tables — the error types the eval
    actually showed:
        earlier_period  same row, an earlier-period column (e.g. prior quarter)
        wrong_row       same column, a different row
        invented        a value that appears nowhere in the sources (+7%)
    and report how often each verifier flags the mutated answer.
    """
    import re
    from generation.verifier import KeywordVerifier, Verifier, _cells_from_chunk

    with open("eval/eval_questions.json", encoding="utf-8") as f:
        gold = {q["id"]: q["gold"] for q in json.load(f)}
    old, new = KeywordVerifier(), Verifier()
    counts: dict[str, list[int]] = {}

    def fmt(v: float) -> str:
        return f"{v:,.2f}".rstrip("0").rstrip(".")

    for e in labelled_examples(run):
        g = gold[e["id"]]
        if not e["correct"] or len(g) != 1 or not re.fullmatch(r"[\d,.]+", g[0][0]):
            continue
        gv = float(g[0][0].replace(",", ""))
        # the gold figure as written in the answer
        m = next((m for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", e["answer"])
                  if float(m.group().replace(",", "")) == gv), None)
        cells = [c for ch in e["chunks"] for c in _cells_from_chunk(ch)]
        gcell = next((c for c in cells if c.is_table and c.value == gv), None)
        if m is None or gcell is None:
            continue
        candidates = {
            "earlier_period": next((c.value for c in cells if c.is_table and c.row_label == gcell.row_label
                                    and c.column != gcell.column and c.value != gv), None),
            "wrong_row": next((c.value for c in cells if c.is_table and c.column == gcell.column
                               and c.row_label != gcell.row_label and c.value != gv), None),
            "invented": round(gv * 1.07, 2),
        }
        for kind, wv in candidates.items():
            if wv is None:
                continue
            mutated = e["answer"][:m.start()] + fmt(wv) + e["answer"][m.end():]
            caught_old = not old.verify(mutated, e["chunks"])["is_grounded"]
            caught_new = not new.verify(mutated, e["chunks"], query=e["question"])["is_grounded"]
            counts.setdefault(kind, [0, 0, 0])
            counts[kind][0] += 1
            counts[kind][1] += caught_old
            counts[kind][2] += caught_new

    print(f"Mutation test on '{run}' correct answers (wrong figure swapped in):\n")
    print(f"{'mutation':16} {'n':>3}   {'keyword (old)':>14}   {'cell-level':>12}")
    tot = [0, 0, 0]
    for kind, (n, o, w) in counts.items():
        print(f"{kind:16} {n:3}   {o:3} ({o/n*100:5.1f}%)   {w:3} ({w/n*100:5.1f}%)")
        tot = [tot[0] + n, tot[1] + o, tot[2] + w]
    n, o, w = tot
    print(f"{'all':16} {n:3}   {o:3} ({o/n*100:5.1f}%)   {w:3} ({w/n*100:5.1f}%)")


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        cache(sys.argv[2])
    elif sys.argv[1] == "mutate":
        mutate(*(sys.argv[2:3] or ["headers"]))
    else:
        score()
