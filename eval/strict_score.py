"""
strict_score.py

Strict answer scoring for the eval set, replacing the original
`keyword_match` as the headline metric.

Why: `keyword_match` (run_eval.py) counts an answer correct if ANY expected
keyword appears in it — and most keyword lists contain words from the
question itself ("EBITDA", "standalone", "revenue from operations"). An
answer that repeats the question with the wrong number still scored as
correct. Re-scored strictly, the first "95.6%" run was far lower.

Each question now carries a `gold` field: a list of groups, ALL of which
must be satisfied; within a group, ANY alternative counts.

    "gold": [["74,163"]]                          one exact figure
    "gold": [["Deloitte"], ["unmodified"]]        two facts, both required
    "gold": [["10", "76"], ["66", "76"]]          "76" alone, or "10" + "66"

Matching rules:
- Numeric alternatives match a NUMBER in the answer by value, so "1,90,845"
  matches "190845" and "5.5" matches "5.50" — but "234" does not match
  "1,234", and "10" does not match "Page 10" (citation brackets are
  stripped before matching).
- Text alternatives match case-insensitively with punctuation and spacing
  removed, so "BSR" matches "B S R & Co." and "BSR & Co".

Usage:
    python eval/strict_score.py eval/eval_results.json   # re-score a saved run
"""

import json
import re
import sys

# a citation is short and on one line; an unbounded match let a stray OCR "["
# in one retrieved chunk swallow everything up to a "]" several chunks later
_CITATION_RE = re.compile(r"\[[^\]\n]{0,120}\]|【[^】\n]{0,120}】")  # gpt-oss cites with 【】
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMERIC_GOLD_RE = re.compile(r"^[\d,]+(?:\.\d+)?$")


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUMBER_RE.findall(text):
        try:
            out.add(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _alt_matches(alt: str, numbers: set[float], norm_text: str) -> bool:
    if _NUMERIC_GOLD_RE.match(alt):
        return float(alt.replace(",", "")) in numbers
    return _norm_text(alt) in norm_text


def strict_match(text: str, gold: list[list[str]]) -> bool:
    """True if `text` satisfies every group in `gold`."""
    text = _CITATION_RE.sub(" ", text)
    numbers = _numbers(text)
    norm = _norm_text(text)
    return all(any(_alt_matches(alt, numbers, norm) for alt in group) for group in gold)


def rescore(results_path: str, questions_path: str = "eval/eval_questions.json") -> None:
    with open(questions_path, encoding="utf-8") as f:
        gold = {q["id"]: q["gold"] for q in json.load(f)}
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    hits = [r for r in results if strict_match(r["answer"], gold[r["id"]])]
    lenient = sum(1 for r in results if r.get("keyword_hit"))
    n = len(results)
    print(f"Strict accuracy  : {len(hits)}/{n} = {len(hits)/n*100:.1f}%")
    print(f"Lenient keyword  : {lenient}/{n} = {lenient/n*100:.1f}%")
    print("Strict misses    :", " ".join(r["id"] for r in results if r not in hits))


if __name__ == "__main__":
    rescore(sys.argv[1] if len(sys.argv) > 1 else "eval/eval_results.json")
