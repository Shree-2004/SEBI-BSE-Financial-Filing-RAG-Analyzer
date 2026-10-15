"""
agent.py

Tool-calling path for questions plain RAG answers badly: comparisons
across companies, growth rates, margins, unit conversions. Retrieval + one
LLM pass has to find several cells, align ₹ crore with ₹ million and do the
arithmetic in its head; here the model only decides WHAT to fetch and
compute, and the tools do the fetching and the maths:

    lookup_metric  exact figures from the metrics table (src/metrics)
    calculate      safe arithmetic (AST-evaluated, + - * / ** only)
    search_filing  the existing hybrid retrieval, for anything not in the
                   table (or when a lookup comes back ambiguous)

Every number in the final answer must come from a tool result; the
verifier checks that against the tool outputs (build_evidence_chunks).

Routing (needs_agent) is a plain rule, not an LLM call: comparison / growth
/ ratio wording, or two or more companies in the question.
"""

import ast
import json
import operator
import re

from groq import Groq

from generation.answer_generator import MODEL
from metrics.store import METRIC_NAMES, MetricStore

MAX_STEPS = 8

_AGENT_PATTERN = re.compile(
    r"\b(compare|comparison|vs\.?|versus|growth|grew|grow|increase[ds]?|decrease[ds]?|declined?|change[ds]?|"
    r"yoy|qoq|y-o-y|year[- ]on[- ]year|quarter[- ]on[- ]quarter|sequential(ly)?|margin|ratio|"
    r"difference|higher|lower|more than|less than|percentage|how much (more|less)|which (company|one))\b",
    re.I,
)


def needs_agent(question: str, companies: list[str]) -> bool:
    return len(companies) >= 2 or bool(_AGENT_PATTERN.search(question))


# ---------------------------------------------------------------------------
# calculate: arithmetic only, no names, no calls
# ---------------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only numbers and + - * / ** ( ) are allowed")
    try:
        tree = ast.parse(expression.replace(",", ""), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"not a valid arithmetic expression: {expression!r}") from e
    return ev(tree)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_metric",
        "description": "Exact reported figure for one company, metric and period from the filings.",
        "parameters": {"type": "object", "properties": {
            "company": {"type": "string"},
            "metric": {"type": "string", "enum": sorted(METRIC_NAMES)},
            "period": {"type": "string", "description": "e.g. Q1FY27, Q3FY25, 9MFY25, FY2025"},
            "basis": {"type": "string", "enum": ["consolidated", "standalone"],
                      "description": "omit unless the question specifies it (defaults to consolidated)"},
        }, "required": ["company", "metric", "period"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression. Use for EVERY difference, ratio, growth rate "
                       "or unit conversion — never compute in your head.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "search_filing",
        "description": "Search a company's filing text and tables. Use only when lookup_metric returns "
                       "nothing or 'ambiguous', or the question is not about a listed metric.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "company": {"type": "string"},
            "period": {"type": "string"}}, "required": ["query", "company"]}}},
]


def _system_prompt(store: MetricStore) -> str:
    periods: dict[str, set[str]] = {}
    for r in store.available():
        periods.setdefault(r["company"], set()).add(r["period"])
    coverage = "\n".join(f"- {c}: {', '.join(sorted(p))}" for c, p in sorted(periods.items()))
    return f"""You answer questions about Indian listed companies' financial filings, using tools.

Rules:
1. Get every reported figure with lookup_metric. If it returns nothing or "ambiguous", use search_filing.
2. NEVER do arithmetic yourself. Call calculate for every difference, ratio, growth rate and unit conversion.
3. Units differ between companies (₹ crore vs ₹ million; 1 crore = 10 million). lookup_metric returns
   value_crore — compare companies on value_crore and say the unit is ₹ crore.
4. Growth % = (current - previous) / previous * 100. Year-ago quarter of Q1FY27 is Q1FY26; the
   previous quarter of Q1FY27 is Q4FY26; of Q3FY25 they are Q3FY24 and Q2FY25.
5. Quote every number exactly as a tool returned it (commas allowed; percentages rounded to 2 decimals
   are fine). Cite each reported figure as [Company | Period | Page N].
6. If a figure is not available, say so plainly. Keep the answer short.

Periods available in the metrics table:
{coverage}"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MetricsAgent:
    def __init__(self, retriever, store: MetricStore | None = None, client: Groq | None = None):
        self.retriever = retriever
        self.store = store or MetricStore()
        self.client = client or Groq()
        self.system = _system_prompt(self.store)

    def _run_tool(self, name: str, args: dict, trace: dict):
        if name == "lookup_metric":
            r = self.store.lookup(args.get("company", ""), args.get("metric", ""),
                                  args.get("period", ""), args.get("basis"))
            if r and r.get("value") is not None:
                trace["lookups"].append(r)
            return r or {"result": "not available in the metrics table"}
        if name == "calculate":
            try:
                v = calculate(args.get("expression", ""))
            except Exception as e:                                   # noqa: BLE001
                return {"error": str(e)}
            trace["calculations"].append({"expression": args["expression"], "result": v})
            return {"expression": args["expression"], "result": round(v, 4)}
        if name == "search_filing":
            f = {"company": args.get("company")}
            if args.get("period"):
                f["period"] = args["period"]
            chunks = self.retriever.retrieve(args.get("query", ""), top_k=4, filters=f)
            trace["chunks"].extend(chunks)
            return [{"page": c.get("page_number"), "text": c["text"][:1200]} for c in chunks]
        return {"error": f"unknown tool {name}"}

    def run(self, question: str) -> dict:
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": question}]
        trace = {"lookups": [], "calculations": [], "chunks": [], "tool_calls": []}
        answer = ""
        for _ in range(MAX_STEPS):
            resp = self.client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
                temperature=0.1, max_tokens=1024)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [t.model_dump() for t in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._run_tool(tc.function.name, args, trace)
                trace["tool_calls"].append({"tool": tc.function.name, "args": args})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, default=str)[:4000]})
        else:
            answer = "I could not complete this question within the tool-call limit."

        return {"answer": answer, "model": MODEL, "query": question, **trace}


# ---------------------------------------------------------------------------
# Verification + citations from the tool trace
# ---------------------------------------------------------------------------

def _variants(v: float) -> list[float]:
    """A figure as an answer may legitimately write it: exact, or rounded."""
    out = {round(v, 4)}
    for d in (2, 1, 0):
        out.add(round(v, d))
    return sorted(out)


def build_evidence_chunks(trace: dict) -> list[dict]:
    """Tool outputs rendered as a table the cell-level Verifier can check
    the answer against, plus any chunks search_filing returned."""
    rows = []
    for r in trace["lookups"]:
        label = f"{r['metric_name']} {r['company']} {r.get('basis', '')} {r['period']}"
        for v in _variants(r["value"]):
            rows.append(f"| {label} | {v} |")
        if r.get("value_crore") is not None and r["value_crore"] != r["value"]:
            for v in _variants(r["value_crore"]):
                rows.append(f"| {label} in crore | {v} |")
    for c in trace["calculations"]:
        # describe a calculation by the looked-up figures that went into it,
        # so the verifier can match "revenue grew 5.6%" to this row
        expr_nums = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", c["expression"])}
        inputs = sorted({f"{r['metric_name']} {r['company']} {r['period']}" for r in trace["lookups"]
                         if r["value"] in expr_nums or r.get("value_crore") in expr_nums})
        label = "calculated growth change difference ratio margin share of " + "; ".join(inputs)
        for v in _variants(c["result"]):
            rows.append(f"| {label} | {v} |")
    chunks = list(trace["chunks"])
    if rows:
        chunks.append({"text": "Tool results\n| Figure | Value |\n|---|---|\n" + "\n".join(rows)})
    return chunks


def citations_from_trace(trace: dict) -> list[dict]:
    cites = []
    for r in trace["lookups"]:
        for p in r.get("pages", [])[:1]:
            cites.append({"company": r["company"], "period": r["period"], "page_number": p,
                          "doc_type": None, "source_file": r.get("source_file"),
                          "excerpt": f"{r['metric_name']} ({r.get('basis')}): {r['value']:,} {r.get('unit')}"})
    for c in trace["chunks"]:
        cites.append({"company": c.get("company"), "period": c.get("period"), "page_number": c.get("page_number"),
                      "doc_type": c.get("doc_type"), "source_file": c.get("source_file"),
                      "excerpt": c.get("text", "")[:300]})
    return cites


def agent_answer(agent: "MetricsAgent", verifier, question: str) -> dict:
    """Run the agent and verify its answer against the tool outputs.
    Same keys as generation.verifier.generate_and_verify, plus the trace."""
    run = agent.run(question)
    evidence = build_evidence_chunks(run)
    ver = verifier.verify(run["answer"], evidence, query=question)
    return {
        "answer": run["answer"],
        "citations": citations_from_trace(run),
        "model": run["model"],
        "chunks_used": len(evidence),
        "query": question,
        "verification": ver,
        "is_grounded": ver["is_grounded"],
        "grounding_confidence": ver["confidence"],
        "warning": ver["warning"],
        "tool_calls": run["tool_calls"],
        "evidence_text": "\n".join(c["text"] for c in evidence),
    }
