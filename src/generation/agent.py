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


# "What was X's <metric>?" -> which metrics-table figure it asks for. Order
# matters: the more specific wording first ("profit before tax" is not
# "net profit", "other income" is not "revenue").
_METRIC_WORDING = [
    ("profit_attributable_to_owners", r"attributable|owners|equity holders|shareholders of the (company|parent)"),
    ("profit_before_tax_and_minority", r"before tax and minority"),
    ("profit_before_tax", r"profit before tax|\bpbt\b|pre-?tax profit"),
    ("current_tax", r"current tax"),
    ("total_tax_expense", r"tax expense|total tax"),
    ("eps_basic", r"\beps\b|earnings per (equity )?share"),
    ("net_profit", r"net profit|profit after tax|\bpat\b|profit for the (period|quarter|year)|bottom[- ]line"),
    ("total_income", r"total income"),
    ("other_income", r"other income"),
    ("interest_earned", r"interest earned"),
    ("employee_benefit_expense", r"employee (benefit|cost)|staff cost"),
    ("finance_costs", r"finance cost|interest (cost|expense)"),
    ("depreciation_amortisation", r"depreciation|amorti[sz]ation"),
    ("total_expenses", r"total (expenses|expenditure)"),
    ("revenue_from_operations", r"revenue|\bsales\b|turnover|top[- ]line"),
]
# figures the table does NOT hold even though the wording overlaps: segment
# or business-line numbers, nine-month / half-year totals, explanations.
# EBITDA is left to retrieval on purpose: most filings print it per segment
# (Reliance O2C / Jio / Retail), so the table's EBITDA rows rarely resolve.
_NOT_A_TABLE_FIGURE = re.compile(
    r"\b(segment|division|business|vertical|subsidiar\w*|geograph\w*|region\w*|nine[- ]months?|9m|"
    r"half[- ]year|h1|why|explain|reason|drivers?|guidance|outlook|arpu|volume|dividend|"
    r"net interest income|nii|order (book|inflow))\b", re.I)


def asks_reported_metric(question: str) -> str | None:
    """The metrics-table figure a single-figure question asks for, if any."""
    if _NOT_A_TABLE_FIGURE.search(question):
        return None
    q = question.lower()
    return next((m for m, pat in _METRIC_WORDING if re.search(pat, q)), None)


def needs_agent(question: str, companies: list[str], store: MetricStore | None = None,
                period: str | None = None) -> bool:
    """
    Agent for comparisons / growth / ratios and multi-company questions; and,
    when a store is given, for a plain "what was X's <metric>" question whose
    figure the metrics table holds — a lookup returns the reported cell
    (voted across every copy in the filing), where top-5 retrieval can hand
    the model a press-release rounding or a figure in the wrong currency.
    """
    if len(companies) >= 2 or _AGENT_PATTERN.search(question):
        return True
    if store is None or len(companies) != 1:
        return False
    metric = asks_reported_metric(question)
    return bool(metric) and any(
        r["metric"] == metric and (period is None or r["period"] == period)
        for r in store.available(companies[0]))


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
            # validated in _run_tool, not as a schema enum: Groq rejects the
            # whole request when the model invents a name outside an enum
            # ("total_revenue_from_operations"), so the user got an error;
            # a tool-result error lets the model correct itself instead
            "metric": {"type": "string", "description": "one of: " + ", ".join(sorted(METRIC_NAMES))},
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
   value_crore — compare companies on value_crore and say the unit is ₹ crore. A single company's
   figure is quoted in the unit its filing reports (value and unit from lookup_metric); you may add
   the ₹ crore equivalent in brackets.
4. Growth % = (current - previous) / previous * 100. Year-ago quarter of Q1FY27 is Q1FY26; the
   previous quarter of Q1FY27 is Q4FY26; of Q3FY25 they are Q3FY24 and Q2FY25. Indian fiscal year:
   quarters ending 30 June / 30 Sep / 31 Dec / 31 March are Q1 / Q2 / Q3 / Q4, and FYxx is named for
   the March it ends in (quarter ended 31 March 2026 = Q4FY26; 30 June 2025 = Q1FY26).
5. Quote every number exactly as a tool returned it (commas allowed; percentages rounded to 2 decimals
   are fine). Cite each reported figure as [Company | Period | Page N].
6. If a figure is not available, say so plainly. Keep the answer short.
7. Metric names: "profit after tax", "PAT", "net profit", "profit for the period" and "bottom line" are
   net_profit (the total, including non-controlling interests). Use profit_attributable_to_owners
   only when the question says owners, shareholders, parent or equity holders.

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
        self.comparatives = self.store.comparative_periods()

    def _run_tool(self, name: str, args: dict, trace: dict):
        if name == "lookup_metric":
            if args.get("metric") not in METRIC_NAMES:
                return {"error": f"unknown metric {args.get('metric')!r}; use one of: "
                                 + ", ".join(sorted(METRIC_NAMES))}
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
                # a prior period (Q4FY26) lives in the filing that prints it as
                # a comparative column (Q1FY27); chunks carry the filing's period
                f["period"] = self.comparatives.get(f["company"], {}).get(args["period"], args["period"])
            chunks = self.retriever.retrieve(args.get("query", ""), top_k=4, filters=f)
            trace["chunks"].extend(chunks)
            return [{"page": c.get("page_number"), "text": c["text"][:1200]} for c in chunks]
        return {"error": f"unknown tool {name}"}

    def run(self, question: str, check=None, period: str | None = None) -> dict:
        """
        check: optional callable(answer, trace) -> feedback str or None. If it
        returns feedback (e.g. "these figures did not come from a tool"), the
        model gets ONE more turn to fix its answer with the tools.
        period: the period detect_period() read from the question. Passed on
        because the model can get the Indian fiscal calendar wrong ("quarter
        ended March 31, 2026" looked up as Q1FY26 instead of Q4FY26).
        """
        content = question if not period else f"{question}\n\n(Period asked about: {period})"
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": content}]
        trace = {"lookups": [], "calculations": [], "chunks": [], "tool_calls": [], "revised": False}
        answer = ""
        for _ in range(MAX_STEPS):
            resp = self.client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
                temperature=0.1, max_tokens=1024)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                answer = (msg.content or "").strip()
                feedback = None if (check is None or trace["revised"]) else check(answer, trace)
                if not feedback:
                    break
                trace["revised"] = True
                messages.append({"role": "assistant", "content": answer})
                messages.append({"role": "user", "content": feedback})
                continue
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


def agent_answer(agent: "MetricsAgent", verifier, question: str, period: str | None = None) -> dict:
    """Run the agent and verify its answer against the tool outputs.
    Same keys as generation.verifier.generate_and_verify, plus the trace."""
    def check(answer, trace):
        # one self-correction round: numbers no tool produced (typically
        # arithmetic done "in its head") go back to the model to redo with
        # calculate / lookup_metric
        ver = verifier.verify(answer, build_evidence_chunks(trace), query=question)
        missing = [c.split(":")[0] for c in ver["unverified_claims"] if c.endswith("not in sources")]
        if not missing:
            return None
        return ("These figures in your answer did not come from a tool result: "
                + ", ".join(missing) + ". Recompute every one of them with calculate (or look them "
                "up with lookup_metric), then give the final answer again, quoting tool results only.")

    run = agent.run(question, check=check, period=period)
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
        "revised": run["revised"],
        "evidence_text": "\n".join(c["text"] for c in evidence),
    }
