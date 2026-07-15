"""
test_pipeline.py — full end-to-end test
Run: set PYTHONPATH=src && python test_pipeline.py
"""
from dotenv import load_dotenv
load_dotenv()

from retrieval.hybrid_retriever import HybridRetriever
from generation.answer_generator import AnswerGenerator
from generation.verifier import Verifier, generate_and_verify

retriever = HybridRetriever()
generator = AnswerGenerator()
verifier = Verifier()

query = "What was TCS revenue and net profit in Q3FY25?"
filters = {"company": "TCS"}

print(f"\nQuery: {query}")
print(f"Filters: {filters}\n")

chunks = retriever.retrieve(query=query, top_k=5, filters=filters)
result = generate_and_verify(query=query, chunks=chunks, generator=generator, verifier=verifier)

print(f"ANSWER:\n{result['answer']}")
print(f"\nGROUNDING: {'✓ Grounded' if result['is_grounded'] else '⚠ Ungrounded'} (confidence: {result['grounding_confidence']})")
print(f"\nCITATIONS ({len(result['citations'])}):")
for c in result["citations"]:
    print(f"  - {c['company']} | {c['period']} | Page {c['page_number']} | {c['source_file']}")

if result["warning"]:
    print(f"\nWARNING: {result['warning']}")