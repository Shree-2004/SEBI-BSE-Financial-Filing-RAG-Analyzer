import json

with open('data/processed/chunked.jsonl', encoding='utf-8') as f:
    chunks = [json.loads(l) for l in f if l.strip()]

hits = [c for c in chunks if 'profit after tax' in c['text'].lower() or 'net profit' in c['text'].lower()]
print(f"Found {len(hits)} chunks mentioning net profit\n")
for h in hits[:5]:
    print(f"Page {h['page_number']} | {h['block_type']}")
    print(h['text'][:300])
    print('---')