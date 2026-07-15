from indexing.vector_store import VectorStore

vs = VectorStore()
pts = vs.client.scroll(collection_name='filing_chunks', limit=3, with_payload=True)
for p in pts[0]:
    print(p.payload.get('company'), '|', p.payload.get('period'), '|', p.payload.get('doc_type'))