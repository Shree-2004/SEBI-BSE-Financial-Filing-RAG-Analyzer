from dotenv import load_dotenv
load_dotenv()
import os
from indexing.embed import default_collection
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

for field in ["company", "period", "doc_type", "block_type"]:
    client.create_payload_index(
        collection_name=os.getenv("QDRANT_COLLECTION") or default_collection(),
        field_name=field,
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(f"✓ Index created: {field}")

print("All indexes created!")