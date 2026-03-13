import os
from qdrant_client import QdrantClient

from src.embedder import embed
from src.config import COLLECTION_NAME, TOP_K


client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


def retrieve(question):

    vector = embed(question)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=TOP_K
    )

    docs = []

    for r in results.points:

        docs.append({
            "text": r.payload["text"]
        })

    return docs