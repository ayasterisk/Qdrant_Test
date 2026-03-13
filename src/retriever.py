from qdrant_client import QdrantClient
from embedder import embed
from config import COLLECTION_NAME, TOP_K

client = QdrantClient("localhost", port=6333)


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
            "text": r.payload["text"],
            "title": r.payload["title"]
        })

    return docs