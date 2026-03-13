from qdrant_client import QdrantClient
from embedder import embed
from config import COLLECTION_NAME, TOP_K
import os


# Kết nối Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


def retrieve(question):

    # embedding câu hỏi
    vector = embed(question)

    # search vector
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