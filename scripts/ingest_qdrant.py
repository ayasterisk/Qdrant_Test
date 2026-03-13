import os
from datasets import load_from_disk

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from src.embedder import embed
from src.config import COLLECTION_NAME


dataset = load_from_disk("hotpot_mini_1k")


client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)


points = []

for idx, item in enumerate(dataset):

    question = item["question"]

    context = str(item["context"])

    text = question + " " + context

    vector = embed(text)

    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={"text": text}
        )
    )


client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("Inserted:", len(points))