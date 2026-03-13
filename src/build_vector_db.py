from datasets import load_from_disk
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from embedder import embed
from config import COLLECTION_NAME

dataset = load_from_disk("hotpot_mini_1k")

client = QdrantClient("localhost", port=6333)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

points = []
id_counter = 0

for item in dataset:

    titles = item["context"]["title"]
    sentences = item["context"]["sentences"]

    for i in range(len(titles)):

        text = " ".join(sentences[i])

        vector = embed(text)

        points.append(
            PointStruct(
                id=id_counter,
                vector=vector,
                payload={
                    "text": text,
                    "title": titles[i]
                }
            )
        )

        id_counter += 1

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("Inserted:", len(points))