from dataset_loader import load_hotpot_dataset, convert_to_documents
from embedding_model import EmbeddingModel
from qdrant_db import QdrantDB

from tqdm import tqdm


dataset = load_hotpot_dataset()

docs = convert_to_documents(dataset)

embedder = EmbeddingModel()
db = QdrantDB()

db.create_collection()

points = []

for i, doc in enumerate(tqdm(docs)):

    vec = embedder.embed(doc["text"])

    points.append({
        "id": i,
        "vector": vec,
        "payload": doc
    })

db.insert_points(points)

print("Inserted", len(points), "documents")