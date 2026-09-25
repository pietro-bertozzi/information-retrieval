"""Explicit text -> embedding -> Qdrant -> similarity search learning demo."""

from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

URL = "http://localhost:6333"
COLLECTION = "qdrant_learning_documents"
MODEL = "BAAI/bge-small-en-v1.5"
DOCUMENTS = [
    (1, "artificial intelligence", "Neural networks learn patterns from training data to make predictions."),
    (2, "automobiles", "An automobile uses an engine and four wheels to transport passengers along roads."),
    (3, "cooking", "To bake bread, mix flour with yeast and water, knead the dough, and put it in an oven."),
    (4, "history", "The Roman Empire built aqueducts and governed territories around the Mediterranean Sea."),
    (5, "astronomy", "Astronomers study distant stars and galaxies using powerful telescopes."),
]
QUERIES = [
    "A car carries people on highways.",
    "How can computers discover patterns and predict outcomes?",
    "What ingredients do I need for homemade bread?",
]


def main():
    # Qdrant runs in Docker. Embedding inference runs here, in Python, on the CPU.
    client = QdrantClient(url=URL, timeout=30)
    try:
        print(f"1. Connect to Qdrant: {URL}", flush=True)
        client.get_collections()

        print(f"\n2. Load local embedding model: {MODEL}", flush=True)
        print("First use downloads model files; later runs reuse model-cache/.", flush=True)
        model = TextEmbedding(
            model_name=MODEL,
            # Preserve the existing cache location; moving source must not redownload weights.
            cache_dir=str(Path(__file__).resolve().parents[2] / "qdrant-experiment/model-cache"),
            threads=2,
        )
        texts = [text for _, _, text in DOCUMENTS]
        vectors = list(model.passage_embed(texts))
        dimension = len(vectors[0])
        if len(vectors) != len(DOCUMENTS) or any(len(v) != dimension for v in vectors):
            raise RuntimeError("Document embeddings have inconsistent shapes")
        print(f"\n3. Embedded {len(texts)} texts: {dimension} numbers per vector.")
        print(f"First vector preview: {vectors[0][:6].tolist()}")

        # Only this dedicated demo collection is recreated on each run.
        print(f"\n4. Recreate collection: {COLLECTION}; distance=Cosine")
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

        print("\n5. Upload points: ID + vector + payload")
        points = []
        for (point_id, topic, text), vector in zip(DOCUMENTS, vectors, strict=True):
            points.append(models.PointStruct(
                id=point_id, vector=vector.tolist(), payload={"topic": topic, "text": text},
            ))
            print(f"  ID={point_id} | topic={topic} | vector length={len(vector)}\n    {text}")
        client.upsert(collection_name=COLLECTION, points=points, wait=True)

        # Read back from the server, not from the Python list just uploaded.
        count = client.count(collection_name=COLLECTION, exact=True).count
        stored = client.retrieve(
            collection_name=COLLECTION, ids=[1], with_payload=True, with_vectors=True,
        )
        if count != len(DOCUMENTS) or not stored or len(stored[0].vector) != dimension:
            raise RuntimeError("Stored point count or vector dimension did not match the upload")
        print(f"\n6. Server confirms {count} stored points.")
        print(f"Read-back ID={stored[0].id}, vector length={len(stored[0].vector)}")
        print(f"Read-back payload: {stored[0].payload}")

        print("\n7. Embed queries with the same model, then search Qdrant")
        for query in QUERIES:
            query_vector = next(model.query_embed(query))
            if len(query_vector) != dimension:
                raise RuntimeError("Query vector does not match collection dimension")
            hits = client.query_points(
                collection_name=COLLECTION,
                query=query_vector.tolist(),
                limit=3,
                with_payload=True,
                with_vectors=False,
            ).points
            print(f"\nQuery: {query}")
            print(f"Query vector: {len(query_vector)} numbers; higher cosine score is closer.")
            for rank, hit in enumerate(hits, 1):
                print(f"  {rank}. score={hit.score:.6f} | ID={hit.id} | {hit.payload['text']}")
        print(f"\nInspect collection '{COLLECTION}' at {URL}/dashboard")
        print("Query embeddings were searched, not stored as new points.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
