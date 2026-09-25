# Qdrant learning demo

A standalone demonstration with five short documents and three queries. It does
not use the repository's datasets, retrieval pipeline, evaluator, or MLflow.

## What happens

```text
documents                           query text
    |                                   |
    v                                   v
local embedding model              same local model
    |                                   |
    v                                   v
document vectors                    query vector
    |                                   |
    v                                   v
Qdrant collection <---------- cosine similarity search
    |                                   |
    v                                   v
stored points                      ranked documents
(ID + vector + payload)            (score + ID + payload text)
```

**The embedding model is not Qdrant.** FastEmbed runs
`BAAI/bge-small-en-v1.5` locally on the CPU using ONNX Runtime. It converts text
into 384-dimensional dense vectors. The first execution downloads model files;
subsequent executions reuse `model-cache/`. No API key, paid API, GPU, or hosted
inference is needed. Downloading the model requires internet access.

**Qdrant stores vectors and finds nearby vectors.** The Docker service receives
numeric vectors, not a request to embed text. This demo uses explicit
`passage_embed`, `create_collection`, `upsert`, `query_embed`, and `query_points`
steps rather than an automatic embedding integration.

| Term | Meaning in this demo |
| --- | --- |
| Collection | `qdrant_learning_documents`, configured for 384-dimensional vectors and cosine similarity. |
| Point | A stored record with an integer ID, a vector, and an optional payload. |
| Vector | The model's 384 floating-point numbers representing a document. |
| Payload | JSON metadata: the original `text` and its `topic`. It is not the embedding. |
| Query vector | A compatible embedding used to search; it is not inserted as another point. |

Documents and queries use the same model and its passage/query embedding methods.
Matching dimension alone is insufficient: vectors from unrelated models occupy
different semantic spaces and should not be compared.

Cosine similarity compares vector direction. Qdrant normalizes cosine vectors
when inserting them and returns higher scores for closer directions. Scores are
similarities, not probabilities. Payload text is returned for display; it is not
used for lexical scoring or filtering here. With only five points, Qdrant may
scan the vectors directly rather than build/use an approximate HNSW index.

Physically, Qdrant persists IDs, vectors, payloads, and internal collection/storage
metadata in the Docker volume `ir-qdrant-learning-data`, mounted at
`/qdrant/storage`. On Windows this volume lives in Docker Desktop's managed Linux
storage. The Python documents list is only the input; the script reads a point
back from Qdrant to demonstrate server-side storage. Model weights remain in the
Python-side cache, separate from Qdrant.

## Setup and run

Run these commands from the **repository root** in PowerShell or Git Bash on
Windows. Start Docker Desktop with Linux containers enabled.

The demo source and requirements live here. The existing local virtual environment
and model cache remain in their legacy location to preserve downloaded files.
The project environment already has the required dependencies:

```text
./.venv/Scripts/python.exe -m pip install -r theory/qdrant-demo/requirements.txt
```

Validate the Compose file and start Qdrant:

```text
docker compose -f infrastructure/qdrant/compose.yaml config --quiet
docker compose -f infrastructure/qdrant/compose.yaml up -d
docker compose -f infrastructure/qdrant/compose.yaml ps
```

Run the demo once the container is running:

```text
./.venv/Scripts/python.exe -B theory/qdrant-demo/demo.py
```

The pinned Qdrant server uses localhost ports **6333** (HTTP/API/dashboard) and
**6334** (gRPC). The Python demo uses HTTP. If startup reports a port conflict,
stop the other service using those ports first. If Python reports a connection
error, check Docker Desktop and the container logs:

```text
docker compose -f infrastructure/qdrant/compose.yaml logs qdrant
```

Every demo run deletes and recreates **only** `qdrant_learning_documents` to start
with the same five points. Treat that collection as disposable learning data.
Other collections are not modified.

## What to look for

The numbered output shows:

1. A successful connection to local Qdrant.
2. The local embedding model being loaded.
3. Five embeddings, their dimension, and the first six values of one vector.
4. Collection creation with the measured dimension and cosine distance.
5. Each inserted point's ID, topic, vector length, and original text.
6. An exact server-side count of five points and a point read back with its vector
   and payload.
7. Three queries, each followed by its top three results: rank, score, point ID,
   and original text.

The first query, `A car carries people on highways.`, should retrieve point 2:
`An automobile uses an engine and four wheels to transport passengers along roads.`
The wording uses synonyms (car/automobile, people/passengers, highways/roads),
rather than identical words. Other queries target machine learning and cooking.
Lower-ranked results are simply the next nearest vectors; top-three search does
not guarantee three relevant answers.

## Observed results

Verified on Windows, Python 3.14.7, with the pinned dependencies and Qdrant image
on 2026-09-23. The server reported five points with 384-dimensional cosine vectors.
For `A car carries people on highways.`, the actual top three were:

| Rank | Point ID | Topic | Cosine score |
| --- | --- | --- | --- |
| 1 | 2 | Automobiles | 0.829058 |
| 2 | 5 | Astronomy | 0.473600 |
| 3 | 1 | Artificial intelligence | 0.444265 |

`How can computers discover patterns and predict outcomes?` returned point 1
first (0.822142). `What ingredients do I need for homemade bread?` returned point 3
first (0.794510). The script prints the complete original text for every hit.
Small numeric differences between machines are possible.

## Inspect the stored data

Open <http://localhost:6333/dashboard> and choose `qdrant_learning_documents`.
Inspect point IDs, payloads, and vectors. The collection configuration shows its
vector size and distance metric. The REST endpoints are also directly readable:

- <http://localhost:6333/collections/qdrant_learning_documents>
- <http://localhost:6333/collections/qdrant_learning_documents/points/2>

The original text appears because we explicitly stored it as payload. Qdrant does
not reconstruct the document text from its vector.

## Shared storage

Stop/remove the experiment container and network, keeping stored points:

```text
docker compose -f infrastructure/qdrant/compose.yaml down
```

Running `up -d` again reuses the same volume. This volume now also stores the
project's vector indexes; do not reset it for the demo. See
[shared Qdrant infrastructure](../../infrastructure/qdrant/README.md).

The demo continues to use `qdrant-experiment/model-cache/` at the repository root.
That directory and the legacy environment are preserved and ignored by Git.

## References

- [Qdrant local quickstart](https://qdrant.tech/documentation/quickstart/)
- [FastEmbed quickstart](https://qdrant.tech/documentation/fastembed/fastembed-quickstart/)
- [Qdrant vector and distance concepts](https://qdrant.tech/documentation/concepts/vectors/)
