"""Dense bi-encoder retrieval with explicit FastEmbed inference and Qdrant storage."""

import argparse
import hashlib
from importlib.metadata import version
from itertools import batched
import json
import math
from pathlib import Path
import re
import tempfile

from fastembed import TextEmbedding
from fastembed.common.model_description import DenseModelDescription, ModelSource
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

MODEL_NAME = "2.1.1-dense-retrieval"
EXPERIMENT_CAPABILITY = {
    "version": 1,
    "dimensions": {"dense_models": "embedding-model"},
    "parameters": {"hnsw_ef": 128, "exact_search": False},
}
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_DIR = Path(__file__).resolve().parents[2] / "data-preparation" / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def read_records(path, id_field):
    seen = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each record must be a JSON object")
                identifier = row[id_field]
                if (
                    not isinstance(identifier, str)
                    or not identifier
                    or any(character.isspace() for character in identifier)
                ):
                    raise ValueError(f"{id_field} must be a nonempty, whitespace-free string")
                if identifier in seen:
                    raise ValueError(f"duplicate {id_field}: {identifier}")
                if not isinstance(row["text"], str):
                    raise ValueError("text must be a string")
                seen.add(identifier)
                yield identifier, row["text"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(model_info, threads):
    # Resolve once, fingerprint the actual weights/tokenizer, and load that same path.
    description = DenseModelDescription(
        **{**model_info, "sources": ModelSource(**model_info["sources"])})
    cache = str(OUTPUT_DIR / ".fastembed-cache")
    directory = TextEmbedding.download_model(description, cache_dir=cache)
    files = sorted(path for path in directory.rglob("*") if path.is_file()
                   and not any(part.startswith(".") for part in path.relative_to(directory).parts))
    if not files:
        raise ValueError(f"No embedding model files found in {directory}")
    hashes = [(path.relative_to(directory).as_posix(), fingerprint(path)) for path in files]
    artifact_hash = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    model = TextEmbedding(
        model_name=model_info["model"], threads=threads, providers=["CPUExecutionProvider"],
        cache_dir=cache, specific_model_path=str(directory),
    )
    return model, artifact_hash


def index_identity(dataset, corpus_hash, count, model_info, artifact_hash):
    # Split/top-k do not change document vectors; train/test share this index.
    manifest = {
        "method": MODEL_NAME, "dataset": dataset, "corpus_sha256": corpus_hash,
        "document_count": count, "embedding_runtime": "FastEmbed",
        "fastembed_version": version("fastembed"), "embedding_model": model_info["model"],
        "model_artifacts_sha256": artifact_hash, "onnxruntime_version": version("onnxruntime"),
        "vector_dimension": model_info["dim"], "distance": "Cosine",
        "encoding": "passage_embed/query_embed", "provider": "CPUExecutionProvider",
        "model_definition_sha256": hashlib.sha256(
            json.dumps(model_info, sort_keys=True).encode()
        ).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    label = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset)[:40]
    return f"ir_dense_{label}_{digest[:32]}", manifest


def validate_collection(client, collection, manifest):
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    expected = {"dense_index": manifest, "complete": True}
    if (info.config.metadata != expected
            or not isinstance(vectors, models.VectorParams)
            or vectors.size != manifest["vector_dimension"]
            or vectors.distance != models.Distance.COSINE
            or client.count(collection_name=collection, exact=True).count != manifest["document_count"]):
        raise ValueError(
            f"Collection {collection} is incomplete or incompatible with the corpus/model/configuration; "
            "inspect it or explicitly use --rebuild-index."
        )


def checked_vector(vector, dimension):
    values = [float(value) for value in vector]
    if len(values) != dimension or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Embedding must contain {dimension} finite values")
    if not any(values):
        raise ValueError("Embedding is a zero vector; cosine similarity is undefined")
    return values


def ensure_index(client, model, corpus, collection, manifest, batch_size, rebuild):
    exists = client.collection_exists(collection)
    if exists and not rebuild:
        validate_collection(client, collection, manifest)
        print(f"Reusing compatible index: {collection}", flush=True)
        return True
    if exists:
        # Only explicit --rebuild-index authorizes deletion of this exact collection.
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=manifest["vector_dimension"], distance=models.Distance.COSINE,
        ),
        metadata={"dense_index": manifest, "complete": False},
    )
    print(f"Indexing {manifest['document_count']} documents into {collection}", flush=True)
    inserted = 0
    for records in batched(read_records(corpus, "doc_id"), batch_size):
        embeddings = model.passage_embed([text for _, text in records], batch_size=batch_size)
        points = [
            models.PointStruct(
                id=inserted + offset,
                vector=checked_vector(vector, manifest["vector_dimension"]),
                payload={"doc_id": doc_id},
            )
            for offset, ((doc_id, _), vector) in enumerate(zip(records, embeddings, strict=True))
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        inserted += len(points)
    if (inserted != manifest["document_count"] or fingerprint(corpus) != manifest["corpus_sha256"]
            or client.count(collection_name=collection, exact=True).count != inserted):
        raise ValueError("Corpus changed during indexing or uploaded point count is incorrect; rebuild the index")
    client.update_collection(
        collection_name=collection, metadata={"dense_index": manifest, "complete": True},
    )
    validate_collection(client, collection, manifest)
    return False


def retrieve(client, model, text, collection, dimension, top_k, document_count):
    vectors = list(model.query_embed(text))
    if len(vectors) != 1:
        raise ValueError("Expected exactly one query embedding")
    hits = client.query_points(
        collection_name=collection, query=checked_vector(vectors[0], dimension),
        limit=min(top_k, document_count), with_payload=True, with_vectors=False,
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
    ).points
    results = []
    for hit in hits:
        doc_id = (hit.payload or {}).get("doc_id")
        if (not isinstance(doc_id, str) or not doc_id or any(c.isspace() for c in doc_id)
                or not math.isfinite(hit.score)):
            raise ValueError("Qdrant returned an invalid document ID/payload or score")
        results.append((doc_id, hit.score))
    if len(results) != min(top_k, document_count) or len({d for d, _ in results}) != len(results):
        raise ValueError("Qdrant returned empty, incomplete, or duplicate results unexpectedly")
    return sorted(results, key=lambda item: (item[1], item[0]), reverse=True)


def boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value.lower() == "true"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="replace ranking/metadata, not the index")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--rebuild-index", nargs="?", const=True, default=False, type=boolean,
                        help="rebuild this collection; accepts =true through --method-option")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    client = None
    try:
        if Path(args.dataset).name != args.dataset or args.dataset in (".", ".."):
            raise ValueError("--dataset must be a directory name, not a path")
        if min(args.top_k, args.batch_size, args.threads) < 1:
            raise ValueError("--top-k, --batch-size and --threads must be positive")
        corpus = DATA_DIR / args.dataset / "corpus.jsonl"
        queries = DATA_DIR / args.dataset / "queries" / f"{args.split}.jsonl"
        for path in (corpus, queries):
            if not path.is_file():
                raise ValueError(f"Prepared input does not exist: {path}")
        output = args.output_dir or OUTPUT_DIR / args.dataset / args.split
        ranking = output / f"{MODEL_NAME}.trec"
        sidecar = ranking.with_suffix(".metadata.json")
        if (ranking.exists() or sidecar.exists()) and not args.overwrite:
            raise ValueError(f"run exists: {ranking}; use --overwrite to replace it")
        supported = {info["model"]: info for info in TextEmbedding.list_supported_models()}
        if args.embedding_model not in supported:
            raise ValueError(f"Unsupported FastEmbed dense model: {args.embedding_model}; "
                             "see TextEmbedding.list_supported_models()")
        # Validate before downloading weights or changing Qdrant. No embeddings held in memory.
        corpus_hash = fingerprint(corpus)
        count = sum(1 for _ in read_records(corpus, "doc_id"))
        query_count = sum(1 for _ in read_records(queries, "query_id"))
        if not count or not query_count:
            raise ValueError("Corpus and query files must contain at least one record")
        client = QdrantClient(url=args.qdrant_url, timeout=120)
        try:
            server_version = client.info().version
        except Exception as error:
            raise ValueError(f"Cannot reach Qdrant at {args.qdrant_url}; start local Docker Qdrant: {error}") from error
        print(f"Loading FastEmbed model: {args.embedding_model}", flush=True)
        model, artifact_hash = load_model(supported[args.embedding_model], args.threads)
        collection, manifest = index_identity(
            args.dataset, corpus_hash, count, supported[args.embedding_model], artifact_hash,
        )
        reused = ensure_index(client, model, corpus, collection, manifest, args.batch_size, args.rebuild_index)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".dense-", dir=output) as temporary:
            staged = Path(temporary) / ranking.name
            results_count = 0
            with staged.open("w", encoding="utf-8") as stream:
                for query_id, text in read_records(queries, "query_id"):
                    results = retrieve(client, model, text, collection, manifest["vector_dimension"], args.top_k, count)
                    for rank, (doc_id, score) in enumerate(results, 1):
                        stream.write(f"{query_id} Q0 {doc_id} {rank} {score:.17g} {MODEL_NAME}\n")
                    results_count += len(results)
            parameters = {
                **manifest, "method": "dense-retrieval", "version": "2.1.1",
                "top_k": args.top_k, "qdrant_collection": collection,
                "qdrant_client_version": version("qdrant-client"), "qdrant_server_version": server_version,
                "batch_size": args.batch_size, "threads": args.threads,
                "hnsw_ef": 128, "exact_search": False, "index_reused": reused,
            }
            parameters.pop("dataset")
            metadata = {"dataset": args.dataset, "split": args.split, "parameters": parameters}
            staged_metadata = Path(temporary) / sidecar.name
            staged_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            staged.replace(ranking)
            staged_metadata.replace(sidecar)
        print(f"{query_count} queries; {results_count} results. Run saved to {ranking.resolve()}")
    except (OSError, ValueError, RuntimeError, ResponseHandlingException, UnexpectedResponse) as error:
        parser.exit(2, f"error: {error}\n")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
