"""Sparse neural retrieval with FastEmbed BM42 and Qdrant sparse vectors."""

import argparse
import hashlib
from importlib.metadata import version
from itertools import batched
import json
from pathlib import Path
import re
import tempfile

from fastembed import SparseTextEmbedding
from fastembed.common.model_description import SparseModelDescription
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from _vector_common import (
    boolean,
    checked_results,
    checked_sparse,
    fingerprint,
    load_embedding_model,
    read_records,
)

MODEL_NAME = "2.2.1-sparse-retrieval"
EXPERIMENT_CAPABILITY = {
    "version": 1,
    "dimensions": {"sparse_models": "embedding-model"},
    "parameter_names": {"embedding-model": "sparse_embedding_model"},
}
DEFAULT_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
VECTOR_NAME = "sparse"
DATA_DIR = Path(__file__).resolve().parents[2] / "data-preparation" / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def index_identity(dataset, corpus_hash, count, model_info, artifact_hash):
    modifier = "idf" if model_info.get("requires_idf") else "none"
    manifest = {
        "method": MODEL_NAME,
        "dataset": dataset,
        "corpus_sha256": corpus_hash,
        "document_count": count,
        "embedding_runtime": "FastEmbed",
        "fastembed_version": version("fastembed"),
        "sparse_embedding_model": model_info["model"],
        "sparse_model_artifacts_sha256": artifact_hash,
        "onnxruntime_version": version("onnxruntime"),
        "vector_type": "sparse",
        "vector_name": VECTOR_NAME,
        "vocabulary_size": model_info.get("vocab_size") or 0,
        "sparse_modifier": modifier,
        "encoding": "passage_embed/query_embed",
        "provider": "CPUExecutionProvider",
        "sparse_model_definition_sha256": hashlib.sha256(
            json.dumps(model_info, sort_keys=True).encode()
        ).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    label = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset)[:40]
    return f"ir_sparse_{label}_{digest[:32]}", manifest


def sparse_params(manifest):
    modifier = (
        models.Modifier.IDF
        if manifest["sparse_modifier"] == "idf"
        else models.Modifier.NONE
    )
    return models.SparseVectorParams(
        index=models.SparseIndexParams(on_disk=False),
        modifier=modifier,
    )


def validate_collection(client, collection, manifest):
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors
    configured = sparse_vectors.get(VECTOR_NAME) if isinstance(sparse_vectors, dict) else None
    expected_modifier = (
        models.Modifier.IDF
        if manifest["sparse_modifier"] == "idf"
        else models.Modifier.NONE
    )
    if (
        info.config.metadata != {"sparse_index": manifest, "complete": True}
        or vectors not in ({}, None)
        or set(sparse_vectors or {}) != {VECTOR_NAME}
        or not isinstance(configured, models.SparseVectorParams)
        or configured.modifier != expected_modifier
        or client.count(collection_name=collection, exact=True).count
        != manifest["document_count"]
    ):
        raise ValueError(
            f"Collection {collection} is incomplete or incompatible with the "
            "corpus/model/configuration; inspect it or explicitly use --rebuild-index."
        )


def ensure_index(client, model, corpus, collection, manifest, batch_size, rebuild):
    exists = client.collection_exists(collection)
    if exists and not rebuild:
        validate_collection(client, collection, manifest)
        print(f"Reusing compatible sparse index: {collection}", flush=True)
        return True
    if exists:
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={},
        sparse_vectors_config={VECTOR_NAME: sparse_params(manifest)},
        metadata={"sparse_index": manifest, "complete": False},
    )
    print(f"Indexing {manifest['document_count']} sparse documents into {collection}", flush=True)
    inserted = 0
    for records in batched(read_records(corpus, "doc_id"), batch_size):
        embeddings = model.passage_embed(
            [text for _, text in records], batch_size=batch_size
        )
        points = []
        for offset, ((doc_id, _), embedding) in enumerate(
            zip(records, embeddings, strict=True)
        ):
            sparse = checked_sparse(embedding)
            vector = {VECTOR_NAME: sparse} if sparse is not None else {}
            points.append(
                models.PointStruct(
                    id=inserted + offset,
                    vector=vector,
                    payload={"doc_id": doc_id},
                )
            )
        client.upsert(collection_name=collection, points=points, wait=True)
        inserted += len(points)
    if (
        inserted != manifest["document_count"]
        or fingerprint(corpus) != manifest["corpus_sha256"]
        or client.count(collection_name=collection, exact=True).count != inserted
    ):
        raise ValueError(
            "Corpus changed during indexing or uploaded point count is incorrect; "
            "rebuild the index"
        )
    client.update_collection(
        collection_name=collection,
        metadata={"sparse_index": manifest, "complete": True},
    )
    validate_collection(client, collection, manifest)
    return False


def retrieve(client, model, text, collection, top_k, document_count):
    embeddings = list(model.query_embed(text))
    if len(embeddings) != 1:
        raise ValueError("Expected exactly one sparse query embedding")
    sparse = checked_sparse(embeddings[0])
    if sparse is None:
        return []
    limit = min(top_k, document_count)
    hits = client.query_points(
        collection_name=collection,
        query=sparse,
        using=VECTOR_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    ).points
    return checked_results(hits, limit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace ranking/metadata, not the index"
    )
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument(
        "--rebuild-index", nargs="?", const=True, default=False, type=boolean,
        help="rebuild this collection; accepts =true through --method-option",
    )
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
        supported = {
            info["model"]: info for info in SparseTextEmbedding.list_supported_models()
        }
        if args.embedding_model not in supported:
            raise ValueError(
                f"Unsupported FastEmbed sparse model: {args.embedding_model}; "
                "see SparseTextEmbedding.list_supported_models()"
            )
        corpus_hash = fingerprint(corpus)
        count = sum(1 for _ in read_records(corpus, "doc_id"))
        query_count = sum(1 for _ in read_records(queries, "query_id"))
        if not count or not query_count:
            raise ValueError("Corpus and query files must contain at least one record")
        client = QdrantClient(url=args.qdrant_url, timeout=120)
        try:
            server_version = client.info().version
        except Exception as error:
            raise ValueError(
                f"Cannot reach Qdrant at {args.qdrant_url}; start local Docker Qdrant: {error}"
            ) from error
        print(f"Loading FastEmbed sparse model: {args.embedding_model}", flush=True)
        model, artifact_hash = load_embedding_model(
            SparseTextEmbedding,
            SparseModelDescription,
            supported[args.embedding_model],
            OUTPUT_DIR / ".fastembed-cache",
            args.threads,
        )
        collection, manifest = index_identity(
            args.dataset,
            corpus_hash,
            count,
            supported[args.embedding_model],
            artifact_hash,
        )
        reused = ensure_index(
            client, model, corpus, collection, manifest, args.batch_size,
            args.rebuild_index,
        )
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".sparse-", dir=output) as temporary:
            staged = Path(temporary) / ranking.name
            result_count = 0
            with staged.open("w", encoding="utf-8") as stream:
                for query_id, text in read_records(queries, "query_id"):
                    results = retrieve(
                        client, model, text, collection, args.top_k, count
                    )
                    for rank, (doc_id, score) in enumerate(results, 1):
                        stream.write(
                            f"{query_id} Q0 {doc_id} {rank} {score:.17g} {MODEL_NAME}\n"
                        )
                    result_count += len(results)
            parameters = {
                **manifest,
                "method": "sparse-retrieval",
                "version": "2.2.1",
                "top_k": args.top_k,
                "qdrant_collection": collection,
                "qdrant_client_version": version("qdrant-client"),
                "qdrant_server_version": server_version,
                "batch_size": args.batch_size,
                "threads": args.threads,
                "sparse_index_on_disk": False,
                "index_reused": reused,
            }
            parameters.pop("dataset")
            metadata = {
                "dataset": args.dataset,
                "split": args.split,
                "parameters": parameters,
            }
            staged_metadata = Path(temporary) / sidecar.name
            staged_metadata.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            staged.replace(ranking)
            staged_metadata.replace(sidecar)
        print(
            f"{query_count} queries; {result_count} results. "
            f"Run saved to {ranking.resolve()}"
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        ResponseHandlingException,
        UnexpectedResponse,
    ) as error:
        parser.exit(2, f"error: {error}\n")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
