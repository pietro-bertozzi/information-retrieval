"""Hybrid dense+sparse retrieval with FastEmbed, Qdrant, and native RRF."""

import argparse
import hashlib
from importlib.metadata import version
from itertools import batched
import json
from pathlib import Path
import re
import tempfile

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.model_description import (
    DenseModelDescription,
    SparseModelDescription,
)
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from _vector_common import (
    boolean,
    checked_dense,
    checked_results,
    checked_sparse,
    fingerprint,
    load_embedding_model,
    read_records,
)

MODEL_NAME = "2.3.1-hybrid-retrieval"
EXPERIMENT_CAPABILITY = {
    "version": 1,
    "dimensions": {
        "dense_models": "dense-embedding-model",
        "sparse_models": "sparse-embedding-model",
    },
    "parameters": {
        "dense_hnsw_ef": 128,
        "dense_exact_search": False,
        "fusion_strategy": "qdrant-native-rrf",
        "fusion_weights": "equal",
        "rrf_k": 60,
    },
}
DEFAULT_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
DATA_DIR = Path(__file__).resolve().parents[2] / "data-preparation" / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"


def index_identity(
    dataset,
    corpus_hash,
    count,
    dense_info,
    dense_artifact_hash,
    sparse_info,
    sparse_artifact_hash,
):
    modifier = "idf" if sparse_info.get("requires_idf") else "none"
    manifest = {
        "method": MODEL_NAME,
        "dataset": dataset,
        "corpus_sha256": corpus_hash,
        "document_count": count,
        "embedding_runtime": "FastEmbed",
        "fastembed_version": version("fastembed"),
        "onnxruntime_version": version("onnxruntime"),
        "dense_embedding_model": dense_info["model"],
        "dense_model_artifacts_sha256": dense_artifact_hash,
        "dense_model_definition_sha256": hashlib.sha256(
            json.dumps(dense_info, sort_keys=True).encode()
        ).hexdigest(),
        "dense_vector_dimension": dense_info["dim"],
        "dense_distance": "Cosine",
        "dense_vector_name": DENSE_VECTOR,
        "sparse_embedding_model": sparse_info["model"],
        "sparse_model_artifacts_sha256": sparse_artifact_hash,
        "sparse_model_definition_sha256": hashlib.sha256(
            json.dumps(sparse_info, sort_keys=True).encode()
        ).hexdigest(),
        "sparse_vocabulary_size": sparse_info.get("vocab_size") or 0,
        "sparse_modifier": modifier,
        "sparse_vector_name": SPARSE_VECTOR,
        "encoding": "passage_embed/query_embed",
        "provider": "CPUExecutionProvider",
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    label = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset)[:40]
    return f"ir_hybrid_{label}_{digest[:32]}", manifest


def expected_sparse_modifier(manifest):
    return (
        models.Modifier.IDF
        if manifest["sparse_modifier"] == "idf"
        else models.Modifier.NONE
    )


def validate_collection(client, collection, manifest):
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors
    dense = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
    sparse = sparse_vectors.get(SPARSE_VECTOR) if isinstance(sparse_vectors, dict) else None
    if (
        info.config.metadata != {"hybrid_index": manifest, "complete": True}
        or set(vectors or {}) != {DENSE_VECTOR}
        or not isinstance(dense, models.VectorParams)
        or dense.size != manifest["dense_vector_dimension"]
        or dense.distance != models.Distance.COSINE
        or set(sparse_vectors or {}) != {SPARSE_VECTOR}
        or not isinstance(sparse, models.SparseVectorParams)
        or sparse.modifier != expected_sparse_modifier(manifest)
        or client.count(collection_name=collection, exact=True).count
        != manifest["document_count"]
    ):
        raise ValueError(
            f"Collection {collection} is incomplete or incompatible with the "
            "corpus/models/configuration; inspect it or explicitly use --rebuild-index."
        )


def ensure_index(
    client,
    dense_model,
    sparse_model,
    corpus,
    collection,
    manifest,
    batch_size,
    rebuild,
):
    exists = client.collection_exists(collection)
    if exists and not rebuild:
        validate_collection(client, collection, manifest)
        print(f"Reusing compatible hybrid index: {collection}", flush=True)
        return True
    if exists:
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=manifest["dense_vector_dimension"],
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
                modifier=expected_sparse_modifier(manifest),
            )
        },
        metadata={"hybrid_index": manifest, "complete": False},
    )
    print(f"Indexing {manifest['document_count']} hybrid documents into {collection}", flush=True)
    inserted = 0
    for records in batched(read_records(corpus, "doc_id"), batch_size):
        texts = [text for _, text in records]
        dense_embeddings = dense_model.passage_embed(texts, batch_size=batch_size)
        sparse_embeddings = sparse_model.passage_embed(texts, batch_size=batch_size)
        points = []
        for offset, ((doc_id, _), dense_embedding, sparse_embedding) in enumerate(
            zip(records, dense_embeddings, sparse_embeddings, strict=True)
        ):
            vectors = {
                DENSE_VECTOR: checked_dense(
                    dense_embedding, manifest["dense_vector_dimension"]
                )
            }
            sparse = checked_sparse(sparse_embedding)
            if sparse is not None:
                vectors[SPARSE_VECTOR] = sparse
            points.append(
                models.PointStruct(
                    id=inserted + offset,
                    vector=vectors,
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
        metadata={"hybrid_index": manifest, "complete": True},
    )
    validate_collection(client, collection, manifest)
    return False


def retrieve(
    client,
    dense_model,
    sparse_model,
    text,
    collection,
    dimension,
    top_k,
    document_count,
    rrf_k,
):
    dense_embeddings = list(dense_model.query_embed(text))
    sparse_embeddings = list(sparse_model.query_embed(text))
    if len(dense_embeddings) != 1 or len(sparse_embeddings) != 1:
        raise ValueError("Expected exactly one dense and one sparse query embedding")
    limit = min(top_k, document_count)
    prefetch = [
        models.Prefetch(
            query=checked_dense(dense_embeddings[0], dimension),
            using=DENSE_VECTOR,
            params=models.SearchParams(hnsw_ef=128, exact=False),
            limit=limit,
        )
    ]
    sparse = checked_sparse(sparse_embeddings[0])
    if sparse is not None:
        prefetch.append(
            models.Prefetch(query=sparse, using=SPARSE_VECTOR, limit=limit)
        )
    hits = client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    ).points
    return checked_results(hits, limit, require_limit=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace ranking/metadata, not the index"
    )
    parser.add_argument("--dense-embedding-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--sparse-embedding-model", default=DEFAULT_SPARSE_MODEL)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument(
        "--rebuild-index", nargs="?", const=True, default=False, type=boolean,
        help="rebuild this collection; accepts =true through --method-option",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args(argv)
    client = None
    try:
        if Path(args.dataset).name != args.dataset or args.dataset in (".", ".."):
            raise ValueError("--dataset must be a directory name, not a path")
        if min(args.top_k, args.batch_size, args.threads, args.rrf_k) < 1:
            raise ValueError("--top-k, --batch-size, --threads and --rrf-k must be positive")
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
        dense_supported = {info["model"]: info for info in TextEmbedding.list_supported_models()}
        sparse_supported = {
            info["model"]: info for info in SparseTextEmbedding.list_supported_models()
        }
        if args.dense_embedding_model not in dense_supported:
            raise ValueError(
                f"Unsupported FastEmbed dense model: {args.dense_embedding_model}; "
                "see TextEmbedding.list_supported_models()"
            )
        if args.sparse_embedding_model not in sparse_supported:
            raise ValueError(
                f"Unsupported FastEmbed sparse model: {args.sparse_embedding_model}; "
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
        print(f"Loading FastEmbed dense model: {args.dense_embedding_model}", flush=True)
        dense_model, dense_artifact_hash = load_embedding_model(
            TextEmbedding,
            DenseModelDescription,
            dense_supported[args.dense_embedding_model],
            OUTPUT_DIR / ".fastembed-cache",
            args.threads,
        )
        print(f"Loading FastEmbed sparse model: {args.sparse_embedding_model}", flush=True)
        sparse_model, sparse_artifact_hash = load_embedding_model(
            SparseTextEmbedding,
            SparseModelDescription,
            sparse_supported[args.sparse_embedding_model],
            OUTPUT_DIR / ".fastembed-cache",
            args.threads,
        )
        collection, manifest = index_identity(
            args.dataset,
            corpus_hash,
            count,
            dense_supported[args.dense_embedding_model],
            dense_artifact_hash,
            sparse_supported[args.sparse_embedding_model],
            sparse_artifact_hash,
        )
        reused = ensure_index(
            client,
            dense_model,
            sparse_model,
            corpus,
            collection,
            manifest,
            args.batch_size,
            args.rebuild_index,
        )
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".hybrid-", dir=output) as temporary:
            staged = Path(temporary) / ranking.name
            result_count = 0
            with staged.open("w", encoding="utf-8") as stream:
                for query_id, text in read_records(queries, "query_id"):
                    results = retrieve(
                        client,
                        dense_model,
                        sparse_model,
                        text,
                        collection,
                        manifest["dense_vector_dimension"],
                        args.top_k,
                        count,
                        args.rrf_k,
                    )
                    for rank, (doc_id, score) in enumerate(results, 1):
                        stream.write(
                            f"{query_id} Q0 {doc_id} {rank} {score:.17g} {MODEL_NAME}\n"
                        )
                    result_count += len(results)
            parameters = {
                **manifest,
                "method": "hybrid-retrieval",
                "version": "2.3.1",
                "top_k": args.top_k,
                "qdrant_collection": collection,
                "qdrant_client_version": version("qdrant-client"),
                "qdrant_server_version": server_version,
                "batch_size": args.batch_size,
                "threads": args.threads,
                "dense_hnsw_ef": 128,
                "dense_exact_search": False,
                "sparse_index_on_disk": False,
                "fusion_strategy": "qdrant-native-rrf",
                "fusion_weights": "equal",
                "rrf_k": args.rrf_k,
                "prefetch_limit": args.top_k,
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
