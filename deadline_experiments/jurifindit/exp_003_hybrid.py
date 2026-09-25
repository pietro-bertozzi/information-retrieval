"""Independent top-10 BM25/dense hybrid experiment using reciprocal rank fusion."""

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import islice
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "scripts"))
import evaluate

DATASET = "jurifindit"
SOURCE_SPLIT = "validation"
METRICS = ["MAP", "Precision@1", "Precision@5", "Precision@10",
           "Recall@1", "Recall@5", "Recall@10", "MRR"]
QDRANT_URL = "http://localhost:6333"
BM25_OPTIONS = {"k": 1.2, "b": 0.75, "avg_len": 256.0, "language": "italian",
                "tokenizer": "word", "lowercase": True, "ascii_folding": False}
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384
TOP_K = 10
RRF_K = 60


def read_records(path, id_field):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            evaluate.check_id(row[id_field], id_field)
            if not isinstance(row["text"], str):
                raise ValueError(f"{path}:{number}: text must be a string")
            yield row[id_field], row["text"]


def load_data(data_dir, smoke):
    qrels = evaluate.read_qrels(data_dir / "qrels" / "test.jsonl")
    records = read_records(data_dir / "corpus.jsonl", "doc_id")
    documents = {}
    for doc_id, text in islice(records, 50000) if smoke else records:
        if doc_id in documents:
            raise ValueError(f"Duplicate document ID: {doc_id}")
        documents[doc_id] = text
    pool_size = len(documents)
    queries = {}
    for query_id, text in read_records(data_dir / "queries" / "test.jsonl", "query_id"):
        if query_id in queries:
            raise ValueError(f"Duplicate query ID: {query_id}")
        queries[query_id] = text
    eligible = sorted(q for q, judgments in qrels.items()
                      if q in queries and any(r >= 1 for r in judgments.values())
                      and (not smoke or judgments.keys() <= documents.keys()))
    if smoke:
        if len(eligible) < 20:
            raise ValueError(f"Only {len(eligible)} eligible queries in the 50,000-record pool; need 20")
        selected = eligible[:20]
    else:
        selected = sorted(qrels)
        if not set(selected) <= queries.keys():
            raise ValueError("Some judged queries have no query text")
    qrels = {q: qrels[q] for q in selected}
    required = {d for judgments in qrels.values() for d in judgments}
    if not required <= documents.keys():
        raise ValueError("Judged documents are missing from the corpus")
    if smoke:
        negatives = list(islice((d for d in documents if d not in required), 1000))
        if len(negatives) != 1000:
            raise ValueError("Need 1,000 unjudged distractors in the smoke pool")
        retained = required | set(negatives)
        documents = {d: text for d, text in documents.items() if d in retained}
    queries = {q: queries[q] for q in selected}
    sampling = {
        "procedure": (
            "First 50,000 corpus records (or EOF); first 20 eligible query IDs in "
            "lexicographic string order, each with positive relevance and all judged "
            "documents in the pool; retain every judgment and judged document; add "
            "first 1,000 corpus-order documents unjudged for all selected queries. "
            "Unjudged distractors count as nonrelevant under the evaluator. No RNG. "
            "Prefix-biased diagnostic sample, not a full-benchmark score."
            if smoke else "Entire prepared corpus and all test-split judged queries; no sampling."
        ),
        "pool_size": pool_size,
        "eligible_query_count": len(eligible),
        "corpus_size": len(documents),
        "query_count": len(queries),
        "qrel_count": sum(map(len, qrels.values())),
        "relevant_document_count": len({d for js in qrels.values() for d, r in js.items() if r >= 1}),
        "unjudged_distractor_count": len(documents.keys() - required),
    }
    return documents, queries, qrels, sampling


def point_id(doc_id):
    # Preserve arbitrary source IDs (including leading zeros) in the payload.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"deadline_experiments:{DATASET}:{doc_id}"))


def document_payload(doc_id, text):
    return {"doc_id": doc_id, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def verify_collection(client, collection, documents):
    if client.count(collection, exact=True).count != len(documents):
        return False
    records = iter(documents.items())
    while batch := list(islice(records, 128)):
        expected = {point_id(d): document_payload(d, text) for d, text in batch}
        points = client.retrieve(collection, ids=list(expected), with_payload=True,
                                 with_vectors=["bm25", "dense"])
        if len(points) != len(expected):
            return False
        for point in points:
            if point.payload != expected.get(str(point.id)) or not isinstance(point.vector, dict):
                return False
            vector = point.vector.get("dense")
            if not isinstance(vector, list) or len(vector) != VECTOR_SIZE:
                return False
            if not isinstance(point.vector.get("bm25"), models.SparseVector):
                return False
    return True


def embed_dense(embedder, texts):
    vectors = []
    for vector in embedder.embed(texts, batch_size=128):
        values = vector.tolist()
        if len(values) != VECTOR_SIZE or not all(math.isfinite(value) for value in values):
            raise RuntimeError("FastEmbed returned an invalid dense vector")
        vectors.append(values)
    if len(vectors) != len(texts):
        raise RuntimeError("FastEmbed returned an unexpected number of dense vectors")
    return vectors


def build_index(client, embedder, documents, mode, server_version):
    digest = hashlib.sha256()
    for doc_id in sorted(documents):
        digest.update((json.dumps([doc_id, documents[doc_id]], ensure_ascii=False) + "\n").encode("utf-8"))
    identity = {"owner": "deadline_experiments", "experiment": "exp_003_hybrid",
                "dataset": DATASET, "mode": mode, "bm25_model": "qdrant/bm25",
                "bm25_options": BM25_OPTIONS, "bm25_modifier": "idf",
                "dense_model": EMBEDDING_MODEL, "distance": "Cosine",
                "dense_inference": "FastEmbed-local", "fastembed_version": version("fastembed"),
                "vector_size": VECTOR_SIZE, "fusion": "RRF", "rrf_k": RRF_K,
                "component_depth": TOP_K, "server_version": server_version,
                "corpus_sha256": digest.hexdigest(), "document_count": len(documents)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    prefix = f"deadline_{DATASET.replace('-', '_')}_exp003_hybrid_{mode}_{fingerprint}"
    existing = {c.name for c in client.get_collections().collections}
    for name in sorted(existing):
        if name != prefix and not name.startswith(prefix + "_"):
            continue
        info = client.get_collection(name)
        vectors = info.config.params.vectors or {}
        sparse = info.config.params.sparse_vectors or {}
        if (info.config.metadata == {"hybrid_experiment": identity, "complete": True}
                and set(vectors) == {"dense"}
                and vectors["dense"].size == VECTOR_SIZE
                and vectors["dense"].distance == models.Distance.COSINE
                and set(sparse) == {"bm25"}
                and sparse["bm25"].modifier == models.Modifier.IDF
                and verify_collection(client, name, documents)):
            return name, {**identity, "collection": name, "reused": True}
    # An incompatible or incomplete collection is never overwritten or deleted.
    name = prefix if prefix not in existing else f"{prefix}_{uuid.uuid4().hex[:12]}"
    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        metadata={"hybrid_experiment": identity, "complete": False},
    )
    records = iter(documents.items())
    while batch := list(islice(records, 128)):
        dense_vectors = embed_dense(embedder, [text for _, text in batch])
        client.upsert(name, points=[models.PointStruct(
            id=point_id(doc_id), payload=document_payload(doc_id, text),
            vector={
                "bm25": models.Document(text=text, model="qdrant/bm25", options=BM25_OPTIONS),
                "dense": dense_vector,
            },
        ) for (doc_id, text), dense_vector in zip(batch, dense_vectors)], wait=True)
    if not verify_collection(client, name, documents):
        raise RuntimeError(f"Qdrant collection verification failed: {name}")
    client.update_collection(name, metadata={"hybrid_experiment": identity, "complete": True})
    return name, {**identity, "collection": name, "reused": False}


def retrieve_component(client, collection, query_vector, documents, vector_name, depth):
    limit = min(depth + 1, len(documents))
    results = []
    if query_vector is not None:
        while True:
            points = client.query_points(
                collection, query=query_vector, using=vector_name,
                limit=limit, with_payload=["doc_id"], with_vectors=False,
            ).points
            results = []
            seen = set()
            for point in points:
                doc_id = (point.payload or {}).get("doc_id")
                if doc_id not in documents or str(point.id) != point_id(doc_id) or doc_id in seen:
                    raise RuntimeError("Qdrant returned an invalid or duplicate source document ID")
                if not math.isfinite(point.score) or (vector_name == "bm25" and point.score < 0):
                    raise RuntimeError(f"Qdrant returned an invalid {vector_name} score")
                seen.add(doc_id)
                results.append((doc_id, point.score))
            results.sort(key=lambda item: (item[1], item[0]), reverse=True)
            if (len(points) < limit or limit == len(documents)
                    or (len(results) > depth and results[-1][1] < results[depth - 1][1])):
                break
            limit = min(limit * 2, len(documents))
    if vector_name == "bm25" and len(results) < depth:
        scores = dict(results)
        for doc_id in sorted(documents, reverse=True):
            scores.setdefault(doc_id, 0.0)
            if len(scores) >= depth:
                break
        results = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    return results[:depth]


def reciprocal_rank_fusion(*rankings, depth=TOP_K):
    scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:depth]


def retrieve(client, embedder, collection, query, documents, depth=TOP_K):
    bm25_query = (models.Document(text=query, model="qdrant/bm25", options=BM25_OPTIONS)
                  if query.strip() else None)
    dense_query = embed_dense(embedder, [query])[0]
    bm25 = retrieve_component(client, collection, bm25_query, documents, "bm25", depth)
    dense = retrieve_component(client, collection, dense_query, documents, "dense", depth)
    return reciprocal_rank_fusion(bm25, dense, depth=depth)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true", help="Entire local corpus; potentially expensive")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data-preparation" / "data" / DATASET)
    args = parser.parse_args(argv)
    mode_name = "smoke" if args.smoke else "full"
    documents, queries, qrels, sampling = load_data(args.data_dir, args.smoke)
    loaded = time.perf_counter()
    print(f"{DATASET} mode={mode_name}: {sampling}", flush=True)
    depth = min(TOP_K, len(documents))
    execution_id = uuid.uuid4().hex
    output = Path(__file__).resolve().parent / "data" / mode_name / execution_id
    output.mkdir(parents=True, exist_ok=False)
    ranking = output / f"exp_003_hybrid_{execution_id}.trec"
    if TextEmbedding.get_embedding_size(EMBEDDING_MODEL) != VECTOR_SIZE:
        raise RuntimeError(f"Unexpected embedding size for {EMBEDDING_MODEL}")
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    # Only native BM25 inference is forwarded to Qdrant; dense vectors are numeric FastEmbed output.
    qdrant = QdrantClient(url=QDRANT_URL, cloud_inference=True, timeout=60)
    try:
        server_version = qdrant.info().version
        collection, index_info = build_index(qdrant, embedder, documents, mode_name, server_version)
        indexed = time.perf_counter()
        with ranking.open("w", encoding="utf-8") as stream:
            for query_id, text in queries.items():
                results = retrieve(qdrant, embedder, collection, text, documents, depth)
                for rank, (doc_id, score) in enumerate(results, 1):
                    stream.write(f"{query_id} Q0 {doc_id} {rank} {score:.17g} hybrid_rrf\n")
        write_json(output / "qdrant.json", index_info)
    finally:
        qdrant.close()
    retrieved = time.perf_counter()
    qrels_path = output / "qrels.jsonl"
    with qrels_path.open("w", encoding="utf-8") as stream:
        for query_id, judgments in qrels.items():
            for doc_id, relevance in judgments.items():
                stream.write(json.dumps(dict(query_id=query_id, doc_id=doc_id, relevance=relevance)) + "\n")
    experiment = f"deadline-{DATASET}"
    parameters = {
        "dataset": DATASET, "mode": mode_name, "method": "hybrid-rrf",
        "engine": "qdrant-server",
        "implementation": "client-side RRF over Qdrant BM25 and local FastEmbed dense rankings",
        "qdrant_url": QDRANT_URL, "qdrant_server_version": server_version,
        "qdrant_client_version": version("qdrant-client"), "collection": collection,
        "collection_reused": index_info["reused"], "corpus_sha256": index_info["corpus_sha256"],
        "fusion_method": "reciprocal-rank-fusion", "rrf_k": RRF_K,
        "retrieval_components": "qdrant/bm25,dense-cosine",
        "component_top_k": depth, "output_top_k": depth,
        "bm25_model": "qdrant/bm25", "bm25_options": json.dumps(BM25_OPTIONS, sort_keys=True),
        "bm25_modifier": "idf", "dense_inference": "FastEmbed-local",
        "fastembed_version": version("fastembed"),
        "embedding_model": EMBEDDING_MODEL, "vector_size": VECTOR_SIZE, "distance": "Cosine",
        "python_version": sys.version.split()[0],
        "top_k": depth, "data_dir": str(args.data_dir.resolve()), **sampling,
    }
    write_json(ranking.with_suffix(".metadata.json"), {
        "dataset": experiment, "split": mode_name, "parameters": parameters,
    })
    manifest = {"dataset": DATASET, "mode": mode_name, "source_split": SOURCE_SPLIT,
                "created_at": datetime.now(timezone.utc).isoformat(), **sampling,
                "query_ids": list(queries),
                "source_files": {str(p.resolve()): {"bytes": p.stat().st_size,
                                  "mtime_ns": p.stat().st_mtime_ns} for p in (
                    args.data_dir / "corpus.jsonl", args.data_dir / "queries" / "test.jsonl",
                    args.data_dir / "qrels" / "test.jsonl")}}
    if args.smoke:
        manifest["document_ids"] = list(documents)
        for filename, records, id_field in (("corpus.jsonl", documents, "doc_id"),
                                             ("queries.jsonl", queries, "query_id")):
            with (output / filename).open("w", encoding="utf-8") as stream:
                for identifier, text in records.items():
                    stream.write(json.dumps({id_field: identifier, "text": text}, ensure_ascii=False) + "\n")
    write_json(output / "sample.json", manifest)
    evaluate.main(["--dataset", experiment, "--split", mode_name, "--source-split", SOURCE_SPLIT,
                   "--qrels", str(qrels_path), "--run", str(ranking), "--metrics", *METRICS,
                   "--output-dir", str(output), "--no-tracking"])
    evaluated = time.perf_counter()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    write_json(output / "metrics.json", report["runs"][0]["aggregate"])
    write_json(output / "retrieval.json", report["runs"][0]["retrieval"])
    write_json(output / "metric-names.json", {name: evaluate.metric_key(name) for name in report["metrics"]})
    from mlflow import MlflowClient

    # Explicit persistent local store: no server or environment override is needed.
    tracking = ROOT / "tracking" / "data"
    tracking.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(tracking / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=uri)
    if client.get_experiment_by_name(experiment) is None:
        client.create_experiment(experiment, artifact_location=(tracking / "artifacts" / experiment).as_uri())
    evaluate.log_tracking(report, output, tracking_uri=uri, log_rankings=True)
    experiment_id = client.get_experiment_by_name(experiment).experiment_id
    runs = client.search_runs([experiment_id], filter_string=f"tags.mlflow.runName = '{ranking.name} - {mode_name}'")
    if len(runs) != 1:
        raise RuntimeError(f"Expected one persisted run, found {len(runs)}")
    run_id = runs[0].info.run_id
    try:
        tags = {"mode": mode_name, "dataset": DATASET, "experiment": "exp_003_hybrid",
                "execution_id": execution_id, "mlflow.runName": f"exp_003_hybrid_{mode_name}",
                "retrieval_components": "bm25,dense", "fusion": "rrf",
                "score_scope": "sampled-corpus diagnostic" if args.smoke else "full corpus top-10"}
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
        extra = ["sample.json", "metrics.json", "qrels.jsonl", "qdrant.json"]
        if args.smoke:
            extra += ["corpus.jsonl", "queries.jsonl"]
        for filename in extra:
            client.log_artifact(run_id, str(output / filename))
        timings = {"load_seconds": loaded - started, "index_seconds": indexed - loaded,
                   "retrieval_seconds": retrieved - indexed,
                   "evaluation_and_export_seconds": evaluated - retrieved,
                   "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "timing.json", timings)
        client.log_artifact(run_id, str(output / "timing.json"))
        for key, value in timings.items():
            client.log_metric(run_id, key, value)
        # Read persisted metrics/parameters, list artifacts, and download/check bytes.
        persisted = client.get_run(run_id)
        expected = {evaluate.metric_key(k): v for k, v in report["runs"][0]["aggregate"].items()}
        if persisted.info.status != "FINISHED" or any(persisted.data.metrics.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Persisted run status/metrics differ from the evaluated results")
        for key, value in parameters.items():
            if persisted.data.params.get(f"retrieval.{key}") != str(value):
                raise RuntimeError(f"Persisted parameter mismatch: {key}")
        for key, value in tags.items():
            if persisted.data.tags.get(key) != value:
                raise RuntimeError(f"Persisted tag mismatch: {key}")
        checksums = {}
        paths = [p.path for p in client.list_artifacts(run_id) if not p.is_dir]
        paths += [p.path for p in client.list_artifacts(run_id, "rankings")]
        required = {*extra, "timing.json", "report.json", "per-query.csv", "summary.csv",
                    "retrieval.json", "metric-names.json", f"rankings/{ranking.name}"}
        if not required <= set(paths):
            raise RuntimeError(f"Missing artifacts: {required - set(paths)}")
        with tempfile.TemporaryDirectory(prefix="deadline-verify-") as temporary:
            for path in sorted(required):
                downloaded = Path(client.download_artifacts(run_id, path, temporary))
                digest = hashlib.sha256(downloaded.read_bytes()).hexdigest()
                local = output / Path(path).name
                if digest != hashlib.sha256(local.read_bytes()).hexdigest():
                    raise RuntimeError(f"Artifact content mismatch: {path}")
                checksums[path] = digest
        verification = {"run_id": run_id, "tracking_uri": uri, "status": persisted.info.status,
                        "metrics": persisted.data.metrics, "artifact_sha256": checksums,
                        "verified_elapsed_seconds": time.perf_counter() - started}
        write_json(output / "verification.json", verification)
        print(json.dumps({"output": str(output), **verification}, indent=2), flush=True)
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise


if __name__ == "__main__":
    main()
