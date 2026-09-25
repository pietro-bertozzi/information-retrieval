"""Independent top-10 dense retrieval experiment using Qdrant and FastEmbed."""

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

from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "scripts"))
import evaluate

DATASET = "scifact"
SOURCE_SPLIT = "test"
METRICS = ["MAP", "Precision@1", "Precision@5", "Precision@10",
           "Recall@1", "Recall@5", "Recall@10", "MRR"]
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384
TOP_K = 10


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
        points = client.retrieve(collection, ids=list(expected), with_payload=True, with_vectors=["dense"])
        if len(points) != len(expected):
            return False
        for point in points:
            if point.payload != expected.get(str(point.id)) or not isinstance(point.vector, dict):
                return False
            vector = point.vector.get("dense")
            if not isinstance(vector, list) or len(vector) != VECTOR_SIZE:
                return False
    return True


def build_index(client, documents, mode, server_version):
    digest = hashlib.sha256()
    for doc_id in sorted(documents):
        digest.update((json.dumps([doc_id, documents[doc_id]], ensure_ascii=False) + "\n").encode("utf-8"))
    identity = {"owner": "deadline_experiments", "experiment": "exp_002_dense",
                "dataset": DATASET, "mode": mode, "model": EMBEDDING_MODEL,
                "distance": "Cosine", "vector_size": VECTOR_SIZE,
                "fastembed_version": version("fastembed"), "server_version": server_version,
                "corpus_sha256": digest.hexdigest(), "document_count": len(documents)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    prefix = f"deadline_{DATASET.replace('-', '_')}_exp002_dense_{mode}_{fingerprint}"
    existing = {c.name for c in client.get_collections().collections}
    for name in sorted(existing):
        if name != prefix and not name.startswith(prefix + "_"):
            continue
        info = client.get_collection(name)
        vectors = info.config.params.vectors or {}
        if (info.config.metadata == {"dense_experiment": identity, "complete": True}
                and set(vectors) == {"dense"}
                and vectors["dense"].size == VECTOR_SIZE
                and vectors["dense"].distance == models.Distance.COSINE
                and not info.config.params.sparse_vectors
                and verify_collection(client, name, documents)):
            return name, {**identity, "collection": name, "reused": True}
    # An incompatible or incomplete collection is never overwritten or deleted.
    name = prefix if prefix not in existing else f"{prefix}_{uuid.uuid4().hex[:12]}"
    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)},
        metadata={"dense_experiment": identity, "complete": False},
    )
    records = iter(documents.items())
    while batch := list(islice(records, 128)):
        client.upsert(name, points=[models.PointStruct(
            id=point_id(d), payload=document_payload(d, text),
            vector={"dense": models.Document(text=text, model=EMBEDDING_MODEL)},
        ) for d, text in batch], wait=True)
    if not verify_collection(client, name, documents):
        raise RuntimeError(f"Qdrant collection verification failed: {name}")
    client.update_collection(name, metadata={"dense_experiment": identity, "complete": True})
    return name, {**identity, "collection": name, "reused": False}


def retrieve(client, collection, query, documents, depth=TOP_K):
    limit = min(depth + 1, len(documents))
    while True:
        points = client.query_points(
            collection, query=models.Document(text=query, model=EMBEDDING_MODEL),
            using="dense", limit=limit, with_payload=["doc_id"], with_vectors=False,
        ).points
        results = []
        seen = set()
        for point in points:
            doc_id = (point.payload or {}).get("doc_id")
            if doc_id not in documents or str(point.id) != point_id(doc_id) or doc_id in seen:
                raise RuntimeError("Qdrant returned an invalid or duplicate source document ID")
            if not math.isfinite(point.score):
                raise RuntimeError("Qdrant returned an invalid cosine score")
            seen.add(doc_id)
            results.append((doc_id, point.score))
        results.sort(key=lambda item: (item[1], item[0]), reverse=True)
        if (len(points) < limit or limit == len(documents)
                or (len(results) > depth and results[-1][1] < results[depth - 1][1])):
            return results[:depth]
        limit = min(limit * 2, len(documents))


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
    ranking = output / f"exp_002_dense_{execution_id}.trec"
    # Document inputs are embedded locally by FastEmbed, then searched in Qdrant.
    qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
    try:
        if qdrant.get_embedding_size(EMBEDDING_MODEL) != VECTOR_SIZE:
            raise RuntimeError(f"Unexpected embedding size for {EMBEDDING_MODEL}")
        server_version = qdrant.info().version
        collection, index_info = build_index(qdrant, documents, mode_name, server_version)
        indexed = time.perf_counter()
        with ranking.open("w", encoding="utf-8") as stream:
            for query_id, text in queries.items():
                results = retrieve(qdrant, collection, text, documents, depth)
                for rank, (doc_id, score) in enumerate(results, 1):
                    stream.write(f"{query_id} Q0 {doc_id} {rank} {score:.17g} dense\n")
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
        "dataset": DATASET, "mode": mode_name, "method": "dense-retrieval",
        "engine": "qdrant-server", "implementation": "FastEmbed embeddings with Qdrant cosine search",
        "qdrant_url": QDRANT_URL, "qdrant_server_version": server_version,
        "qdrant_client_version": version("qdrant-client"), "collection": collection,
        "collection_reused": index_info["reused"], "corpus_sha256": index_info["corpus_sha256"],
        "embedding_runtime": "FastEmbed", "fastembed_version": version("fastembed"),
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
        tags = {"mode": mode_name, "dataset": DATASET, "experiment": "exp_002_dense",
                "execution_id": execution_id, "mlflow.runName": f"exp_002_dense_{mode_name}",
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
