"""Independent BM25 experiment; smoke scores are sampled-corpus diagnostics."""

import argparse
import hashlib
import heapq
import json
import math
import re
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation" / "scripts"))
import evaluate

DATASET = "scifact"
SOURCE_SPLIT = "test"
METRICS = ["MAP", "Precision@1", "Precision@5", "Precision@10",
           "Recall@1", "Recall@5", "Recall@10", "MRR"]
TOKEN_PATTERN = re.compile(r"[^\W_]+")


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


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def build_index(documents):
    postings = defaultdict(list)
    lengths = {}
    for doc_id, text in documents.items():
        counts = Counter(tokenize(text))
        lengths[doc_id] = sum(counts.values())
        for term, count in counts.items():
            postings[term].append((doc_id, count))
    average = sum(lengths.values()) / len(lengths) if lengths else 0
    if not average:
        raise ValueError("Corpus has no tokens")
    return postings, lengths, average


def retrieve(query, postings, lengths, average, depth):
    # BM25 with positive Robertson IDF, k1=1.2, b=0.75, unique query terms.
    scores = defaultdict(float)
    count = len(lengths)
    for term in sorted(set(tokenize(query))):
        matches = postings.get(term, ())
        idf = math.log1p((count - len(matches) + 0.5) / (len(matches) + 0.5))
        for doc_id, frequency in matches:
            norm = 1.2 * (0.25 + 0.75 * lengths[doc_id] / average)
            scores[doc_id] += idf * frequency * 2.2 / (frequency + norm)
    # Include zero-score documents, with the evaluator's descending-ID tie rule.
    return heapq.nlargest(depth, ((d, scores.get(d, 0.0)) for d in lengths),
                         key=lambda item: (item[1], item[0]))


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
    postings, lengths, average = build_index(documents)
    depth = len(documents) if args.smoke else min(1000, len(documents))
    execution_id = uuid.uuid4().hex
    output = Path(__file__).resolve().parent / "data" / mode_name / execution_id
    output.mkdir(parents=True, exist_ok=False)
    ranking = output / f"exp_001_baseline_{execution_id}.trec"
    with ranking.open("w", encoding="utf-8") as stream:
        for query_id, text in queries.items():
            for rank, (doc_id, score) in enumerate(retrieve(text, postings, lengths, average, depth), 1):
                stream.write(f"{query_id} Q0 {doc_id} {rank} {score:.17g} bm25\n")
    retrieved = time.perf_counter()
    qrels_path = output / "qrels.jsonl"
    with qrels_path.open("w", encoding="utf-8") as stream:
        for query_id, judgments in qrels.items():
            for doc_id, relevance in judgments.items():
                stream.write(json.dumps(dict(query_id=query_id, doc_id=doc_id, relevance=relevance)) + "\n")
    experiment = f"deadline-{DATASET}"
    parameters = {
        "dataset": DATASET, "mode": mode_name, "method": "bm25", "k1": 1.2, "b": 0.75,
        "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
        "tokenization": "lowercase Unicode alphanumeric words; distinct query terms; no stemming or stopwords",
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
        for key, value in {"mode": mode_name, "dataset": DATASET, "experiment": "exp_001_baseline",
                           "score_scope": "sampled-corpus diagnostic" if args.smoke else "full corpus, top 1000"}.items():
            client.set_tag(run_id, key, value)
        extra = ["sample.json", "metrics.json", "qrels.jsonl"]
        if args.smoke:
            extra += ["corpus.jsonl", "queries.jsonl"]
        for filename in extra:
            client.log_artifact(run_id, str(output / filename))
        timings = {"load_seconds": loaded - started, "retrieval_seconds": retrieved - loaded,
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
                if local.exists() and digest != hashlib.sha256(local.read_bytes()).hexdigest():
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
