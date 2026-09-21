"""Evaluate saved retrieval runs against JSONL or TREC relevance judgments."""

import argparse
import csv
import gzip
import json
import math
import re
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import ir_measures
from ir_measures.providers import FallbackProvider

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data-preparation" / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CUTOFFS = (1, 3, 5, 10, 20, 100, 1000)
FULL_METRICS = ("AP", "RR", "nDCG", "Rprec", "Bpref", "SetP", "SetR", "SetF")
CUTOFF_METRICS = ("nDCG", "AP", "RR", "P", "R", "Success", "Judged")
ALIASES = {"MAP": "AP", "MRR": "RR", "Precision": "P", "Recall": "R"}
PROVIDERS = ("pytrec_eval", "msmarco", "judged")


def open_text(path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def file_format(path, requested):
    if requested != "auto":
        return requested
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes.pop()
    return "jsonl" if suffixes and suffixes[-1] == ".jsonl" else "trec"


def check_id(value, field):
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError(f"{field} must be a nonempty string without whitespace")
    return value


def read_rows(path, kind, format_name="auto"):
    """Validate records without changing string identifiers or score precision."""
    format_name = file_format(path, format_name)
    run_tag = None
    with open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                if format_name == "jsonl":
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("each JSONL record must be an object")
                    query_id = check_id(row["query_id"], "query_id")
                    doc_id = check_id(row["doc_id"], "doc_id")
                    value = row["relevance" if kind == "qrels" else "score"]
                else:
                    fields = line.split()
                    expected = 4 if kind == "qrels" else 6
                    if len(fields) != expected:
                        raise ValueError(f"expected {expected} TREC columns")
                    query_id, _, doc_id = fields[:3]
                    if kind == "qrels":
                        value = int(fields[3])
                    else:
                        if int(fields[3]) < 1:
                            raise ValueError("rank must be a positive integer")
                        value = float(fields[4])
                        if run_tag is not None and fields[5] != run_tag:
                            raise ValueError("a run file must contain only one run tag")
                        run_tag = fields[5]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("relevance/score must be numeric")
                if not math.isfinite(value):
                    raise ValueError("relevance/score must be finite")
                if kind == "qrels":
                    if value != int(value) or not -1 <= value <= 2147483647:
                        raise ValueError(
                            "relevance must be an integer from -1 to 2147483647"
                        )
                    value = int(value)
                yield query_id, doc_id, value
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def read_qrels(path, format_name="auto"):
    qrels = {}
    for query_id, doc_id, relevance in read_rows(path, "qrels", format_name):
        documents = qrels.setdefault(query_id, {})
        if doc_id in documents:
            raise ValueError(f"{path}: duplicate judgment for {query_id}/{doc_id}")
        documents[doc_id] = relevance
    # TREC uses -1 for unjudged documents; these must not count as judgments.
    qrels = {
        query_id: {doc_id: rel for doc_id, rel in docs.items() if rel >= 0}
        for query_id, docs in qrels.items()
    }
    qrels = {query_id: docs for query_id, docs in qrels.items() if docs}
    if not qrels:
        raise ValueError(f"{path}: no judged queries")
    return qrels


def read_run(path, format_name="auto"):
    run = {}
    for query_id, doc_id, score in read_rows(path, "run", format_name):
        documents = run.setdefault(query_id, {})
        if doc_id in documents:
            raise ValueError(f"{path}: duplicate result for {query_id}/{doc_id}")
        documents[doc_id] = score
    return run


def build_metrics(names=None, cutoffs=DEFAULT_CUTOFFS, relevance_level=1):
    """Construct public metric objects without the Python-version-sensitive parser."""
    if relevance_level < 1:
        raise ValueError("relevance level must be positive")
    if any(cutoff < 1 for cutoff in cutoffs):
        raise ValueError("cutoffs must be positive integers")
    if names is None:
        names = list(FULL_METRICS) + [
            f"{name}@{cutoff}"
            for cutoff in sorted(set(cutoffs))
            for name in CUTOFF_METRICS
        ]
    metrics = []
    for name in names:
        match = re.fullmatch(r"([A-Za-z]+)(?:@([1-9][0-9]*))?", name)
        if not match:
            raise ValueError(f"invalid metric {name!r}; use NAME or NAME@K")
        base, cutoff = match.groups()
        base = ALIASES.get(base, base)
        if base not in set(FULL_METRICS + CUTOFF_METRICS):
            raise ValueError(f"unsupported metric {name!r}")
        measure = getattr(ir_measures, base)
        if "rel" in measure.SUPPORTED_PARAMS:
            measure = measure(rel=relevance_level)
        if cutoff is not None:
            if base not in CUTOFF_METRICS:
                raise ValueError(f"{base} does not accept a cutoff")
            measure = measure @ int(cutoff)
        elif base in ("P", "R", "Success"):
            raise ValueError(f"{base} requires a cutoff")
        if measure not in metrics:
            metrics.append(measure)
    if not metrics:
        raise ValueError("select at least one metric")
    return metrics


def create_evaluator(metrics, qrels):
    provider = FallbackProvider([
        ir_measures.providers.registry[name] for name in PROVIDERS
    ])
    return provider.evaluator(metrics, qrels)


def evaluate_run(qrels, run, metrics, relevance_level=1, evaluator=None):
    """Score every judged query, including queries missing from the run."""
    query_ids = set(qrels)
    evaluated_run = {}
    for query_id in query_ids.intersection(run):
        documents = run[query_id]
        if not documents:
            continue
        # trec_eval orders by descending score, then descending document ID.
        # Unique ordinal scores make every backend use exactly this ordering.
        ordered = sorted(documents, key=lambda doc: (documents[doc], doc), reverse=True)
        evaluated_run[query_id] = {
            doc_id: float(len(ordered) - index)
            for index, doc_id in enumerate(ordered)
        }
    evaluator = evaluator or create_evaluator(metrics, qrels)
    per_query = {query_id: {} for query_id in sorted(qrels)}
    aggregators = {measure: measure.aggregator() for measure in metrics}
    for result in evaluator.iter_calc(evaluated_run):
        if not math.isfinite(result.value):
            raise ValueError(f"nonfinite metric {result.measure} for {result.query_id}")
        per_query[result.query_id][str(result.measure)] = result.value
        aggregators[result.measure].add(result.value)
    expected = {str(measure) for measure in metrics}
    if any(set(values) != expected for values in per_query.values()):
        raise ValueError("evaluation backend did not return every requested metric")
    coverage = {
        "judged_queries": len(qrels),
        "queries_with_relevant_documents": sum(
            any(rel >= relevance_level for rel in docs.values())
            for docs in qrels.values()
        ),
        "run_queries": len(run),
        "evaluated_queries_with_results": len(evaluated_run),
        "missing_judged_queries": len(qrels) - len(evaluated_run),
        "ignored_unjudged_queries": len(set(run) - query_ids),
        "judged_query_coverage": len(evaluated_run) / len(qrels),
        "retrieved_documents": sum(len(docs) for docs in evaluated_run.values()),
        "minimum_depth": min((len(evaluated_run.get(q, {})) for q in qrels), default=0),
        "maximum_depth": max((len(docs) for docs in evaluated_run.values()), default=0),
    }
    aggregate = {
        str(measure): aggregators[measure].result() for measure in metrics
    }
    return {"aggregate": aggregate, "coverage": coverage}, per_query


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset", help="dataset directory under data-preparation/data"
    )
    source.add_argument("--qrels", type=Path, help="external JSONL or TREC judgments")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--run", type=Path, nargs="+", required=True, help="one or more run files"
    )
    parser.add_argument(
        "--qrels-format", choices=("auto", "jsonl", "trec"), default="auto"
    )
    parser.add_argument(
        "--run-format", choices=("auto", "jsonl", "trec"), default="auto"
    )
    parser.add_argument(
        "--metrics", nargs="+", help="replace defaults with NAME or NAME@K"
    )
    parser.add_argument(
        "--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS,
        help="cutoffs for the default suite; ignored with --metrics",
    )
    parser.add_argument(
        "--relevance-level", type=int, default=1,
        help="minimum relevance grade for binary metrics (default: 1)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing reports"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.dataset and (
            Path(args.dataset).name != args.dataset or args.dataset in (".", "..")
        ):
            raise ValueError("--dataset must be a directory name, not a path")
        qrels_path = args.qrels or (
            DATA_DIR / args.dataset / "qrels" / f"{args.split}.jsonl"
        )
        run_names = [path.name for path in args.run]
        if len(set(run_names)) != len(run_names):
            raise ValueError("run filenames must be unique for report labels")
        inputs = {path.resolve() for path in [qrels_path, *args.run]}
        report_names = ("report.json", "summary.csv", "per-query.csv")
        targets = [args.output_dir / name for name in report_names]
        for target in targets:
            if target.resolve() in inputs:
                raise ValueError(f"output would overwrite input: {target}")
            if target.exists() and not args.overwrite:
                raise ValueError(
                    f"report exists: {target}; use --overwrite to replace it"
                )
        metrics = build_metrics(args.metrics, args.cutoffs, args.relevance_level)
        qrels = read_qrels(qrels_path, args.qrels_format)
        evaluator = create_evaluator(metrics, qrels)
        metric_names = [str(measure) for measure in metrics]
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "qrels": str(qrels_path.resolve()),
            "qrels_format": file_format(qrels_path, args.qrels_format),
            "dataset": args.dataset,
            "split": args.split if args.dataset else None,
            "metrics": metric_names,
            "relevance_level": args.relevance_level,
            "query_policy": "all judged queries; missing results score zero",
            "tie_policy": "descending score, then descending document ID",
            "ndcg_gain": "linear relevance grades (trec_eval default)",
            "providers": list(PROVIDERS),
            "versions": {
                name: version(name)
                for name in ("ir-measures", "pytrec-eval-terrier")
            },
            "python_version": sys.version.split()[0],
            "runs": [],
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Stage complete reports so a malformed later run preserves prior reports.
        with tempfile.TemporaryDirectory(
            prefix=".evaluation-", dir=args.output_dir
        ) as temporary:
            staging = Path(temporary)
            with (staging / "per-query.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["run", "query_id", *metric_names]
                )
                writer.writeheader()
                for path in args.run:
                    run = read_run(path, args.run_format)
                    result, per_query = evaluate_run(
                        qrels, run, metrics, args.relevance_level, evaluator
                    )
                    result.update({
                        "name": path.name,
                        "path": str(path.resolve()),
                        "format": file_format(path, args.run_format),
                    })
                    report["runs"].append(result)
                    for query_id, values in per_query.items():
                        writer.writerow({
                            "run": path.name, "query_id": query_id, **values
                        })
                    coverage = result["coverage"]
                    print(
                        f"{path.name}: "
                        f"{coverage['evaluated_queries_with_results']}/"
                        f"{coverage['judged_queries']} judged queries have results; "
                        f"{coverage['ignored_unjudged_queries']} "
                        "unjudged queries ignored"
                    )
                    for name, value in result["aggregate"].items():
                        print(f"  {name}: {value:.6f}")
                    del run, per_query
            with (staging / "report.json").open("w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            with (staging / "summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                coverage_names = list(report["runs"][0]["coverage"])
                writer = csv.DictWriter(
                    stream, fieldnames=["run", *coverage_names, *metric_names]
                )
                writer.writeheader()
                for result in report["runs"]:
                    writer.writerow({
                        "run": result["name"],
                        **result["coverage"],
                        **result["aggregate"],
                    })
            for name in report_names:
                (staging / name).replace(args.output_dir / name)
        print(f"Reports saved to {args.output_dir.resolve()}")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
