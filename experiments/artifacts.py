"""Validate, publish, and reuse retrieval ranking and metadata pairs."""

import json
import math
import re
from pathlib import Path
import tempfile


def staging_directory(output_dir):
    """Keep temporary outputs on the publication filesystem."""
    return tempfile.TemporaryDirectory(prefix=".experiment-", dir=output_dir)


def artifact_paths(output_dir, identifier):
    return (
        output_dir / f"{identifier}.trec",
        output_dir / f"{identifier}.metadata.json",
    )


def prepare_output(directory, dataset, split, experiment):
    """Validate fresh script outputs before publishing or attaching identity."""
    ranking, metadata = artifact_paths(Path(directory), experiment.method)
    if not ranking.is_file() or not metadata.is_file():
        raise RuntimeError(
            f"method must create {ranking.name} and {metadata.name} "
            "in its supplied --output-dir"
        )
    if metadata.stat().st_size == 0:
        raise RuntimeError(f"empty metadata: {metadata.name}")
    validate_ranking(ranking)
    add_experiment_metadata(metadata, dataset, split, experiment)
    return ranking, metadata


def read_metadata(path, dataset, split):
    """Validate the evaluator's scalar sidecar contract without changing the file."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid retrieval metadata {path}: {error}") from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("dataset") != dataset
        or metadata.get("split") != split
        or not isinstance(metadata.get("parameters"), dict)
    ):
        raise ValueError(
            f"retrieval metadata {path.name} must match dataset/split and contain parameters"
        )
    for key, value in metadata["parameters"].items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError(f"{path}: invalid parameter name {key!r}")
        if not isinstance(value, (str, int, float, bool)) or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            raise ValueError(f"{path}: parameters must be finite scalar values")
    top_k = metadata["parameters"].get("top_k")
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError(f"{path}: top_k must be a positive integer")
    return metadata


def add_experiment_metadata(path, dataset, split, experiment):
    """Attach orchestration identity while preserving retrieval-owned metadata."""
    metadata = read_metadata(path, dataset, split)
    for key, value in (
        ("retrieval_method", experiment.method),
        ("experiment_id", experiment.experiment_id),
    ):
        if key in metadata["parameters"] and metadata["parameters"][key] != value:
            raise ValueError(f"retrieval metadata contains conflicting {key}")
        metadata["parameters"][key] = value
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def validate_ranking(path):
    """Validate the TREC contract before a ranking can be published."""
    seen = set()
    run_tag = None
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                fields = line.split()
                if len(fields) != 6:
                    raise ValueError("expected 6 TREC columns")
                query_id, _, document_id, rank, score, tag = fields
                if int(rank) < 1:
                    raise ValueError("rank must be a positive integer")
                if not math.isfinite(float(score)):
                    raise ValueError("score must be finite")
                if run_tag is not None and tag != run_tag:
                    raise ValueError("a run file must contain only one run tag")
                run_tag = tag
                identity = (query_id, document_id)
                if identity in seen:
                    raise ValueError(
                        f"duplicate result for {query_id}/{document_id}"
                    )
                seen.add(identity)
            except (ValueError, OverflowError) as error:
                raise ValueError(f"invalid ranking {path}:{line_number}: {error}") from error


def publish_pair(ranking, metadata, output_dir, experiment_id):
    """Publish one validated ranking/metadata pair, restoring old files on failure."""
    destinations = artifact_paths(output_dir, experiment_id)
    backups = []
    published = []
    try:
        for destination in destinations:
            if destination.exists():
                backup = ranking.parent / f"previous-{destination.name}"
                destination.replace(backup)
                backups.append((backup, destination))
        for source, destination in zip((ranking, metadata), destinations, strict=True):
            source.replace(destination)
            published.append(destination)
    except (OSError, KeyboardInterrupt):
        for destination in published:
            if destination.exists():
                destination.unlink()
        for backup, destination in backups:
            if backup.exists():
                backup.replace(destination)
        raise


def reusable_benchmark_output(output_dir, dataset, split, experiment, top_k):
    """Accept only valid pairs matching identity, model values and declared settings."""
    ranking, metadata_path = artifact_paths(output_dir, experiment.experiment_id)
    if not ranking.is_file() or not metadata_path.is_file():
        return False
    try:
        if ranking.stat().st_size == 0 or metadata_path.stat().st_size == 0:
            return False
        validate_ranking(ranking)
        metadata = read_metadata(metadata_path, dataset, split)
        parameters = metadata["parameters"]
        return (
            metadata.get("dataset") == dataset
            and metadata.get("split") == split
            and isinstance(parameters, dict)
            and parameters.get("experiment_id") == experiment.experiment_id
            and parameters.get("retrieval_method") == experiment.method
            and parameters.get("top_k") == top_k
            and all(
                key in parameters and str(parameters[key]) == str(value)
                for key, value in experiment.parameters
            )
        )
    except (OSError, ValueError, TypeError, UnicodeError):
        return False
