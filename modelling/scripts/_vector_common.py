"""Narrow helpers shared by sparse and hybrid vector retrieval scripts."""

import hashlib
import json
import math

from fastembed.common.model_description import ModelSource
from qdrant_client import models


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


def load_embedding_model(embedding_class, description_class, model_info, cache, threads):
    """Download once, fingerprint resolved files, then load exactly those files."""
    description = description_class(
        **{**model_info, "sources": ModelSource(**model_info["sources"])})
    directory = embedding_class.download_model(description, cache_dir=str(cache))
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )
    if not files:
        raise ValueError(f"No embedding model files found in {directory}")
    hashes = [(path.relative_to(directory).as_posix(), fingerprint(path)) for path in files]
    artifact_hash = hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    model = embedding_class(
        model_name=model_info["model"], threads=threads,
        providers=["CPUExecutionProvider"], cache_dir=str(cache),
        specific_model_path=str(directory),
    )
    return model, artifact_hash


def checked_dense(vector, dimension):
    values = [float(value) for value in vector]
    if len(values) != dimension or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Dense embedding must contain {dimension} finite values")
    if not any(values):
        raise ValueError("Dense embedding is a zero vector; cosine similarity is undefined")
    return values


def checked_sparse(vector):
    indices = [int(index) for index in vector.indices]
    values = [float(value) for value in vector.values]
    if len(indices) != len(values):
        raise ValueError("Sparse embedding indices and values have different lengths")
    if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("Sparse embedding indices must be unique nonnegative integers")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Sparse embedding values must be finite")
    if not indices:
        return None
    ordered = sorted(zip(indices, values))
    return models.SparseVector(
        indices=[index for index, _ in ordered],
        values=[value for _, value in ordered],
    )


def checked_results(hits, limit, require_limit=False):
    results = []
    for hit in hits:
        doc_id = (hit.payload or {}).get("doc_id")
        if (
            not isinstance(doc_id, str)
            or not doc_id
            or any(character.isspace() for character in doc_id)
            or not math.isfinite(hit.score)
        ):
            raise ValueError("Qdrant returned an invalid document ID/payload or score")
        results.append((doc_id, float(hit.score)))
    if len(results) > limit or len({doc_id for doc_id, _ in results}) != len(results):
        raise ValueError("Qdrant returned too many or duplicate results")
    if require_limit and len(results) != limit:
        raise ValueError("Qdrant returned incomplete results unexpectedly")
    return sorted(results, key=lambda item: (item[1], item[0]), reverse=True)


def boolean(value):
    if value.lower() not in ("true", "false"):
        raise ValueError("expected true or false")
    return value.lower() == "true"
