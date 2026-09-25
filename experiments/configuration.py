"""Load and validate the researcher-owned benchmark variant catalog."""

import json
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_CONFIG = Path("modelling/configs/benchmark-variants.json")


def resolve_config_path(path=None, root=ROOT_DIR):
    """Resolve catalog paths independently of the working directory."""
    path = Path(path) if path is not None else DEFAULT_BENCHMARK_CONFIG
    return path if path.is_absolute() else root / path


def load_variants(path):
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in benchmark configuration {path}: {error}") from error
    if not isinstance(config, dict) or set(config) != {"version", "variants"}:
        raise ValueError("benchmark configuration must contain exactly version and variants")
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("benchmark configuration version must be 1")
    variants = config["variants"]
    if not isinstance(variants, dict):
        raise ValueError("benchmark configuration variants must be an object")
    used_ids = set()
    validated = {}
    for dimension, entries in variants.items():
        if not isinstance(dimension, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", dimension):
            raise ValueError(f"invalid benchmark dimension {dimension!r}")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"benchmark dimension {dimension} must be a nonempty array")
        validated[dimension] = []
        for index, entry in enumerate(entries):
            location = f"variants.{dimension}[{index}]"
            if not isinstance(entry, dict) or set(entry) != {"id", "value"}:
                raise ValueError(f"{location} must contain exactly id and value")
            identifier, value = entry["id"], entry["value"]
            if not isinstance(identifier, str) or not re.fullmatch(
                r"[a-z0-9][a-z0-9-]{0,62}", identifier
            ):
                raise ValueError(f"{location}.id must be a lowercase filesystem-safe slug")
            if identifier in used_ids:
                raise ValueError(f"duplicate benchmark variant id: {identifier}")
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{location}.value must be a nonempty string")
            used_ids.add(identifier)
            validated[dimension].append((identifier, value))
    return validated
