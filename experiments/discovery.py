"""Discover prepared datasets and public retrieval script capabilities."""

import ast
import math
import re

CAPABILITY_NAME = "EXPERIMENT_CAPABILITY"


def discover_datasets(data_dir, split):
    """Check prepared file structure; modelling/evaluation validate the contents."""
    ready, unavailable = {}, {}
    if not data_dir.is_dir():
        return ready, unavailable
    required = ("corpus.jsonl", f"queries/{split}.jsonl", f"qrels/{split}.jsonl")
    for directory in sorted(data_dir.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        missing = [
            name for name in required
            if not (directory / name).is_file() or (directory / name).stat().st_size == 0
        ]
        if missing:
            unavailable[directory.name] = missing
        else:
            ready[directory.name] = directory
    return ready, unavailable


def discover_methods(scripts_dir):
    """Public Python scripts are entry points; underscore-prefixed files are helpers."""
    return {
        path.stem: path for path in sorted(scripts_dir.glob("*.py"))
        if path.is_file() and not path.name.startswith("_")
    }


def read_experiment_capability(script):
    """Read a script's literal capability declaration without importing the script."""
    try:
        tree = ast.parse(script.read_text(encoding="utf-8-sig"), filename=str(script))
    except SyntaxError as error:
        raise ValueError(f"cannot inspect experiment capability in {script}: {error}") from error
    declarations = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(
            isinstance(target, ast.Name) and target.id == CAPABILITY_NAME
            for target in targets
        ):
            try:
                declarations.append(ast.literal_eval(node.value))
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"{script.name}: {CAPABILITY_NAME} must be a literal value"
                ) from error
    if not declarations:
        return None
    if len(declarations) != 1:
        raise ValueError(f"{script.name}: declare {CAPABILITY_NAME} exactly once")
    capability = declarations[0]
    required = {"version", "dimensions"}
    optional = {"parameter_names", "parameters"}
    if (
        not isinstance(capability, dict)
        or not required <= capability.keys()
        or capability.keys() - required - optional
    ):
        raise ValueError(
            f"{script.name}: {CAPABILITY_NAME} requires version and dimensions; "
            "optional fields are parameter_names and parameters"
        )
    if type(capability["version"]) is not int or capability["version"] != 1:
        raise ValueError(f"{script.name}: unsupported {CAPABILITY_NAME} version")
    dimensions = capability["dimensions"]
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError(f"{script.name}: capability dimensions must be a nonempty object")
    for dimension, option in dimensions.items():
        if not isinstance(dimension, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", dimension):
            raise ValueError(f"{script.name}: invalid experiment dimension {dimension!r}")
        if not isinstance(option, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", option):
            raise ValueError(f"{script.name}: invalid CLI option for {dimension}")
        if any(name.startswith(option) for name in (
            "dataset", "split", "top-k", "output-dir", "overwrite", "help"
        )):
            raise ValueError(f"{script.name}: capability cannot override --{option}")
    if len(set(dimensions.values())) != len(dimensions):
        raise ValueError(f"{script.name}: capability CLI options must be unique")
    names = capability.get("parameter_names", {})
    parameters = capability.get("parameters", {})
    if not isinstance(names, dict) or set(names) - set(dimensions.values()):
        raise ValueError(f"{script.name}: parameter_names must map declared CLI options")
    if not isinstance(parameters, dict):
        raise ValueError(f"{script.name}: parameters must be a scalar metadata object")
    for key in (*names.values(), *parameters.keys()):
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError(f"{script.name}: invalid metadata parameter name")
    mapped_names = [names.get(option, option.replace("-", "_"))
                    for option in dimensions.values()]
    if len(set(mapped_names)) != len(mapped_names):
        raise ValueError(f"{script.name}: dimension metadata parameter names must be unique")
    for value in parameters.values():
        if not isinstance(value, (str, int, float, bool)) or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            raise ValueError(f"{script.name}: parameters must contain finite scalars")
    return capability


def discover_benchmark_methods(methods):
    """Return only methods that opt in through the generic capability contract."""
    return {
        method: capability
        for method, script in methods.items()
        if (capability := read_experiment_capability(script)) is not None
    }
