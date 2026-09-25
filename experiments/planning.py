"""Combine discovered capabilities and selected values into stable configurations."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

from .configuration import load_variants
from .discovery import discover_benchmark_methods


@dataclass(frozen=True)
class Experiment:
    method: str
    script: Path
    experiment_id: str
    options: tuple[str, ...] = ()
    parameters: tuple[tuple[str, object], ...] = ()


def load_benchmark(path, methods):
    """Expand configured dimensions for every method that declares capabilities."""
    return plan_benchmark(methods, discover_benchmark_methods(methods), load_variants(path))


def plan_benchmark(methods, capabilities, variants):
    """Build configurations without reading files or invoking retrieval."""
    if not capabilities:
        raise ValueError("no discovered retrieval method exposes experiment dimensions")
    consumed = {dimension for capability in capabilities.values()
                for dimension in capability["dimensions"]}
    unused = sorted(set(variants) - consumed)
    if unused:
        raise ValueError(f"benchmark dimension is not consumed by any method: {unused[0]}")
    experiments = []
    for method, capability in capabilities.items():
        dimensions = tuple(capability["dimensions"].items())
        missing = [dimension for dimension, _ in dimensions if dimension not in variants]
        if missing:
            raise ValueError(
                f"benchmark method {method} requires missing dimension: {missing[0]}"
            )
        for combination in product(*(variants[name] for name, _ in dimensions)):
            identifiers = [identifier for identifier, _ in combination]
            options = tuple(
                f"--{option}={value}"
                for (_, option), (_, value) in zip(dimensions, combination, strict=True)
            )
            parameters = dict(capability.get("parameters", {}))
            names = capability.get("parameter_names", {})
            for (_, option), (_, value) in zip(dimensions, combination, strict=True):
                parameters[names.get(option, option.replace("-", "_"))] = value
            experiments.append(Experiment(
                method, methods[method], f"{method}__{'__'.join(identifiers)}",
                options, tuple(parameters.items()),
            ))
    identities = [experiment.experiment_id for experiment in experiments]
    if len(set(identities)) != len(identities):
        raise ValueError("benchmark expands to duplicate experiment IDs")
    return experiments
