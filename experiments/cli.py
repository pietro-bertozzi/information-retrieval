"""Command-line selection and validation for retrieval experiments."""

import argparse
from pathlib import Path
import re

from .configuration import DEFAULT_BENCHMARK_CONFIG, ROOT_DIR, resolve_config_path
from .discovery import discover_datasets, discover_methods, discover_benchmark_methods
from .planning import Experiment, load_benchmark
from .runner import run_benchmark, run_standard


def method_options(values, methods):
    """Forward method-specific scalar options without knowing individual methods."""
    options = {name: [] for name in methods}
    reserved = {"dataset", "split", "top-k", "output-dir", "overwrite", "help"}
    for value in values:
        method, separator, setting = value.partition(":")
        key, equals, argument = setting.partition("=")
        if not separator or not equals or not re.fullmatch(r"[a-z][a-z0-9-]*", key):
            raise ValueError("--method-option must use METHOD:OPTION=VALUE")
        if method not in options:
            raise ValueError(f"--method-option refers to an unselected method: {method}")
        if any(name.startswith(key) for name in reserved):
            raise ValueError(f"--method-option cannot override orchestrator option --{key}")
        options[method].append(f"--{key}={argument}")
    return options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="one prepared dataset; omit to select all runnable datasets")
    parser.add_argument("--method", help="one script stem; omit to select all methods")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="run every dataset/method combination")
    mode.add_argument("--list", action="store_true", help="list datasets and methods without running them")
    parser.add_argument(
        "--benchmark", action="store_true",
        help="run configured variants for methods that expose experiment dimensions",
    )
    parser.add_argument(
        "--benchmark-config", type=Path, metavar="PATH",
        help=("variant configuration relative to the repository "
              f"(default: {DEFAULT_BENCHMARK_CONFIG})"),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="with --benchmark, reuse complete validated outputs instead of rerunning them",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument(
        "--method-option", action="append", default=[], metavar="METHOD:OPTION=VALUE",
        help="pass a method-specific option; may be repeated",
    )
    parser.add_argument("--no-tracking", action="store_true", help="ask evaluation for local reports only")
    parser.add_argument("--tracking-uri", help="forward an MLflow URI to the evaluator")
    parser.add_argument("--log-rankings", action="store_true", help="ask evaluation to upload rankings too")
    args = parser.parse_args(argv)
    if (args.all or args.list) and (args.dataset or args.method):
        parser.error("--all and --list cannot be combined with --dataset or --method")
    if args.resume and not args.benchmark:
        parser.error("--resume requires --benchmark")
    if args.benchmark and args.method:
        parser.error("--benchmark cannot be combined with --method")
    if args.benchmark and args.method_option:
        parser.error("--benchmark cannot be combined with --method-option; edit the configuration")
    if args.benchmark and args.list:
        parser.error("--benchmark cannot be combined with --list; --list shows benchmark discovery")
    if args.benchmark_config and not (args.benchmark or args.list):
        parser.error("--benchmark-config requires --benchmark or --list")
    if not (args.all or args.list or args.dataset or args.method):
        parser.error("select --dataset, --method, --all, or --list")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.no_tracking and (args.tracking_uri or args.log_rankings):
        parser.error("--no-tracking cannot be combined with tracking options")
    try:
        datasets, unavailable = discover_datasets(
            ROOT_DIR / "data-preparation" / "data", args.split
        )
        methods = discover_methods(ROOT_DIR / "modelling" / "scripts")
        if args.dataset:
            if args.dataset in unavailable:
                raise ValueError(
                    f"dataset {args.dataset!r} is not runnable for {args.split}: "
                    f"missing or empty {', '.join(unavailable[args.dataset])}"
                )
            if args.dataset not in datasets:
                raise ValueError(f"unknown dataset {args.dataset!r}; use --list")
            datasets = {args.dataset: datasets[args.dataset]}
        if args.method:
            if args.method not in methods:
                raise ValueError(f"unknown method {args.method!r}; use --list")
            methods = {args.method: methods[args.method]}
        config_path = resolve_config_path(args.benchmark_config, ROOT_DIR)
        if args.list:
            print(f"Runnable datasets ({args.split}):")
            print("\n".join(f"  {name}" for name in datasets) or "  (none)")
            print("Methods (discovered from modelling/scripts):")
            print("\n".join(f"  {name}" for name in methods) or "  (none)")
            benchmark_methods = discover_benchmark_methods(methods)
            print("Benchmark-capable methods:")
            print("\n".join(
                f"  {name}: {', '.join(capability['dimensions'])}"
                for name, capability in benchmark_methods.items()
            ) or "  (none)")
            if config_path.is_file():
                benchmark = load_benchmark(config_path, methods)
                print(f"Benchmark configuration: {config_path}")
                print(f"  {len(benchmark)} configurations")
                for item in benchmark:
                    print(f"  {item.experiment_id}")
            elif args.benchmark_config:
                raise ValueError(f"benchmark configuration does not exist: {config_path}")
            else:
                print(f"Benchmark configuration: not found ({config_path})")
        if not args.dataset:
            for name, missing in unavailable.items():
                print(f"Unavailable dataset {name}: missing or empty {', '.join(missing)}")
        if args.list:
            return 0
        if not datasets:
            raise ValueError(f"no runnable datasets for {args.split}; check data-preparation/data")
        if not methods:
            raise ValueError("no methods found under modelling/scripts")
        if not (ROOT_DIR / "evaluation" / "scripts" / "evaluate.py").is_file():
            raise ValueError("missing evaluator: evaluation/scripts/evaluate.py")
        if args.benchmark:
            if not config_path.is_file():
                raise ValueError(f"benchmark configuration does not exist: {config_path}")
            experiments = load_benchmark(config_path, methods)
        else:
            options = method_options(args.method_option, methods)
            experiments = [
                Experiment(method, script, method, tuple(options[method]))
                for method, script in methods.items()
            ]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Running {len(datasets)} dataset(s) x {len(experiments)} "
        f"{'configuration(s)' if args.benchmark else 'method(s)'}; "
        f"split={args.split}, top_k={args.top_k}", flush=True,
    )
    runner = run_benchmark if args.benchmark else run_standard
    return runner(datasets, experiments, args, ROOT_DIR)
