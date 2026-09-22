"""Run discovered retrieval methods and evaluate their outputs by dataset."""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT_DIR = Path(__file__).resolve().parent


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


def run_selected(datasets, methods, args, options):
    summaries = []
    exit_code = 0
    evaluator = ROOT_DIR / "evaluation" / "scripts" / "evaluate.py"
    for dataset in datasets:
        output_dir = ROOT_DIR / "modelling" / "data" / dataset / args.split
        report_dir = ROOT_DIR / "evaluation" / "data" / dataset / args.split
        completed = []
        phase = "preparing output directory"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".experiment-", dir=output_dir) as temp:
                produced = []
                for method, script in methods.items():
                    phase = f"retrieval {method}"
                    print(f"[{dataset}/{args.split}] {phase}", flush=True)
                    method_dir = Path(temp) / method
                    method_dir.mkdir()
                    subprocess.run([
                        sys.executable, "-B", str(script), "--dataset", dataset,
                        "--split", args.split, "--top-k", str(args.top_k),
                        "--output-dir", str(method_dir), "--overwrite", *options[method],
                    ], cwd=ROOT_DIR, check=True)
                    ranking = method_dir / f"{method}.trec"
                    metadata = method_dir / f"{method}.metadata.json"
                    if not ranking.is_file() or not metadata.is_file():
                        raise RuntimeError(
                            f"method must create {ranking.name} and {metadata.name} "
                            "in its supplied --output-dir"
                        )
                    if metadata.stat().st_size == 0:
                        raise RuntimeError(f"empty metadata: {metadata.name}")
                    # Empty rankings are valid: the evaluator scores missing results zero.
                    produced.append((ranking, metadata))
                    completed.append(method)
                phase = "publishing rankings"
                for ranking, metadata in produced:
                    ranking.replace(output_dir / ranking.name)
                    metadata.replace(output_dir / metadata.name)
            rankings = [output_dir / f"{method}.trec" for method in methods]
            phase = "evaluation / MLflow logging"
            print(f"[{dataset}/{args.split}] evaluate {len(rankings)} rankings", flush=True)
            command = [
                sys.executable, "-B", str(evaluator), "--dataset", dataset,
                "--split", args.split, "--run", *map(str, rankings),
                "--output-dir", str(report_dir), "--overwrite",
            ]
            if args.no_tracking:
                command.append("--no-tracking")
            if args.tracking_uri:
                command.extend(["--tracking-uri", args.tracking_uri])
            if args.log_rankings:
                command.append("--log-rankings")
            subprocess.run(command, cwd=ROOT_DIR, check=True)
            summaries.append(
                f"{dataset}/{args.split}: SUCCEEDED; {len(completed)} methods evaluated\n"
                f"  Methods: {', '.join(completed)}\n"
                f"  Rankings: {output_dir}\n  Reports: {report_dir}"
            )
        except (OSError, RuntimeError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
            if isinstance(error, subprocess.CalledProcessError):
                detail = f"exit code {error.returncode}; see the child process error above"
            else:
                detail = str(error) or "interrupted"
            print(f"error: {dataset}/{args.split}: {phase}: {detail}", file=sys.stderr)
            summaries.append(
                f"{dataset}/{args.split}: FAILED during {phase}; "
                f"retrieval succeeded for {len(completed)}/{len(methods)} methods\n"
                f"  Completed retrieval: {', '.join(completed) or '(none)'}\n"
                f"  Rankings directory: {output_dir}\n  Reports directory: {report_dir}"
            )
            if phase.startswith("retrieval") or phase == "preparing output directory":
                summaries[-1] += "\n  No staged rankings published; evaluation skipped."
            exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
            break
    print("\nSummary:", flush=True)
    for summary in summaries:
        print(summary)
    remaining = list(datasets)[len(summaries):]
    if remaining:
        print(f"Not run after failure: {', '.join(remaining)}")
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="one prepared dataset; omit to select all runnable datasets")
    parser.add_argument("--method", help="one script stem; omit to select all methods")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="run every dataset/method combination")
    mode.add_argument("--list", action="store_true", help="list datasets and methods without running them")
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
    if not (args.all or args.list or args.dataset or args.method):
        parser.error("select --dataset, --method, --all, or --list")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.no_tracking and (args.tracking_uri or args.log_rankings):
        parser.error("--no-tracking cannot be combined with tracking options")
    try:
        datasets, unavailable = discover_datasets(ROOT_DIR / "data-preparation" / "data", args.split)
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
        if args.list:
            print(f"Runnable datasets ({args.split}):")
            print("\n".join(f"  {name}" for name in datasets) or "  (none)")
            print("Methods:")
            print("\n".join(f"  {name}" for name in methods) or "  (none)")
        if not args.dataset:
            for name, missing in unavailable.items():
                print(f"Unavailable dataset {name}: missing or empty {', '.join(missing)}")
        if args.list:
            return 0
        if not datasets:
            raise ValueError(f"no runnable datasets for {args.split}; check data-preparation/data")
        if not methods:
            raise ValueError("no methods found under modelling/scripts")
        if not (ROOT_DIR / "evaluation/scripts/evaluate.py").is_file():
            raise ValueError("missing evaluator: evaluation/scripts/evaluate.py")
        options = method_options(args.method_option, methods)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Running {len(datasets)} dataset(s) x {len(methods)} method(s); "
        f"split={args.split}, top_k={args.top_k}", flush=True,
    )
    return run_selected(datasets, methods, args, options)


if __name__ == "__main__":
    sys.exit(main())
