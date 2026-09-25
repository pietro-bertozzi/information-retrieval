"""Execute retrieval plans and preserve independent benchmark successes."""

from pathlib import Path
import subprocess
import sys

from .artifacts import (
    prepare_output, publish_pair, reusable_benchmark_output, staging_directory,
)
from .evaluation import evaluate_outputs


def _run_retrieval(dataset, experiment, args, directory, root):
    subprocess.run([
        sys.executable, "-B", str(experiment.script), "--dataset", dataset,
        "--split", args.split, "--top-k", str(args.top_k),
        "--output-dir", str(directory), "--overwrite", *experiment.options,
    ], cwd=root, check=True)
    return prepare_output(directory, dataset, args.split, experiment)


def _error_detail(error):
    if isinstance(error, subprocess.CalledProcessError):
        return f"exit code {error.returncode}; see the child process error above"
    return str(error) or "interrupted"


def run_standard(datasets, experiments, args, root):
    """Preserve the original all-or-nothing dataset transaction for manual runs."""
    summaries = []
    exit_code = 0
    for dataset in datasets:
        output_dir = root / "modelling" / "data" / dataset / args.split
        report_dir = root / "evaluation" / "data" / dataset / args.split
        completed = []
        phase = "preparing output directory"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with staging_directory(output_dir) as temp:
                produced = []
                for experiment in experiments:
                    phase = f"retrieval {experiment.experiment_id}"
                    print(f"[{dataset}/{args.split}] {phase}", flush=True)
                    method_dir = Path(temp) / experiment.experiment_id
                    method_dir.mkdir()
                    produced.append((experiment, *_run_retrieval(
                        dataset, experiment, args, method_dir, root
                    )))
                    completed.append(experiment.experiment_id)
                phase = "publishing rankings"
                for experiment, ranking, metadata in produced:
                    publish_pair(ranking, metadata, output_dir, experiment.experiment_id)
            rankings = [output_dir / f"{item.experiment_id}.trec" for item in experiments]
            phase = "evaluation / MLflow logging"
            print(f"[{dataset}/{args.split}] evaluate {len(rankings)} rankings", flush=True)
            evaluate_outputs(dataset, rankings, report_dir, args, root)
            summaries.append(
                f"{dataset}/{args.split}: SUCCEEDED; {len(completed)} methods evaluated\n"
                f"  Methods: {', '.join(completed)}\n"
                f"  Rankings: {output_dir}\n  Reports: {report_dir}"
            )
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
            print(
                f"error: {dataset}/{args.split}: {phase}: {_error_detail(error)}",
                file=sys.stderr,
            )
            summary = (
                f"{dataset}/{args.split}: FAILED during {phase}; retrieval succeeded for "
                f"{len(completed)}/{len(experiments)} methods\n"
                f"  Completed retrieval: {', '.join(completed) or '(none)'}\n"
                f"  Rankings directory: {output_dir}\n  Reports directory: {report_dir}"
            )
            if phase.startswith("retrieval") or phase == "preparing output directory":
                summary += "\n  No staged rankings published; evaluation skipped."
            summaries.append(summary)
            exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
            break
    print("\nSummary:", flush=True)
    for summary in summaries:
        print(summary)
    remaining = list(datasets)[len(summaries):]
    if remaining:
        print(f"Not run after failure: {', '.join(remaining)}")
    return exit_code


def run_benchmark(datasets, experiments, args, root):
    """Run and publish benchmark configurations independently, then evaluate successes."""
    summaries = []
    exit_code = 0
    interrupted = False
    for dataset in datasets:
        output_dir = root / "modelling" / "data" / dataset / args.split
        report_dir = root / "evaluation" / "data" / dataset / args.split
        succeeded = []
        reused = []
        failures = []
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            failures.append(("output directory", _error_detail(error)))
        else:
            for experiment in experiments:
                if args.resume and reusable_benchmark_output(
                    output_dir, dataset, args.split, experiment, args.top_k
                ):
                    print(
                        f"[{dataset}/{args.split}] reuse {experiment.experiment_id}",
                        flush=True,
                    )
                    succeeded.append(experiment)
                    reused.append(experiment.experiment_id)
                    continue
                print(
                    f"[{dataset}/{args.split}] retrieval {experiment.experiment_id}",
                    flush=True,
                )
                try:
                    with staging_directory(output_dir) as temp:
                        ranking, metadata = _run_retrieval(
                            dataset, experiment, args, Path(temp), root
                        )
                        publish_pair(
                            ranking, metadata, output_dir, experiment.experiment_id
                        )
                    succeeded.append(experiment)
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    subprocess.CalledProcessError,
                    KeyboardInterrupt,
                ) as error:
                    detail = _error_detail(error)
                    failures.append((experiment.experiment_id, detail))
                    print(
                        f"error: {dataset}/{args.split}: retrieval "
                        f"{experiment.experiment_id}: {detail}",
                        file=sys.stderr,
                    )
                    if isinstance(error, KeyboardInterrupt):
                        interrupted = True
                        break
        rankings = [output_dir / f"{item.experiment_id}.trec" for item in succeeded]
        if rankings:
            print(f"[{dataset}/{args.split}] evaluate {len(rankings)} rankings", flush=True)
            try:
                evaluate_outputs(dataset, rankings, report_dir, args, root)
            except (OSError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
                detail = _error_detail(error)
                failures.append(("evaluation / MLflow logging", detail))
                print(
                    f"error: {dataset}/{args.split}: evaluation / MLflow logging: {detail}",
                    file=sys.stderr,
                )
                interrupted = interrupted or isinstance(error, KeyboardInterrupt)
        status = "SUCCEEDED" if not failures else "COMPLETED WITH FAILURES"
        summaries.append(
            f"{dataset}/{args.split}: {status}; {len(succeeded)}/{len(experiments)} "
            "configurations available"
            f"\n  Reused: {', '.join(reused) or '(none)'}"
            f"\n  Current outputs submitted to evaluation: "
            f"{', '.join(item.experiment_id for item in succeeded) or '(none)'}"
            f"\n  Failed: "
            f"{'; '.join(f'{name} ({detail})' for name, detail in failures) or '(none)'}"
            f"\n  Rankings: {output_dir}\n  Reports: {report_dir}"
        )
        if failures:
            exit_code = 130 if interrupted else 1
        if interrupted:
            break
    print("\nSummary:", flush=True)
    for summary in summaries:
        print(summary)
    remaining = list(datasets)[len(summaries):]
    if remaining:
        print(f"Not run after interruption: {', '.join(remaining)}")
    return exit_code
