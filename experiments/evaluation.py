"""Invoke the existing evaluator, which owns scoring and MLflow logging."""

import subprocess
import sys


def evaluation_command(dataset, rankings, report_dir, args, root):
    command = [
        sys.executable, "-B", str(root / "evaluation" / "scripts" / "evaluate.py"),
        "--dataset", dataset, "--split", args.split, "--run", *map(str, rankings),
        "--output-dir", str(report_dir), "--overwrite",
    ]
    if args.no_tracking:
        command.append("--no-tracking")
    if args.tracking_uri:
        command.extend(["--tracking-uri", args.tracking_uri])
    if args.log_rankings:
        command.append("--log-rankings")
    return command


def evaluate_outputs(dataset, rankings, report_dir, args, root):
    subprocess.run(
        evaluation_command(dataset, rankings, report_dir, args, root),
        cwd=root, check=True,
    )
