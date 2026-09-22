"""Build an offline dashboard from evaluation reports, without computing metrics."""

import argparse
import csv
import hashlib
import json
import math
import tempfile
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def json_text(value):
    """Encode data safely for both inline HTML and external JavaScript."""
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expected a finite number, got {value!r}")
    return value


def load_report(path):
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if report.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported report schema")
    metrics = report.get("metrics")
    runs = report.get("runs")
    if not isinstance(metrics, list) or not metrics or any(not isinstance(m, str) or not m for m in metrics):
        raise ValueError(f"{path}: expected nonempty metric names")
    if len(set(metrics)) != len(metrics) or not isinstance(runs, list) or not runs:
        raise ValueError(f"{path}: duplicate metrics or missing runs")
    names = set()
    for run in runs:
        name = run.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"{path}: missing or duplicate run name")
        names.add(name)
        for metric in metrics:
            number(run["aggregate"][metric])
        count = run["coverage"]["judged_queries"]
        if type(count) is not int or count < 1:
            raise ValueError(f"{path}: invalid judged query count")
    return report


def write_queries(report_path, report, staging):
    """Stream one CSV to separate model payloads; retain only IDs for validation."""
    metrics = report["metrics"]
    runs = report["runs"]
    by_name = {run["name"]: i for i, run in enumerate(runs)}
    seen = [set() for _ in runs]
    temporary_paths = [staging / f"query-{i}.tmp" for i in range(len(runs))]
    csv_path = report_path.with_name("per-query.csv")
    with ExitStack() as stack:
        outputs = [stack.enter_context(path.open("w", encoding="utf-8", newline="\n")) for path in temporary_paths]
        for stream in outputs:
            stream.write("window.receiveQueries([\n")
        source = stack.enter_context(csv_path.open(encoding="utf-8-sig", newline=""))
        reader = csv.DictReader(source)
        if reader.fieldnames != ["run", "query_id", *metrics]:
            raise ValueError(f"{csv_path}: columns do not match report metrics")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{csv_path}:{line}: malformed CSV row")
            if row["run"] not in by_name:
                raise ValueError(f"{csv_path}:{line}: unknown run")
            index = by_name[row["run"]]
            query = row["query_id"]
            if not query or any(c.isspace() for c in query) or query in seen[index]:
                raise ValueError(f"{csv_path}:{line}: empty or duplicate query ID")
            values = [number(float(row[metric])) for metric in metrics]
            if seen[index]:
                outputs[index].write(",\n")
            outputs[index].write(json_text([query, *values]))
            seen[index].add(query)
        for stream in outputs:
            stream.write("\n]);\n")
    for index, run in enumerate(runs):
        if len(seen[index]) != run["coverage"]["judged_queries"]:
            raise ValueError(f"{csv_path}: query count mismatch for {run['name']}")
        if seen[index] != seen[0]:
            raise ValueError(f"{csv_path}: inconsistent query populations")
        path = temporary_paths[index]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        filename = f"queries-{digest}.js"
        destination = staging / filename
        if destination.exists():
            path.unlink()
        else:
            path.replace(destination)
        run["query_file"] = filename
        # Absolute source paths are not needed in the portable dashboard.
        run.pop("path", None)


def build(input_dir, output_dir, overwrite=False):
    paths = sorted(input_dir.rglob("report.json"))
    if not paths:
        raise ValueError(f"no report.json files found under {input_dir}")
    if output_dir.resolve() == input_dir.resolve() or input_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("output must be outside the evaluation input directory")
    target = output_dir / "index.html"
    if target.exists() and not overwrite:
        raise ValueError(f"{target} exists; use --overwrite to replace it")
    # Validate and prepare all reports before touching existing dashboard files.
    with tempfile.TemporaryDirectory(prefix="ir-dashboard-") as temporary:
        staging = Path(temporary)
        reports = []
        for path in paths:
            report = load_report(path)
            report["source"] = path.relative_to(input_dir).as_posix()
            write_queries(path, report, staging)
            report.pop("qrels", None)
            reports.append(report)
        template = (ASSETS / "index.html").read_text(encoding="utf-8")
        html = template.replace("/* DASHBOARD_CSS */", (ASSETS / "dashboard.css").read_text(encoding="utf-8"))
        html = html.replace("/* DASHBOARD_JS */", (ASSETS / "dashboard.js").read_text(encoding="utf-8"))
        html = html.replace("__REPORT_DATA__", json_text(reports))
        output_dir.mkdir(parents=True, exist_ok=True)
        # Content-addressed payloads leave the old index usable during a rebuild.
        for path in staging.glob("queries-*.js"):
            destination = output_dir / path.name
            if not destination.exists():
                with path.open("rb") as source, destination.open("wb") as stream:
                    while chunk := source.read(1024 * 1024):
                        stream.write(chunk)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, suffix=".tmp", delete=False) as stream:
            pending = Path(stream.name)
            stream.write(html)
        try:
            pending.replace(target)
        finally:
            pending.unlink(missing_ok=True)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "evaluation" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "visualization" / "data")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        reports = build(args.input_dir, args.output_dir, args.overwrite)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Built {len(reports)} reports with {sum(len(r['runs']) for r in reports)} model runs.")
    print(f"Open {(args.output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
