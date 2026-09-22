"""Regression tests for report validation and portable dashboard generation."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("dashboard_build", Path(__file__).resolve().parents[1] / "scripts" / "build.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "evaluation"
        self.output = self.root / "visualization"
        self.report_dir = self.inputs / "example" / "test"
        self.report_dir.mkdir(parents=True)
        self.report = {
            "schema_version": 1, "dataset": "example", "split": "test",
            "metrics": ["AP", "nDCG@10"],
            "runs": [{"name": "model.trec", "aggregate": {"AP": 0.5, "nDCG@10": 0.75},
                      "coverage": {"judged_queries": 2}, "path": "/private/run.trec"}],
            "qrels": "/private/qrels.jsonl",
        }
        self.rows = [["model.trec", "001", "1", "1"], ["model.trec", "002", "0", "0.5"]]
        self.save()

    def save(self):
        (self.report_dir / "report.json").write_text(json.dumps(self.report), encoding="utf-8")
        with (self.report_dir / "per-query.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run", "query_id", "AP", "nDCG@10"])
            writer.writerows(self.rows)

    def test_preserves_values_and_ids_and_separates_query_payload(self):
        reports = builder.build(self.inputs, self.output)
        run = reports[0]["runs"][0]
        self.assertEqual(run["aggregate"], self.report["runs"][0]["aggregate"])
        payload = (self.output / run["query_file"]).read_text()
        rows = json.loads(payload.removeprefix("window.receiveQueries(").removesuffix(");\n"))
        self.assertEqual(rows, [["001", 1.0, 1.0], ["002", 0.0, 0.5]])
        html = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("/private/", html)
        self.assertNotIn("/* DASHBOARD_JS */", html)
        self.assertNotIn('"001"', html)
        self.assertIn("Explore your evaluations.", html)

    def test_discovers_multiple_datasets(self):
        other = self.inputs / "other" / "train"
        other.mkdir(parents=True)
        for path in self.report_dir.iterdir():
            (other / path.name).write_bytes(path.read_bytes())
        self.assertEqual(len(builder.build(self.inputs, self.output)), 2)

    def test_escapes_script_terminators(self):
        self.report["dataset"] = '</script><script>alert("x")</script>'
        self.save()
        builder.build(self.inputs, self.output)
        html = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertNotIn(self.report["dataset"], html)
        self.assertIn(r"\u003c/script>", html)

    def test_refuses_overwrite_without_flag(self):
        builder.build(self.inputs, self.output)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            builder.build(self.inputs, self.output)

    def test_invalid_rebuild_preserves_previous_dashboard(self):
        builder.build(self.inputs, self.output)
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.rows.append(self.rows[0])
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            builder.build(self.inputs, self.output, overwrite=True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_rejects_unknown_schema_and_nonfinite_values(self):
        self.report["schema_version"] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, "schema"):
            builder.build(self.inputs, self.output)
        self.report["schema_version"] = 1
        self.report["runs"][0]["aggregate"]["AP"] = float("nan")
        self.save()
        with self.assertRaisesRegex(ValueError, "finite"):
            builder.build(self.inputs, self.output)

    def test_rejects_missing_queries_and_nonfinite_query_values(self):
        self.rows.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            builder.build(self.inputs, self.output)
        self.rows[0][2] = "inf"
        self.save()
        with self.assertRaisesRegex(ValueError, "finite"):
            builder.build(self.inputs, self.output)

    def test_rejects_unknown_model_and_wrong_columns(self):
        self.rows[0][0] = "unknown"
        self.save()
        with self.assertRaisesRegex(ValueError, "unknown run"):
            builder.build(self.inputs, self.output)
        (self.report_dir / "per-query.csv").write_text("run,query_id,AP\n")
        with self.assertRaisesRegex(ValueError, "columns"):
            builder.build(self.inputs, self.output)

    def test_rejects_different_query_populations(self):
        self.report["runs"].append({"name": "second", "aggregate": {"AP": 0.5, "nDCG@10": 0.5}, "coverage": {"judged_queries": 2}})
        self.rows.extend([["second", "001", "0", "0"], ["second", "003", "1", "1"]])
        self.save()
        with self.assertRaisesRegex(ValueError, "populations"):
            builder.build(self.inputs, self.output)

    def test_rebuild_reuses_payload_and_preserves_unrelated_files(self):
        builder.build(self.inputs, self.output)
        (self.output / "notes.txt").write_text("keep")
        payloads = sorted(p.name for p in self.output.glob("queries-*.js"))
        builder.build(self.inputs, self.output, overwrite=True)
        self.assertEqual(payloads, sorted(p.name for p in self.output.glob("queries-*.js")))
        self.assertEqual((self.output / "notes.txt").read_text(), "keep")

    def test_rejects_missing_reports_and_output_inside_inputs(self):
        with self.assertRaisesRegex(ValueError, "no report"):
            builder.build(self.root / "missing", self.output)
        with self.assertRaisesRegex(ValueError, "outside"):
            builder.build(self.inputs, self.inputs / "dashboard")


if __name__ == "__main__":
    unittest.main()
