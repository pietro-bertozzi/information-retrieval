"""Regression tests for metric semantics, validation, and report generation."""

import contextlib
import csv
import gzip
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytrec_eval

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate", SCRIPT)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write(self, name, content):
        path = self.directory / name
        path.write_text(content, encoding="utf-8")
        return path

    def score(self, qrels, run, names, relevance_level=1):
        metrics = evaluate.build_metrics(names, relevance_level=relevance_level)
        return evaluate.evaluate_run(qrels, run, metrics, relevance_level)

    def invoke(self, arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate.main([*arguments, "--no-tracking"])

    def test_default_outputs_are_grouped_by_dataset_and_split(self):
        data_dir = self.directory / "prepared"
        output_dir = self.directory / "reports"
        run = self.write("model.trec", "q Q0 a 1 1 model\n")
        for dataset in ("first", "second"):
            for split in ("train", "test"):
                qrels = data_dir / dataset / "qrels" / f"{split}.jsonl"
                qrels.parent.mkdir(parents=True, exist_ok=True)
                qrels.write_text(
                    json.dumps({"query_id": "q", "doc_id": "a", "relevance": 1}),
                    encoding="utf-8",
                )
                with patch.object(evaluate, "DATA_DIR", data_dir):
                    with patch.object(evaluate, "OUTPUT_DIR", output_dir):
                        self.invoke([
                            "--dataset", dataset, "--split", split,
                            "--run", str(run), "--metrics", "AP",
                        ])
                destination = output_dir / dataset / split
                self.assertEqual(
                    {path.name for path in destination.iterdir()},
                    {"report.json", "summary.csv", "per-query.csv"},
                )
                report = json.loads((destination / "report.json").read_text())
                self.assertEqual(report["runs"][0]["aggregate"]["AP"], 1)
        self.assertFalse((output_dir / "report.json").exists())

    def test_known_binary_scores(self):
        qrels = {"q": {"a": 1, "b": 1, "z": 0}}
        run = {"q": {"a": 3.0, "z": 2.0, "b": 1.0}}
        result, _ = self.score(qrels, run, ["AP", "RR", "P@2", "R@2", "nDCG@3"])
        scores = result["aggregate"]
        self.assertAlmostEqual(scores["AP"], (1 + 2 / 3) / 2)
        self.assertEqual(scores["RR"], 1)
        self.assertEqual(scores["P@2"], 0.5)
        self.assertEqual(scores["R@2"], 0.5)
        self.assertAlmostEqual(scores["nDCG@3"], 1.5 / (1 + 1 / math.log2(3)))

    def test_missing_query_counts_as_zero(self):
        result, per_query = self.score(
            {"q1": {"a": 1}, "q2": {"b": 1}},
            {"q1": {"a": 1.0}}, ["AP", "RR@10"],
        )
        self.assertEqual(result["aggregate"]["AP"], 0.5)
        self.assertEqual(per_query["q2"]["AP"], 0)
        self.assertEqual(result["coverage"]["missing_judged_queries"], 1)
        self.assertEqual(result["coverage"]["judged_query_coverage"], 0.5)

    def test_macro_average_is_not_weighted_by_relevant_count(self):
        result, _ = self.score(
            {"q1": {"a": 1}, "q2": {"b": 1, "c": 1, "d": 1}},
            {"q1": {"a": 1.0}}, ["Recall@10"],
        )
        self.assertEqual(result["aggregate"]["R@10"], 0.5)

    def test_unjudged_queries_are_excluded(self):
        result, per_query = self.score(
            {"q": {"a": 1}}, {"q": {"a": 1.0}, "unknown": {"z": 9.0}}, ["AP"],
        )
        self.assertEqual(result["aggregate"]["AP"], 1)
        self.assertEqual(set(per_query), {"q"})
        self.assertEqual(result["coverage"]["ignored_unjudged_queries"], 1)

    def test_no_relevant_documents_score_zero(self):
        metrics = evaluate.build_metrics()
        result, _ = evaluate.evaluate_run({"q": {"a": 0}}, {"q": {"a": 1.0}}, metrics)
        for name, value in result["aggregate"].items():
            self.assertEqual(value, 1 if name.startswith("Judged") else 0, name)

    def test_empty_run_scores_zero_for_entire_default_suite(self):
        result, per_query = evaluate.evaluate_run(
            {"q": {"a": 1}}, {}, evaluate.build_metrics(),
        )
        self.assertEqual(len(result["aggregate"]), 57)
        self.assertTrue(all(value == 0 for value in result["aggregate"].values()))
        self.assertEqual(set(per_query), {"q"})

    def test_ties_use_reverse_document_id_for_every_backend(self):
        result, _ = self.score(
            {"q": {"a": 1, "z": 0}}, {"q": {"a": 4.0, "z": 4.0}},
            ["RR", "RR@1", "RR@2", "P@1", "nDCG@1"],
        )
        self.assertEqual(result["aggregate"], {
            "RR": 0.5, "RR@1": 0, "RR@2": 0.5, "P@1": 0, "nDCG@1": 0,
        })

    def test_graded_ndcg_uses_linear_gain(self):
        result, _ = self.score(
            {"q": {"a": 3, "b": 1}}, {"q": {"b": 2.0, "a": 1.0}}, ["nDCG@2"],
        )
        expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(result["aggregate"]["nDCG@2"], expected)

    def test_relevance_threshold_changes_binary_metrics_only(self):
        qrels = {"q": {"a": 2, "b": 1}}
        run = {"q": {"b": 2.0, "a": 1.0}}
        first, _ = self.score(qrels, run, ["P@1", "RR@1", "R@1", "nDCG@2"])
        second, _ = self.score(qrels, run, ["P@1", "RR@1", "R@1", "nDCG@2"], 2)
        self.assertEqual(first["aggregate"]["P@1"], 1)
        self.assertEqual(second["aggregate"]["P(rel=2)@1"], 0)
        self.assertEqual(second["aggregate"]["RR(rel=2)@1"], 0)
        self.assertEqual(second["aggregate"]["R(rel=2)@1"], 0)
        self.assertEqual(first["aggregate"]["nDCG@2"], second["aggregate"]["nDCG@2"])

    def test_unjudged_documents_occupy_ranks(self):
        result, _ = self.score(
            {"q": {"a": 1}}, {"q": {"unknown": 2.0, "a": 1.0}},
            ["RR@2", "Judged@2"],
        )
        self.assertEqual(result["aggregate"]["RR@2"], 0.5)
        self.assertEqual(result["aggregate"]["Judged@2"], 0.5)

    def test_short_run_precision_uses_requested_cutoff(self):
        result, _ = self.score({"q": {"a": 1}}, {"q": {"a": 1.0}}, ["P@10", "Judged@10"])
        self.assertEqual(result["aggregate"]["P@10"], 0.1)
        self.assertEqual(result["aggregate"]["Judged@10"], 1)

    def test_scores_match_direct_trec_eval(self):
        qrels = {"q": {"a": 2, "b": 1, "c": 0}, "r": {"x": 1, "y": 0}}
        run = {"q": {"c": 4.0, "a": 3.0, "b": 1.0}, "r": {"y": 4.0, "x": 2.0}}
        mapping = {"AP": "map", "RR": "recip_rank", "nDCG@10": "ndcg_cut_10",
                   "P@10": "P_10", "R@10": "recall_10", "Rprec": "Rprec", "Bpref": "bpref"}
        _, per_query = self.score(qrels, run, list(mapping))
        expected = pytrec_eval.RelevanceEvaluator(qrels, set(mapping.values())).evaluate(run)
        for query_id, values in per_query.items():
            for metric, value in values.items():
                self.assertAlmostEqual(value, expected[query_id][mapping[metric]])

    def test_aliases_and_duplicate_metrics(self):
        metrics = evaluate.build_metrics(["MAP", "AP", "MRR@10", "Precision@5", "Recall@100"])
        self.assertEqual(list(map(str, metrics)), ["AP", "RR@10", "P@5", "R@100"])

    def test_invalid_metrics_and_cutoffs(self):
        for name in ("unknown", "P", "R", "AP@0", "RR@-1", "Rprec@10", "__import__('os')"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                evaluate.build_metrics([name])
        with self.assertRaises(ValueError):
            evaluate.build_metrics(cutoffs=[0])
        with self.assertRaises(ValueError):
            evaluate.build_metrics(relevance_level=0)

    def test_trec_and_jsonl_inputs_match_and_preserve_ids(self):
        trec = self.write("qrels.txt", "001 0 002 1\n001 0 003 0\n")
        jsonl = self.write("qrels.jsonl", '\ufeff{"query_id":"001","doc_id":"002","relevance":1}\n'
                           '{"query_id":"001","doc_id":"003","relevance":0}\n')
        self.assertEqual(evaluate.read_qrels(trec), evaluate.read_qrels(jsonl))
        self.assertIn("001", evaluate.read_qrels(trec))
        run_trec = self.write("run.trec", "001 Q0 002 2 0.25 model\n001 Q0 003 1 0.1 model\n")
        run_jsonl = self.write("run.jsonl", '{"query_id":"001","doc_id":"002","score":0.25}\n'
                               '{"query_id":"001","doc_id":"003","score":0.1}\n')
        self.assertEqual(evaluate.read_run(run_trec), evaluate.read_run(run_jsonl))
        result, _ = self.score(evaluate.read_qrels(trec), evaluate.read_run(run_trec), ["P@1"])
        self.assertEqual(result["aggregate"]["P@1"], 1)

    def test_gzip_and_explicit_format(self):
        path = self.directory / "judgments.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write('{"query_id":"q","doc_id":"a","relevance":1}\n')
        self.assertEqual(evaluate.read_qrels(path), {"q": {"a": 1}})
        renamed = self.write("qrels.data", '{"query_id":"q","doc_id":"a","relevance":1}\n')
        self.assertEqual(evaluate.read_qrels(renamed, "jsonl"), {"q": {"a": 1}})

    def test_minus_one_is_unjudged(self):
        path = self.write("qrels.txt", "q 0 a 1\nq 0 b -1\nu 0 c -1\n")
        self.assertEqual(evaluate.read_qrels(path), {"q": {"a": 1}})

    def test_empty_or_unjudged_only_qrels_rejected(self):
        for content in ("", "q 0 a -1\n"):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "no judged"):
                evaluate.read_qrels(self.write("qrels.txt", content))

    def test_duplicate_judgments_and_results_rejected(self):
        for content, reader in (("q 0 a 1\nq 0 a 1\n", evaluate.read_qrels),
                                ("q Q0 a 1 2 m\nq Q0 a 2 1 m\n", evaluate.read_run)):
            with self.subTest(reader=reader), self.assertRaisesRegex(ValueError, "duplicate"):
                reader(self.write("input.txt", content))

    def test_malformed_jsonl_has_line_context(self):
        bad_rows = ['{}', '[]', '{broken',
                    '{"query_id":2,"doc_id":"a","score":1}',
                    '{"query_id":"q q","doc_id":"a","score":1}',
                    '{"query_id":"q","doc_id":"a","score":NaN}',
                    '{"query_id":"q","doc_id":"a","score":true}']
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, r'input.jsonl:2:'):
                evaluate.read_run(self.write("input.jsonl", "\n" + row + "\n"))

    def test_invalid_trec_and_grades(self):
        for row in ("q Q0 a 1 NaN m", "q Q0 a 0 1 m", "q Q0 a 1 1", "q Q0 a x 1 m"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                evaluate.read_run(self.write("run.trec", row))
        for grade in (-2, 0.5, True, 2147483648):
            row = json.dumps({"query_id": "q", "doc_id": "a", "relevance": grade})
            with self.subTest(grade=grade), self.assertRaises(ValueError):
                evaluate.read_qrels(self.write("qrels.jsonl", row))

    def test_mixed_run_tags_rejected(self):
        path = self.write("run.trec", "q Q0 a 1 1 first\nq Q0 b 2 0 second\n")
        with self.assertRaisesRegex(ValueError, "one run tag"):
            evaluate.read_run(path)

    def test_cli_multiple_runs_reports_and_overwrite(self):
        qrels = self.write("qrels.txt", "q 0 a 1\nr 0 b 1\n")
        run = self.write("model.trec", "q Q0 a 1 5 model\n")
        empty = self.write("empty.trec", "")
        output = self.directory / "reports"
        arguments = ["--qrels", str(qrels), "--run", str(run), str(empty),
                     "--metrics", "AP", "RR@10", "--output-dir", str(output)]
        self.invoke(arguments)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual([r["aggregate"]["AP"] for r in report["runs"]], [0.5, 0])
        with (output / "per-query.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        with (output / "summary.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            self.invoke(arguments)
        self.assertEqual(error.exception.code, 2)
        self.invoke(arguments + ["--overwrite"])

    def test_invalid_later_run_preserves_existing_reports(self):
        qrels = self.write("qrels.txt", "q 0 a 1\n")
        run = self.write("model.trec", "q Q0 a 1 5 model\n")
        output = self.directory / "reports"
        arguments = ["--qrels", str(qrels), "--run", str(run),
                     "--metrics", "AP", "--output-dir", str(output)]
        self.invoke(arguments)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        malformed = self.write("bad.trec", "invalid\n")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(["--qrels", str(qrels), "--run", str(run), str(malformed),
                         "--metrics", "AP", "--output-dir", str(output), "--overwrite"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_output_cannot_overwrite_input(self):
        qrels = self.write("qrels.txt", "q 0 a 1\n")
        run = self.write("summary.csv", "q Q0 a 1 5 model\n")
        original = run.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(["--qrels", str(qrels), "--run", str(run),
                         "--output-dir", str(self.directory), "--overwrite"])
        self.assertEqual(original, run.read_bytes())


if __name__ == "__main__":
    unittest.main()
