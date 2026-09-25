"""Integration checks for MLflow persistence without a server or real datasets."""

import contextlib
import csv
import gzip
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("tracked_evaluate", SCRIPT)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.tracking = cls.root / "tracking"
        cls.tracking.mkdir()
        cls.uri = f"sqlite:///{(cls.tracking / 'mlflow.db').as_posix()}"
        cls.client = MlflowClient(tracking_uri=cls.uri)

    @classmethod
    def tearDownClass(cls):
        # Release SQLite handles before removing the temporary directory on Windows.
        cls.client._tracking_client.store._dispose_engine()
        cls.temporary.cleanup()

    def setUp(self):
        self.directory = self.root / self._testMethodName
        self.directory.mkdir()
        self.dataset = self._testMethodName
        self.qrels = self.directory / "qrels.txt"
        self.qrels.write_text("001 0 a 2\n002 0 b 1\n", encoding="utf-8")
        self.run = self.directory / "model.trec"
        self.run.write_text("001 Q0 a 1 4 model\n", encoding="utf-8")
        self.output = self.directory / "reports"
        self.arguments = [
            "--qrels", str(self.qrels), "--dataset", self.dataset, "--split", "test",
            "--run", str(self.run), "--output-dir", str(self.output),
        ]
        self.addCleanup(patch.stopall)
        patch.object(evaluate, "TRACKING_DIR", self.tracking).start()
        patch.dict(os.environ, {"MLFLOW_TRACKING_URI": ""}).start()

    def invoke(self, arguments=None):
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate.main(arguments or self.arguments)

    def runs(self, dataset=None):
        experiment = self.client.get_experiment_by_name(dataset or self.dataset)
        if experiment is None:
            return []
        return self.client.search_runs([experiment.experiment_id])

    def artifact(self, run_id, name):
        destination = self.directory / "downloads"
        destination.mkdir(exist_ok=True)
        return Path(self.client.download_artifacts(run_id, name, str(destination)))

    def test_multiple_runs_metrics_and_artifacts_match_local_results(self):
        metadata = {"dataset": self.dataset, "split": "test", "parameters": {
            "method": "example", "version": "1", "top_k": 1000, "language": "it",
        }}
        self.run.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        empty = self.directory / "empty.trec"
        empty.write_text("", encoding="utf-8")
        self.invoke(self.arguments + ["--run", str(self.run), str(empty)])
        report = json.loads((self.output / "report.json").read_text())
        runs = self.runs()
        self.assertEqual(len(runs), 2)
        by_name = {run.data.tags["mlflow.runName"].split(" - ")[0]: run for run in runs}
        for result in report["runs"]:
            run = by_name[result["name"]]
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(run.data.tags["dataset"], self.dataset)
            self.assertEqual(run.data.tags["split"], "test")
            expected = {evaluate.metric_key(k): v for k, v in result["aggregate"].items()}
            expected.update({f"coverage.{k}": v for k, v in result["coverage"].items()})
            self.assertEqual(run.data.metrics, expected)
            artifact = json.loads(self.artifact(run.info.run_id, "report.json").read_text())
            self.assertEqual(artifact["runs"], [result])
            with self.artifact(run.info.run_id, "per-query.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["query_id"] for row in rows], ["001", "002"])
            self.assertEqual({row["run"] for row in rows}, {result["name"]})
            self.assertEqual(float(rows[1]["AP"]), 0)
            with self.artifact(run.info.run_id, "summary.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            mapping = json.loads(self.artifact(run.info.run_id, "metric-names.json").read_text())
            self.assertEqual(mapping["nDCG@10"], "ir.nDCG_at_10")
        self.assertEqual(by_name["model.trec"].data.params["retrieval.top_k"], "1000")
        self.assertEqual(by_name["model.trec"].data.params["retrieval.language"], "it")
        stored = json.loads(self.artifact(by_name["model.trec"].info.run_id, "retrieval.json").read_text())
        self.assertEqual(stored, metadata)
        self.assertNotIn("retrieval.top_k", by_name["empty.trec"].data.params)
        self.assertEqual(by_name["empty.trec"].data.tags["retrieval.metadata"], "unknown")
        self.assertNotIn("rankings", {a.path for a in self.client.list_artifacts(by_name["model.trec"].info.run_id)})

    def test_dense_model_is_a_run_parameter_in_the_dataset_experiment(self):
        metadata = {"dataset": self.dataset, "split": "test", "parameters": {
            "method": "dense-retrieval", "version": "2.1.1", "top_k": 10,
            "embedding_runtime": "FastEmbed", "embedding_model": "example/multilingual",
            "vector_dimension": 384, "distance": "Cosine", "qdrant_collection": "example",
        }}
        self.run.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.invoke()
        run = self.runs()[0]
        self.assertEqual(run.data.params["retrieval.embedding_model"], "example/multilingual")
        self.assertEqual(run.data.params["retrieval.vector_dimension"], "384")
        self.assertEqual(run.data.tags["dataset"], self.dataset)

    def test_benchmark_configurations_have_distinct_run_names_and_identity_parameters(self):
        rankings = []
        for identifier, model in (("dense-a", "example/a"), ("dense-b", "example/b")):
            ranking = self.directory / f"2.1.1-dense-retrieval__{identifier}.trec"
            ranking.write_bytes(self.run.read_bytes())
            metadata = {
                "dataset": self.dataset,
                "split": "test",
                "parameters": {
                    "method": "dense-retrieval",
                    "retrieval_method": "2.1.1-dense-retrieval",
                    "experiment_id": ranking.stem,
                    "embedding_model": model,
                    "top_k": 10,
                },
            }
            ranking.with_suffix(".metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            rankings.append(ranking)
        self.invoke([
            "--qrels", str(self.qrels), "--dataset", self.dataset, "--split", "test",
            "--run", *map(str, rankings), "--output-dir", str(self.output),
        ])
        runs = self.runs()
        self.assertEqual(
            {run.data.tags["mlflow.runName"] for run in runs},
            {f"{ranking.name} - test" for ranking in rankings},
        )
        by_id = {run.data.params["retrieval.experiment_id"]: run for run in runs}
        self.assertEqual(set(by_id), {ranking.stem for ranking in rankings})
        self.assertEqual(by_id[rankings[0].stem].data.params["retrieval.embedding_model"], "example/a")
        self.assertEqual(by_id[rankings[1].stem].data.params["retrieval.embedding_model"], "example/b")

    def test_nondefault_relevance_and_optional_ranking_artifact(self):
        self.invoke(self.arguments + ["--relevance-level", "2", "--log-rankings"])
        run = self.runs()[0]
        self.assertEqual(run.data.metrics["ir.AP_rel_2_at_10"], 0.5)
        self.assertEqual(run.data.params["evaluation.relevance_level"], "2")
        self.assertEqual(self.artifact(run.info.run_id, "rankings/model.trec").read_bytes(), self.run.read_bytes())

    def test_prepared_dataset_experiments_and_source_splits(self):
        for dataset, source in (("jurifindit", "validation"), ("scifact", "test"), ("msmarco", "dev"), ("mmarco-it", "dev")):
            for split in ("test", "train"):
                path = self.directory / dataset / "qrels" / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"query_id": "001", "doc_id": "a", "relevance": 1}), encoding="utf-8")
                with patch.object(evaluate, "DATA_DIR", self.directory):
                    self.invoke(["--dataset", dataset, "--split", split, "--run", str(self.run),
                                 "--metrics", "AP", "--output-dir", str(self.output), "--overwrite"])
            runs = self.runs(dataset)
            self.assertEqual(len(runs), 2)
            self.assertEqual({r.data.tags["split"] for r in runs}, {"test", "train"})
            self.assertEqual({r.data.tags["source_split"] for r in runs}, {source, "train"})

    def test_repeated_evaluation_preserves_prior_mlflow_runs(self):
        self.invoke()
        original = self.runs()[0].info.run_id
        self.invoke(self.arguments + ["--overwrite"])
        self.assertEqual(len(self.runs()), 2)
        self.assertIn(original, {r.info.run_id for r in self.runs()})

    def test_external_identity_and_tracking_options_are_validated(self):
        cases = [
            ["--qrels", str(self.qrels), "--run", str(self.run)],
            ["--qrels", str(self.qrels), "--dataset", self.dataset, "--run", str(self.run)],
            self.arguments + ["--no-tracking", "--log-rankings"],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(arguments)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.runs(), [])

    def test_bad_sidecars_and_later_inputs_do_not_log_or_replace_reports(self):
        self.invoke(self.arguments + ["--no-tracking"])
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        sidecar = self.run.with_suffix(".metadata.json")
        for metadata in ({"parameters": {}}, {"dataset": self.dataset, "split": "test", "parameters": {"top_k": 0}}, {"dataset": self.dataset, "split": "test", "parameters": {"x": float("nan")}}):
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(self.arguments + ["--overwrite"])
        sidecar.unlink()
        bad = self.directory / "bad.trec"
        bad.write_text("invalid", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(self.arguments + ["--run", str(self.run), str(bad), "--overwrite"])
        self.assertEqual(self.runs(), [])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_local_only_does_not_call_tracking(self):
        with patch.object(evaluate, "log_tracking") as logger:
            self.invoke(self.arguments + ["--no-tracking"])
        logger.assert_not_called()
        self.assertTrue((self.output / "report.json").exists())
        self.assertEqual(self.runs(), [])

    def test_upload_failure_keeps_reports_and_marks_run_failed(self):
        with patch.object(MlflowClient, "log_artifacts", side_effect=MlflowException("upload unavailable")):
            with contextlib.redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as failure:
                self.invoke()
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("Local reports remain", errors.getvalue())
        self.assertEqual(self.runs()[0].info.status, "FAILED")
        self.assertTrue((self.output / "per-query.csv").exists())

    def test_compressed_external_run_reads_its_sidecar(self):
        compressed = self.directory / "model.jsonl.gz"
        with gzip.open(compressed, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"query_id": "001", "doc_id": "a", "score": 1}) + "\n")
        metadata = {"dataset": self.dataset, "split": "test", "parameters": {
            "method": "external", "model": "example/encoder", "top_k": 50,
        }}
        self.run.with_suffix(".metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.invoke(self.arguments + ["--run", str(compressed)])
        self.assertEqual(self.runs()[0].data.params["retrieval.model"], "example/encoder")

    def test_later_upload_failure_preserves_completed_runs(self):
        other = self.directory / "other.trec"
        other.write_bytes(self.run.read_bytes())
        original = MlflowClient.log_artifacts
        calls = []

        def fail_second(client, run_id, *args, **kwargs):
            calls.append(run_id)
            if len(calls) == 2:
                raise MlflowException("second upload unavailable")
            return original(client, run_id, *args, **kwargs)

        with patch.object(MlflowClient, "log_artifacts", fail_second):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(self.arguments + ["--run", str(self.run), str(other)])
        self.assertEqual(self.client.get_run(calls[0]).info.status, "FINISHED")
        self.assertEqual(self.client.get_run(calls[1]).info.status, "FAILED")
        self.assertEqual(len(json.loads((self.output / "report.json").read_text())["runs"]), 2)

    def test_explicit_tracking_uri_overrides_environment(self):
        # Precreate the experiment to give an explicit URI its own artifact location.
        self.client.create_experiment(self.dataset, artifact_location=(self.directory / "artifacts").as_uri())
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://127.0.0.1:1"}):
            self.invoke(self.arguments + ["--tracking-uri", self.uri, "--source-split", "validation"])
        self.assertEqual(self.runs()[0].data.tags["source_split"], "validation")


if __name__ == "__main__":
    unittest.main()
