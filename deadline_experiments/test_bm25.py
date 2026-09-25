"""Focused checks for the independent experiments and sampled-corpus protocol."""

import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from qdrant_client import models


class BM25Tests(unittest.TestCase):
    def modules(self):
        for dataset in ("scifact", "jurifindit", "msmarco", "mmarco_it"):
            path = Path(__file__).parent / dataset / "exp_001_bm25.py"
            spec = importlib.util.spec_from_file_location(dataset, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            yield module

    def test_native_queries_real_ids_and_cutoff_ties(self):
        for module in self.modules():
            with self.subTest(dataset=module.DATASET):
                documents = {"001": "apple", "1": "apple", "z": "apple", "a": "pear", "b": "plum"}
                client = MagicMock()

                def point(doc_id, score):
                    return SimpleNamespace(id=module.point_id(doc_id), payload={"doc_id": doc_id}, score=score)

                client.query_points.side_effect = [
                    SimpleNamespace(points=[point("001", 2), point("1", 2)]),
                    SimpleNamespace(points=[point("001", 2), point("1", 2), point("z", 2), point("a", 1)]),
                ]
                self.assertEqual(module.retrieve(client, "collection", "apple", documents, 1), [("z", 2)])
                self.assertEqual([c.kwargs["limit"] for c in client.query_points.call_args_list], [2, 4])
                request = client.query_points.call_args.kwargs
                self.assertIsInstance(request["query"], models.Document)
                self.assertEqual(request["query"].model, "qdrant/bm25")
                self.assertEqual(request["query"].text, "apple")
                self.assertEqual(request["query"].options, module.BM25_OPTIONS)
                self.assertEqual(request["using"], "bm25")
                self.assertNotEqual(module.point_id("001"), module.point_id("1"))
                client.query_points.side_effect = None
                client.query_points.return_value = SimpleNamespace(points=[point("001", 3)])
                self.assertEqual(module.retrieve(client, "collection", "apple", documents, 3),
                                 [("001", 3), ("z", 0), ("b", 0)])
                client.query_points.return_value = SimpleNamespace(points=[point("missing", 3)])
                with self.assertRaisesRegex(RuntimeError, "invalid or duplicate"):
                    module.retrieve(client, "collection", "apple", documents, 3)

    def test_native_index_reuse_and_incompatible_collection_safety(self):
        for module in self.modules():
            with self.subTest(dataset=module.DATASET):
                documents = {"001": "apple", "1": "pear"}
                client = MagicMock()
                client.get_collections.return_value = SimpleNamespace(collections=[])
                client.count.return_value = SimpleNamespace(count=2)
                client.retrieve.return_value = [SimpleNamespace(
                    id=module.point_id(d), payload=module.document_payload(d, text),
                    vector={"bm25": models.SparseVector(indices=[1], values=[1.0])},
                ) for d, text in documents.items()]
                name, manifest = module.build_index(client, documents, "smoke", "1.19.1")
                self.assertFalse(manifest["reused"])
                config = client.create_collection.call_args.kwargs
                self.assertEqual(config["sparse_vectors_config"]["bm25"].modifier, models.Modifier.IDF)
                self.assertFalse(config["metadata"]["complete"])
                for point in client.upsert.call_args.kwargs["points"]:
                    self.assertIsInstance(point.vector["bm25"], models.Document)
                    self.assertEqual(point.vector["bm25"].model, "qdrant/bm25")
                    self.assertEqual(point.vector["bm25"].options, module.BM25_OPTIONS)
                metadata = client.update_collection.call_args.kwargs["metadata"]
                self.assertTrue(metadata["complete"])
                client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name=name)])
                info = SimpleNamespace(config=SimpleNamespace(metadata=metadata, params=SimpleNamespace(
                    vectors={}, sparse_vectors=config["sparse_vectors_config"])))
                client.get_collection.return_value = info
                reused, manifest = module.build_index(client, documents, "smoke", "1.19.1")
                self.assertEqual(reused, name)
                self.assertTrue(manifest["reused"])
                self.assertEqual(client.upsert.call_count, 1)
                # A colliding unrelated collection must never receive writes.
                info.config.metadata = {"owner": "someone_else"}
                replacement, _ = module.build_index(client, documents, "smoke", "1.19.1")
                self.assertNotEqual(replacement, name)
                self.assertEqual(client.upsert.call_args.args[0], replacement)
                self.assertEqual(client.update_collection.call_args.args[0], replacement)
                client.delete_collection.assert_not_called()
                # The payload hash check rejects stale text even when the count matches.
                client.retrieve.return_value[0].payload = {"doc_id": "001", "text_sha256": "stale"}
                self.assertFalse(module.verify_collection(client, replacement, documents))

    def test_sampling_retains_every_judgment_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "queries").mkdir()
            (root / "qrels").mkdir()

            def write(path, records):
                path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

            write(root / "corpus.jsonl", [dict(doc_id=str(i), text=f"word {i}") for i in range(1030)])
            write(root / "queries" / "test.jsonl", [dict(query_id=str(i), text="word") for i in range(22)])
            judgments = [dict(query_id=str(i), doc_id=str(i), relevance=1) for i in range(21)]
            judgments += [dict(query_id="0", doc_id="25", relevance=1),
                          dict(query_id="0", doc_id="26", relevance=0),
                          dict(query_id="21", doc_id="missing", relevance=1)]
            write(root / "qrels" / "test.jsonl", judgments)
            for module in self.modules():
                with self.subTest(dataset=module.DATASET):
                    first = module.load_data(root, True)
                    self.assertEqual(first, module.load_data(root, True))
                    documents, queries, qrels, sampling = first
                    self.assertEqual(list(queries), sorted(str(i) for i in range(21))[:20])
                    self.assertEqual(sampling["qrel_count"], 22)
                    self.assertEqual(sampling["corpus_size"], 1022)
                    self.assertEqual(sampling["unjudged_distractor_count"], 1000)
                    self.assertEqual(qrels["0"], {"0": 1, "25": 1, "26": 0})
                    self.assertTrue(all(d in documents for js in qrels.values() for d in js))
                    with self.assertRaisesRegex(ValueError, "missing from the corpus"):
                        module.load_data(root, False)

    def test_shared_evaluation_semantics(self):
        for module in self.modules():
            with self.subTest(dataset=module.DATASET):
                result, _ = module.evaluate.evaluate_run(
                    {"q": {"a": 1, "b": 1}}, {"q": {"a": 3, "x": 2, "b": 1}},
                    module.evaluate.build_metrics(module.METRICS))
                metrics = result["aggregate"]
                self.assertEqual(len(metrics), 8)
                self.assertAlmostEqual(metrics["AP"], (1 + 2 / 3) / 2)
                self.assertEqual(metrics["RR"], 1)
                self.assertEqual(metrics["P@5"], 2 / 5)
                self.assertEqual(metrics["R@1"], 1 / 2)

    def test_smoke_tracking_names_engine_and_output_isolation(self):
        from mlflow import MlflowClient

        module = next(self.modules())
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as cleanup:
            root = Path(temporary)
            tracking = root / "tracking" / "data"
            tracking.mkdir(parents=True)
            client = MlflowClient(tracking_uri=f"sqlite:///{(tracking / 'mlflow.db').as_posix()}")
            # Release SQLite handles before temporary-directory cleanup on Windows.
            cleanup.callback(client._tracking_client.store._dispose_engine)
            script = root / "deadline_experiments" / "scifact" / "exp_001_bm25.py"
            # Tiny fixture isolates tracking from real dataset sampling, tested above.
            fixture = ({"001": "apple", "z": "pear"}, {"q": "apple"},
                       {"q": {"001": 1}}, {"corpus_size": 2, "query_count": 1, "qrel_count": 1})
            qdrant = cleanup.enter_context(patch.object(module, "QdrantClient"))
            qdrant.return_value.info.return_value.version = "test"
            cleanup.enter_context(patch.object(module, "build_index", return_value=(
                "test-collection", {"reused": False, "corpus_sha256": "fixture"})))
            cleanup.enter_context(patch.object(module, "retrieve", return_value=[("001", 1), ("z", 0)]))
            data = root / "data-preparation" / "data" / "scifact"
            for name in ("corpus.jsonl", "queries/test.jsonl", "qrels/test.jsonl"):
                path = data / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            with patch.object(module, "ROOT", root), patch.object(module, "__file__", str(script)), \
                    patch.object(module, "load_data", return_value=fixture), \
                    contextlib.redirect_stdout(io.StringIO()):
                module.main(["--smoke"])
                before = {p: p.read_bytes() for p in script.parent.rglob("*") if p.is_file()}
                module.main(["--smoke"])
                self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))
                qdrant.assert_called_with(url="http://localhost:6333", cloud_inference=True, timeout=60)
            experiment = client.get_experiment_by_name("deadline-scifact")
            runs = client.search_runs([experiment.experiment_id])
            self.assertEqual(len(runs), 2)
            for run in runs:
                self.assertEqual(run.info.status, "FINISHED")
                self.assertEqual(run.data.tags["mlflow.runName"], "exp_001_bm25_smoke")
                self.assertEqual(run.data.tags["mode"], "smoke")
                self.assertEqual(run.data.params["retrieval.engine"], "qdrant-server")
                self.assertEqual(run.data.params["retrieval.implementation"], "server-native qdrant/bm25")
                self.assertEqual(run.data.metrics["ir.AP"], 1)
            self.assertEqual(len(list(script.parent.rglob("verification.json"))), 2)

            original = MlflowClient.download_artifacts
            for artifact in ("retrieval.json", "metric-names.json"):
                def corrupt(client, run_id, path, destination):
                    downloaded = original(client, run_id, path, destination)
                    if path == artifact:
                        Path(downloaded).write_text("{}\n", encoding="utf-8")
                    return downloaded

                with self.subTest(artifact=artifact), patch.object(module, "ROOT", root), \
                        patch.object(module, "__file__", str(script)), \
                        patch.object(module, "load_data", return_value=fixture), \
                        patch.object(MlflowClient, "download_artifacts", corrupt), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaisesRegex(RuntimeError, "Artifact content mismatch"):
                    module.main(["--smoke"])
            runs = client.search_runs([experiment.experiment_id])
            self.assertEqual(sum(run.info.status == "FAILED" for run in runs), 2)
            self.assertEqual(sum(run.info.status == "FINISHED" for run in runs), 2)


if __name__ == "__main__":
    unittest.main()
