"""Focused checks for the four independent dense retrieval experiments."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client import models


class DenseTests(unittest.TestCase):
    def modules(self, experiment="exp_002_dense.py"):
        for dataset in ("scifact", "jurifindit", "msmarco", "mmarco_it"):
            path = Path(__file__).parent / dataset / experiment
            spec = importlib.util.spec_from_file_location(f"{dataset}_{experiment}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            yield module

    def test_same_sampling_and_evaluation_as_bm25(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "queries").mkdir()
            (root / "qrels").mkdir()

            def write(path, rows):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            write(root / "corpus.jsonl", [
                {"doc_id": str(i), "text": f"document {i}"} for i in range(1030)
            ])
            write(root / "queries" / "test.jsonl", [
                {"query_id": str(i), "text": f"query {i}"} for i in range(21)
            ])
            write(root / "qrels" / "test.jsonl", [
                {"query_id": str(i), "doc_id": str(i), "relevance": 1} for i in range(21)
            ])
            for dense, bm25 in zip(self.modules(), self.modules("exp_001_bm25.py")):
                with self.subTest(dataset=dense.DATASET):
                    self.assertEqual(dense.load_data(root, True), bm25.load_data(root, True))
                    self.assertEqual(dense.METRICS, bm25.METRICS)
                    self.assertEqual(dense.TOP_K, 10)

    def test_dense_query_uses_model_and_returns_deterministic_top_ten(self):
        for module in self.modules():
            with self.subTest(dataset=module.DATASET):
                documents = {str(i): f"document {i}" for i in range(12)}
                client = MagicMock()

                def point(doc_id, score):
                    return SimpleNamespace(
                        id=module.point_id(doc_id), payload={"doc_id": doc_id}, score=score
                    )

                first = [point(str(i), 1.0) for i in range(11)]
                second = first + [point("11", 1.0)]
                client.query_points.side_effect = [
                    SimpleNamespace(points=first), SimpleNamespace(points=second)
                ]
                result = module.retrieve(client, "collection", "query text", documents)
                self.assertEqual([doc_id for doc_id, _ in result],
                                 ["9", "8", "7", "6", "5", "4", "3", "2", "11", "10"])
                self.assertEqual([call.kwargs["limit"] for call in client.query_points.call_args_list],
                                 [11, 12])
                request = client.query_points.call_args.kwargs
                self.assertEqual(request["using"], "dense")
                self.assertEqual(request["query"], models.Document(
                    text="query text", model=module.EMBEDDING_MODEL
                ))
                self.assertFalse(request["with_vectors"])
                self.assertNotEqual(module.point_id("001"), module.point_id("1"))

    def test_collection_creation_reuse_and_collision_safety(self):
        for module in self.modules():
            with self.subTest(dataset=module.DATASET):
                documents = {"001": "first", "1": "second"}
                client = MagicMock()
                client.get_collections.return_value = SimpleNamespace(collections=[])
                client.count.return_value = SimpleNamespace(count=2)
                client.retrieve.return_value = [SimpleNamespace(
                    id=module.point_id(doc_id),
                    payload=module.document_payload(doc_id, text),
                    vector={"dense": [0.0] * module.VECTOR_SIZE},
                ) for doc_id, text in documents.items()]
                name, manifest = module.build_index(client, documents, "smoke", "1.19.1")
                self.assertFalse(manifest["reused"])
                config = client.create_collection.call_args.kwargs
                vector = config["vectors_config"]["dense"]
                self.assertEqual(vector.size, module.VECTOR_SIZE)
                self.assertEqual(vector.distance, models.Distance.COSINE)
                self.assertFalse(config["metadata"]["complete"])
                for point in client.upsert.call_args.kwargs["points"]:
                    document = point.vector["dense"]
                    self.assertIsInstance(document, models.Document)
                    self.assertEqual(document.model, module.EMBEDDING_MODEL)
                completed = client.update_collection.call_args.kwargs["metadata"]
                self.assertTrue(completed["complete"])

                client.get_collections.return_value = SimpleNamespace(
                    collections=[SimpleNamespace(name=name)]
                )
                client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(
                    metadata=completed,
                    params=SimpleNamespace(
                        vectors=config["vectors_config"], sparse_vectors=None,
                    ),
                ))
                reused, manifest = module.build_index(client, documents, "smoke", "1.19.1")
                self.assertEqual(reused, name)
                self.assertTrue(manifest["reused"])
                self.assertEqual(client.upsert.call_count, 1)

                client.get_collection.return_value.config.metadata = {"owner": "unrelated"}
                replacement, _ = module.build_index(client, documents, "smoke", "1.19.1")
                self.assertNotEqual(replacement, name)
                self.assertEqual(client.upsert.call_args.args[0], replacement)
                client.delete_collection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
