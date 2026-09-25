"""Focused hybrid retrieval tests with embedding and Qdrant boundaries stubbed."""

import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from qdrant_client import models

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "hybrid_retrieval", SCRIPTS / "2.3.1-hybrid-retrieval.py"
)
hybrid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hybrid)


class FakeDenseModel:
    calls = 0
    passage_inputs = []
    query_inputs = []
    dimensions = {hybrid.DEFAULT_DENSE_MODEL: 2, "other/dense": 3}

    @staticmethod
    def list_supported_models():
        return [
            {
                "model": name,
                "sources": {"hf": "fixture"},
                "model_file": "dense.onnx",
                "description": "fixture",
                "license": "apache-2.0",
                "size_in_GB": 0.0,
                "additional_files": [],
                "dim": dimension,
                "tasks": {},
            }
            for name, dimension in FakeDenseModel.dimensions.items()
        ]

    @staticmethod
    def download_model(description, cache_dir):
        directory = Path(cache_dir) / description.model.replace("/", "-")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "dense.onnx").write_text(description.model, encoding="utf-8")
        return directory

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name

    def passage_embed(self, texts, **kwargs):
        FakeDenseModel.calls += 1
        texts = list(texts)
        FakeDenseModel.passage_inputs.extend(texts)
        dimension = FakeDenseModel.dimensions[self.model_name]
        return iter([[1.0] + [0.25] * (dimension - 1) for _ in texts])

    def query_embed(self, text):
        FakeDenseModel.query_inputs.append(text)
        dimension = FakeDenseModel.dimensions[self.model_name]
        return iter([[1.0] + [0.25] * (dimension - 1)])


class FakeSparseModel:
    calls = 0
    passage_inputs = []
    query_inputs = []
    definitions = {
        hybrid.DEFAULT_SPARSE_MODEL: (30522, True),
        "other/sparse": (50000, False),
    }

    @staticmethod
    def list_supported_models():
        return [
            {
                "model": name,
                "sources": {"hf": "fixture"},
                "model_file": "sparse.onnx",
                "description": "fixture",
                "license": "apache-2.0",
                "size_in_GB": 0.0,
                "additional_files": [],
                "requires_idf": requires_idf,
                "vocab_size": vocabulary,
            }
            for name, (vocabulary, requires_idf) in FakeSparseModel.definitions.items()
        ]

    @staticmethod
    def download_model(description, cache_dir):
        directory = Path(cache_dir) / description.model.replace("/", "-")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sparse.onnx").write_text(description.model, encoding="utf-8")
        return directory

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name

    def passage_embed(self, texts, **kwargs):
        FakeSparseModel.calls += 1
        texts = list(texts)
        FakeSparseModel.passage_inputs.extend(texts)
        return iter([NS(indices=[8, 3], values=[0.8, 0.3]) for _ in texts])

    def query_embed(self, text):
        FakeSparseModel.query_inputs.append(text)
        return iter([NS(indices=[8, 3], values=[1.0, 1.0])])


class FakeHybridClient:
    def __init__(self, **kwargs):
        self.collections = {}
        self.deleted = []
        self.queries = []

    def info(self):
        return NS(version="1.19.1")

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(
        self, collection_name, vectors_config, sparse_vectors_config, metadata
    ):
        self.collections[collection_name] = NS(
            config=NS(
                params=NS(vectors=vectors_config, sparse_vectors=sparse_vectors_config),
                metadata=deepcopy(metadata),
            ),
            points={},
        )

    def delete_collection(self, name):
        self.deleted.append(name)
        del self.collections[name]

    def get_collection(self, name):
        return self.collections[name]

    def upsert(self, collection_name, points, wait):
        self.collections[collection_name].points.update({point.id: point for point in points})

    def count(self, collection_name, exact):
        return NS(count=len(self.collections[collection_name].points))

    def update_collection(self, collection_name, metadata):
        self.collections[collection_name].config.metadata = deepcopy(metadata)

    def query_points(self, collection_name, **kwargs):
        self.queries.append(kwargs)
        points = list(self.collections[collection_name].points.values())
        ranked = list(reversed(points))[:kwargs["limit"]]
        return NS(points=[
            NS(payload=point.payload, score=0.0325 - offset * 0.0008)
            for offset, point in enumerate(ranked)
        ])

    def close(self):
        pass


class HybridTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        dataset = self.root / "tiny"
        (dataset / "queries").mkdir(parents=True)
        self.corpus = dataset / "corpus.jsonl"
        self.corpus.write_text(
            '{"doc_id":"001","text":"neural retrieval"}\n'
            '{"doc_id":"002","text":"bread recipe"}\n', encoding="utf-8"
        )
        (dataset / "queries/test.jsonl").write_text(
            '{"query_id":"q1","text":"semantic search"}\n', encoding="utf-8"
        )
        self.output = self.root / "out"
        self.client = FakeHybridClient()
        for fake in (FakeDenseModel, FakeSparseModel):
            fake.calls = 0
            fake.passage_inputs = []
            fake.query_inputs = []
        for name, value in (
            ("DATA_DIR", self.root),
            ("OUTPUT_DIR", self.root / "cache"),
            ("TextEmbedding", FakeDenseModel),
            ("SparseTextEmbedding", FakeSparseModel),
            ("QdrantClient", lambda **kwargs: self.client),
        ):
            patcher = patch.object(hybrid, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.args = [
            "--dataset", "tiny", "--top-k", "2", "--output-dir", str(self.output)
        ]

    def run_hybrid(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            hybrid.main(self.args + list(extra))

    def metadata(self):
        return json.loads(
            (self.output / f"{hybrid.MODEL_NAME}.metadata.json").read_text()
        )

    def test_native_rrf_uses_both_signals_and_writes_trec_metadata(self):
        self.run_hybrid()
        lines = (self.output / f"{hybrid.MODEL_NAME}.trec").read_text().splitlines()
        self.assertEqual(lines[0].split()[:4], ["q1", "Q0", "002", "1"])
        self.assertEqual(len(lines[0].split()), 6)
        request = self.client.queries[0]
        self.assertEqual([item.using for item in request["prefetch"]], ["dense", "sparse"])
        self.assertIsInstance(request["prefetch"][0].query, list)
        self.assertIsInstance(request["prefetch"][1].query, models.SparseVector)
        self.assertIsInstance(request["query"], models.RrfQuery)
        self.assertEqual(request["query"].rrf.k, 60)
        parameters = self.metadata()["parameters"]
        self.assertEqual(parameters["method"], "hybrid-retrieval")
        self.assertEqual(parameters["dense_embedding_model"], hybrid.DEFAULT_DENSE_MODEL)
        self.assertEqual(parameters["sparse_embedding_model"], hybrid.DEFAULT_SPARSE_MODEL)
        self.assertEqual(parameters["fusion_strategy"], "qdrant-native-rrf")
        self.assertEqual(parameters["fusion_weights"], "equal")

    def test_dense_and_sparse_models_are_independent_configuration(self):
        self.run_hybrid()
        original = self.metadata()["parameters"]["qdrant_collection"]
        self.run_hybrid(
            "--overwrite",
            "--dense-embedding-model", "other/dense",
            "--sparse-embedding-model", "other/sparse",
            "--rrf-k", "30",
        )
        parameters = self.metadata()["parameters"]
        self.assertEqual(parameters["dense_embedding_model"], "other/dense")
        self.assertEqual(parameters["sparse_embedding_model"], "other/sparse")
        self.assertEqual(parameters["dense_vector_dimension"], 3)
        self.assertEqual(parameters["sparse_vocabulary_size"], 50000)
        self.assertEqual(parameters["rrf_k"], 30)
        self.assertNotEqual(parameters["qdrant_collection"], original)

    def test_query_document_paths_and_compatible_reuse(self):
        self.run_hybrid()
        expected_documents = ["neural retrieval", "bread recipe"]
        self.assertEqual(FakeDenseModel.passage_inputs, expected_documents)
        self.assertEqual(FakeSparseModel.passage_inputs, expected_documents)
        self.assertEqual(FakeDenseModel.query_inputs, ["semantic search"])
        self.assertEqual(FakeSparseModel.query_inputs, ["semantic search"])
        self.run_hybrid("--overwrite")
        self.assertEqual(FakeDenseModel.calls, 1)
        self.assertEqual(FakeSparseModel.calls, 1)
        self.assertTrue(self.metadata()["parameters"]["index_reused"])

    def test_incompatible_index_is_rejected_and_explicit_rebuild_repairs_it(self):
        self.run_hybrid()
        name = self.metadata()["parameters"]["qdrant_collection"]
        self.client.collections[name].config.params.vectors["dense"].size = 99
        with self.assertRaises(SystemExit):
            self.run_hybrid("--overwrite")
        self.assertFalse(self.client.deleted)
        self.run_hybrid("--overwrite", "--rebuild-index=true")
        self.assertEqual(self.client.deleted, [name])
        self.assertFalse(self.metadata()["parameters"]["index_reused"])

    def test_invalid_models_and_incomplete_fusion_results_fail(self):
        with self.assertRaises(SystemExit):
            self.run_hybrid("--dense-embedding-model", "unsupported")
        with self.assertRaises(SystemExit):
            self.run_hybrid("--sparse-embedding-model", "unsupported")
        with patch.object(self.client, "query_points", return_value=NS(points=[])):
            with self.assertRaises(SystemExit):
                self.run_hybrid()
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
