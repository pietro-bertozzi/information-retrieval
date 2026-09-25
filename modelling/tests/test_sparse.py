"""Focused sparse retrieval tests with FastEmbed and Qdrant stubbed."""

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
    "sparse_retrieval", SCRIPTS / "2.2.1-sparse-retrieval.py"
)
sparse = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sparse)


class FakeSparseModel:
    calls = 0
    weights = "original"
    passage_inputs = []
    query_inputs = []
    definitions = {
        sparse.DEFAULT_MODEL: (30522, True),
        "other/sparse": (50000, False),
    }

    @staticmethod
    def list_supported_models():
        return [
            {
                "model": name,
                "sources": {"hf": "fixture"},
                "model_file": "model.onnx",
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
        (directory / "model.onnx").write_text(FakeSparseModel.weights, encoding="utf-8")
        return directory

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name

    def passage_embed(self, texts, **kwargs):
        FakeSparseModel.calls += 1
        texts = list(texts)
        FakeSparseModel.passage_inputs.extend(texts)
        vectors = [
            NS(indices=[7, 2], values=[0.7, 0.2]),
            NS(indices=[9], values=[0.9]),
        ]
        return iter(vectors[:len(texts)])

    def query_embed(self, text):
        FakeSparseModel.query_inputs.append(text)
        return iter([NS(indices=[7, 2], values=[1.0, 1.0])])


class FakeSparseClient:
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
                params=NS(
                    vectors=vectors_config,
                    sparse_vectors=sparse_vectors_config,
                ),
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
        return NS(points=[
            NS(payload=point.payload, score=1.2 - offset * 0.4)
            for offset, point in enumerate(points[:kwargs["limit"]])
        ])

    def close(self):
        pass


class SparseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        dataset = self.root / "tiny"
        (dataset / "queries").mkdir(parents=True)
        self.corpus = dataset / "corpus.jsonl"
        self.corpus.write_text(
            '{"doc_id":"001","text":"neural retrieval"}\n'
            '{"doc_id":"002","text":"bread recipe"}\n',
            encoding="utf-8",
        )
        (dataset / "queries/test.jsonl").write_text(
            '{"query_id":"q1","text":"semantic search"}\n', encoding="utf-8"
        )
        self.output = self.root / "out"
        self.client = FakeSparseClient()
        FakeSparseModel.calls = 0
        FakeSparseModel.weights = "original"
        FakeSparseModel.passage_inputs = []
        FakeSparseModel.query_inputs = []
        for name, value in (
            ("DATA_DIR", self.root),
            ("OUTPUT_DIR", self.root / "cache"),
            ("SparseTextEmbedding", FakeSparseModel),
            ("QdrantClient", lambda **kwargs: self.client),
        ):
            patcher = patch.object(sparse, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.args = [
            "--dataset", "tiny", "--top-k", "2", "--output-dir", str(self.output)
        ]

    def run_sparse(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sparse.main(self.args + list(extra))

    def metadata(self):
        return json.loads(
            (self.output / f"{sparse.MODEL_NAME}.metadata.json").read_text()
        )

    def test_sparse_vectors_trec_and_metadata(self):
        self.run_sparse()
        lines = (self.output / f"{sparse.MODEL_NAME}.trec").read_text().splitlines()
        self.assertEqual(lines[0].split()[:4], ["q1", "Q0", "001", "1"])
        self.assertEqual(len(lines[0].split()), 6)
        parameters = self.metadata()["parameters"]
        self.assertEqual(parameters["method"], "sparse-retrieval")
        self.assertEqual(parameters["sparse_embedding_model"], sparse.DEFAULT_MODEL)
        self.assertEqual(parameters["vector_type"], "sparse")
        self.assertEqual(parameters["sparse_modifier"], "idf")
        collection = self.client.collections[parameters["qdrant_collection"]]
        stored = collection.points[0].vector[sparse.VECTOR_NAME]
        self.assertIsInstance(stored, models.SparseVector)
        self.assertEqual(stored.indices, [2, 7])
        self.assertEqual(stored.values, [0.2, 0.7])

    def test_query_and_document_paths_and_compatible_reuse(self):
        self.run_sparse()
        self.assertEqual(FakeSparseModel.passage_inputs, ["neural retrieval", "bread recipe"])
        self.assertEqual(FakeSparseModel.query_inputs, ["semantic search"])
        query = self.client.queries[0]
        self.assertIsInstance(query["query"], models.SparseVector)
        self.assertEqual(query["using"], sparse.VECTOR_NAME)
        self.run_sparse("--overwrite")
        self.assertEqual(FakeSparseModel.calls, 1)
        self.assertTrue(self.metadata()["parameters"]["index_reused"])

    def test_model_is_configurable_and_changes_collection_and_modifier(self):
        self.run_sparse()
        original = self.metadata()["parameters"]["qdrant_collection"]
        self.run_sparse("--overwrite", "--embedding-model", "other/sparse")
        parameters = self.metadata()["parameters"]
        self.assertEqual(parameters["sparse_embedding_model"], "other/sparse")
        self.assertEqual(parameters["vocabulary_size"], 50000)
        self.assertEqual(parameters["sparse_modifier"], "none")
        self.assertNotEqual(parameters["qdrant_collection"], original)

    def test_incompatible_index_is_rejected_and_explicit_rebuild_repairs_it(self):
        self.run_sparse()
        name = self.metadata()["parameters"]["qdrant_collection"]
        self.client.collections[name].config.metadata["complete"] = False
        with self.assertRaises(SystemExit):
            self.run_sparse("--overwrite")
        self.assertFalse(self.client.deleted)
        self.run_sparse("--overwrite", "--rebuild-index=true")
        self.assertEqual(self.client.deleted, [name])
        self.assertFalse(self.metadata()["parameters"]["index_reused"])

    def test_invalid_sparse_shapes_and_unsupported_model_fail(self):
        with patch.object(
            FakeSparseModel, "query_embed",
            return_value=iter([NS(indices=[1, 2], values=[1.0])]),
        ):
            with self.assertRaises(SystemExit):
                self.run_sparse()
        with self.assertRaises(SystemExit):
            self.run_sparse("--embedding-model", "unsupported")
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
