"""Dense integration tests with embedding and remote-storage boundaries stubbed."""
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fastembed import TextEmbedding as FastEmbedTextEmbedding

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/2.1.1-dense-retrieval.py"
SPEC = importlib.util.spec_from_file_location("dense", SCRIPT)
dense = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense)

RECOMMENDED_MODELS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "minishlab/potion-multilingual-128M": 256,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
}


class FakeModel:
    calls = 0
    weights = "original"
    dimensions = {dense.DEFAULT_MODEL: 2, "other/model": 3}
    passage_inputs = []
    query_inputs = []

    @staticmethod
    def list_supported_models():
        return [{"model": name, "dim": dimension, "sources": {"hf": "fixture"},
                 "model_file": "model.onnx", "description": "fixture", "license": "mit",
                 "size_in_GB": 0.0} for name, dimension in FakeModel.dimensions.items()]

    @staticmethod
    def download_model(description, cache_dir):
        directory = Path(cache_dir) / description.model.replace("/", "-")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.onnx").write_text(FakeModel.weights, encoding="utf-8")
        return directory

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name

    def passage_embed(self, texts, **kwargs):
        FakeModel.calls += 1
        texts = list(texts)
        FakeModel.passage_inputs.extend(texts)
        dimension = FakeModel.dimensions[self.model_name]
        return iter([[1.0] + [0.5] * (dimension - 1) for text in texts])

    def query_embed(self, text):
        FakeModel.query_inputs.append(text)
        dimension = FakeModel.dimensions[self.model_name]
        return iter([[1.0] + [0.5] * (dimension - 1)])


class FakeClient:
    def __init__(self, **kwargs):
        self.collections = {}
        self.deleted = []

    def info(self):
        return NS(version="1.19.1")

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config, metadata):
        self.collections[collection_name] = NS(
            config=NS(params=NS(vectors=vectors_config), metadata=deepcopy(metadata)), points={})

    def delete_collection(self, name):
        self.deleted.append(name)
        del self.collections[name]

    def get_collection(self, name):
        return self.collections[name]

    def upsert(self, collection_name, points, wait):
        self.collections[collection_name].points.update({p.id: p for p in points})

    def count(self, collection_name, exact):
        return NS(count=len(self.collections[collection_name].points))

    def update_collection(self, collection_name, metadata):
        self.collections[collection_name].config.metadata = deepcopy(metadata)

    def query_points(self, collection_name, limit, **kwargs):
        points = list(self.collections[collection_name].points.values())[:limit]
        return NS(points=[NS(payload=p.payload, score=0.9 - i * 0.1) for i, p in enumerate(points)])

    def close(self):
        pass


class DenseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "tiny"
        (self.dataset / "queries").mkdir(parents=True)
        self.corpus = self.dataset / "corpus.jsonl"
        self.corpus.write_text(' {"doc_id":"001","text":"automobile"}\n'
                               '{"doc_id":"italiano","text":"una macchina"}\n', encoding="utf-8")
        (self.dataset / "queries/test.jsonl").write_text(
            '{"query_id":"002","text":"car"}\n', encoding="utf-8")
        self.output = self.root / "out"
        self.client = FakeClient()
        FakeModel.calls = 0
        FakeModel.weights = "original"
        FakeModel.passage_inputs = []
        FakeModel.query_inputs = []
        for name, value in (("DATA_DIR", self.root), ("OUTPUT_DIR", self.root / "cache"), ("TextEmbedding", FakeModel),
                            ("QdrantClient", lambda **kwargs: self.client)):
            patcher = patch.object(dense, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.args = ["--dataset", "tiny", "--top-k", "2", "--output-dir", str(self.output)]

    def run_dense(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            dense.main(self.args + list(extra))

    def metadata(self):
        return json.loads((self.output / f"{dense.MODEL_NAME}.metadata.json").read_text())

    def test_trec_metadata_and_reuse_without_document_embedding(self):
        self.run_dense()
        lines = (self.output / f"{dense.MODEL_NAME}.trec").read_text().splitlines()
        self.assertEqual(lines[0].split()[:4], ["002", "Q0", "001", "1"])
        self.assertEqual(lines[1].split()[:4], ["002", "Q0", "italiano", "2"])
        self.assertEqual(len(lines[0].split()), 6)
        params = self.metadata()["parameters"]
        self.assertEqual(params["embedding_model"], dense.DEFAULT_MODEL)
        self.assertEqual(params["embedding_runtime"], "FastEmbed")
        self.assertEqual(params["vector_dimension"], 2)
        self.assertEqual(params["distance"], "Cosine")
        self.assertFalse(params["index_reused"])
        self.run_dense("--overwrite")
        self.assertEqual(FakeModel.calls, 1)
        self.assertTrue(self.metadata()["parameters"]["index_reused"])
        self.assertFalse(self.client.deleted)

    def test_models_and_changed_corpus_get_distinct_collections(self):
        self.run_dense()
        original = self.metadata()["parameters"]["qdrant_collection"]
        self.run_dense("--overwrite", "--embedding-model", "other/model")
        other = self.metadata()["parameters"]
        self.assertEqual(other["embedding_model"], "other/model")
        self.assertEqual(other["vector_dimension"], 3)
        self.assertEqual(self.client.collections[other["qdrant_collection"]].config.params.vectors.size, 3)
        self.assertNotEqual(original, other["qdrant_collection"])
        self.corpus.write_text(self.corpus.read_text().replace("automobile", "bread"), encoding="utf-8")
        self.run_dense("--overwrite")
        self.assertEqual(len(self.client.collections), 3)
        self.assertFalse(self.client.deleted)

    def test_documents_and_queries_use_their_explicit_embedding_apis(self):
        self.run_dense()
        self.assertEqual(FakeModel.passage_inputs, ["automobile", "una macchina"])
        self.assertEqual(FakeModel.query_inputs, ["car"])

    def test_recommended_models_match_the_pinned_fastembed_registry(self):
        registry = {model["model"]: model for model in FastEmbedTextEmbedding.list_supported_models()}
        self.assertEqual({name: registry[name]["dim"] for name in RECOMMENDED_MODELS},
                         RECOMMENDED_MODELS)
        for name in RECOMMENDED_MODELS:
            self.assertIn("Prefixes for queries/documents: not necessary",
                          registry[name]["description"])

    def test_changed_weights_under_same_model_name_get_a_new_index(self):
        self.run_dense()
        original = self.metadata()["parameters"]
        FakeModel.weights = "updated weights"
        self.run_dense("--overwrite")
        current = self.metadata()["parameters"]
        self.assertEqual(original["embedding_model"], current["embedding_model"])
        self.assertNotEqual(original["model_artifacts_sha256"], current["model_artifacts_sha256"])
        self.assertNotEqual(original["qdrant_collection"], current["qdrant_collection"])
        self.assertFalse(self.client.deleted)

    def test_incompatible_metadata_config_count_and_incomplete_index_are_rejected(self):
        self.run_dense()
        name = self.metadata()["parameters"]["qdrant_collection"]
        pristine = deepcopy(self.client.collections[name])
        for defect in ("model", "dimension", "distance", "count", "complete", "missing_metadata"):
            with self.subTest(defect=defect):
                state = deepcopy(pristine)
                self.client.collections[name] = state
                if defect == "model":
                    state.config.metadata["dense_index"]["embedding_model"] = "wrong"
                elif defect == "dimension":
                    state.config.params.vectors.size = 3
                elif defect == "distance":
                    state.config.params.vectors.distance = dense.models.Distance.DOT
                elif defect == "count":
                    state.points.pop(0)
                elif defect == "complete":
                    state.config.metadata["complete"] = False
                else:
                    state.config.metadata = None
                with self.assertRaises(SystemExit):
                    self.run_dense("--overwrite")
                self.assertEqual(FakeModel.calls, 1)
                self.assertFalse(self.client.deleted)
        self.run_dense("--overwrite", "--rebuild-index=true")
        self.assertEqual(self.client.deleted, [name])
        self.assertEqual(FakeModel.calls, 2)

    def test_failures_preserve_published_outputs(self):
        self.run_dense()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        for query_vectors in ([[1.0]], [[float("nan"), 1.0]], [[0.0, 0.0]]):
            with patch.object(FakeModel, "query_embed", return_value=iter(query_vectors)):
                with self.assertRaises(SystemExit):
                    self.run_dense("--overwrite")
        with patch.object(self.client, "query_points", return_value=NS(points=[])):
            with self.assertRaises(SystemExit):
                self.run_dense("--overwrite")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_failed_upload_is_not_reused(self):
        with patch.object(self.client, "upsert", side_effect=RuntimeError("upload interrupted")):
            with self.assertRaises(SystemExit):
                self.run_dense()
        self.assertFalse(self.output.exists())
        with self.assertRaises(SystemExit):
            self.run_dense()
        self.run_dense("--rebuild-index")
        self.assertEqual(len(self.client.deleted), 1)

    def test_invalid_model_missing_input_and_unreachable_server(self):
        with self.assertRaises(SystemExit):
            self.run_dense("--embedding-model", "unsupported")
        self.assertFalse(self.client.collections)
        with patch.object(self.client, "info", side_effect=RuntimeError("offline")):
            with self.assertRaises(SystemExit):
                self.run_dense()
        self.corpus.unlink()
        with self.assertRaises(SystemExit):
            self.run_dense()
        self.assertFalse(self.client.collections)


if __name__ == "__main__":
    unittest.main()
