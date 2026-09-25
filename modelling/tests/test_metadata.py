"""Check metadata emitted by every standalone retrieval script."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_dense import FakeClient, FakeModel
from test_hybrid import FakeDenseModel, FakeHybridClient, FakeSparseModel as HybridSparseModel
from test_sparse import FakeSparseClient, FakeSparseModel

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class MetadataTests(unittest.TestCase):
    def test_all_methods_write_settings_and_preserve_outputs_on_invalid_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "jurifindit"
            (dataset / "queries").mkdir(parents=True)
            (dataset / "corpus.jsonl").write_text(
                json.dumps({"doc_id": "001", "text": "legal evidence"}) + "\n",
                encoding="utf-8",
            )
            queries = dataset / "queries/test.jsonl"
            query = json.dumps({"query_id": "002", "text": "legal evidence"}) + "\n"
            for script in sorted(
                path for path in SCRIPTS.glob("*.py") if not path.name.startswith("_")
            ):
                with self.subTest(script=script.name):
                    spec = importlib.util.spec_from_file_location("retriever", script)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if script.stem == "2.1.1-dense-retrieval":
                        # Keep the common contract checks; only external boundaries are stubbed.
                        module.TextEmbedding = FakeModel
                        module.QdrantClient = FakeClient
                        module.OUTPUT_DIR = root / "cache"
                    elif script.stem == "2.2.1-sparse-retrieval":
                        module.SparseTextEmbedding = FakeSparseModel
                        module.QdrantClient = FakeSparseClient
                        module.OUTPUT_DIR = root / "cache"
                    elif script.stem == "2.3.1-hybrid-retrieval":
                        module.TextEmbedding = FakeDenseModel
                        module.SparseTextEmbedding = HybridSparseModel
                        module.QdrantClient = FakeHybridClient
                        module.OUTPUT_DIR = root / "cache"
                    self.assertEqual(module.MODEL_NAME, script.stem)
                    queries.write_text(query, encoding="utf-8")
                    output = root / script.stem
                    args = ["--dataset", "jurifindit", "--top-k", "7", "--output-dir", str(output)]
                    with patch.object(module, "DATA_DIR", root), contextlib.redirect_stdout(io.StringIO()):
                        module.main(args)
                        metadata = json.loads((output / f"{script.stem}.metadata.json").read_text())
                        self.assertEqual(metadata["dataset"], "jurifindit")
                        self.assertEqual(metadata["split"], "test")
                        self.assertEqual(metadata["parameters"]["top_k"], 7)
                        capability = getattr(module, "EXPERIMENT_CAPABILITY", None)
                        if capability:
                            for key, value in capability.get("parameters", {}).items():
                                self.assertEqual(metadata["parameters"][key], value)
                            names = capability.get("parameter_names", {})
                            for option in capability["dimensions"].values():
                                self.assertIn(
                                    names.get(option, option.replace("-", "_")),
                                    metadata["parameters"],
                                )
                        self.assertEqual(metadata["parameters"]["version"], script.stem.split("-", 1)[0])
                        self.assertEqual(metadata["parameters"].get("language"), "it" if "stopword" in script.stem else None)
                        run = output / f"{script.stem}.trec"
                        self.assertEqual(run.read_text().split()[:4], ["002", "Q0", "001", "1"])
                        before = {p.name: p.read_bytes() for p in output.iterdir()}
                        queries.write_text("invalid", encoding="utf-8")
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                            module.main(args + ["--overwrite"])
                        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
                        queries.write_text(query, encoding="utf-8")
                        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                            module.main(args)
                        if "stopword" in script.stem:
                            module.main(args + ["--overwrite", "--language", "en"])
                            metadata = json.loads((output / f"{script.stem}.metadata.json").read_text())
                            self.assertEqual(metadata["parameters"]["language"], "en")


if __name__ == "__main__":
    unittest.main()
