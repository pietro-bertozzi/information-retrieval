"""Check metadata emitted by every standalone retrieval script."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            for script in sorted(SCRIPTS.glob("*.py")):
                with self.subTest(script=script.name):
                    spec = importlib.util.spec_from_file_location("retriever", script)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
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
