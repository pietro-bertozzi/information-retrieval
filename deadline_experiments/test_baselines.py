"""Focused checks for the independent experiments and sampled-corpus protocol."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


class BaselineTests(unittest.TestCase):
    def modules(self):
        for dataset in ("scifact", "jurifindit", "msmarco", "mmarco_it"):
            path = Path(__file__).parent / dataset / "exp_001_baseline.py"
            spec = importlib.util.spec_from_file_location(dataset, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            yield module

    def test_bm25_formula_real_ids_and_ties(self):
        for module in self.modules():
            with self.subTest(dataset=module.DATASET):
                documents = {"001": "apple apple pear", "z": "pear", "a": "plum"}
                index = module.build_index(documents)
                result = module.retrieve("apple apple", *index, 3)
                expected = math.log1p(2.5 / 1.5) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / (5 / 3)))
                self.assertEqual([d for d, _ in result], ["001", "z", "a"])
                self.assertAlmostEqual(result[0][1], expected)
                self.assertEqual(result[1][1], 0)
                self.assertEqual(module.tokenize("L'acqua È_utile"), ["l", "acqua", "è", "utile"])

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


if __name__ == "__main__":
    unittest.main()
