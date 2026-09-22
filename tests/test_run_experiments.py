"""Test orchestration using temporary repositories and tiny prepared fixtures."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_experiments", REPOSITORY / "run_experiments.py")
orchestrator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(orchestrator)


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="experiment-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data-preparation/data"
        self.scripts = self.root / "modelling/scripts"
        self.scripts.mkdir(parents=True)
        for name in ("alpha", "beta"):
            self.dataset(name)
        for name in ("first", "second"):
            (self.scripts / f"{name}.py").write_text("# Public retrieval entry point.\n", encoding="utf-8")
        evaluator = self.root / "evaluation/scripts/evaluate.py"
        evaluator.parent.mkdir(parents=True)
        evaluator.write_text("# Evaluator stand-in.\n", encoding="utf-8")
        self.commands = []

    def dataset(self, name, split="test"):
        path = self.data / name
        rows = {
            "corpus.jsonl": {"doc_id": "001", "text": "legal evidence"},
            f"queries/{split}.jsonl": {"query_id": "002", "text": "legal evidence"},
            f"qrels/{split}.jsonl": {"query_id": "002", "doc_id": "001", "relevance": 1},
        }
        for relative, row in rows.items():
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    def fake_run(self, command, *, cwd, check):
        self.assertEqual(cwd, self.root)
        self.assertTrue(check)
        self.assertEqual(command[:2], [sys.executable, "-B"])
        self.commands.append(command)
        script = Path(command[2])
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if script.parent.name == "scripts" and script.parent.parent.name == "modelling":
            (output / f"{script.stem}.trec").write_text("fresh ranking\n", encoding="utf-8")
            (output / f"{script.stem}.metadata.json").write_text("{}", encoding="utf-8")
        else:
            for ranking in self.rankings(command):
                self.assertEqual(ranking.read_text(), "fresh ranking\n")
                self.assertTrue(ranking.with_suffix(".metadata.json").is_file())
            for name in ("report.json", "summary.csv", "per-query.csv"):
                (output / name).write_text("fresh report\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    def invoke(self, arguments, runner=None):
        with patch.object(orchestrator, "ROOT_DIR", self.root):
            with patch.object(orchestrator.subprocess, "run", side_effect=runner or self.fake_run):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    with contextlib.redirect_stderr(io.StringIO()) as errors:
                        code = orchestrator.main(arguments)
        return code, output.getvalue(), errors.getvalue()

    @staticmethod
    def rankings(command):
        return [Path(p) for p in command[command.index("--run") + 1:command.index("--output-dir")]]

    def evaluation_commands(self):
        return [command for command in self.commands if Path(command[2]).name == "evaluate.py"]

    def pairs(self):
        return [
            (command[command.index("--dataset") + 1], Path(command[2]).stem)
            for command in self.commands if Path(command[2]).name != "evaluate.py"
        ]

    def test_dataset_discovery_requires_nonempty_files_for_selected_split(self):
        self.dataset("train-only", "train")
        incomplete = self.dataset("incomplete")
        (incomplete / "qrels/test.jsonl").unlink()
        empty = self.dataset("empty")
        (empty / "corpus.jsonl").write_text("", encoding="utf-8")
        wrong_type = self.dataset("directory-instead-of-file")
        (wrong_type / "queries/test.jsonl").unlink()
        (wrong_type / "queries/test.jsonl").mkdir()
        (self.data / "notes.txt").write_text("not a dataset", encoding="utf-8")
        ready, rejected = orchestrator.discover_datasets(self.data, "test")
        self.assertEqual(list(ready), ["alpha", "beta"])
        self.assertEqual(rejected["incomplete"], ["qrels/test.jsonl"])
        self.assertEqual(rejected["empty"], ["corpus.jsonl"])
        self.assertIn("directory-instead-of-file", rejected)
        ready, _ = orchestrator.discover_datasets(self.data, "train")
        self.assertEqual(list(ready), ["train-only"])
        self.assertEqual(orchestrator.discover_datasets(self.root / "absent", "test"), ({}, {}))

    def test_method_discovery_and_new_files_are_automatic(self):
        for name in ("__init__.py", "_helper.py", "notes.md"):
            (self.scripts / name).write_text("", encoding="utf-8")
        (self.scripts / "subdirectory.py").mkdir()
        self.assertEqual(list(orchestrator.discover_methods(self.scripts)), ["first", "second"])
        (self.scripts / "new-method.py").write_text("", encoding="utf-8")
        self.dataset("new-dataset")
        code, _, _ = self.invoke(["--dataset", "new-dataset", "--method", "new-method"])
        self.assertEqual(code, 0)
        self.assertEqual(self.pairs(), [("new-dataset", "new-method")])

    def test_dataset_selects_every_method_and_evaluates_once(self):
        code, output, _ = self.invoke(["--dataset", "alpha"])
        self.assertEqual(code, 0)
        self.assertEqual(self.pairs(), [("alpha", "first"), ("alpha", "second")])
        self.assertEqual(len(self.evaluation_commands()), 1)
        self.assertIn("alpha/test: SUCCEEDED", output)
        self.assertNotIn("--no-tracking", self.evaluation_commands()[0])

    def test_method_selects_all_runnable_datasets(self):
        code, _, _ = self.invoke(["--method", "second"])
        self.assertEqual(code, 0)
        self.assertEqual(self.pairs(), [("alpha", "second"), ("beta", "second")])
        self.assertEqual(len(self.evaluation_commands()), 2)

    def test_dataset_and_method_select_one_combination(self):
        code, _, _ = self.invoke(["--dataset", "beta", "--method", "first"])
        self.assertEqual(code, 0)
        self.assertEqual(self.pairs(), [("beta", "first")])
        self.assertEqual(len(self.evaluation_commands()), 1)

    def test_all_selects_cartesian_product(self):
        code, _, _ = self.invoke(["--all"])
        self.assertEqual(code, 0)
        self.assertEqual(self.pairs(), [
            ("alpha", "first"), ("alpha", "second"),
            ("beta", "first"), ("beta", "second"),
        ])
        self.assertEqual(len(self.evaluation_commands()), 2)

    def test_list_reports_discovery_without_running_or_writing_outputs(self):
        (self.data / "incomplete").mkdir()
        code, output, _ = self.invoke(["--list"])
        self.assertEqual(code, 0)
        for text in ("Runnable datasets (test)", "alpha", "beta", "first", "second",
                     "Unavailable dataset incomplete", "qrels/test.jsonl"):
            self.assertIn(text, output)
        self.assertEqual(self.commands, [])
        self.assertFalse((self.root / "modelling/data").exists())

    def test_invalid_selections_fail_before_subprocesses(self):
        (self.data / "incomplete").mkdir()
        cases = [
            [], ["--dataset", "unknown"], ["--method", "unknown"],
            ["--dataset", "incomplete"], ["--dataset", "../alpha"],
            ["--all", "--dataset", "alpha"], ["--list", "--method", "first"],
            ["--all", "--top-k", "0"], ["--all", "--split", "../test"],
            ["--all", "--no-tracking", "--tracking-uri", "http://example"],
            ["--all", "--method-option", "first:output=/tmp"],
            ["--method", "first", "--method-option", "second:language=it"],
            ["--all", "--method-option", "invalid"],
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                self.invoke(args)
            self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.commands, [])

    def test_stale_files_are_not_evaluated_or_deleted(self):
        output = self.root / "modelling/data/alpha/test"
        output.mkdir(parents=True)
        stale = output / "retired.trec"
        stale.write_text("stale", encoding="utf-8")
        selected_old = output / "first.trec"
        selected_old.write_text("old selected", encoding="utf-8")
        code, _, _ = self.invoke(["--dataset", "alpha"])
        self.assertEqual(code, 0)
        self.assertEqual(self.rankings(self.evaluation_commands()[0]), [
            output / "first.trec", output / "second.trec",
        ])
        self.assertEqual(stale.read_text(), "stale")
        self.assertEqual(selected_old.read_text(), "fresh ranking\n")
        self.assertFalse(list(output.glob(".experiment-*")))

    def test_modelling_failure_preserves_previous_outputs_and_stops_batch(self):
        output = self.root / "modelling/data/alpha/test"
        output.mkdir(parents=True)
        old = output / "first.trec"
        old.write_text("old ranking", encoding="utf-8")
        report = self.root / "evaluation/data/alpha/test/summary.csv"
        report.parent.mkdir(parents=True)
        report.write_text("old report", encoding="utf-8")

        def fail_second(command, **kwargs):
            if Path(command[2]).stem == "second":
                raise subprocess.CalledProcessError(7, command)
            return self.fake_run(command, **kwargs)

        code, summary, errors = self.invoke(["--all"], fail_second)
        self.assertEqual(code, 1)
        self.assertIn("FAILED during retrieval second", summary)
        self.assertIn("Not run after failure: beta", summary)
        self.assertIn("exit code 7", errors)
        self.assertEqual(self.evaluation_commands(), [])
        self.assertEqual(old.read_text(), "old ranking")
        self.assertEqual(report.read_text(), "old report")
        self.assertFalse(list(output.glob(".experiment-*")))

    def test_exit_zero_without_outputs_cannot_reuse_stale_rankings(self):
        output = self.root / "modelling/data/alpha/test"
        output.mkdir(parents=True)
        for name in ("first.trec", "first.metadata.json"):
            (output / name).write_text("old", encoding="utf-8")
        code, _, errors = self.invoke(
            ["--dataset", "alpha", "--method", "first"],
            lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
        )
        self.assertEqual(code, 1)
        self.assertIn("must create first.trec and first.metadata.json", errors)
        self.assertEqual(self.evaluation_commands(), [])
        self.assertEqual((output / "first.trec").read_text(), "old")

    def test_evaluator_failure_is_reported_and_later_datasets_are_not_run(self):
        def fail_evaluator(command, **kwargs):
            if Path(command[2]).name == "evaluate.py":
                raise subprocess.CalledProcessError(2, command)
            return self.fake_run(command, **kwargs)

        code, summary, _ = self.invoke(["--all"], fail_evaluator)
        self.assertEqual(code, 1)
        self.assertIn("FAILED during evaluation / MLflow logging", summary)
        self.assertIn("Not run after failure: beta", summary)
        self.assertTrue((self.root / "modelling/data/alpha/test/first.trec").is_file())

    def test_split_depth_and_method_options_are_forwarded(self):
        self.dataset("alpha", "train")
        code, _, _ = self.invoke([
            "--dataset", "alpha", "--split", "train", "--top-k", "7",
            "--method-option", "second:language=it", "--no-tracking",
        ])
        self.assertEqual(code, 0)
        for command in self.commands:
            self.assertEqual(command[command.index("--split") + 1], "train")
            self.assertIn("--overwrite", command)
            if Path(command[2]).name != "evaluate.py":
                self.assertEqual(command[command.index("--top-k") + 1], "7")
                self.assertEqual("--language=it" in command, Path(command[2]).stem == "second")
                self.assertNotIn("--no-tracking", command)
        self.assertIn("--no-tracking", self.evaluation_commands()[0])

    def test_tracking_options_are_only_forwarded_to_evaluator(self):
        code, _, _ = self.invoke([
            "--dataset", "alpha", "--tracking-uri", "http://localhost:5000", "--log-rankings",
        ])
        self.assertEqual(code, 0)
        for command in self.commands:
            expected = Path(command[2]).name == "evaluate.py"
            self.assertEqual("--tracking-uri" in command, expected)
            self.assertEqual("--log-rankings" in command, expected)

    def test_real_scripts_and_evaluator_on_a_tiny_new_dataset(self):
        # A separate repository copy keeps real datasets, reports, and MLflow untouched.
        isolated = self.root / "integration"
        (isolated / "modelling/scripts").mkdir(parents=True)
        (isolated / "evaluation/scripts").mkdir(parents=True)
        shutil.copyfile(REPOSITORY / "run_experiments.py", isolated / "run_experiments.py")
        for script in (REPOSITORY / "modelling/scripts").glob("*.py"):
            shutil.copyfile(script, isolated / "modelling/scripts" / script.name)
        shutil.copyfile(REPOSITORY / "evaluation/scripts/evaluate.py", isolated / "evaluation/scripts/evaluate.py")
        shutil.copytree(self.data / "alpha", isolated / "data-preparation/data/new-dataset")
        result = subprocess.run([
            sys.executable, "-B", str(isolated / "run_experiments.py"),
            "--dataset", "new-dataset", "--top-k", "3", "--no-tracking",
            "--method-option", "1.7.1-stopword-overlap:language=it",
        ], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((isolated / "evaluation/data/new-dataset/test/report.json").read_text())
        expected = sorted(p.stem for p in (REPOSITORY / "modelling/scripts").glob("*.py"))
        self.assertEqual([r["name"] for r in report["runs"]], [name + ".trec" for name in expected])
        self.assertTrue(all(r["aggregate"]["AP"] == 1 for r in report["runs"]))
        self.assertTrue(all(r["retrieval"]["parameters"]["top_k"] == 3 for r in report["runs"]))
        self.assertFalse((isolated / "tracking").exists())


if __name__ == "__main__":
    unittest.main()
