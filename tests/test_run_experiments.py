"""Test generic experiment discovery, expansion, and transactional execution."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments import artifacts, cli, configuration, discovery, planning

REPOSITORY = Path(__file__).resolve().parents[1]


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
            self.script(name)
        evaluator = self.root / "evaluation/scripts/evaluate.py"
        evaluator.parent.mkdir(parents=True)
        evaluator.write_text("# Evaluator stand-in.\n", encoding="utf-8")
        self.commands = []

    def dataset(self, name, split="test"):
        path = self.data / name
        rows = {
            "corpus.jsonl": {"doc_id": "001", "text": "legal evidence"},
            f"queries/{split}.jsonl": {"query_id": "002", "text": "legal evidence"},
            f"qrels/{split}.jsonl": {
                "query_id": "002", "doc_id": "001", "relevance": 1,
            },
        }
        for relative, row in rows.items():
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    def script(self, name, dimensions=None):
        declaration = ""
        if dimensions is not None:
            declaration = (
                "EXPERIMENT_CAPABILITY = "
                + repr({"version": 1, "dimensions": dimensions})
                + "\n"
            )
        path = self.scripts / f"{name}.py"
        path.write_text(declaration + "# Retrieval entry point.\n", encoding="utf-8")
        return path

    def config(self, variants):
        path = self.root / "variants.json"
        path.write_text(
            json.dumps({"version": 1, "variants": variants}, indent=2),
            encoding="utf-8",
        )
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
            (output / f"{script.stem}.trec").write_text(
                f"002 Q0 001 1 1 {script.stem}\n", encoding="utf-8"
            )
            parameters = {
                "method": script.stem,
                "top_k": int(command[command.index("--top-k") + 1]),
            }
            for argument in command:
                if argument.startswith("--embedding-model="):
                    parameters["embedding_model"] = argument.split("=", 1)[1]
                elif argument.startswith("--dense-embedding-model="):
                    parameters["dense_embedding_model"] = argument.split("=", 1)[1]
                elif argument.startswith("--sparse-embedding-model="):
                    parameters["sparse_embedding_model"] = argument.split("=", 1)[1]
            metadata = {
                "dataset": command[command.index("--dataset") + 1],
                "split": command[command.index("--split") + 1],
                "parameters": parameters,
            }
            (output / f"{script.stem}.metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
        else:
            for ranking in self.rankings(command):
                self.assertTrue(ranking.read_text().startswith("002 Q0 001 1 1 "))
                self.assertTrue(ranking.with_suffix(".metadata.json").is_file())
            for name in ("report.json", "summary.csv", "per-query.csv"):
                (output / name).write_text("fresh report\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    def invoke(self, arguments, runner=None):
        with patch.object(cli, "ROOT_DIR", self.root):
            with patch.object(
                subprocess, "run", side_effect=runner or self.fake_run
            ):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    with contextlib.redirect_stderr(io.StringIO()) as errors:
                        code = cli.main(arguments)
        return code, output.getvalue(), errors.getvalue()

    @staticmethod
    def rankings(command):
        return [
            Path(value)
            for value in command[
                command.index("--run") + 1:command.index("--output-dir")
            ]
        ]

    def retrieval_commands(self):
        return [
            command for command in self.commands
            if Path(command[2]).name != "evaluate.py"
        ]

    def evaluation_commands(self):
        return [
            command for command in self.commands
            if Path(command[2]).name == "evaluate.py"
        ]

    def test_dataset_discovery_requires_nonempty_files_for_selected_split(self):
        self.dataset("train-only", "train")
        incomplete = self.dataset("incomplete")
        (incomplete / "qrels/test.jsonl").unlink()
        empty = self.dataset("empty")
        (empty / "corpus.jsonl").write_text("", encoding="utf-8")
        ready, rejected = discovery.discover_datasets(self.data, "test")
        self.assertEqual(list(ready), ["alpha", "beta"])
        self.assertEqual(rejected["incomplete"], ["qrels/test.jsonl"])
        self.assertEqual(rejected["empty"], ["corpus.jsonl"])

    def test_filesystem_is_the_only_method_inventory(self):
        for name in ("__init__.py", "_helper.py", "notes.md"):
            (self.scripts / name).write_text("", encoding="utf-8")
        self.assertEqual(
            list(discovery.discover_methods(self.scripts)), ["first", "second"]
        )
        self.script("future-ordinary")
        methods = discovery.discover_methods(self.scripts)
        self.assertIn("future-ordinary", methods)
        self.assertIsNone(discovery.read_experiment_capability(methods["first"]))

    def test_generic_capabilities_expand_one_and_multiple_dimensions(self):
        self.script("consumer-a", {"models_a": "model-a"})
        self.script("consumer-b", {"models_b": "model-b"})
        self.script(
            "consumer-both", {"models_a": "left-model", "models_b": "right-model"}
        )
        config = self.config({
            "models_a": [
                {"id": "a1", "value": "models/a1"},
                {"id": "a2", "value": "models/a2"},
            ],
            "models_b": [
                {"id": "b1", "value": "models/b1"},
                {"id": "b2", "value": "models/b2"},
            ],
        })
        methods = discovery.discover_methods(self.scripts)
        experiments = planning.load_benchmark(config, methods)
        by_method = {}
        for experiment in experiments:
            by_method.setdefault(experiment.method, []).append(experiment)
        self.assertNotIn("first", by_method)
        self.assertEqual(len(by_method["consumer-a"]), 2)
        self.assertEqual(len(by_method["consumer-b"]), 2)
        self.assertEqual(len(by_method["consumer-both"]), 4)
        self.assertEqual(
            by_method["consumer-both"][-1].options,
            ("--left-model=models/a2", "--right-model=models/b2"),
        )

    def test_future_configurable_method_opts_in_without_orchestrator_changes(self):
        future = self.script("9.9.9-future", {"rerankers": "reranker"})
        config = self.config({
            "rerankers": [
                {"id": "small", "value": "example/small"},
                {"id": "large", "value": "example/large"},
            ]
        })
        methods = discovery.discover_methods(self.scripts)
        experiments = planning.load_benchmark(config, methods)
        self.assertEqual(
            [(item.script, item.experiment_id, item.options) for item in experiments],
            [
                (future, "9.9.9-future__small", ("--reranker=example/small",)),
                (future, "9.9.9-future__large", ("--reranker=example/large",)),
            ],
        )

    def test_orchestrator_has_no_specific_vector_method_mapping(self):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (REPOSITORY / "experiments").glob("*.py")
        )
        for method in (
            "2.1.1-dense-retrieval",
            "2.2.1-sparse-retrieval",
            "2.3.1-hybrid-retrieval",
        ):
            self.assertNotIn(method, source)
        self.assertNotIn("startswith(\"2.", source)

    def test_benchmark_runs_only_capable_methods_and_cartesian_product(self):
        self.script("configurable", {"models_a": "embedding-model"})
        self.script(
            "combined", {"models_a": "dense-model", "models_b": "sparse-model"}
        )
        self.config({
            "models_a": [
                {"id": "a1", "value": "models/a1"},
                {"id": "a2", "value": "models/a2"},
            ],
            "models_b": [{"id": "b1", "value": "models/b1"}],
        })
        code, output, _ = self.invoke([
            "--dataset", "alpha", "--benchmark",
            "--benchmark-config", "variants.json",
        ])
        self.assertEqual(code, 0)
        methods = [Path(command[2]).stem for command in self.retrieval_commands()]
        self.assertEqual(methods, ["combined", "combined", "configurable", "configurable"])
        self.assertNotIn("first", methods)
        self.assertIn("4/4 configurations available", output)
        rankings = self.rankings(self.evaluation_commands()[0])
        self.assertEqual(len(rankings), len(set(rankings)))

    def test_experiment_ids_are_deterministic_human_readable_and_unique(self):
        self.script("configurable", {"models": "model"})
        config = self.config({
            "models": [
                {"id": "model-a", "value": "example/a"},
                {"id": "model-b", "value": "example/b"},
            ]
        })
        methods = discovery.discover_methods(self.scripts)
        first = planning.load_benchmark(config, methods)
        second = planning.load_benchmark(config, methods)
        expected = ["configurable__model-a", "configurable__model-b"]
        self.assertEqual([item.experiment_id for item in first], expected)
        self.assertEqual([item.experiment_id for item in second], expected)
        self.assertEqual(len(expected), len(set(expected)))

    def test_invalid_duplicate_ids_and_missing_dimensions_are_clear(self):
        self.script("configurable", {"models": "model", "prompts": "prompt"})
        methods = discovery.discover_methods(self.scripts)
        duplicate = self.config({
            "models": [
                {"id": "same", "value": "example/a"},
                {"id": "same", "value": "example/b"},
            ],
            "prompts": [{"id": "prompt", "value": "query"}],
        })
        with self.assertRaisesRegex(ValueError, "duplicate benchmark variant id"):
            planning.load_benchmark(duplicate, methods)
        missing = self.config({
            "models": [{"id": "model", "value": "example/a"}],
        })
        with self.assertRaisesRegex(ValueError, "requires missing dimension: prompts"):
            planning.load_benchmark(missing, methods)

    def test_success_is_published_and_evaluated_after_later_failure(self):
        self.script("configurable", {"models": "embedding-model"})
        self.config({
            "models": [
                {"id": "good", "value": "models/good"},
                {"id": "bad", "value": "models/bad"},
            ]
        })
        output = self.root / "modelling/data/alpha/test"
        output.mkdir(parents=True)
        stale_ranking = output / "configurable__bad.trec"
        stale_metadata = output / "configurable__bad.metadata.json"
        stale_ranking.write_text("stale ranking\n", encoding="utf-8")
        stale_metadata.write_text("stale metadata\n", encoding="utf-8")

        def fail_bad(command, **kwargs):
            if "--embedding-model=models/bad" in command:
                directory = Path(command[command.index("--output-dir") + 1])
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "configurable.trec").write_text(
                    "incomplete\n", encoding="utf-8"
                )
                raise subprocess.CalledProcessError(7, command)
            return self.fake_run(command, **kwargs)

        code, summary, errors = self.invoke([
            "--dataset", "alpha", "--benchmark",
            "--benchmark-config", "variants.json",
        ], fail_bad)
        self.assertEqual(code, 1)
        self.assertIn("COMPLETED WITH FAILURES", summary)
        self.assertIn("exit code 7", errors)
        current = output / "configurable__good.trec"
        self.assertEqual(current.read_text(), "002 Q0 001 1 1 configurable\n")
        self.assertEqual(stale_ranking.read_text(), "stale ranking\n")
        self.assertEqual(stale_metadata.read_text(), "stale metadata\n")
        self.assertEqual(self.rankings(self.evaluation_commands()[0]), [current])
        self.assertFalse(list(output.glob(".configurable*")))

    def test_zero_exit_without_both_outputs_never_publishes_or_evaluates_stale(self):
        self.script("configurable", {"models": "model"})
        self.config({"models": [{"id": "only", "value": "example/model"}]})
        output = self.root / "modelling/data/alpha/test"
        output.mkdir(parents=True)
        stale = output / "configurable__only.trec"
        stale.write_text("stale\n", encoding="utf-8")
        code, _, errors = self.invoke(
            [
                "--dataset", "alpha", "--benchmark",
                "--benchmark-config", "variants.json",
            ],
            lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
        )
        self.assertEqual(code, 1)
        self.assertIn("must create configurable.trec", errors)
        self.assertEqual(stale.read_text(), "stale\n")
        self.assertEqual(self.evaluation_commands(), [])

    def test_invalid_ranking_is_not_published_or_evaluated(self):
        self.script("configurable", {"models": "model"})
        self.config({"models": [{"id": "only", "value": "example/model"}]})

        def malformed(command, **kwargs):
            result = self.fake_run(command, **kwargs)
            script = Path(command[2])
            if script.stem == "configurable":
                output = Path(command[command.index("--output-dir") + 1])
                (output / "configurable.trec").write_text(
                    "not a TREC ranking\n", encoding="utf-8"
                )
            return result

        code, _, errors = self.invoke([
            "--dataset", "alpha", "--benchmark",
            "--benchmark-config", "variants.json",
        ], malformed)
        self.assertEqual(code, 1)
        self.assertIn("expected 6 TREC columns", errors)
        self.assertFalse(
            (self.root / "modelling/data/alpha/test/configurable__only.trec").exists()
        )
        self.assertEqual(self.evaluation_commands(), [])

    def test_manual_lexical_and_vector_execution_and_options_remain_generic(self):
        vector = "2.1.1-dense-retrieval"
        self.script(vector, {"dense_models": "embedding-model"})
        code, _, _ = self.invoke(["--dataset", "alpha", "--method", "first"])
        self.assertEqual(code, 0)
        self.assertEqual(Path(self.retrieval_commands()[0][2]).stem, "first")
        self.commands.clear()
        code, _, _ = self.invoke([
            "--dataset", "alpha", "--method", vector,
            "--method-option", f"{vector}:embedding-model=other/model",
            "--method-option", f"{vector}:rebuild-index=true",
        ])
        self.assertEqual(code, 0)
        self.assertIn("--embedding-model=other/model", self.retrieval_commands()[0])
        self.assertIn("--rebuild-index=true", self.retrieval_commands()[0])

    def test_metadata_distinguishes_method_configuration_and_model(self):
        self.script("configurable", {"models": "embedding-model"})
        self.config({
            "models": [
                {"id": "a", "value": "example/a"},
                {"id": "b", "value": "example/b"},
            ]
        })
        code, _, _ = self.invoke([
            "--dataset", "alpha", "--benchmark",
            "--benchmark-config", "variants.json",
        ])
        self.assertEqual(code, 0)
        for ranking in self.rankings(self.evaluation_commands()[0]):
            parameters = json.loads(
                ranking.with_suffix(".metadata.json").read_text(encoding="utf-8")
            )["parameters"]
            self.assertEqual(parameters["retrieval_method"], "configurable")
            self.assertEqual(parameters["experiment_id"], ranking.stem)
            self.assertIn(parameters["embedding_model"], {"example/a", "example/b"})

    def test_list_reports_methods_capabilities_and_expansion_without_running(self):
        self.script("configurable", {"models": "model"})
        self.config({"models": [{"id": "one", "value": "example/one"}]})
        code, output, _ = self.invoke([
            "--list", "--benchmark-config", "variants.json",
        ])
        self.assertEqual(code, 0)
        for text in (
            "Methods (discovered from modelling/scripts)",
            "first",
            "Benchmark-capable methods",
            "configurable: models",
            "configurable__one",
        ):
            self.assertIn(text, output)
        self.assertEqual(self.commands, [])
        self.assertFalse((self.root / "modelling/data").exists())

    def test_invalid_cli_combinations_fail_before_subprocesses(self):
        cases = [
            [],
            ["--dataset", "unknown"],
            ["--method", "unknown"],
            ["--all", "--dataset", "alpha"],
            ["--dataset", "alpha", "--benchmark", "--method", "first"],
            ["--dataset", "alpha", "--benchmark", "--method-option", "first:x=y"],
            ["--dataset", "alpha", "--benchmark-config", "variants.json"],
            ["--all", "--top-k", "0"],
            ["--all", "--no-tracking", "--tracking-uri", "http://example"],
            ["--all", "--method-option", "first:output=/tmp"],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as error:
                self.invoke(arguments)
            self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.commands, [])

    def test_standard_mode_preserves_dataset_level_transaction(self):
        output = self.root / "modelling/data/alpha/test"
        output.mkdir(parents=True)
        old = output / "first.trec"
        old.write_text("old ranking", encoding="utf-8")

        def fail_second(command, **kwargs):
            if Path(command[2]).stem == "second":
                raise subprocess.CalledProcessError(7, command)
            return self.fake_run(command, **kwargs)

        code, summary, _ = self.invoke(["--all"], fail_second)
        self.assertEqual(code, 1)
        self.assertIn("No staged rankings published", summary)
        self.assertIn("Not run after failure: beta", summary)
        self.assertEqual(old.read_text(), "old ranking")
        self.assertEqual(self.evaluation_commands(), [])

    def test_tracking_options_are_forwarded_only_to_evaluator(self):
        code, _, _ = self.invoke([
            "--dataset", "alpha", "--tracking-uri", "http://localhost:5000",
            "--log-rankings",
        ])
        self.assertEqual(code, 0)
        for command in self.commands:
            expected = Path(command[2]).name == "evaluate.py"
            self.assertEqual("--tracking-uri" in command, expected)
            self.assertEqual("--log-rankings" in command, expected)

    def test_repository_capabilities_expand_selected_models_without_imports(self):
        methods = discovery.discover_methods(REPOSITORY / "modelling/scripts")
        experiments = planning.load_benchmark(
            configuration.resolve_config_path(), methods
        )
        self.assertEqual(len(experiments), 5)
        counts = {}
        for experiment in experiments:
            counts[experiment.method] = counts.get(experiment.method, 0) + 1
        self.assertEqual(sorted(counts.values()), [1, 2, 2])
        self.assertTrue(all(item.experiment_id.startswith(item.method + "__") for item in experiments))

    def test_resume_reuses_artifacts_without_retrieval_or_rewriting(self):
        self.script("configurable", {"models": "embedding-model"})
        self.config({"models": [{"id": "one", "value": "example/one"}]})
        args = ["--dataset", "alpha", "--benchmark", "--benchmark-config", "variants.json"]
        self.assertEqual(self.invoke(args)[0], 0)
        directory = self.root / "modelling/data/alpha/test"
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
        self.commands.clear()
        code, summary, _ = self.invoke(args + ["--resume"])
        self.assertEqual(code, 0)
        self.assertEqual(self.retrieval_commands(), [])
        self.assertEqual(len(self.evaluation_commands()), 1)
        self.assertIn("Reused: configurable__one", summary)
        self.assertEqual(
            before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.iterdir()}
        )

    def test_resume_changed_value_or_depth_runs_retrieval(self):
        self.script("configurable", {"models": "embedding-model"})
        self.config({"models": [{"id": "same-id", "value": "example/old"}]})
        args = ["--dataset", "alpha", "--benchmark", "--benchmark-config", "variants.json"]
        self.assertEqual(self.invoke(args)[0], 0)
        self.config({"models": [{"id": "same-id", "value": "example/new"}]})
        self.commands.clear()
        self.assertEqual(self.invoke(args + ["--resume"])[0], 0)
        self.assertEqual(len(self.retrieval_commands()), 1)
        self.assertIn("--embedding-model=example/new", self.retrieval_commands()[0])
        self.commands.clear()
        self.assertEqual(self.invoke(args + ["--resume", "--top-k", "7"])[0], 0)
        self.assertEqual(len(self.retrieval_commands()), 1)

    def test_resume_requires_benchmark(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke(["--dataset", "alpha", "--resume"])
        self.assertEqual(error.exception.code, 2)

    def test_interruption_evaluates_successes_and_returns_130(self):
        self.script("configurable", {"models": "embedding-model"})
        self.config({"models": [
            {"id": "good", "value": "example/good"},
            {"id": "interrupted", "value": "example/interrupted"},
        ]})

        def interrupt(command, **kwargs):
            if "--embedding-model=example/interrupted" in command:
                raise KeyboardInterrupt
            return self.fake_run(command, **kwargs)

        code, _, _ = self.invoke([
            "--dataset", "alpha", "--benchmark", "--benchmark-config", "variants.json",
        ], interrupt)
        self.assertEqual(code, 130)
        self.assertEqual(len(self.evaluation_commands()), 1)
        self.assertEqual(
            [path.stem for path in self.rankings(self.evaluation_commands()[0])],
            ["configurable__good"],
        )

    def test_evaluation_failure_is_nonzero_and_keeps_rankings(self):
        def fail_evaluation(command, **kwargs):
            if Path(command[2]).name == "evaluate.py":
                raise subprocess.CalledProcessError(9, command)
            return self.fake_run(command, **kwargs)

        code, summary, _ = self.invoke(["--dataset", "alpha"], fail_evaluation)
        self.assertEqual(code, 1)
        self.assertIn("FAILED during evaluation", summary)
        self.assertTrue((self.root / "modelling/data/alpha/test/first.trec").is_file())

    def test_split_depth_and_local_evaluation_flags(self):
        self.dataset("alpha", "train")
        code, _, _ = self.invoke([
            "--dataset", "alpha", "--split", "train", "--top-k", "7",
            "--method", "first", "--no-tracking",
        ])
        self.assertEqual(code, 0)
        for command in self.commands:
            self.assertEqual(command[command.index("--split") + 1], "train")
        command = self.retrieval_commands()[0]
        self.assertEqual(command[command.index("--top-k") + 1], "7")
        self.assertIn("--no-tracking", self.evaluation_commands()[0])


if __name__ == "__main__":
    unittest.main()
