"""Focused contract, catalog, artifact, and package-entry checks."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments import artifacts, cli, configuration, discovery, planning

ROOT = Path(__file__).resolve().parents[1]


class ModuleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / "future.py"

    def capability(self, value):
        self.script.write_text(
            "raise RuntimeError('must not execute during discovery')\n"
            + "EXPERIMENT_CAPABILITY = " + value + "\n", encoding="utf-8"
        )
        return discovery.read_experiment_capability(self.script)

    def test_capability_is_literal_and_does_not_import(self):
        result = self.capability(
            repr({"version": 1, "dimensions": {"encoders": "encoder"}})
        )
        self.assertEqual(result["dimensions"], {"encoders": "encoder"})

    def test_invalid_capabilities_are_rejected(self):
        values = [
            "dict(version=1)",
            repr({"version": 2, "dimensions": {"encoders": "encoder"}}),
            repr({"version": 1, "dimensions": {}}),
            repr({"version": 1, "dimensions": {"encoders": "output-dir"}}),
            repr({"version": 1, "dimensions": {"a": "model", "b": "model"}}),
            repr({"version": 1, "dimensions": {"a": "model"}, "parameter_names": {"other": "x"}}),
            repr({"version": 1, "dimensions": {"a": "model"}, "parameters": {"x": []}}),
            repr({"version": 1, "dimensions": {"a": "left", "b": "right"},
                  "parameter_names": {"left": "same", "right": "same"}}),
        ]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.capability(value)

    def test_config_path_and_errors(self):
        path = configuration.resolve_config_path()
        self.assertEqual(path, ROOT / "modelling/configs/benchmark-variants.json")
        self.assertEqual(
            configuration.resolve_config_path("custom.json", self.root),
            self.root / "custom.json",
        )
        self.assertEqual(configuration.resolve_config_path(path, self.root), path)
        self.assertEqual(len(configuration.load_variants(path)["dense_models"]), 2)
        missing = self.root / "missing.json"
        with self.assertRaises(FileNotFoundError):
            configuration.load_variants(missing)
        for content in ("{", "[]", '{"version":2,"variants":{}}'):
            missing.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                configuration.load_variants(missing)

    def test_config_rejects_invalid_ids(self):
        path = self.root / "variants.json"
        for identifier in ("../bad", "Uppercase", "has space", "", "x" * 64):
            path.write_text(json.dumps({
                "version": 1,
                "variants": {"models": [{"id": identifier, "value": "example/model"}]},
            }), encoding="utf-8")
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "slug"):
                configuration.load_variants(path)

    def test_plan_maps_metadata_names_and_search_parameters_generically(self):
        capability = {
            "version": 1, "dimensions": {"encoders": "encoder"},
            "parameter_names": {"encoder": "actual_encoder"},
            "parameters": {"search_depth": 64},
        }
        plan = planning.plan_benchmark(
            {"future": self.script}, {"future": capability},
            {"encoders": [("small", "example/encoder")]},
        )
        self.assertEqual(plan[0].options, ("--encoder=example/encoder",))
        self.assertEqual(dict(plan[0].parameters), {
            "actual_encoder": "example/encoder", "search_depth": 64,
        })

    def test_module_cli_and_retired_root_entrypoint(self):
        self.assertFalse((ROOT / "run_experiments.py").exists())
        result = subprocess.run(
            [sys.executable, "-B", "-m", "experiments.run_experiments", "--list"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("5 configurations", result.stdout)
        self.assertIn("1.1.1-term-overlap", result.stdout)
        self.assertNotIn("multilingual-mpnet", result.stdout)
        result = subprocess.run(
            [sys.executable, "-B", "-m", "experiments.run_experiments",
             "--dataset", "scifact", "--resume"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--resume requires --benchmark", result.stderr)

    def test_list_uses_package_root_even_from_another_working_directory(self):
        # The package must already be importable, as for an installed/imported module.
        import os
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(["--list"]), 0)
            self.assertIn("5 configurations", output.getvalue())
        finally:
            os.chdir(previous)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.experiment = planning.Experiment(
            "future", self.root / "future.py", "future__small",
            ("--encoder=example/model",),
            (("actual_model", "example/model"), ("search_depth", 64)),
        )
        self.ranking, self.sidecar = artifacts.artifact_paths(
            self.root, self.experiment.experiment_id
        )
        self.ranking.write_text("q Q0 d 1 1 future\n", encoding="utf-8")
        self.metadata = {
            "dataset": "fixture", "split": "test",
            "parameters": {
                "retrieval_method": "future", "experiment_id": "future__small",
                "top_k": 1000, "actual_model": "example/model", "search_depth": 64,
            },
        }
        self.write_metadata()

    def write_metadata(self):
        self.sidecar.write_text(json.dumps(self.metadata), encoding="utf-8")

    def reusable(self, **kwargs):
        return artifacts.reusable_benchmark_output(
            self.root, "fixture", "test", kwargs.get("experiment", self.experiment),
            kwargs.get("top_k", 1000),
        )

    def test_valid_pair_and_changed_parameters(self):
        self.assertTrue(self.reusable())
        self.assertFalse(self.reusable(top_k=10))
        self.assertFalse(self.reusable(experiment=replace(
            self.experiment, parameters=(("actual_model", "another/model"),)
        )))
        self.metadata["parameters"]["search_depth"] = 32
        self.write_metadata()
        self.assertFalse(self.reusable())

    def test_invalid_metadata_never_reuses(self):
        for value in ([], None, {}, {"dataset": "wrong"}, {
            "dataset": "fixture", "split": "test", "parameters": {"x": float("nan")},
        }):
            self.sidecar.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(value=value):
                self.assertFalse(self.reusable())
        self.sidecar.write_text("{", encoding="utf-8")
        self.assertFalse(self.reusable())

    def test_identity_mismatch_never_reuses(self):
        for key, value in (("retrieval_method", "other"), ("experiment_id", "other"),
                           ("actual_model", "changed")):
            original = self.metadata["parameters"][key]
            self.metadata["parameters"][key] = value
            self.write_metadata()
            self.assertFalse(self.reusable())
            self.metadata["parameters"][key] = original

    def test_partial_and_invalid_pairs_never_reuse(self):
        self.sidecar.unlink()
        self.assertFalse(self.reusable())
        self.write_metadata()
        for content in ("", "invalid", "q Q0 d 1 nan tag\n",
                        "q Q0 d 1 1 tag\nq Q0 d 2 0 tag\n"):
            self.ranking.write_text(content, encoding="utf-8")
            with self.subTest(content=content):
                self.assertFalse(self.reusable())
        self.ranking.unlink()
        self.assertFalse(self.reusable())

    def test_failed_second_publication_restores_previous_pair(self):
        staging = self.root / "staging"
        staging.mkdir()
        ranking = staging / "future.trec"
        metadata = staging / "future.metadata.json"
        ranking.write_text("new ranking", encoding="utf-8")
        metadata.write_text("new metadata", encoding="utf-8")
        before = (self.ranking.read_bytes(), self.sidecar.read_bytes())
        original = Path.replace

        def fail_second(path, destination):
            if path == metadata:
                raise OSError("injected publication failure")
            return original(path, destination)

        with patch.object(Path, "replace", fail_second), self.assertRaises(OSError):
            artifacts.publish_pair(ranking, metadata, self.root, self.experiment.experiment_id)
        self.assertEqual((self.ranking.read_bytes(), self.sidecar.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
