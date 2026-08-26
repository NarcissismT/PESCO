"""Fast provenance-manifest tests (no model or benchmark rerun)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_strategy_optimization.utils.run_manifest import (
    RUN_MANIFEST_SCHEMA,
    build_run_manifest,
    checkpoint_inventory,
    digest_paths,
    manifest_digest,
    seed_inventory,
    write_run_manifest,
)


class RunManifestTests(unittest.TestCase):
    def test_seed_inventory_preserves_categories_and_deduplicates_all(self) -> None:
        seeds = seed_inventory({"training": [17, 17], "exploration": (29, 41), "confirmation": [103]})
        self.assertEqual(seeds["training"], [17, 17])
        self.assertEqual(seeds["all"], [17, 29, 41, 103])

    def test_checkpoint_inventory_is_explicit_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            inventory = checkpoint_inventory(Path(temporary) / "missing")
        self.assertTrue(inventory["supplied"])
        self.assertFalse(inventory["available"])
        self.assertIsNone(inventory["full_checkpoint_digest"])
        self.assertEqual(inventory["reason"], "checkpoint_path_missing")

    def test_checkpoint_inventory_separates_weights_and_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model-00001.safetensors").write_bytes(b"weights-1")
            (root / "model-00002.safetensors").write_bytes(b"weights-2")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "spiece.model").write_bytes(b"sentencepiece")
            (root / "config.json").write_text("{}", encoding="utf-8")
            inventory = checkpoint_inventory(root)
        self.assertTrue(inventory["available"])
        self.assertEqual(inventory["weights_file_count"], 2)
        self.assertEqual(inventory["tokenizer_file_count"], 3)
        self.assertIsNotNone(inventory["weights_digest"])
        self.assertIsNotNone(inventory["tokenizer_digest"])
        self.assertIsNotNone(inventory["config_digest"])
        self.assertNotEqual(inventory["weights_digest"], inventory["tokenizer_digest"])

    def test_manifest_contains_required_runtime_command_source_and_data_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runner.py"
            data = root / "dataset.json"
            source.write_text("print('runner')\n", encoding="utf-8")
            data.write_text('{"n": 1}\n', encoding="utf-8")
            manifest = build_run_manifest(
                experiment="A_test",
                repo_root=root,
                command=["python", "runner.py", "--seed", "17"],
                runner_paths=[source],
                data_paths=[data],
                seeds={"training": [17], "exploration": [17, 29]},
                diagnostics={"posthoc": True},
                generated_at_utc="2026-01-01T00:00:00+00:00",
            )
            output = root / "run_manifest.json"
            write_run_manifest(output, manifest)
            loaded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], RUN_MANIFEST_SCHEMA)
        self.assertEqual(loaded["manifest_digest"], manifest["manifest_digest"])
        self.assertEqual(manifest["manifest_digest"], manifest_digest(manifest))
        self.assertEqual(manifest["command"]["argv"], ["python", "runner.py", "--seed", "17"])
        self.assertEqual(manifest["seeds"]["all"], [17, 29])
        self.assertTrue(manifest["source"]["available"])
        self.assertTrue(manifest["data"]["available"])
        self.assertIn("python_version", manifest["runtime"])
        self.assertIn("numpy", manifest["runtime"]["packages"])
        self.assertIn("torch", manifest["runtime"]["packages"])
        self.assertEqual(manifest["git_sha"], manifest["git"]["sha"])
        self.assertEqual(manifest["data_digest"], manifest["data"]["digest"])
        self.assertEqual(manifest["training_seed"], [17])
        self.assertIn("dependency_versions_digest", manifest["runtime"])
        self.assertIn("dependency_spec_digest", manifest)

    def test_digest_paths_is_deterministic_for_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            left = digest_paths([first, second], root=root)
            right = digest_paths([second, first], root=root)
        self.assertEqual(left["digest"], right["digest"])
        self.assertEqual(left["files"], right["files"])


if __name__ == "__main__":
    unittest.main()
