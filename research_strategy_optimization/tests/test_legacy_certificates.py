from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_strategy_optimization.evaluation.legacy_certificates import (
    LEGACY_CERTIFICATE_ARTIFACT_SCOPE,
    annotate_legacy_certificate,
    build_legacy_certificate_manifest,
)


class LegacyCertificateScopeTests(unittest.TestCase):
    def test_annotation_preserves_historical_pass_but_fails_current_scope_closed(self):
        payload = annotate_legacy_certificate({"certificate_pass": True, "autonomous": True})
        self.assertTrue(payload["certificate_pass"])
        self.assertTrue(payload["autonomous"])
        self.assertTrue(payload["legacy"])
        self.assertEqual(payload["artifact_scope"], LEGACY_CERTIFICATE_ARTIFACT_SCOPE)
        self.assertFalse(payload["current_formal_claim_authorized"])
        self.assertFalse(payload["legacy_scope"]["current_formal_claim_authorized"])

    def test_manifest_counts_legacy_pass_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "certificate_old.json").write_text(
                json.dumps(annotate_legacy_certificate({"certificate_pass": True, "autonomous": True})),
                encoding="utf-8",
            )
            manifest = build_legacy_certificate_manifest(root)
            self.assertEqual(manifest["certificate_count"], 1)
            self.assertEqual(manifest["certificate_pass_true_count"], 1)
            self.assertTrue(manifest["legacy"])
            self.assertEqual(manifest["files"][0]["artifact_scope"], LEGACY_CERTIFICATE_ARTIFACT_SCOPE)

    def test_manifest_ignores_explicitly_current_certificate_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "certificate_current.json").write_text(
                json.dumps({
                    "certificate_pass": True,
                    "artifact_scope": "current_open_ended_certificate",
                }),
                encoding="utf-8",
            )
            manifest = build_legacy_certificate_manifest(root)
            self.assertEqual(manifest["certificate_count"], 0)


if __name__ == "__main__":
    unittest.main()
