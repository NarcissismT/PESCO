from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ArtifactBoundaryTests(unittest.TestCase):
    def test_tier1_artifact_granularities_are_explicit(self):
        counts = json.loads((ROOT / "artifacts/tier1_v03/counts.json").read_text())
        audit = json.loads((ROOT / "artifacts/tier1_v03/count_semantics_audit.json").read_text())
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(counts["branch_groups"], 48)
        self.assertEqual(counts["question_world_group_count"], 48)
        self.assertEqual(counts["action_level_row_count"], 192)
        self.assertEqual(counts["seed_level_observation_count"], 768)
        groups = json.loads((ROOT / "artifacts/tier1_v03/question_world_groups.json").read_text())
        action_rows = json.loads((ROOT / "artifacts/tier1_v03/action_level_branches.json").read_text())
        self.assertEqual(len(groups), 48)
        self.assertEqual(len(action_rows), 192)
        self.assertTrue(all(row["record_granularity"] == "action_level" for row in action_rows))

    def test_legacy_certificate_scope_conflict_is_machine_readable(self):
        policy = json.loads((ROOT / "artifacts/pesco_pilot/discovery_policy.json").read_text())
        manifest = json.loads((ROOT / "artifacts/pesco_pilot/legacy_certificates_manifest.json").read_text())
        self.assertFalse(policy["current_certificate_claim_authorized"])
        self.assertEqual(policy["legacy_certificate_artifact_scope"], "legacy_certificate_evidence")
        self.assertEqual(manifest["certificate_pass_true_count"], 4)
        for path in (ROOT / "artifacts/pesco_pilot").glob("certificate_*.json"):
            payload = json.loads(path.read_text())
            self.assertTrue(payload["legacy"])
            self.assertFalse(payload["current_formal_claim_authorized"])


if __name__ == "__main__":
    unittest.main()
