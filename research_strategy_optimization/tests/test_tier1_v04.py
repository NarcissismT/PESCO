from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_strategy_optimization.environments.tier1_benchmark import (
    FAMILY_CAUSAL_CONFOUNDING,
)
from research_strategy_optimization.evaluation.tier1_v04 import (
    TRACK_ORACLE_STATE,
    TRACK_RAW_EVIDENCE,
    build_tier1_v04_benchmark,
    candidate_scenarios,
    posterior_from_evidence,
    tier1_v04_manifest,
)
from research_strategy_optimization.evaluation.tier1_v04_extended import (
    V04_EXTENDED_FAMILIES,
    V04_EXTENDED_EXPLORATION_SEEDS,
    build_tier1_v04_extended_benchmark,
)
from scripts.run_tier1_v04_extended import _select_decision_questions
from research_strategy_optimization.schemas import ResearchAction


class Tier1V04HardeningTests(unittest.TestCase):
    def test_manifest_has_dual_tracks_and_removes_legacy_target_table(self) -> None:
        benchmark = build_tier1_v04_benchmark()
        hidden = tier1_v04_manifest(benchmark, include_hidden=True)
        public = tier1_v04_manifest(benchmark, include_hidden=False)
        self.assertEqual(hidden["schema_version"], "pesco_tier1_benchmark_v0.4")
        self.assertEqual(
            set(hidden["dual_tracks"]),
            {TRACK_ORACLE_STATE, TRACK_RAW_EVIDENCE},
        )
        self.assertTrue(hidden["posterior"]["method_b_hindsight_excluded"])
        self.assertTrue(all("target_actions" not in q for q in hidden["questions"]))
        encoded_public = json.dumps(public, sort_keys=True)
        self.assertNotIn("target_actions", encoded_public)
        self.assertNotIn("__invalid", encoded_public)

    def test_candidate_bank_is_leave_one_question_out_same_family(self) -> None:
        benchmark = build_tier1_v04_benchmark()
        question = next(q for q in benchmark.questions if q.family == FAMILY_CAUSAL_CONFOUNDING)
        candidates = candidate_scenarios(benchmark, question)
        self.assertEqual(len(candidates), 8)
        self.assertTrue(all(item.family == question.family for item in candidates))
        self.assertTrue(all(item.source_question_id != question.question_id for item in candidates))
        self.assertAlmostEqual(sum(item.prior for item in candidates), 1.0, places=12)

    def test_posteriors_normalize_for_both_tracks_without_target_labels(self) -> None:
        benchmark = build_tier1_v04_benchmark()
        question = next(q for q in benchmark.questions if q.family == FAMILY_CAUSAL_CONFOUNDING)
        candidates = candidate_scenarios(benchmark, question)
        raw = {
            "effect_estimate": -0.08,
            "confidence_interval": [-0.10, -0.06],
            "sample_size": 1000,
            "validity_signals": ["tier1_numpy_backend", "treatment_assignment_independent"],
            "task_family": question.family,
        }
        oracle = {"evidence_state": "refuted", "task_family": question.family}
        for track, evidence in ((TRACK_RAW_EVIDENCE, raw), (TRACK_ORACLE_STATE, oracle)):
            posterior = posterior_from_evidence(evidence, candidates, track=track)
            self.assertEqual(set(posterior), {item.key for item in candidates})
            self.assertAlmostEqual(sum(posterior.values()), 1.0, places=12)
            self.assertTrue(all(value >= 0.0 for value in posterior.values()))

    def test_causal_refuted_variant_has_no_hindsight_target_dependency(self) -> None:
        # This is intentionally a structural test rather than a performance claim:
        # v0.4's planner source is the leave-one-question-out posterior rule, not the
        # v0.3 hidden method-B target table.
        benchmark = build_tier1_v04_benchmark()
        question = next(
            q for q in benchmark.questions
            if q.family == FAMILY_CAUSAL_CONFOUNDING and q.variant == 3
        )
        world = next(w for w in question.worlds if w.kind == "refuted")
        self.assertEqual(question.target_action(world.world_id), ResearchAction.SWITCH)
        candidates = candidate_scenarios(benchmark, question)
        self.assertTrue(all(item.source_question_id != question.question_id for item in candidates))
        # The current variant's positive method-B effect is absent from the candidate
        # bank, so a planner cannot obtain it by looking up the current hidden world.
        self.assertNotIn(world.world_id, {item.world_id for item in candidates})

    def test_extended_benchmark_has_required_coverage_and_closed_formal_clusters(self) -> None:
        benchmark = build_tier1_v04_extended_benchmark()
        self.assertEqual(len(benchmark.questions), 64)
        self.assertEqual(len(benchmark.worlds), 256)
        self.assertEqual(len(V04_EXTENDED_FAMILIES), 8)
        self.assertEqual(
            {family: sum(q.family == family for q in benchmark.questions) for family in V04_EXTENDED_FAMILIES},
            {family: 8 for family in V04_EXTENDED_FAMILIES},
        )
        manifest = benchmark.manifest(exploration_seeds=V04_EXTENDED_EXPLORATION_SEEDS)
        self.assertEqual(manifest["counts_by_split"], {
            "train": 24,
            "dev": 20,
            "diagnostic_ood": 20,
        })
        self.assertEqual(set(manifest["counts_by_split"]), {"train", "dev", "diagnostic_ood"})
        self.assertFalse(manifest["formal_comparison_authorized"])

    def test_extended_manifest_has_same_question_pairs_and_no_targets(self) -> None:
        benchmark = build_tier1_v04_extended_benchmark()
        public = benchmark.manifest(include_hidden=False, exploration_seeds=tuple(range(8)))
        self.assertTrue(all(q["world_count"] == 4 for q in public["questions"]))
        encoded = json.dumps(public, sort_keys=True)
        self.assertNotIn("legacy_target_actions_audit_only", encoded)
        self.assertNotIn("true_effect_b", encoded)

    def test_bounded_decision_subset_is_family_stratified(self) -> None:
        benchmark = build_tier1_v04_extended_benchmark()
        selected = _select_decision_questions(benchmark.questions, 16)
        self.assertEqual(len(selected), 16)
        self.assertEqual({question.family for question in selected}, set(V04_EXTENDED_FAMILIES))
        self.assertTrue(any(question.split == "diagnostic_ood" for question in selected))
        self.assertTrue(any(question.family == FAMILY_CAUSAL_CONFOUNDING for question in selected))


if __name__ == "__main__":
    unittest.main()
