#!/usr/bin/env python3
"""Finalize P2.3 promotion-v2 collection after both tracks are present."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p22_diagnostics import P22Config, _canonical_reversals
from research_strategy_optimization.evaluation.tier1_p23_dataset import P23_COUNTS, P23_GENERATOR_VERSION, build_tier1_p23_promotion_v2_benchmark
from research_strategy_optimization.evaluation.tier1_v04_extended import counterfactual_raw_observation_audit
from research_strategy_optimization.schemas import Protocol
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def _dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p23_promotion_v2"); args = parser.parse_args(argv)
    out = args.output_dir
    raw_path, oracle_path = out / "dataset_raw_evidence.json", out / "dataset_oracle_state.json"
    if not raw_path.exists() or not oracle_path.exists():
        raise SystemExit("both raw and oracle track files are required")
    benchmark = build_tier1_p23_promotion_v2_benchmark()
    raw = DecisionDataset.from_json(raw_path)
    canonical, audit = _canonical_reversals(raw, P22Config(top1_gap_threshold=0.0, max_pairs_per_question=1))
    promotion_pairs = [pair for pair in canonical if raw.examples[pair.left].split == "promotion" and raw.examples[pair.right].split == "promotion"]
    promotion_questions = sorted({raw.examples[pair.left].question_id for pair in promotion_pairs})
    protocol = Protocol(protocol_version="pesco_v0_2", max_budget=6)
    rows = [counterfactual_raw_observation_audit(question, world, protocol) for question in benchmark.questions for world in question.worlds]
    leakage = {"schema_version": "pesco_p23_counterfactual_leakage_audit_v0.1", "generator_version": P23_GENERATOR_VERSION, "audited_world_count": len(rows), "pass_count": sum(bool(row["pass"]) for row in rows), "pass": bool(rows) and all(bool(row["pass"]) for row in rows), "decision_before_candidate_branches": True, "feature_removed": "log_confirmation_pass_rate", "rows": rows}
    audit.update({"stage": "p2.3_promotion_v2", "promotion_selected_reversal_count": len(promotion_pairs), "promotion_question_cluster_count": len(promotion_questions), "promotion_pair_minimum": 30, "promotion_question_cluster_minimum": 20, "promotion_power_boundary_pass": len(promotion_pairs) >= 30 and len(promotion_questions) >= 20})
    _dump(out / "raw_canonical_audit.json", audit)
    _dump(out / "counterfactual_leakage_audit.json", leakage)
    _dump(out / "benchmark_manifest_hidden.json", benchmark.manifest(include_hidden=True))
    _dump(out / "benchmark_manifest_public.json", benchmark.manifest(include_hidden=False))
    result = {"schema_version": "pesco_tier1_p23_promotion_v2_collection_v0.1", "status": "completed_cpu_diagnostic" if audit["promotion_power_boundary_pass"] and leakage["pass"] else "failed_closed_underpowered_or_leaky", "generator_version": P23_GENERATOR_VERSION, "counts_by_split": dict(P23_COUNTS), "question_count": len(benchmark.questions), "world_count": len(benchmark.worlds), "tracks": ["raw_evidence", "oracle_state"], "raw_canonical_promotion_pairs": len(promotion_pairs), "raw_canonical_promotion_question_clusters": len(promotion_questions), "canonical_promotion_gate": audit["promotion_power_boundary_pass"], "counterfactual_leakage": {"pass": leakage["pass"], "audited_world_count": leakage["audited_world_count"]}, "diagnostic_only": True, "formal_comparison_authorized": False, "model_evaluation_authorized": False}
    _dump(out / "p23_promotion_v2_collection.json", result)
    manifest = build_run_manifest(experiment="tier1_p23_promotion_v2_collection", repo_root=ROOT, command=sys.argv, runner_paths=[ROOT / "scripts/prepare_tier1_p23_track.py", ROOT / "scripts/finalize_tier1_p23_promotion_v2.py", ROOT / "research_strategy_optimization/evaluation/tier1_p23_dataset.py", ROOT / "research_strategy_optimization/evaluation/tier1_v04_extended.py"], data_paths=[path for path in sorted(out.iterdir()) if path.is_file() and path.name != "run_manifest.json"], seeds={"exploration": list(protocol.exploration_seeds), "confirmation": list(protocol.confirmation_seeds)}, checkpoint=None, status=result["status"], diagnostics={"generator_version": P23_GENERATOR_VERSION, "counts_by_split": dict(P23_COUNTS), "tracks": result["tracks"], "canonical_promotion_gate": result["canonical_promotion_gate"], "counterfactual_leakage_pass": leakage["pass"], "formal_comparison_authorized": False, "diagnostic_only": True})
    write_run_manifest(out / "run_manifest.json", manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed_cpu_diagnostic" else 2


if __name__ == "__main__": raise SystemExit(main())
