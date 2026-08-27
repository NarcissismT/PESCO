#!/usr/bin/env python3
"""Collect the independent P2.3 promotion-v2 raw/oracle tracks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
from research_strategy_optimization.evaluation.tier1_p21_diagnostics import select_top_candidate_reversals
from research_strategy_optimization.evaluation.tier1_p22_diagnostics import P22Config, _canonical_reversals
from research_strategy_optimization.evaluation.tier1_p23_dataset import (
    P23_COUNTS,
    P23_GENERATOR_VERSION,
    build_tier1_p23_promotion_v2_benchmark,
)
from research_strategy_optimization.evaluation.tier1_v04_extended import (
    TRACK_ORACLE_STATE,
    TRACK_RAW_EVIDENCE,
    V04_EXTENDED_CONFIRMATION_SEEDS,
    V04_EXTENDED_EXPLORATION_SEEDS,
    collect_tier1_v04_extended,
    counterfactual_raw_observation_audit,
)
from research_strategy_optimization.schemas import Protocol
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _canonical_audit(dataset: DecisionDataset) -> dict:
    config = P22Config(top1_gap_threshold=0.0, max_pairs_per_question=1)
    selected, audit = _canonical_reversals(dataset, config)
    promotion_pairs = [
        pair for pair in selected
        if dataset.examples[int(pair.left)].split == "promotion"
        and dataset.examples[int(pair.right)].split == "promotion"
    ]
    promotion_questions = sorted({dataset.examples[int(pair.left)].question_id for pair in promotion_pairs})
    audit.update({
        "stage": "p2.3_promotion_v2",
        "max_pairs_per_question": 1,
        "promotion_selected_reversal_count": len(promotion_pairs),
        "promotion_question_cluster_count": len(promotion_questions),
        "promotion_pair_minimum": 30,
        "promotion_question_cluster_minimum": 20,
        "promotion_power_boundary_pass": bool(len(promotion_pairs) >= 30 and len(promotion_questions) >= 20),
    })
    return audit


def run(output_dir: Path, *, counterfactual_limit: int | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = build_tier1_p23_promotion_v2_benchmark()
    protocol = Protocol(
        protocol_version="pesco_v0_2",
        exploration_seeds=V04_EXTENDED_EXPLORATION_SEEDS,
        confirmation_seeds=V04_EXTENDED_CONFIRMATION_SEEDS,
        max_budget=6,
    )
    tracks = {}
    audits = {}
    for track, filename in ((TRACK_RAW_EVIDENCE, "dataset_raw_evidence.json"), (TRACK_ORACLE_STATE, "dataset_oracle_state.json")):
        dataset, collection_audit = collect_tier1_v04_extended(benchmark, protocol, track=track)
        dataset.schema_version = "pesco_decision_dataset_p2.3_promotion_v2"
        dataset.provenance.update({
            "schema_version": dataset.schema_version,
            "generator_version": P23_GENERATOR_VERSION,
            "fresh_generator": True,
            "not_reused_p21_or_p22_promotion": True,
            "split_contract": ["train", "tune", "promotion"],
            "counts_by_split": dict(P23_COUNTS),
            "question_count": len(benchmark.questions),
            "world_count": len(dataset.examples),
            "mechanism_family_count": len({q.family for q in benchmark.questions}),
            "mechanism_families": sorted({q.family for q in benchmark.questions}),
            "promotion_consumed_before_generation": True,
            "track": track,
        })
        dataset.save_json(output_dir / filename, include_audit=True)
        dataset.save_json(output_dir / filename.replace(".json", "_public.json"), include_audit=False)
        audits[track] = {"collection": collection_audit, "canonical": _canonical_audit(dataset)}
        tracks[track] = dataset

    # Only raw evidence is used for the leakage audit; oracle-state is an upper
    # bound and cannot authorize an autonomous/raw claim.
    raw_questions = benchmark.questions
    rows = []
    limit = None if counterfactual_limit is None else max(0, int(counterfactual_limit))
    audited = 0
    for question in raw_questions:
        for world in question.worlds:
            if limit is not None and audited >= limit:
                break
            rows.append(counterfactual_raw_observation_audit(question, world, protocol))
            audited += 1
        if limit is not None and audited >= limit:
            break
    leakage = {
        "schema_version": "pesco_p23_counterfactual_leakage_audit_v0.1",
        "generator_version": P23_GENERATOR_VERSION,
        "audited_world_count": len(rows),
        "pass_count": sum(bool(row["pass"]) for row in rows),
        "pass": bool(rows) and all(bool(row["pass"]) for row in rows),
        "decision_before_candidate_branches": True,
        "feature_removed": "log_confirmation_pass_rate",
        "rows": rows,
    }
    _dump(output_dir / "counterfactual_leakage_audit.json", leakage)
    _dump(output_dir / "benchmark_manifest_hidden.json", benchmark.manifest(include_hidden=True, exploration_seeds=protocol.exploration_seeds))
    _dump(output_dir / "benchmark_manifest_public.json", benchmark.manifest(include_hidden=False, exploration_seeds=protocol.exploration_seeds))
    canonical_raw = audits[TRACK_RAW_EVIDENCE]["canonical"]
    result = {
        "schema_version": "pesco_tier1_p23_promotion_v2_collection_v0.1",
        "status": "completed_cpu_diagnostic",
        "generator_version": P23_GENERATOR_VERSION,
        "counts_by_split": dict(P23_COUNTS),
        "question_count": len(benchmark.questions),
        "world_count": len(benchmark.worlds),
        "tracks": [TRACK_RAW_EVIDENCE, TRACK_ORACLE_STATE],
        "raw_canonical_promotion_pairs": canonical_raw["promotion_selected_reversal_count"],
        "raw_canonical_promotion_question_clusters": canonical_raw["promotion_question_cluster_count"],
        "canonical_promotion_gate": canonical_raw["promotion_power_boundary_pass"],
        "counterfactual_leakage": {"pass": leakage["pass"], "audited_world_count": leakage["audited_world_count"]},
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "model_evaluation_authorized": False,
    }
    _dump(output_dir / "p23_promotion_v2_collection.json", result)
    _dump(output_dir / "track_audits.json", audits)
    manifest = build_run_manifest(
        experiment="tier1_p23_promotion_v2_collection",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[ROOT / "scripts/prepare_tier1_p23_promotion_v2.py", ROOT / "research_strategy_optimization/evaluation/tier1_p23_dataset.py", ROOT / "research_strategy_optimization/evaluation/tier1_v04_extended.py"],
        data_paths=[path for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "run_manifest.json"],
        seeds={"exploration": list(protocol.exploration_seeds), "confirmation": list(protocol.confirmation_seeds)},
        checkpoint=None,
        status="completed_diagnostic" if result["canonical_promotion_gate"] and result["counterfactual_leakage"]["pass"] else "failed_closed_underpowered_or_leaky",
        diagnostics={
            "generator_version": P23_GENERATOR_VERSION,
            "counts_by_split": dict(P23_COUNTS),
            "tracks": [TRACK_RAW_EVIDENCE, TRACK_ORACLE_STATE],
            "canonical_promotion_gate": result["canonical_promotion_gate"],
            "counterfactual_leakage_pass": result["counterfactual_leakage"]["pass"],
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
        },
    )
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p23_promotion_v2")
    parser.add_argument("--counterfactual-limit", type=int, default=None)
    args = parser.parse_args(argv)
    result = run(args.output_dir, counterfactual_limit=args.counterfactual_limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["canonical_promotion_gate"] and result["counterfactual_leakage"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
