#!/usr/bin/env python3
"""Generate the frozen promotion-v3 final-ID/final-OOD dataset after P2.3.1 GO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_p23_v3_dataset import (
    V3_GENERATOR_VERSION,
    V3_ID_FAMILIES,
    V3_OOD_FAMILIES,
    build_tier1_p23_promotion_v3_benchmark,
)
from research_strategy_optimization.evaluation.tier1_v04_extended import TRACK_RAW_EVIDENCE, collect_tier1_v04_extended
from research_strategy_optimization.schemas import Protocol
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p23_promotion_v3")
    args = parser.parse_args(argv)
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    benchmark = build_tier1_p23_promotion_v3_benchmark()
    protocol = Protocol(
        exploration_seeds=(17, 29, 41, 53, 67, 71, 83, 97),
        confirmation_seeds=(103, 107, 109, 113, 127, 131, 137, 139),
        max_budget=6,
    )
    dataset, audit = collect_tier1_v04_extended(benchmark, protocol, track=TRACK_RAW_EVIDENCE)
    dataset.schema_version = "pesco_decision_dataset_p2.3_promotion_v3"
    dataset.provenance.update({
        "schema_version": dataset.schema_version,
        "generator_version": V3_GENERATOR_VERSION,
        "fresh_generator": True,
        "consumed_p231_gate_status": "GO_P2_3_1_10SEED_AUTHORIZED",
        "id_families": list(V3_ID_FAMILIES),
        "ood_families": list(V3_OOD_FAMILIES),
        "whole_family_holdout": True,
        "final_access_locked": True,
        "formal_comparison_authorized": False,
        "diagnostic_only": False,
    })
    dataset.save_json(output / "dataset_raw_evidence.json", include_audit=True)
    dataset.save_json(output / "dataset_raw_evidence_public.json", include_audit=False)
    (output / "benchmark_manifest_hidden.json").write_text(json.dumps(benchmark.manifest(include_hidden=True), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (output / "benchmark_manifest_public.json").write_text(json.dumps(benchmark.manifest(include_hidden=False), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    result = {
        "schema_version": "pesco_tier1_p23_promotion_v3_collection_v0.1",
        "status": "completed_locked_final_dataset",
        "generator_version": V3_GENERATOR_VERSION,
        "counts_by_split": {split: sum(example.split == split for example in dataset.examples) for split in benchmark.split_names},
        "question_counts_by_split": {split: sum(question.split == split for question in benchmark.questions) for split in benchmark.split_names},
        "world_count": len(dataset.examples),
        "id_families": list(V3_ID_FAMILIES),
        "ood_families": list(V3_OOD_FAMILIES),
        "ood_family_count": len(V3_OOD_FAMILIES),
        "whole_family_holdout": True,
        "final_access": {"locked": True, "first_access_completed": False, "authorization_required": True},
        "formal_comparison_authorized": False,
        "audit": audit,
    }
    (output / "promotion_v3_collection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    manifest = build_run_manifest(
        experiment="tier1_p23_promotion_v3_collection",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[ROOT / "scripts/prepare_tier1_p23_promotion_v3.py", ROOT / "research_strategy_optimization/evaluation/tier1_p23_v3_dataset.py"],
        data_paths=[output / "dataset_raw_evidence.json", output / "benchmark_manifest_hidden.json", output / "benchmark_manifest_public.json"],
        seeds={"exploration": list(protocol.exploration_seeds), "confirmation": list(protocol.confirmation_seeds)},
        checkpoint=None,
        status="completed_locked_final_dataset",
        diagnostics={"formal_comparison_authorized": False, "final_access_locked": True, "ood_family_count": len(V3_OOD_FAMILIES)},
    )
    write_run_manifest(output / "run_manifest.json", manifest)
    print(json.dumps({"status": result["status"], "question_counts_by_split": result["question_counts_by_split"], "world_count": result["world_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
