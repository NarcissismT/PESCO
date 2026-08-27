#!/usr/bin/env python3
"""Collect the fresh P2.1 train/tune/promotion diagnostic dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_p21_dataset import (
    P21_GENERATOR_VERSION,
    build_tier1_p21_diagnostic_benchmark,
    p21_latent_signature,
)
from research_strategy_optimization.evaluation.tier1_v04_extended import (
    TRACK_RAW_EVIDENCE,
    V04_EXTENDED_CONFIRMATION_SEEDS,
    V04_EXTENDED_EXPLORATION_SEEDS,
    collect_tier1_v04_extended,
    counterfactual_raw_observation_audit,
)
from research_strategy_optimization.schemas import Protocol
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def run(output_dir: Path, *, question_limit: int | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = Protocol(
        protocol_version="pesco_v0_2",
        exploration_seeds=V04_EXTENDED_EXPLORATION_SEEDS,
        confirmation_seeds=V04_EXTENDED_CONFIRMATION_SEEDS,
        max_budget=6,
    )
    benchmark = build_tier1_p21_diagnostic_benchmark()
    # Keep every downstream audit/count aligned with the exact subset that was
    # collected.  ``collect_tier1_v04_extended`` applies ``question_limit`` to
    # the serialized dataset, so iterating over the full benchmark here would
    # make limited diagnostic runs report boundary rows, split counts, and
    # latent signatures for uncollected worlds.
    questions = (
        benchmark.questions
        if question_limit is None
        else benchmark.questions[: max(0, int(question_limit))]
    )
    dataset, collection_audit = collect_tier1_v04_extended(
        benchmark,
        protocol,
        track=TRACK_RAW_EVIDENCE,
        question_limit=question_limit,
    )
    counts = {
        split: sum(question.split == split for question in questions)
        for split in ("train", "tune", "promotion")
    }
    selected_families = tuple(sorted({str(question.family) for question in questions}))
    signatures = [
        p21_latent_signature(world)
        for question in questions
        for world in question.worlds
    ]
    # Make the split/generator contract explicit in the serialized diagnostic.
    dataset.schema_version = "pesco_decision_dataset_p21_fresh_diagnostic"
    dataset.provenance.update({
        "schema_version": dataset.schema_version,
        "generator_version": P21_GENERATOR_VERSION,
        "fresh_generator": True,
        "not_reused_v04_formal": True,
        "split_contract": ["train", "tune", "promotion"],
        "formal_comparison_authorized": False,
        "diagnostic_only": True,
        # ``collect_tier1_v04_extended`` receives the full generator object even
        # for bounded runs, so its provenance starts with full-benchmark counts.
        # Overwrite those fields with the actually collected subset and retain the
        # full generator dimensions under explicit audit-only names.
        "question_count": len(questions),
        "world_count": len(dataset.examples),
        "counts_by_split": counts,
        "mechanism_family_count": len(selected_families),
        "mechanism_families": list(selected_families),
        "benchmark_question_count": len(benchmark.questions),
        "benchmark_world_count": len(benchmark.worlds),
        "question_limit": None if question_limit is None else max(0, int(question_limit)),
    })
    dataset.save_json(output_dir / "dataset_raw_evidence.json", include_audit=True)
    dataset.save_json(output_dir / "dataset_raw_evidence_public.json", include_audit=False)

    benchmark_manifest = benchmark.manifest(include_hidden=True, exploration_seeds=protocol.exploration_seeds)
    public_manifest = benchmark.manifest(include_hidden=False, exploration_seeds=protocol.exploration_seeds)
    _dump(output_dir / "benchmark_manifest.json", benchmark_manifest)
    _dump(output_dir / "benchmark_public_manifest.json", public_manifest)

    # Run the boundary test over every collected world, not just one unit-test case.
    boundary_rows = [
        counterfactual_raw_observation_audit(question, world, protocol)
        for question in questions
        for world in question.worlds
    ]
    boundary = {
        "schema_version": "pesco_p21_counterfactual_leakage_audit_v0.1",
        "generator_version": P21_GENERATOR_VERSION,
        "row_count": len(boundary_rows),
        "pass_count": sum(bool(row["pass"]) for row in boundary_rows),
        "pass": bool(boundary_rows) and all(bool(row["pass"]) for row in boundary_rows),
        "rows": boundary_rows,
    }
    _dump(output_dir / "counterfactual_leakage_audit.json", boundary)

    # Keep the collection audit explicit about bounded-vs-full generator scope.
    collection_audit.update({
        "collected_question_count": len(questions),
        "collected_world_count": len(dataset.examples),
        "question_limit": None if question_limit is None else max(0, int(question_limit)),
        "benchmark_question_count": len(benchmark.questions),
        "benchmark_world_count": len(benchmark.worlds),
    })
    _dump(output_dir / "collection_audit.json", collection_audit)
    result = {
        "schema_version": "pesco_tier1_p21_diagnostic_result_v0.1",
        "status": "completed_cpu_diagnostic",
        "generator_version": P21_GENERATOR_VERSION,
        "question_count": len(questions),
        "world_count": len(dataset.examples),
        "counts_by_split": counts,
        "dataset_schema": dataset.schema_version,
        "benchmark_manifest_digest": benchmark_manifest["manifest_digest"],
        "public_manifest_digest": public_manifest["manifest_digest"],
        "latent_signature_count": len(set(signatures)),
        "counterfactual_leakage_audit": {
            "pass": boundary["pass"],
            "row_count": boundary["row_count"],
            "pass_count": boundary["pass_count"],
        },
        "same_question_reversal_count": len(dataset.reversals),
        "reversal_weights_sum_by_question": {
            qid: sum(float(pair.weight) for pair in dataset.reversals if dataset.examples[pair.left].question_id == qid)
            for qid in sorted({dataset.examples[pair.left].question_id for pair in dataset.reversals})
        },
        "formal_comparison_authorized": False,
        "diagnostic_only": True,
    }
    _dump(output_dir / "p21_diagnostic_result.json", result)
    manifest = build_run_manifest(
        experiment="tier1_p21_fresh_diagnostic_collection",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[
            ROOT / "scripts/prepare_tier1_p21_diagnostic.py",
            ROOT / "research_strategy_optimization/evaluation/tier1_p21_dataset.py",
            ROOT / "research_strategy_optimization/evaluation/tier1_v04_extended.py",
            ROOT / "research_strategy_optimization/algorithms/differentiable_strategy.py",
        ],
        data_paths=[path for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "run_manifest.json"],
        seeds={"exploration": list(protocol.exploration_seeds), "confirmation": list(protocol.confirmation_seeds)},
        checkpoint=None,
        status="completed_diagnostic",
        diagnostics={
            "generator_version": P21_GENERATOR_VERSION,
            "formal_comparison_authorized": False,
            "diagnostic_only": True,
            "counterfactual_leakage_pass": boundary["pass"],
            "split_contract": ["train", "tune", "promotion"],
        },
    )
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p21_diagnostic")
    parser.add_argument("--question-limit", type=int, default=None)
    args = parser.parse_args(argv)
    result = run(args.output_dir, question_limit=args.question_limit)
    print(json.dumps({key: result[key] for key in ("status", "question_count", "world_count", "same_question_reversal_count", "counterfactual_leakage_audit")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
