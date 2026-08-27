#!/usr/bin/env python3
"""Collect the fresh P2.1 diagnostic and run tabular shortcut probes.

This is a diagnostic-only convenience runner.  It writes the evaluator/audit
dataset alongside the probe result so the exact train/tune/promotion boundary can
be replayed.  It never touches the v0.5 frozen-final bundle or authorizes model
scaling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.shortcut_probes import (  # noqa: E402
    FEATURE_SETS,
    MODEL_NAMES,
    run_shortcut_probe,
)
from research_strategy_optimization.evaluation.tier1_p21_dataset import (  # noqa: E402
    P21_GENERATOR_VERSION,
    build_tier1_p21_diagnostic_benchmark,
)
from research_strategy_optimization.evaluation.tier1_v04_extended import (  # noqa: E402
    TRACK_RAW_EVIDENCE,
    V04_EXTENDED_CONFIRMATION_SEEDS,
    V04_EXTENDED_EXPLORATION_SEEDS,
    collect_tier1_v04_extended,
)
from research_strategy_optimization.schemas import Protocol  # noqa: E402
from research_strategy_optimization.utils.run_manifest import (  # noqa: E402
    build_run_manifest,
    write_run_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/tier1_p21_shortcut_probe")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--feature-set", dest="feature_sets", action="append", choices=FEATURE_SETS, default=None)
    parser.add_argument("--model", dest="models", action="append", choices=MODEL_NAMES, default=None)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = Protocol(
        protocol_version="pesco_v0_2",
        exploration_seeds=V04_EXTENDED_EXPLORATION_SEEDS,
        confirmation_seeds=V04_EXTENDED_CONFIRMATION_SEEDS,
        max_budget=6,
    )
    benchmark = build_tier1_p21_diagnostic_benchmark()
    dataset, audit = collect_tier1_v04_extended(
        benchmark,
        protocol,
        track=TRACK_RAW_EVIDENCE,
        question_limit=None,
    )
    # Match the canonical P2.1 collection contract exactly.  The shortcut probe
    # must consume the same fresh generator/schema and split provenance as the
    # algorithm diagnostics; otherwise a metadata-only digest difference can make
    # it appear to be a separate benchmark.
    questions = benchmark.questions
    counts = {
        split: sum(question.split == split for question in questions)
        for split in ("train", "tune", "promotion")
    }
    selected_families = tuple(sorted({str(question.family) for question in questions}))
    dataset.schema_version = "pesco_decision_dataset_p21_fresh_diagnostic"
    dataset.provenance.update({
        "schema_version": dataset.schema_version,
        "generator_version": P21_GENERATOR_VERSION,
        "fresh_generator": True,
        "not_reused_v04_formal": True,
        "split_contract": ["train", "tune", "promotion"],
        "formal_comparison_authorized": False,
        "diagnostic_only": True,
        "question_count": len(questions),
        "world_count": len(dataset.examples),
        "counts_by_split": counts,
        "mechanism_family_count": len(selected_families),
        "mechanism_families": list(selected_families),
        "benchmark_question_count": len(benchmark.questions),
        "benchmark_world_count": len(benchmark.worlds),
        "question_limit": None,
    })
    dataset_path = output / "dataset_raw_evidence.json"
    dataset_public_path = output / "dataset_raw_evidence_public.json"
    audit_path = output / "audit_raw_evidence.json"
    dataset.save_json(dataset_path, include_audit=True)
    dataset.save_json(dataset_public_path, include_audit=False)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    result = run_shortcut_probe(
        dataset_path,
        output_dir=output,
        feature_sets=tuple(args.feature_sets or FEATURE_SETS),
        models=tuple(args.models or MODEL_NAMES),
        train_split="train",
        eval_splits=("tune", "promotion"),
        seed=int(args.seed),
        bootstrap_replicates=max(1, int(args.bootstrap_replicates)),
        max_examples=args.max_examples,
    )
    manifest = build_run_manifest(
        experiment="tier1_p21_shortcut_probes",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[
            Path(__file__),
            ROOT / "research_strategy_optimization/evaluation/shortcut_probes.py",
            ROOT / "research_strategy_optimization/evaluation/tier1_p21_dataset.py",
            ROOT / "research_strategy_optimization/evaluation/tier1_v04_extended.py",
            ROOT / "requirements-optional.txt",
        ],
        data_paths=[dataset_path, audit_path],
        seeds={"probe": int(args.seed), "bootstrap": int(args.bootstrap_replicates)},
        status=str(result.get("status", "unknown")),
        diagnostics={
            "result_schema": result.get("schema_version"),
            "dataset_schema": dataset.schema_version,
            "dataset_example_count": len(dataset.examples),
            "dataset_reversal_count": len(dataset.reversals),
            "splits": {split: sum(example.split == split for example in dataset.examples) for split in ("train", "tune", "promotion")},
            "fallback_used": result.get("fallback_used"),
        },
    )
    write_run_manifest(output / "run_manifest.json", manifest)
    print(json.dumps({
        "output": str(output),
        "status": result.get("status"),
        "fallback_used": result.get("fallback_used", False),
        "sklearn_available": result.get("dependency", {}).get("sklearn", {}).get("available"),
        "split_counts": {split: sum(example.split == split for example in dataset.examples) for split in ("train", "tune", "promotion")},
        "model_statuses": {key: value.get("status") for key, value in result.get("models", {}).items()},
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
