#!/usr/bin/env python3
"""Collect one P2.3 promotion-v2 observation track.

Raw and oracle tracks are intentionally runnable in separate processes because
the full promotion-v2 collection is larger than the earlier P2.1 diagnostic.
Both tracks still call the same evaluator/executor and use identical seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import DecisionDataset
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
)
from research_strategy_optimization.schemas import Protocol


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=(TRACK_RAW_EVIDENCE, TRACK_ORACLE_STATE), required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p23_promotion_v2")
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    benchmark = build_tier1_p23_promotion_v2_benchmark()
    protocol = Protocol(protocol_version="pesco_v0_2", exploration_seeds=V04_EXTENDED_EXPLORATION_SEEDS, confirmation_seeds=V04_EXTENDED_CONFIRMATION_SEEDS, max_budget=6)
    dataset, collection = collect_tier1_v04_extended(benchmark, protocol, track=args.track)
    dataset.schema_version = "pesco_decision_dataset_p2.3_promotion_v2"
    dataset.provenance.update({
        "schema_version": dataset.schema_version,
        "generator_version": P23_GENERATOR_VERSION,
        "fresh_generator": True,
        "not_reused_p21_or_p22_promotion": True,
        "promotion_consumed_before_generation": True,
        "split_contract": ["train", "tune", "promotion"],
        "counts_by_split": dict(P23_COUNTS),
        "question_count": len(benchmark.questions),
        "world_count": len(dataset.examples),
        "mechanism_family_count": len({q.family for q in benchmark.questions}),
        "mechanism_families": sorted({q.family for q in benchmark.questions}),
        "track": args.track,
    })
    stem = "dataset_raw_evidence" if args.track == TRACK_RAW_EVIDENCE else "dataset_oracle_state"
    dataset.save_json(out / f"{stem}.json", include_audit=True)
    dataset.save_json(out / f"{stem}_public.json", include_audit=False)
    canonical, audit = _canonical_reversals(dataset, P22Config(top1_gap_threshold=0.0, max_pairs_per_question=1))
    promotion_pairs = [pair for pair in canonical if dataset.examples[pair.left].split == "promotion" and dataset.examples[pair.right].split == "promotion"]
    audit.update({"track": args.track, "promotion_selected_reversal_count": len(promotion_pairs), "promotion_question_cluster_count": len({dataset.examples[pair.left].question_id for pair in promotion_pairs}), "promotion_power_boundary_pass": len(promotion_pairs) >= 30 and len({dataset.examples[pair.left].question_id for pair in promotion_pairs}) >= 20})
    collection.update({"track": args.track, "generator_version": P23_GENERATOR_VERSION, "counts_by_split": dict(P23_COUNTS)})
    _dump(out / f"{stem}_canonical_audit.json", audit)
    _dump(out / f"{stem}_collection_audit.json", collection)
    print(json.dumps({"track": args.track, "examples": len(dataset.examples), "reversals": len(dataset.reversals), "promotion_pairs": len(promotion_pairs), "promotion_questions": audit["promotion_question_cluster_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
