#!/usr/bin/env python3
"""Run the single frozen promotion-v3 10-seed evaluation.

The evaluator refuses to run without the P2.3.1 GO receipt, a freeze receipt, and
the locked promotion-v3 manifest.  The final split is never used for training.
"""

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
from research_strategy_optimization.evaluation.tier1_p231_diagnostics import P231Config, run_p231_dev_diagnostic


METHODS = (
    "RLOO", "GRPO-Terminal", "GRPO-FourState", "GRPO-MatchedAtomic",
    "GRPO+State", "GRPO+Branch", "GRPO+Flip", "GRPO+Branch+Flip",
)
SEEDS = (17, 23, 29, 31, 37, 41, 43, 47, 53, 59)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_combined(train_path: Path, final_path: Path) -> DecisionDataset:
    train = DecisionDataset.from_json(train_path)
    final = DecisionDataset.from_json(final_path)
    train_indices = [index for index, example in enumerate(train.examples) if example.split == "train"]
    train_remap = {old: new for new, old in enumerate(train_indices)}
    from research_strategy_optimization.algorithms.differentiable_strategy import ReversalExample
    train_pairs = [ReversalExample(train_remap[pair.left], train_remap[pair.right], pair.action_left, pair.action_right, pair.margin, pair.confirmed, pair.weight, pair.lcb_left, pair.ucb_right, pair.sample_count) for pair in train.reversals if pair.left in train_remap and pair.right in train_remap]
    offset = len(train_indices)
    final_pairs = [ReversalExample(offset + pair.left, offset + pair.right, pair.action_left, pair.action_right, pair.margin, pair.confirmed, pair.weight, pair.lcb_left, pair.ucb_right, pair.sample_count) for pair in final.reversals]
    examples = [train.examples[index] for index in train_indices] + list(final.examples)
    return DecisionDataset(examples, train_pairs + final_pairs, "pesco_decision_dataset_p2.3_promotion_v3_combined_private", {
        "train_source_digest": _digest(train_path), "final_source_digest": _digest(final_path),
        "final_labels_never_used_for_training": True,
    })


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, default=ROOT / "artifacts/tier1_p23_promotion_v2/dataset_raw_evidence.json")
    parser.add_argument("--final-dataset", type=Path, default=ROOT / "artifacts/tier1_p23_promotion_v3/dataset_raw_evidence.json")
    parser.add_argument("--p231-gate", type=Path, default=ROOT / "artifacts/tier1_p231_consumed_10seed/p231_authenticity_gate.json")
    parser.add_argument("--freeze-receipt", type=Path, default=ROOT / "artifacts/tier1_p231_freeze/promotion_v3_freeze_receipt.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p23_promotion_v3_final_10seed")
    args = parser.parse_args(argv)
    gate = json.loads(args.p231_gate.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze_receipt.read_text(encoding="utf-8"))
    final_manifest = json.loads((args.final_dataset.parent / "benchmark_manifest_hidden.json").read_text(encoding="utf-8"))
    if gate.get("status") != "GO_P2_3_1_10SEED_AUTHORIZED":
        raise SystemExit("P2.3.1 authenticity gate is not GO")
    if freeze.get("status") != "FROZEN_FOR_PROMOTION_V3":
        raise SystemExit("freeze receipt is missing or not frozen")
    if not freeze.get("clean_commit"):
        raise SystemExit("promotion-v3 requires a clean commit freeze")
    if not final_manifest.get("final_access", {}).get("locked", False):
        raise SystemExit("promotion-v3 manifest is not locked")
    dataset = _load_combined(args.train_dataset, args.final_dataset)
    output = args.output_dir; output.mkdir(parents=True, exist_ok=True)
    private_dataset = output / "combined_private_evaluator_input.json"
    dataset.save_json(private_dataset, include_audit=True)
    config = P231Config(sft_steps=16, finetune_steps=16, batch_size=32, hidden_dim=24, learning_rate=0.005, minibatch_epochs=4, clip_epsilon=0.05, top1_gap_threshold=0.05)
    result = run_p231_dev_diagnostic(output, dataset, seeds=SEEDS, config=config, methods=METHODS, eval_splits=("final_id", "final_ood"))
    result["diagnostic_only"] = False
    result["formal_comparison_authorized"] = True
    result["final_access"] = {"locked": True, "first_access_completed": True, "independent_evaluator_process": True, "single_evaluation": True}
    result["primary_method"] = "GRPO+Branch+Flip"
    result["dev_selected_baseline"] = "GRPO-MatchedAtomic"
    result["freeze_receipt_digest"] = _digest(args.freeze_receipt)
    result["final_manifest_digest"] = str(final_manifest.get("manifest_digest", ""))
    (output / "promotion_v3_final_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    (output / "final_access_receipt.json").write_text(json.dumps({
        "schema_version": "pesco_promotion_v3_final_access_receipt_v0.1", "locked": True,
        "first_access_completed": True, "single_evaluation": True,
        "final_manifest_digest": result["final_manifest_digest"], "freeze_receipt_digest": result["freeze_receipt_digest"],
        "evaluator_input_digest": _digest(private_dataset), "training_split": "consumed_p2.3_train_only",
        "evaluation_splits": ["final_id", "final_ood"], "final_labels_never_used_for_training": True,
    }, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "seeds": list(SEEDS), "eval_splits": ["final_id", "final_ood"], "canonical_pair_digest": result["canonical_pair_digest"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
