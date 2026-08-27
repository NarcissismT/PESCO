#!/usr/bin/env python3
"""Run P2.2 seed/method cells in isolated processes and aggregate them.

The CPU PyTorch runtime can retain a large allocator arena after evaluator calls.
Isolation makes the matrix reproducible and fail-closed instead of allowing one
long-lived process to be killed after several methods.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

P22_METHODS = (
    "SFT-Continued", "SFT→BranchOnly", "SFT→NoFlip", "SFT→Pairwise-Full",
    "SFT→Listwise-Full", "SFT→Listwise+PCGrad",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/tier1_p21_diagnostic/dataset_raw_evidence.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p22_screening")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 29])
    parser.add_argument("--stage", choices=("screening", "frozen_10_seed"), default="screening")
    parser.add_argument("--sft-steps", type=int, default=32)
    parser.add_argument("--finetune-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--pairwise-weight", type=float, default=0.35)
    parser.add_argument("--listwise-weight", type=float, default=0.65)
    parser.add_argument("--entropy-floor", type=float, default=0.55)
    parser.add_argument("--utility-temperature", type=float, default=0.25)
    parser.add_argument("--utility-target-weight", type=float, default=1.0)
    parser.add_argument("--utility-hard-weight", type=float, default=0.0)
    parser.add_argument("--top1-margin-weight", type=float, default=0.0)
    parser.add_argument("--top1-margin", type=float, default=0.05)
    parser.add_argument("--branch-loss-weight", type=float, default=0.25)
    parser.add_argument("--repair-safety-weight", type=float, default=0.0)
    parser.add_argument("--top1-gap-threshold", type=float, default=0.05)
    parser.add_argument("--eval-splits", nargs="+", default=["tune", "promotion"])
    parser.add_argument("--method", action="append", dest="methods", default=None)
    parser.add_argument("--reuse-existing", action="store_true", help="reuse successful cell JSONs instead of deleting/rerunning them")
    args = parser.parse_args(argv)
    methods = tuple(args.methods) if args.methods else P22_METHODS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = args.output_dir / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    all_logs: dict[str, dict] = {}
    failures: list[dict] = []
    sft_seen: set[int] = set()
    for seed in args.seeds:
        for method in methods:
            cell = (cells / f"seed_{int(seed)}_{method.replace('/', '_').replace('→', 'to').replace('+', '_') }").resolve()
            command = [
                sys.executable, str(ROOT / "scripts/run_tier1_p22_diagnostics.py"),
                "--dataset", str(args.dataset), "--output-dir", str(cell),
                "--seeds", str(int(seed)), "--stage", str(args.stage),
                "--sft-steps", str(max(1, int(args.sft_steps))),
                "--finetune-steps", str(max(1, int(args.finetune_steps))),
                "--batch-size", str(max(1, int(args.batch_size))),
                "--hidden-dim", str(max(4, int(args.hidden_dim))),
                "--learning-rate", str(float(args.learning_rate)),
                "--pairwise-weight", str(float(args.pairwise_weight)),
                "--listwise-weight", str(float(args.listwise_weight)),
                "--entropy-floor", str(float(args.entropy_floor)),
                "--utility-temperature", str(float(args.utility_temperature)),
                "--utility-target-weight", str(float(args.utility_target_weight)),
                "--utility-hard-weight", str(float(args.utility_hard_weight)),
                "--top1-margin-weight", str(float(args.top1_margin_weight)),
                "--top1-margin", str(float(args.top1_margin)),
                "--branch-loss-weight", str(float(args.branch_loss_weight)),
                "--repair-safety-weight", str(float(args.repair_safety_weight)),
                "--top1-gap-threshold", str(float(args.top1_gap_threshold)),
                "--eval-splits", *[str(split) for split in args.eval_splits],
                "--method", method,
            ]
            result_path = cell / "p22_result.json"
            if args.reuse_existing and result_path.exists():
                completed = None
            else:
                if cell.exists():
                    shutil.rmtree(cell)
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            if not result_path.exists() or (completed is not None and completed.returncode != 0):
                failures.append({"seed": int(seed), "method": method, "returncode": None if completed is None else completed.returncode, "stdout": "" if completed is None else completed.stdout[-2000:], "stderr": "" if completed is None else completed.stderr[-4000:]})
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            allowed_splits = {str(split) for split in args.eval_splits}
            all_records.extend(
                row for row in payload.get("records", [])
                if str(row.get("split")) in allowed_splits
                and (row.get("method") != "SFT" or int(seed) not in sft_seen)
            )
            sft_seen.add(int(seed))
            # Each isolated worker contains the common SFT log plus its single
            # method log.  Merge by seed instead of overwriting the previous
            # method; otherwise the final matrix silently loses the
            # batch-level gradient-cosine evidence required by P2.2.
            seed_logs = payload.get("training_logs", {}).get(str(seed), {})
            all_logs.setdefault(str(seed), {}).update(seed_logs)
    # Obtain the canonical-pair audit from the first successful cell.
    first_result = next((cells / name / "p22_result.json" for name in __import__("os").listdir(cells) if (cells / name / "p22_result.json").exists()), None)
    reversal_audit = json.loads(first_result.read_text(encoding="utf-8")).get("canonical_reversal_audit", {}) if first_result else {}
    # Import torch/P2.2 only after all worker processes have exited; keeping the
    # parent lightweight avoids overlapping two ~0.9GB CPU allocator arenas.
    from research_strategy_optimization.evaluation.tier1_p22_diagnostics import P22Config, _aggregate
    from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest
    config = P22Config(sft_steps=max(1, int(args.sft_steps)), finetune_steps=max(1, int(args.finetune_steps)), batch_size=max(1, int(args.batch_size)), hidden_dim=max(4, int(args.hidden_dim)), learning_rate=max(1e-6, float(args.learning_rate)), pairwise_weight=max(0.0, float(args.pairwise_weight)), listwise_weight=max(0.0, float(args.listwise_weight)), entropy_floor=max(0.0, float(args.entropy_floor)), utility_temperature=max(1e-6, float(args.utility_temperature)), utility_target_weight=max(0.0, float(args.utility_target_weight)), utility_hard_weight=max(0.0, float(args.utility_hard_weight)), top1_margin_weight=max(0.0, float(args.top1_margin_weight)), top1_margin=max(0.0, float(args.top1_margin)), branch_loss_weight=max(0.0, float(args.branch_loss_weight)), repair_safety_weight=max(0.0, float(args.repair_safety_weight)), top1_gap_threshold=max(0.0, float(args.top1_gap_threshold)))
    aggregate = _aggregate(all_records, methods, args.seeds, bootstrap_replicates=2000)
    result = {
        "schema_version": "pesco_tier1_p22_common_sft_matrix_v0.1",
        "stage": args.stage,
        "config": {**config.__dict__, "isolated_process_per_seed_method": True},
        "seeds": [int(s) for s in args.seeds],
        "methods": list(methods),
        "canonical_reversal_audit": reversal_audit,
        "records": all_records,
        "training_logs": all_logs,
        "aggregation": aggregate,
        "cell_failures": failures,
        "matrix_complete": not failures and len(all_records) == len(args.seeds) * (len(methods) + 1) * len(args.eval_splits),
        "eval_splits": list(args.eval_splits),
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "final_evaluation_authorized": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "p22_matrix_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    manifest = build_run_manifest(
        experiment="tier1_p22_common_sft_matrix",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[ROOT / "scripts/run_tier1_p22_matrix.py", ROOT / "scripts/run_tier1_p22_diagnostics.py", ROOT / "research_strategy_optimization/evaluation/tier1_p22_diagnostics.py"],
        data_paths=[args.dataset],
        seeds={"training": [int(s) for s in args.seeds]},
        checkpoint=None,
        status="completed_diagnostic" if result["matrix_complete"] else "failed_closed_incomplete_matrix",
        diagnostics={
            "stage": args.stage,
            "methods": list(methods),
            "eval_splits": list(args.eval_splits),
            "matrix_complete": bool(result["matrix_complete"]),
            "cell_failures": len(failures),
            "common_sft_initialization": True,
            "reference_policy": "SFT",
            "utility_floor_reference": "SFT",
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
            "final_evaluation_authorized": False,
        },
    )
    write_run_manifest(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps({"status": result["schema_version"], "matrix_complete": result["matrix_complete"], "cell_failures": failures, "promotion_summary": aggregate["promotion_summary"], "gate_checks": aggregate["gate_checks"]}, ensure_ascii=False, indent=2))
    return 0 if result["matrix_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
