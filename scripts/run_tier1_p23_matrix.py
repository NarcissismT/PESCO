#!/usr/bin/env python3
"""Run P2.3 seed/method cells in isolated processes and aggregate them."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_p23_diagnostics import P23_METHODS, P23Config, _aggregate_p23
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest


def _safe_name(method: str) -> str:
    return method.replace("/", "_").replace("+", "plus").replace(" ", "_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 29])
    parser.add_argument("--stage", choices=("screening", "frozen_10_seed"), default="screening")
    parser.add_argument("--method", action="append", dest="methods", default=None)
    parser.add_argument("--sft-steps", type=int, default=16)
    parser.add_argument("--finetune-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--pairwise-weight", type=float, default=0.35)
    parser.add_argument("--branch-loss-weight", type=float, default=0.25)
    parser.add_argument("--entropy-floor", type=float, default=0.55)
    parser.add_argument("--eval-splits", nargs="+", default=["tune", "promotion"])
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args(argv)
    methods = tuple(args.methods) if args.methods else P23_METHODS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = args.output_dir / "cells"; cells.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []; logs: dict[str, dict] = {}; failures: list[dict] = []; sft_seen: set[tuple[int, str]] = set()
    for seed in args.seeds:
        for method in methods:
            cell = cells / f"seed_{int(seed)}_{_safe_name(method)}"
            result_path = cell / "p23_result.json"
            completed = None
            if not (args.reuse_existing and result_path.exists()):
                if cell.exists(): shutil.rmtree(cell)
                command = [sys.executable, str(ROOT / "scripts/run_tier1_p23_diagnostics.py"),
                    "--dataset", str(args.dataset), "--output-dir", str(cell), "--seeds", str(int(seed)),
                    "--stage", args.stage, "--method", method, "--sft-steps", str(args.sft_steps),
                    "--finetune-steps", str(args.finetune_steps), "--batch-size", str(args.batch_size),
                    "--hidden-dim", str(args.hidden_dim), "--learning-rate", str(args.learning_rate),
                    "--pairwise-weight", str(args.pairwise_weight), "--branch-loss-weight", str(args.branch_loss_weight),
                    "--entropy-floor", str(args.entropy_floor), "--eval-splits", *args.eval_splits]
                completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            if not result_path.exists() or (completed is not None and completed.returncode != 0):
                failures.append({"seed": int(seed), "method": method, "returncode": None if completed is None else completed.returncode, "stdout": "" if completed is None else completed.stdout[-2000:], "stderr": "" if completed is None else completed.stderr[-4000:]})
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            allowed = set(args.eval_splits)
            for row in payload.get("records", []):
                if row.get("split") not in allowed: continue
                if row.get("method") == "SFT" and (int(seed), str(row.get("split"))) in sft_seen: continue
                records.append(row)
                if row.get("method") == "SFT": sft_seen.add((int(seed), str(row.get("split"))))
            sft_seen.add(int(seed))
            logs.setdefault(str(seed), {}).update(payload.get("training_logs", {}).get(str(seed), {}))
    first = next((cells / name / "p23_result.json" for name in os.listdir(cells) if (cells / name / "p23_result.json").exists()), None)
    payload = json.loads(first.read_text(encoding="utf-8")) if first else {}
    pairs_by_question = {str(qid): 1 for qid in payload.get("canonical_reversal_audit", {}).get("promotion_question_ids", [])}
    if not pairs_by_question:
        pairs_by_question = {str(qid): 1 for qid in payload.get("canonical_reversal_audit", {}).get("weights_sum_by_question", {}) if str(qid).startswith("p23_")}
    # The exact question map is available in each row; count only promotion
    # questions with a canonical pair from the audit when the compact audit was
    # emitted by an older worker.
    if not pairs_by_question:
        pairs_by_question = {str(q["question_id"]): int(q.get("pair_count", 1) or 1) for r in records if r.get("method") == "SFT" for q in r.get("question_metric_rows", []) if q.get("pair_count")}
    aggregate = _aggregate_p23(records, methods, args.seeds, pairs_by_question=pairs_by_question, bootstrap_replicates=max(100, int(args.bootstrap_replicates)))
    result = {
        "schema_version": "pesco_tier1_p23_common_protocol_matrix_v0.1", "stage": args.stage,
        "config": {"sft_steps": args.sft_steps, "finetune_steps": args.finetune_steps, "batch_size": args.batch_size, "hidden_dim": args.hidden_dim, "learning_rate": args.learning_rate, "pairwise_weight": args.pairwise_weight, "branch_loss_weight": args.branch_loss_weight, "entropy_floor": args.entropy_floor, "isolated_process_per_seed_method": True},
        "seeds": [int(s) for s in args.seeds], "methods": list(methods), "records": records, "training_logs": logs,
        "aggregation": aggregate, "cell_failures": failures, "matrix_complete": not failures,
        "eval_splits": list(args.eval_splits), "diagnostic_only": True, "formal_comparison_authorized": False,
    }
    (args.output_dir / "p23_matrix_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    manifest = build_run_manifest(experiment="tier1_p23_common_protocol_matrix", repo_root=ROOT, command=sys.argv, runner_paths=[ROOT / "scripts/run_tier1_p23_matrix.py", ROOT / "scripts/run_tier1_p23_diagnostics.py", ROOT / "research_strategy_optimization/evaluation/tier1_p23_diagnostics.py"], data_paths=[args.dataset], seeds={"training": [int(s) for s in args.seeds]}, checkpoint=None, status="completed_diagnostic" if result["matrix_complete"] else "failed_closed_incomplete_matrix", diagnostics={"stage": args.stage, "methods": list(methods), "matrix_complete": result["matrix_complete"], "cell_failures": len(failures), "common_sft_initialization": True, "formal_comparison_authorized": False})
    write_run_manifest(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps({"matrix_complete": result["matrix_complete"], "cell_failures": failures, "promotion_summary": aggregate["promotion_summary"], "gate_checks": aggregate["gate_checks"]}, ensure_ascii=False))
    return 0 if result["matrix_complete"] else 2


if __name__ == "__main__": raise SystemExit(main())
