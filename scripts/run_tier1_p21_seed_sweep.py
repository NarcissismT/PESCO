#!/usr/bin/env python3
"""Run a small isolated P2.1 seed sweep with seed×question uncertainty.

This is a diagnostic supplement to the one-seed method table.  Each worker process
trains one method/seed and exits, avoiding retained PyTorch allocator state.  The
result deliberately remains non-formal unless the preregistered ten-seed/final gates
are satisfied elsewhere.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_p2_experiments import (  # noqa: E402
    _family_leave_one_out_cis,
    _two_level_seed_question_ci,
)
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest  # noqa: E402


DEFAULT_METHODS = ("SFT", "PESCO-BranchOnly", "PESCO-NoFlipLoss", "PESCO-Full")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/tier1_p21_diagnostic/dataset_raw_evidence.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p21_seed_sweep_diagnostic")
    parser.add_argument("--seed", dest="seeds", action="append", type=int, default=None)
    parser.add_argument("--method", dest="methods", action="append", default=None)
    parser.add_argument("--max-optimizer-steps", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    seeds = tuple(args.seeds or (17, 23, 29))
    methods = tuple(args.methods or DEFAULT_METHODS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    children = args.output_dir / "workers"
    children.mkdir(parents=True, exist_ok=True)
    worker_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            slug = f"seed_{seed}_{method.lower().replace(' ', '_').replace('-', '_')}"
            child = children / slug
            command = [
                sys.executable,
                str(ROOT / "scripts/run_tier1_p21_diagnostics.py"),
                "--dataset", str(args.dataset),
                "--output-dir", str(child),
                "--method", method,
                "--no-gradient", "--no-constrained",
                "--max-optimizer-steps", str(max(1, int(args.max_optimizer_steps))),
                "--epochs", str(max(1, int(args.epochs))),
                "--batch-size", str(max(1, int(args.batch_size))),
                "--seed", str(seed),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={**__import__("os").environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=240,
                check=False,
            )
            result_path = child / "p21_result.json"
            if completed.returncode != 0 or not result_path.exists():
                failures.append({"seed": seed, "method": method, "returncode": completed.returncode})
                continue
            payload = _load(result_path)
            for row in payload.get("records", []):
                if row.get("split") in {"tune", "promotion"}:
                    worker_rows.append({
                        "seed": int(seed),
                        "method": method,
                        "split": row.get("split"),
                        "mean_regret": row.get("mean_regret"),
                        "normalized_regret": row.get("normalized_regret"),
                        "pairrank_acc": row.get("pairwise_reversal_ranking_accuracy"),
                        "question_metric_rows": row.get("question_metric_rows", []),
                    })

    baseline_methods = [method for method in methods if method != "PESCO-Full"]
    selected_by_seed: dict[int, str | None] = {}
    regret_deltas: list[dict[str, Any]] = []
    pair_deltas: list[dict[str, Any]] = []
    for seed in seeds:
        tune = [row for row in worker_rows if row["seed"] == seed and row["split"] == "tune" and row["method"] in baseline_methods and row.get("normalized_regret") is not None]
        selected = min(tune, key=lambda row: float(row["normalized_regret"]))["method"] if tune else None
        selected_by_seed[int(seed)] = selected
        full = next((row for row in worker_rows if row["seed"] == seed and row["split"] == "promotion" and row["method"] == "PESCO-Full"), None)
        baseline = next((row for row in worker_rows if row["seed"] == seed and row["split"] == "promotion" and row["method"] == selected), None)
        if full and baseline:
            refs = {str(row.get("question_id")): row for row in baseline.get("question_metric_rows", [])}
            for row in full.get("question_metric_rows", []):
                ref = refs.get(str(row.get("question_id")))
                if ref is not None and row.get("normalized_regret") is not None and ref.get("normalized_regret") is not None:
                    regret_deltas.append({
                        "seed": int(seed),
                        "question_id": str(row.get("question_id")),
                        "family": str(row.get("family", ref.get("family", "unknown"))),
                        "value": float(row["normalized_regret"]) - float(ref["normalized_regret"]),
                    })
        noflip = next((row for row in worker_rows if row["seed"] == seed and row["split"] == "promotion" and row["method"] == "PESCO-NoFlipLoss"), None)
        if full and noflip:
            refs = {str(row.get("question_id")): row for row in noflip.get("question_metric_rows", [])}
            for row in full.get("question_metric_rows", []):
                ref = refs.get(str(row.get("question_id")))
                if ref is not None and row.get("pairwise_reversal_ranking_accuracy") is not None and ref.get("pairwise_reversal_ranking_accuracy") is not None:
                    pair_deltas.append({
                        "seed": int(seed),
                        "question_id": str(row.get("question_id")),
                        "family": str(row.get("family", ref.get("family", "unknown"))),
                        "value": float(row["pairwise_reversal_ranking_accuracy"]) - float(ref["pairwise_reversal_ranking_accuracy"]),
                    })

    regret_ci = _two_level_seed_question_ci(regret_deltas, value_key="value", seed=202621, replicates=args.bootstrap_replicates)
    regret_ci["family_leave_one_out"] = _family_leave_one_out_cis(regret_deltas, value_key="value", seed=202622, replicates=args.bootstrap_replicates)
    pair_ci = _two_level_seed_question_ci(pair_deltas, value_key="value", seed=202623, replicates=args.bootstrap_replicates)
    pair_ci["family_leave_one_out"] = _family_leave_one_out_cis(pair_deltas, value_key="value", seed=202624, replicates=args.bootstrap_replicates)
    result = {
        "schema_version": "pesco_tier1_p21_seed_sweep_diagnostic_v0.1",
        "status": "completed_cpu_diagnostic" if not failures else "completed_cpu_diagnostic_with_failures",
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "dataset": str(args.dataset),
        "seeds": list(seeds),
        "methods": list(methods),
        "optimizer_steps": int(args.max_optimizer_steps),
        "worker_failures": failures,
        "selected_baseline_by_seed": selected_by_seed,
        "worker_rows": worker_rows,
        "promotion_regret_delta_full_minus_tune_selected": regret_ci,
        "promotion_pairrank_delta_full_minus_noflip": pair_ci,
        "uncertainty_method": "two-level seed resampling then question-cluster resampling; family leave-one-out",
        "interpretation": "diagnostic only; this bounded sweep does not open v0.5 formal comparison or model scaling",
    }
    (args.output_dir / "seed_sweep_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    manifest = build_run_manifest(
        experiment="tier1_p21_seed_sweep_diagnostic",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[Path(__file__), ROOT / "scripts/run_tier1_p21_diagnostics.py", ROOT / "research_strategy_optimization/evaluation/tier1_p2_experiments.py"],
        data_paths=[args.dataset],
        seeds={"training": list(seeds), "bootstrap": int(args.bootstrap_replicates)},
        status=result["status"],
        diagnostics={"diagnostic_only": True, "formal_comparison_authorized": False, "worker_failures": failures},
    )
    write_run_manifest(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps({"output": str(args.output_dir), "status": result["status"], "worker_count": len(worker_rows), "failures": failures, "regret_ci": regret_ci, "pairrank_ci": pair_ci}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
