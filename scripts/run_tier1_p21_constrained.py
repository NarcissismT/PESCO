#!/usr/bin/env python3
"""Run the bounded P2.1 constrained/PCGrad diagnostic in isolation.

The constrained policy is intentionally executed in its own process because the
CPU PyTorch allocator can retain memory across several independently initialized
policies.  This artifact is diagnostic-only and cannot authorize Tier-2 scaling.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.algorithms.differentiable_strategy import (  # noqa: E402
    DecisionDataset,
)
from research_strategy_optimization.evaluation.tier1_differentiable_suite import (  # noqa: E402
    evaluate_differentiable_policy,
)
from research_strategy_optimization.evaluation.tier1_p21_diagnostics import (  # noqa: E402
    P21Config,
    train_constrained_pesco,
)
from research_strategy_optimization.evaluation.tier1_p21_diagnostics import _add_normalized_metrics  # noqa: E402
from research_strategy_optimization.utils.run_manifest import (  # noqa: E402
    build_run_manifest,
    write_run_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts/tier1_p21_diagnostic/dataset_raw_evidence.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tier1_p21_constrained_diagnostic")
    parser.add_argument("--max-optimizer-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-pcgrad", action="store_true")
    args = parser.parse_args(argv)

    dataset = DecisionDataset.from_json(args.dataset)
    config = P21Config(
        seed=int(args.seed),
        max_optimizer_steps=max(1, int(args.max_optimizer_steps)),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        pcgrad=not bool(args.no_pcgrad),
    )
    policy, log = train_constrained_pesco(dataset, config=config)
    records = []
    for split in sorted({str(example.split) for example in dataset.examples}):
        row = evaluate_differentiable_policy(policy, dataset, split)
        row = _add_normalized_metrics(row, policy, dataset, split)
        row.update({"method": log["method"], "seed": int(args.seed), "diagnostic_only": True})
        # Keep aggregate/question rows but omit action probabilities from this
        # canonical diagnostic to keep the artifact compact and reproducible.
        row["record_count"] = len(row.get("records", ()))
        row.pop("records", None)
        records.append(row)
    result = {
        "schema_version": "pesco_tier1_p21_constrained_diagnostic_v0.1",
        "status": "completed_cpu_diagnostic",
        "config": config.__dict__,
        "method": log["method"],
        "records": records,
        "training_log": log,
        "diagnostic_only": True,
        "formal_comparison_authorized": False,
        "dataset": str(args.dataset),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "constrained_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = build_run_manifest(
        experiment="tier1_p21_constrained_diagnostic",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[
            Path(__file__),
            ROOT / "research_strategy_optimization/evaluation/tier1_p21_diagnostics.py",
            ROOT / "research_strategy_optimization/evaluation/tier1_differentiable_suite.py",
        ],
        data_paths=[args.dataset],
        seeds={"training": int(args.seed)},
        status="completed_cpu_diagnostic",
        diagnostics={
            "method": log["method"],
            "diagnostic_only": True,
            "formal_comparison_authorized": False,
            "optimizer_step_cap": int(config.max_optimizer_steps),
        },
    )
    write_run_manifest(args.output_dir / "run_manifest.json", manifest)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "method": log["method"],
        "optimizer_steps": log.get("optimizer_steps"),
        "mean_cos_branch_flip": log.get("mean_cos_branch_flip"),
        "mean_cos_state_flip": log.get("mean_cos_state_flip"),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    del policy
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
