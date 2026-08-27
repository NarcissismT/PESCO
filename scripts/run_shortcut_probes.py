#!/usr/bin/env python3
"""Run tabular shortcut probes on a Tier-1 decision dataset.

The runner fits Logistic Regression, Random Forest, and Gradient Boosting probes on
the registered training split and evaluates untouched diagnostic/holdout splits.
Only public observations are used as features; branch utilities are evaluator-side
labels and regret receipts.  If sklearn is absent, the module runs explicitly
labelled NumPy implementations for a bounded diagnostic.  ``--strict-sklearn``
turns that condition into a fail-closed result instead.
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
from research_strategy_optimization.utils.run_manifest import build_run_manifest, write_run_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="artifacts/tier1_v04_extended/dataset_raw_evidence.json",
        help="audit dataset containing public observations and evaluator branch utilities",
    )
    parser.add_argument("--output", default="artifacts/tier1_shortcut_probe")
    parser.add_argument("--train-split", default="train")
    parser.add_argument(
        "--eval-split",
        dest="eval_splits",
        action="append",
        default=None,
        help="evaluation split (repeatable); defaults to all non-training splits",
    )
    parser.add_argument(
        "--feature-set",
        dest="feature_sets",
        action="append",
        choices=FEATURE_SETS,
        default=None,
        help="repeatable; defaults to all_raw and without_confirmation",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        choices=MODEL_NAMES,
        default=None,
        help="repeatable; defaults to all three shortcut probes",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--strict-sklearn",
        action="store_true",
        help="fail closed instead of running explicitly labelled NumPy fallbacks",
    )
    args = parser.parse_args(argv)
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    result = run_shortcut_probe(
        dataset_path,
        output_dir=output_dir,
        feature_sets=tuple(args.feature_sets or FEATURE_SETS),
        models=tuple(args.models or MODEL_NAMES),
        train_split=str(args.train_split),
        eval_splits=tuple(args.eval_splits) if args.eval_splits else None,
        seed=int(args.seed),
        bootstrap_replicates=max(1, int(args.bootstrap_replicates)),
        strict_sklearn=bool(args.strict_sklearn),
        max_examples=args.max_examples,
    )
    manifest = build_run_manifest(
        experiment="tier1_shortcut_probes",
        repo_root=ROOT,
        command=sys.argv,
        runner_paths=[
            Path(__file__),
            ROOT / "research_strategy_optimization/evaluation/shortcut_probes.py",
            ROOT / "requirements-optional.txt",
        ],
        data_paths=[dataset_path],
        seeds={"probe": int(args.seed), "bootstrap": int(args.bootstrap_replicates)},
        status=str(result.get("status", "unknown")),
        diagnostics={
            "result_schema": result.get("schema_version"),
            "fallback_used": result.get("fallback_used"),
            "strict_sklearn": bool(args.strict_sklearn),
            "feature_sets": list(args.feature_sets or FEATURE_SETS),
            "models": list(args.models or MODEL_NAMES),
        },
    )
    write_run_manifest(output_dir / "run_manifest.json", manifest)
    summary = {
        "output": str(output_dir),
        "status": result.get("status"),
        "sklearn_available": result.get("dependency", {}).get("sklearn", {}).get("available"),
        "fallback_used": result.get("fallback_used", False),
        "eval_splits": result.get("eval_splits", []),
        "model_statuses": {
            key: value.get("status") for key, value in result.get("models", {}).items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    # A missing optional dependency or a scientifically valid NO-GO is represented
    # inside the artifact; it is not a process failure for batch experiment runs.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
