#!/usr/bin/env python3
"""Select one global P2.2 baseline using only aggregated tune clusters.

The selection is intentionally separate from promotion/final evaluation.  It
cannot select a different baseline per seed or inspect promotion metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def select(result: dict) -> dict:
    rows = [row for row in result.get("records", []) if row.get("split") == "tune"]
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("method"))].append(row)
    summary = {}
    for method, method_rows in sorted(by_method.items()):
        values = [float(row["normalized_regret"]) for row in method_rows if row.get("normalized_regret") is not None]
        accuracy = [float(row.get("utility_winner_accuracy") or 0.0) for row in method_rows]
        summary[method] = {
            "seed_count": len({int(row["seed"]) for row in method_rows}),
            "question_cluster_count": sum(len(row.get("question_metric_rows", [])) for row in method_rows),
            "mean_normalized_regret": sum(values) / len(values) if values else None,
            "mean_utility_winner_accuracy": sum(accuracy) / len(accuracy) if accuracy else None,
        }
    eligible = [method for method, metrics in summary.items() if metrics["mean_normalized_regret"] is not None]
    selected = min(eligible, key=lambda method: (summary[method]["mean_normalized_regret"], -float(summary[method]["mean_utility_winner_accuracy"] or 0.0))) if eligible else None
    return {
        "schema_version": "pesco_tier1_p22_global_baseline_selection_v0.1",
        "selection_split": "tune",
        "selected_baseline": selected,
        "selection_is_global_across_seeds": True,
        "promotion_metrics_used": False,
        "candidate_summary": summary,
        "formal_comparison_authorized": False,
        "status": "selected_for_diagnostic_promotion" if selected else "selection_failed_closed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--result", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    output = select(json.loads(args.result.read_text(encoding="utf-8"))); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(output, ensure_ascii=False, indent=2)); return 0 if output["selected_baseline"] else 2


if __name__ == "__main__": raise SystemExit(main())
