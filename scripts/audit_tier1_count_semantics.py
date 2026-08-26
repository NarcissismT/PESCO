#!/usr/bin/env python3
"""Audit the Tier-1 v0.3 48/192/768 artifact boundary.

This command is metadata-only.  It does not rerun the benchmark or inspect hidden
target labels; it checks that the generated files agree on their declared
granularity and writes a compact machine-readable audit beside the A artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(artifact_dir: Path | None = None) -> dict[str, Any]:
    directory = Path(artifact_dir or ROOT / "artifacts/tier1_v03")
    counts = _load(directory / "counts.json")
    groups = _load(directory / "question_world_groups.json")
    action_rows = _load(directory / "action_level_branches.json")
    seed_rows = [
        json.loads(line)
        for line in (directory / "seed_level_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    suite_path = directory.parent / "tier1_differentiable_suite/suite.json"
    suite = _load(suite_path) if suite_path.exists() else {}
    provenance = suite.get("dataset_provenance", {})
    checks = {
        "counts_branch_groups_is_48": counts.get("branch_groups") == 48,
        "counts_question_world_groups_is_48": counts.get("question_world_group_count") == 48,
        "counts_action_rows_is_192": counts.get("action_level_row_count") == 192,
        "counts_seed_observations_is_768": counts.get("seed_level_observation_count") == 768,
        "group_index_has_48_records": len(groups) == 48,
        "action_file_has_192_records": len(action_rows) == 192,
        "seed_file_has_768_records": len(seed_rows) == 768,
        "action_rows_are_explicit": all(row.get("record_granularity") == "action_level" for row in action_rows),
        "seed_rows_are_explicit": all(row.get("record_granularity") == "seed_level" for row in seed_rows),
        "suite_provenance_matches": {
            "question_world_group_count": provenance.get("question_world_group_count") == 48,
            "action_level_row_count": provenance.get("action_level_row_count") == 192,
            "seed_level_observation_count": provenance.get("seed_level_observation_count") == 768,
        },
    }
    suite_checks = checks.pop("suite_provenance_matches")
    checks.update({f"suite_{key}": value for key, value in suite_checks.items()})
    return {
        "schema_version": "pesco_tier1_count_semantics_audit_v0.1",
        "status": "pass" if all(bool(value) for value in checks.values()) else "fail",
        "expected": {
            "question_world_group_count": 48,
            "action_level_row_count": 192,
            "seed_level_observation_count": 768,
        },
        "checks": checks,
        "sources": [
            "counts.json",
            "question_world_groups.json",
            "action_level_branches.json",
            "branch_groups.json",
            "seed_level_results.jsonl",
            "../tier1_differentiable_suite/dataset.json",
            "../tier1_differentiable_suite/suite.json",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    output_dir = Path(argv[0]) if argv else ROOT / "artifacts/tier1_v03"
    result = audit(output_dir)
    (output_dir / "count_semantics_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
