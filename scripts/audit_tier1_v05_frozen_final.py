#!/usr/bin/env python3
"""Audit the historical v0.5 consumption marker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_strategy_optimization.evaluation.tier1_v05_frozen_final import public_contract


def audit(repo_root: Path | None = None) -> dict:
    contract = dict(public_contract())
    notice_path = (repo_root or Path(__file__).resolve().parents[1]) / "artifacts/tier1_v05_frozen_final/consumption_notice.json"
    try:
        notice = json.loads(notice_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        notice = {}
    passed = (
        contract.get("status") == "final_boundary_rehearsal_consumed"
        and not contract.get("generator_available")
        and notice.get("status") == "final_boundary_rehearsal_consumed"
        and not notice.get("formal_comparison_authorized")
    )
    return {"status": "pass" if passed else "fail_closed", "passed": passed, "public_contract": contract, "consumption_notice": notice}


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 2)
