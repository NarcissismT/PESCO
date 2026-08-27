#!/usr/bin/env python3
"""Fail-closed compatibility entry point for the consumed v0.5 baseline gate."""

import json


def main() -> int:
    payload = {
        "status": "final_boundary_rehearsal_consumed",
        "baseline_selection_authorized": False,
        "message": "v0.5 baseline selection is historical; select baselines only for v0.6 development",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
