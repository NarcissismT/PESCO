#!/usr/bin/env python3
"""Fail-closed compatibility entry point for the consumed v0.5 rehearsal."""

import json
from pathlib import Path


def main() -> int:
    payload = {
        "status": "final_boundary_rehearsal_consumed",
        "formal_comparison_authorized": False,
        "message": "v0.5 is historical and cannot be regenerated; use the private v0.6 evaluator",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
