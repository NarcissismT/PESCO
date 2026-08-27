#!/usr/bin/env python3
"""Write a v0.5 consumption audit; no historical evaluator is re-run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_tier1_v05_frozen_final import audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tier1_v05_frozen_final/independent_audit.json"))
    args = parser.parse_args(argv)
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
