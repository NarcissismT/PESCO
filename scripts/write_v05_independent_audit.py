#!/usr/bin/env python3
"""Persist the independent v0.5 audit result without changing the freeze bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_tier1_v05_frozen_final import audit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=ROOT / "artifacts/tier1_v05_frozen_final")
    parser.add_argument("--evaluator", type=Path, default=ROOT / "artifacts/tier1_v05_evaluator_private")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or (args.public / "independent_audit.json")
    result = audit(args.public, args.evaluator)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "pass": result.get("pass"), "gate_count": len(result.get("gates", {}))}, ensure_ascii=False, indent=2))
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
