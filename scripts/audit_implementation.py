#!/usr/bin/env python3
"""Validate the machine-readable PESCO implementation manifest and artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def audit(root: Path) -> dict:
    manifest_path = root / "docs/implementation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = []
    for requirement in manifest.get("requirements", []):
        for relative in requirement.get("evidence", []):
            if not (root / relative).exists():
                missing.append({"requirement": requirement.get("id"), "path": relative})
    result = {
        "manifest": str(manifest_path),
        "requirement_count": len(manifest.get("requirements", [])),
        "missing_evidence": missing,
        "pass": not missing,
        "completion_boundary": manifest.get("completion_boundary", {}),
    }
    return result


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parents[1]
    result = audit(workspace)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["pass"] else 2)
