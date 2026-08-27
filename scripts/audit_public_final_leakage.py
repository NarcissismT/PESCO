#!/usr/bin/env python3
"""Repository-wide audit that public code cannot reconstruct consumed v0.5/v0.6 finals."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FORBIDDEN_PATTERNS = (
    r"build_tier1_v06.*benchmark",
    r"V06_(?:FINAL|GENERATOR|LATENT|TARGET|SEED)",
    r"v06_(?:recipe|latent|target|world|seed)",
)
V05_FORBIDDEN_PATTERNS = (
    r"build_tier1_v05.*benchmark",
    r"V05_(?:FINAL|GENERATOR|LATENT|TARGET|SEED)",
    r"v05_(?:recipe|latent|target|world|seed)",
)


def audit(repo_root: Path = ROOT) -> dict:
    public_files = [
        path.relative_to(repo_root)
        for root in (repo_root / "research_strategy_optimization", repo_root / "scripts", repo_root / "docs")
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    public_files.extend(path for path in (Path("README.md"), Path("pyproject.toml")) if (repo_root / path).exists())
    findings = []
    for path in public_files:
        if path.suffix not in {".py", ".json", ".yaml", ".yml", ".toml", ".md"}:
            continue
        full = repo_root / path
        if path.as_posix() == "scripts/audit_public_final_leakage.py":
            continue
        try:
            text = full.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        patterns = V05_FORBIDDEN_PATTERNS if path.as_posix() == "research_strategy_optimization/evaluation/tier1_v05_frozen_final.py" else FORBIDDEN_PATTERNS
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                # The public contract itself documents excluded field names.  It
                # is safe because no values/recipes are present; only flag source
                # code outside the explicit interface module.
                if path.as_posix() == "research_strategy_optimization/evaluation/tier1_v06_frozen_final.py":
                    continue
                findings.append({"path": path.as_posix(), "pattern": pattern})
    module = importlib.import_module("research_strategy_optimization.evaluation.tier1_v06_frozen_final")
    contract = module.public_contract()
    forbidden_exports = [name for name in dir(module) if re.search(r"(?:build|generate|recipe|latent|target|world|seed)", name, re.I) and not name.startswith("__")]
    # ``verify_commitment`` and ``load_public_commitment`` are allowed helpers;
    # any generator-like export is a hard failure.
    forbidden_exports = [name for name in forbidden_exports if name not in {"load_public_commitment", "verify_commitment", "public_contract"}]
    v05_notice_path = repo_root / "artifacts/tier1_v05_frozen_final/consumption_notice.json"
    try:
        v05_notice = json.loads(v05_notice_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        v05_notice = {}
    v05_consumed = v05_notice.get("status") == "final_boundary_rehearsal_consumed" and not bool(v05_notice.get("formal_comparison_authorized"))
    v05_module = importlib.import_module("research_strategy_optimization.evaluation.tier1_v05_frozen_final")
    v05_contract = v05_module.public_contract()
    v05_public_exports = [name for name in dir(v05_module) if re.search(r"(?:build|generate|recipe|latent|target|world|seed)", name, re.I) and not name.startswith("__")]
    v05_public_exports = [name for name in v05_public_exports if name not in {"load_public_commitment", "public_contract"}]
    v05_contract_pass = v05_contract.get("status") == "final_boundary_rehearsal_consumed" and not v05_contract.get("generator_available") and not v05_public_exports
    passed = not findings and not forbidden_exports and not v05_public_exports and contract.get("status") == "private_evaluator_only" and v05_consumed and v05_contract_pass
    result = {
        "schema_version": "pesco_public_final_leakage_audit_v0.1",
        "status": "pass" if passed else "fail_closed_public_leakage",
        "passed": passed,
        "tracked_public_file_count": len(public_files),
        "findings": findings,
        "forbidden_exports": forbidden_exports,
        "v06_public_contract": contract,
        "v05_status": v05_notice.get("status", "missing_consumption_notice"),
        "v05_consumption_notice_pass": v05_consumed,
        "v05_public_contract": v05_contract,
        "v05_forbidden_exports": v05_public_exports,
        "v05_public_contract_pass": v05_contract_pass,
        "private_evaluator_required": True,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/tier1_v06_public_leakage_audit.json")
    args = parser.parse_args(argv)
    result = audit(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
