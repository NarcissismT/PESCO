#!/usr/bin/env python3
"""Private dataset commitment created *after* code freeze.

P2.3.4 requires the commitment to be created after the r1 code freeze.  This
script hashes the current PESCO code and produces a private dataset commitment
record that is bound to the r1 checkpoint digest and the p234 gate receipt.  No
private data is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _canonical_digest(payload) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _hash_tree(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
            try:
                entries.append({"path": str(path.relative_to(root)), "sha256": _sha256_path(path)})
            except OSError:
                continue
    return _canonical_digest(entries)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gate", type=Path, required=True, help="Path to the r1/p234 gate receipt whose status must be NO_GO")
    p.add_argument("--checkpoint-root", type=Path, required=True, help="SFT checkpoint bundle produced by the r1 matrix")
    p.add_argument("--matrix", type=Path, required=True, help="r1 matrix receipt used for p234 aggregation")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)

    gate = json.loads(a.gate.read_text())
    # The commitment is only meaningful when the gate is final (i.e. NO_GO for p233-r1)
    # and after the code has been frozen.  We do not require GO here; that would be
    # promotion logic.  Instead we record the binding at code-freeze time.
    code_root = Path(__file__).resolve().parents[1]
    code_digest = _hash_tree(code_root / "research_strategy_optimization")
    scripts_digest = _hash_tree(code_root / "scripts")
    matrix = json.loads(a.matrix.read_text())

    checkpoint_digests = []
    for sft in sorted(a.checkpoint_root.glob("sft_seed_*.json")):
        manifest = json.loads(sft.read_text())
        checkpoint_digests.append({"seed": manifest.get("seed"), "state_dict_sha256": manifest.get("state_dict_sha256")})

    commitment = {
        "schema_version": "pesco_p234_private_commitment_v1",
        "status": "CREATED_AFTER_CODE_FREEZE",
        "private_dataset_accessed": False,
        "code_freeze": {
            "code_root_digest": code_digest,
            "scripts_digest": scripts_digest,
            "git_head": os.popen("git -C " + str(code_root) + " rev-parse HEAD").read().strip(),
            "git_status_dirty": bool(os.popen("git -C " + str(code_root) + " status --porcelain").read().strip()),
        },
        "bound_r1_checkpoint_digests": checkpoint_digests,
        "matrix_digest": matrix.get("config_digest"),
        "gate_sha256": _sha256_path(a.gate),
        "promotion_target": "promotion_v6",
        "commitment_created_at": datetime.now(timezone.utc).isoformat(),
        "commitment_purpose": "reserve a separate private ID+OOD dataset for p234 promotion-v6 evaluation; no access performed here",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(commitment, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps({"output": str(a.output), "commitment_digest": _canonical_digest(commitment), "status": commitment["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
