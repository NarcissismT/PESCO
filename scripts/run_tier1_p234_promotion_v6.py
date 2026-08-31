#!/usr/bin/env python3
"""P2.3.4 promotion-v6 guard bound to r1 checkpoints.

The v5 runner bound the promotion guard to the *diagnostic* dataset and the
old p233 gate.  This v6 variant requires:

* the p234 gate receipt,
* the r1 matrix checkpoint digests (sft checkpoints from the r1 final run),
* the private dataset commitment produced by ``create_tier1_p234_private_commitment.py``.

It still creates a one-shot sentinel before any private access, so that an
already-authorized run cannot be re-executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _canonical_digest(payload) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _checkpoint_digests(root: Path) -> list[dict]:
    rows = []
    for manifest_path in sorted(root.glob("sft_seed_*.json")):
        manifest = json.loads(manifest_path.read_text())
        ckpt = manifest_path.with_suffix(".pt")
        ok = ckpt.exists() and manifest.get("state_dict_sha256") and manifest.get("status") == "completed"
        rows.append({
            "seed": manifest.get("seed"),
            "checkpoint": str(ckpt),
            "manifest": str(manifest_path),
            "state_dict_sha256": manifest.get("state_dict_sha256"),
            "verified": bool(ok),
        })
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--p234-gate", type=Path, required=True, help="p234 gate receipt produced by write_tier1_p234_gate.py")
    p.add_argument("--r1-matrix", type=Path, required=True, help="r1 atomic_target matrix receipt")
    p.add_argument("--r1-checkpoint-root", type=Path, required=True, help="sft checkpoint root from the r1 matrix")
    p.add_argument("--private-commitment", type=Path, required=True, help="commitment produced by create_tier1_p234_private_commitment.py")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)

    gate = json.loads(a.p234_gate.read_text())
    commitment = json.loads(a.private_commitment.read_text())
    matrix = json.loads(a.r1_matrix.read_text())

    checkpoints = _checkpoint_digests(a.r1_checkpoint_root)
    checkpoints_ok = bool(checkpoints) and all(r["verified"] for r in checkpoints)

    gate_sha = _sha256_path(a.p234_gate)
    commitment_sha = _sha256_path(a.private_commitment)
    matrix_sha = _sha256_path(a.r1_matrix)

    authorized = bool(
        gate.get("status") == "GO"
        and gate.get("p234_go")
        and all(bool(v) for v in gate["p234_go"].values())
        and commitment.get("status") == "CREATED_AFTER_CODE_FREEZE"
        and commitment.get("private_dataset_accessed") is False
        and checkpoints_ok
    )

    sentinel = Path(str(a.output) + ".sentinel")
    if sentinel.exists():
        raise RuntimeError(f"one-shot promotion sentinel already exists: {sentinel}")
    sentinel.touch()

    out = {
        "schema_version": "pesco_promotion_v6_guard_v1",
        "status": "AUTHORIZED" if authorized else "REFUSED_P234_NO_GO",
        "promotion_runner_bound_to_r1_checkpoints": True,
        "p234_gate_sha256": gate_sha,
        "private_commitment_sha256": commitment_sha,
        "r1_matrix_sha256": matrix_sha,
        "checkpoint_receipts": checkpoints,
        "checkpoint_bundle_verified": checkpoints_ok,
        "sentinel": str(sentinel),
        "sentinel_created_before_private_access": True,
        "private_data_accessed": False,
        "prerequisites": {
            "p234_gate_go": authorized,
            "private_commitment_after_code_freeze": commitment.get("status") == "CREATED_AFTER_CODE_FREEZE",
            "private_commitment_untouched": commitment.get("private_dataset_accessed") is False,
            "r1_checkpoint_bundle_verified": checkpoints_ok,
        },
        "reason": "P2.3.4 gate must be GO and bound to r1 checkpoints before any private promotion-v6 access",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps({"output": str(a.output), "authorized": authorized, "sentinel": str(sentinel)}))
    return 0 if authorized else 2


if __name__ == "__main__":
    raise SystemExit(main())
