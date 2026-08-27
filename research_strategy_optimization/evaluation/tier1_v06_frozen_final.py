"""Public v0.6 final-boundary interface.

The v0.6 generator, world recipes, target mapping, and random seeds live only in
the evaluator service/private bundle.  This module intentionally exposes counts,
schema names, and commitment verification helpers, but no constructor that can
materialize a final benchmark.  A missing private evaluator is a hard failure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


V06_SCHEMA = "pesco_tier1_benchmark_v0.6_private_final"
V06_BOUNDARY_STATUS = "private_evaluator_only"
V06_PUBLIC_CONTRACT = {
    "schema_version": V06_SCHEMA,
    "status": V06_BOUNDARY_STATUS,
    "public_exposes": ("split_counts", "question_cluster_count", "world_count", "commitment_digest", "evaluator_digest"),
    "public_excludes": ("recipes", "coefficients", "latent_parameters", "target_actions", "world_ids", "seeds", "family_labels"),
}


def public_contract() -> dict[str, Any]:
    return {**V06_PUBLIC_CONTRACT, "public_exposes": list(V06_PUBLIC_CONTRACT["public_exposes"]), "public_excludes": list(V06_PUBLIC_CONTRACT["public_excludes"])}


def load_public_commitment(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    forbidden = {"recipes", "coefficients", "latent_parameters", "target_actions", "world_ids", "seeds", "family_labels"}
    leaked = sorted(key for key in forbidden if key in payload)
    if leaked:
        raise ValueError(f"v0.6 public commitment leaks evaluator fields: {leaked}")
    return payload


def verify_commitment(payload: Mapping[str, Any], expected_digest: str) -> bool:
    canonical = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    actual = "sha256:" + hashlib.sha256(canonical).hexdigest()
    return actual == str(expected_digest)


__all__ = ["V06_SCHEMA", "V06_BOUNDARY_STATUS", "V06_PUBLIC_CONTRACT", "public_contract", "load_public_commitment", "verify_commitment"]
