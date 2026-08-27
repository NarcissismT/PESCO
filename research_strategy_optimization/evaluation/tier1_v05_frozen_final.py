"""Opaque v0.5 boundary marker.

The v0.5 rehearsal final was intentionally consumed after review found that its
public source could reconstruct the latent generator.  The hidden benchmark and
receipts remain historical artifacts, but this importable package no longer
contains recipes, world parameters, or a generator.  Any attempt to regenerate
v0.5 fails closed; v0.6 is the only boundary eligible for a future formal run.
"""

from __future__ import annotations

from typing import Any, Mapping

V05_SCHEMA = "pesco_tier1_benchmark_v0.5_frozen_final"
V05_BOUNDARY_STATUS = "final_boundary_rehearsal_consumed"
V05_PUBLIC = False
V05_FORMAL_COMPARISON_AUTHORIZED = False


def public_contract() -> Mapping[str, Any]:
    return {
        "schema_version": V05_SCHEMA,
        "status": V05_BOUNDARY_STATUS,
        "public": V05_PUBLIC,
        "formal_comparison_authorized": V05_FORMAL_COMPARISON_AUTHORIZED,
        "generator_available": False,
        "reason": "v0.5 public source was reconstructible and is consumed",
    }


def load_public_commitment(path: str | None = None) -> Mapping[str, Any]:
    """Return only the opaque historical commitment, never hidden benchmark data."""

    return public_contract()


__all__ = [
    "V05_SCHEMA",
    "V05_BOUNDARY_STATUS",
    "V05_PUBLIC",
    "V05_FORMAL_COMPARISON_AUTHORIZED",
    "public_contract",
    "load_public_commitment",
]
