"""Machine-readable scope markers for historical discovery certificates.

The fixed-action MVP keeps ``switch_to_alternative_method`` inside a registered
action set, so it is not open-ended discovery.  A small set of old pilot files was
written before that boundary was made explicit and may still contain
``certificate_pass=true`` and ``autonomous=true``.  Those values are preserved as
historical facts, but every reader must use the scope marker below before treating a
certificate as current evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


LEGACY_CERTIFICATE_ARTIFACT_SCOPE = "legacy_certificate_evidence"
LEGACY_CERTIFICATE_STATUS = "legacy_out_of_current_discovery_scope"
DISCOVERY_POLICY_ID = "fixed_mvp_action_space_v1"
DISCOVERY_POLICY_STATUS = "disabled"


def legacy_scope_payload(*, discovery_policy_ref: str = "discovery_policy.json") -> Dict[str, Any]:
    """Return the stable scope object attached to historical certificate files."""

    return {
        "is_legacy": True,
        "status": LEGACY_CERTIFICATE_STATUS,
        "artifact_scope": LEGACY_CERTIFICATE_ARTIFACT_SCOPE,
        "discovery_policy_id": DISCOVERY_POLICY_ID,
        "discovery_policy_status": DISCOVERY_POLICY_STATUS,
        "discovery_policy_ref": discovery_policy_ref,
        "current_formal_claim_authorized": False,
        "certificate_pass_interpretation": "historical_only_not_current_discovery",
    }


def annotate_legacy_certificate(
    payload: Mapping[str, Any],
    *,
    discovery_policy_ref: str = "discovery_policy.json",
) -> Dict[str, Any]:
    """Add an idempotent legacy marker without changing historical certificate flags."""

    annotated = dict(payload)
    annotated["legacy"] = True
    annotated["artifact_scope"] = LEGACY_CERTIFICATE_ARTIFACT_SCOPE
    annotated["legacy_scope"] = legacy_scope_payload(discovery_policy_ref=discovery_policy_ref)
    # A duplicated explicit boolean is intentional: simple JSON consumers should
    # not have to parse the nested scope object to fail closed.
    annotated["current_formal_claim_authorized"] = False
    return annotated


def build_legacy_certificate_manifest(
    directory: str | Path,
    *,
    discovery_policy_ref: str = "discovery_policy.json",
) -> Dict[str, Any]:
    """Scan and describe historical ``certificate_*.json`` files in ``directory``."""

    root = Path(directory)
    files = []
    for path in sorted(root.glob("certificate_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # A future open-ended runner may place explicitly current certificates in
        # the same directory.  This manifest is only the historical quarantine;
        # leave explicitly scoped current artifacts to their own manifest.
        if payload.get("artifact_scope") not in (None, LEGACY_CERTIFICATE_ARTIFACT_SCOPE):
            continue
        files.append({
            "path": path.name,
            "certificate_pass": bool(payload.get("certificate_pass", False)),
            "autonomous": bool(payload.get("autonomous", False)),
            "legacy": bool(payload.get("legacy", False)),
            "artifact_scope": payload.get("artifact_scope"),
            "current_formal_claim_authorized": bool(
                payload.get("current_formal_claim_authorized", False)
            ),
        })
    scope = legacy_scope_payload(discovery_policy_ref=discovery_policy_ref)
    return {
        "schema_version": "pesco_legacy_certificate_manifest_v0.1",
        "artifact_scope": LEGACY_CERTIFICATE_ARTIFACT_SCOPE,
        "legacy": True,
        "scope": scope,
        "discovery_policy_ref": discovery_policy_ref,
        "discovery_policy_status": DISCOVERY_POLICY_STATUS,
        "certificate_count": len(files),
        "certificate_pass_true_count": sum(item["certificate_pass"] for item in files),
        "files": files,
        "interpretation": (
            "Historical certificate outputs are retained for implementation evidence. "
            "Their pass/autonomous fields do not authorize current fixed-action "
            "discovery claims."
        ),
    }


__all__ = [
    "DISCOVERY_POLICY_ID",
    "DISCOVERY_POLICY_STATUS",
    "LEGACY_CERTIFICATE_ARTIFACT_SCOPE",
    "LEGACY_CERTIFICATE_STATUS",
    "annotate_legacy_certificate",
    "build_legacy_certificate_manifest",
    "legacy_scope_payload",
]
