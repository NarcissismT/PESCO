"""Stage gates from the plan; gates are evidence-producing, not decorative flags."""

from __future__ import annotations

from typing import Mapping, Sequence


def freeze_check(manifest: Mapping[str, object]) -> dict:
    checks = {
        "question_manifest_is_sealed": bool(manifest.get("question_manifest_sealed", False)),
        "final_split_is_inaccessible": bool(manifest.get("final_split_inaccessible", False)),
        "world_id_hidden": bool(manifest.get("world_id_hidden", False)),
        "verifier_immutable": bool(manifest.get("verifier_immutable", False)),
        "contamination_audit_pass": bool(manifest.get("contamination_audit_pass", False)),
        "resource_budget_defined": bool(manifest.get("resource_budget_defined", False)),
    }
    # If a runner supplies immutable content digests, bind them to the freeze result.
    # Older callers may omit these optional fields; the boolean protocol checks above
    # remain backwards compatible.
    for field in ("protocol_digest", "verifier_digest", "question_manifest_digest", "world_manifest_digest"):
        if field in manifest:
            digest = str(manifest.get(field, ""))
            checks[field] = digest
            checks[f"{field}_bound"] = digest.startswith("sha256:") and len(digest) > len("sha256:")
    # A frozen run must not mix a v0.1 runtime protocol with the checked-in v0.2
    # manifests.  The expected value is supplied by the runner/config rather than
    # hidden in this generic gate so older external callers remain compatible.
    if "expected_protocol_version" in manifest or "protocol_version" in manifest:
        actual = str(manifest.get("protocol_version", ""))
        expected = str(manifest.get("expected_protocol_version", actual))
        checks["protocol_version"] = actual
        checks["expected_protocol_version"] = expected
        checks["protocol_version_consistent"] = bool(actual and actual == expected)
    checks["pass"] = all(checks.values())
    return checks


def mvp_gate(results: Mapping[str, object]) -> dict:
    # Older callers only reported the original eight booleans.  Keep that API
    # compatible while making the pilot's explicit 16-group/64-seed counts mandatory
    # whenever they are supplied (the pilot always supplies them).
    legacy_world_check = bool(results.get("all_worlds_execute", False))
    required = {
        "all_worlds_execute": legacy_world_check,
        "branch_group_count_16": bool(results.get("branch_group_count_16", legacy_world_check)),
        "exploration_experiments_64": bool(results.get("exploration_experiments_64", legacy_world_check)),
        "scientific_verifier_independent": bool(results.get("scientific_verifier_independent", False)),
        "invalid_world_detected": bool(results.get("invalid_world_detected", False)),
        "insufficient_not_refuted": bool(results.get("insufficient_not_refuted", False)),
        "supported_refuted_distinguishable": bool(results.get("supported_refuted_distinguishable", False)),
        "confirmed_reversal": bool(results.get("confirmed_reversal", False)),
        "no_world_identifier_leakage": bool(results.get("no_world_identifier_leakage", False)),
        "reproducible_branches": bool(results.get("reproducible_branches", False)),
        "negative_controls_pass": bool(results.get("negative_controls_pass", True)),
    }
    required["pass"] = all(required.values())
    # Passing the Tier-0 gate only authorises the transparent CPU reference loop.  It
    # never authorises Tier-1/2 online LLM RL or QLoRA; those require a separate
    # scientific-hard-gate artifact.
    required["cpu_reference_loop_authorized"] = required["pass"]
    required["tier2_llm_training_authorized"] = False
    return required


def stage_status(stage: str, evidence: Mapping[str, object]) -> dict:
    passed = bool(evidence.get("pass", False))
    # Diagnostic-only records (for example a CPU BasePolicy run without a frozen
    # model) cannot open a model-dependent stage even if a caller accidentally leaves
    # a stale ``pass: true`` field in the payload.
    if bool(evidence.get("diagnostic_only", False)) and not bool(evidence.get("real_model_zero_shot_completed", False)):
        passed = False
    return {"stage": stage, "status": "GO" if passed else "NO-GO", "evidence": dict(evidence)}


def assert_training_allowed(
    freeze: Mapping[str, object],
    environment: object,
    verifier: object,
    *,
    require_scientific_gate: bool = True,
) -> None:
    """Fail closed before formal online training.

    CPU reference loops may call this with ``require_scientific_gate=False``.  A real
    Tier-2 runner must provide every immutable/negative-control flag and explicitly open
    the scientific hard gate.
    """

    checks = {
        "freeze_pass": bool(freeze.get("pass", False)),
        "environment_snapshot": hasattr(environment, "snapshot") and hasattr(environment, "restore"),
        "environment_hidden_world": bool(getattr(environment, "world_id_hidden", not require_scientific_gate)),
        "verifier_immutable": bool(getattr(verifier, "immutable", not require_scientific_gate)),
        "verifier_evaluate": hasattr(verifier, "evaluate") or hasattr(verifier, "assess_validity"),
    }
    if require_scientific_gate:
        checks["scientific_hard_gate"] = bool(freeze.get("scientific_hard_gate_pass", False))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("formal online training is NO-GO; failed checks: " + ", ".join(failed))
