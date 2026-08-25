"""Validated novelty certificate; text novelty alone is never rewarded."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Set

from ..schemas import DiscoveryCertificate, Verdict


def method_structure_signature(
    method_family: str,
    estimand: str = "group_held_out_accuracy_delta",
    intervention: str = "train",
    data_regime: str = "group_split",
    evaluation_protocol: str = "frozen_test",
) -> tuple:
    return (method_family, estimand, intervention, data_regime, evaluation_protocol)


def make_discovery_certificate(
    *,
    method_family: str,
    proposed_without_method_hint: bool,
    structurally_distinct: bool,
    actually_executed: bool,
    verdict: Verdict,
    lower_confidence_gain: float,
    discovery_margin: float = 0.05,
    source: str = "policy_on_policy",
) -> DiscoveryCertificate:
    independently_confirmed = bool(verdict.independent_confirmation_passed)
    autonomous = source == "policy_on_policy" and proposed_without_method_hint
    passed = bool(
        autonomous
        and structurally_distinct
        and actually_executed
        and verdict.validity_pass
        and independently_confirmed
        and lower_confidence_gain > discovery_margin
    )
    reason = "passed" if passed else "one or more execution/validity/confirmation/novelty gates failed"
    return DiscoveryCertificate(
        method_family=method_family,
        proposed_without_method_hint=proposed_without_method_hint,
        structurally_distinct=structurally_distinct,
        actually_executed=actually_executed,
        valid_experiment=verdict.validity_pass,
        independently_confirmed=independently_confirmed,
        lower_confidence_gain=float(lower_confidence_gain),
        autonomous=autonomous,
        certificate_pass=passed,
        reason=reason,
        proposal_source=source,
    )
