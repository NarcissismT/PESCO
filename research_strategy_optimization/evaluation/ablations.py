"""Preregistered ablation registry for the PESCO mechanism study.

The CPU pilot runs the core mechanism and negative controls.  Formal ablations require
the reserved multi-question splits, so this module keeps their definitions and primary
endpoints machine-readable without fabricating unrun results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class AblationSpec:
    name: str
    removed_mechanism: str
    primary_metrics: tuple[str, ...]
    status: str = "registered_not_run_on_final_splits"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CORE_ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("No-PairedWorld", "paired_world_training", ("flip_accuracy", "vrs")),
    AblationSpec("No-FlipLoss", "preference_reversal_loss", ("flip_accuracy", "effective_switch_rate")),
    AblationSpec("No-Branch", "same_state_real_branches", ("research_regret", "invalid_repair_rate")),
    AblationSpec("No-ProperScore", "proper_log_score", ("refutation_acceptance", "invalid_claim_rate")),
    AblationSpec("No-ValidityGate", "validity_hard_gate", ("invalid_claim_rate", "fdr")),
    AblationSpec("No-Replication", "independent_confirmation", ("fdr", "replication_rate")),
    AblationSpec("No-NoveltyCertificate", "novelty_certificate", ("vnpr", "fdr")),
    AblationSpec("No-StrategyMask", "strategy_token_mask", ("vrs", "cost")),
    AblationSpec("No-CostPenalty", "cost_penalty", ("cost", "vrs")),
    AblationSpec("No-FactorizedEvidence", "factorized_evidence_state", ("state_macro_f1", "invalid_repair_rate")),
    AblationSpec("StaticTeacherMix", "dynamic_teacher_conditioning", ("flip_accuracy", "vrs")),
    AblationSpec("StateGateOnly", "branch_value_learning", ("research_regret", "effective_switch_rate")),
)


def ablation_manifest(specs: Sequence[AblationSpec] = CORE_ABLATIONS) -> dict[str, object]:
    return {
        "schema_version": "pesco_ablations_v0.2",
        "formal_results_authorized": False,
        "reason": "final ID/OOD and promotion splits remain reserved in the CPU pilot",
        "ablations": [spec.to_dict() for spec in specs],
    }


def compare_metric(
    baseline: Mapping[str, float],
    variant: Mapping[str, float],
    metric: str,
) -> float | None:
    """Return variant-minus-baseline only when both values are finite."""

    import math

    left, right = baseline.get(metric), variant.get(metric)
    if left is None or right is None:
        return None
    left, right = float(left), float(right)
    if not (math.isfinite(left) and math.isfinite(right)):
        return None
    return right - left


__all__ = ["AblationSpec", "CORE_ABLATIONS", "ablation_manifest", "compare_metric"]
