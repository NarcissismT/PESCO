"""Evidence rules, proper scoring and append-only hypothesis registration."""

from .evidence_classifier import classify_evidence, evidence_factors
from .proper_scoring import log_score, belief_delta, multiclass_log_score
from .hypothesis_registry import HypothesisRegistry
from .evidence_schema import EvidenceDecision, EvidenceFactors
from .equivalence_tests import EquivalenceDecision, interval_inside_equivalence, practical_equivalence
from .optional_stopping_controls import StoppingSchedule, validate_fixed_horizon

__all__ = [
    "classify_evidence",
    "evidence_factors",
    "log_score",
    "belief_delta",
    "multiclass_log_score",
    "HypothesisRegistry",
    "EvidenceDecision",
    "EvidenceFactors",
    "EquivalenceDecision",
    "interval_inside_equivalence",
    "practical_equivalence",
    "StoppingSchedule",
    "validate_fixed_horizon",
]
