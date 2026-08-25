"""PESCO: a reproducible, CPU-first prototype of evidence-conditioned strategy optimization.

The package intentionally separates the research policy from the trusted verifier.  The
default implementation is a small synthetic Tier-0/Tier-1 environment that can be run
without a language model; it is useful for validating the algorithmic mechanism before
authorising expensive model training.
"""

from .schemas import (
    EvidenceState,
    ResearchAction,
    Observation,
    Verdict,
    WorldSpec,
    Protocol,
    Hypothesis,
    ExperimentOutput,
    Trajectory,
)

__all__ = [
    "EvidenceState",
    "ResearchAction",
    "Observation",
    "Verdict",
    "WorldSpec",
    "Protocol",
    "Hypothesis",
    "ExperimentOutput",
    "Trajectory",
]

__version__ = "0.1.0"
