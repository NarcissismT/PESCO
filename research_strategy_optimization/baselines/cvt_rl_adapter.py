"""Named CPU adapter for the CVT-RL comparison row."""

from .policies import EvidenceHeuristicPolicy

IMPLEMENTATION_STATUS = "reference_cpu_adapter"
policy_class = EvidenceHeuristicPolicy

__all__ = ["IMPLEMENTATION_STATUS", "policy_class"]
