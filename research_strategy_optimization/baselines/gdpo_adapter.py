"""Named CPU adapter for the GDPO comparison row.

This is a protocol-matched reference policy, not an external-paper reimplementation.
"""

from .policies import EvidenceHeuristicPolicy

IMPLEMENTATION_STATUS = "reference_cpu_adapter"
policy_class = EvidenceHeuristicPolicy

__all__ = ["IMPLEMENTATION_STATUS", "policy_class"]
