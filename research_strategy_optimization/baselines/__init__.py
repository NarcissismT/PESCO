"""Comparable CPU baselines for the MVP.

The adapters share the same environment and verifier; differences are restricted to how
they choose a high-level action.  They are intentionally lightweight stand-ins for the
full LLM methods named in the plan and are labelled as such in reports.
"""

from .policies import (
    BasePolicy,
    EvidenceHeuristicPolicy,
    OracleSearchPolicy,
    PESCOPolicy,
    RandomPolicy,
)

__all__ = ["BasePolicy", "EvidenceHeuristicPolicy", "OracleSearchPolicy", "PESCOPolicy", "RandomPolicy"]
