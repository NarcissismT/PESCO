"""Compatibility entry point for the Tier-1 confounding task family.

The implementation lives in :mod:`tier1_tabular_env` so both named Tier-1 adapters
share the exact same budget, snapshot, and verifier contract.  Keeping this module
separate mirrors the research-plan layout and gives experiment runners a stable import
path without duplicating the simulator.
"""

from .tier1_tabular_env import Tier1ConfoundingEnvironment

__all__ = ["Tier1ConfoundingEnvironment"]
