"""Core PESCO strategy-level rollout and preference-learning utilities."""

from .branch_rollout import BranchExecution, BranchRolloutManager
from .leave_one_out_advantage import (
    assign_leave_one_out_advantages,
    leave_one_out_advantages,
    leave_one_out_baselines,
)
from .paired_world_sampler import (
    PairedWorldPreferenceBuilder,
    PairedWorldSample,
    PairedWorldSampler,
    construct_reversal_pairs,
    build_reversal_pair,
    identify_confirmed_reversal,
    sample_paired_worlds,
)
from .preference_reversal_loss import (
    PreferenceReversalExample,
    batched_preference_reversal_loss,
    preference_reversal_loss,
    reversal_accuracy,
)
from .objectives import (
    ObjectiveWeights,
    clipped_option_loss,
    factorized_evidence_loss,
    masked_token_advantages,
    pesco_objective,
    reference_kl,
)
from .option_executor import ExecutionReceipt, OptionExecutor

__all__ = [
    "BranchExecution",
    "BranchRolloutManager",
    "assign_leave_one_out_advantages",
    "leave_one_out_advantages",
    "leave_one_out_baselines",
    "PairedWorldSample",
    "PairedWorldPreferenceBuilder",
    "PairedWorldSampler",
    "construct_reversal_pairs",
    "build_reversal_pair",
    "identify_confirmed_reversal",
    "sample_paired_worlds",
    "PreferenceReversalExample",
    "batched_preference_reversal_loss",
    "preference_reversal_loss",
    "reversal_accuracy",
    "ObjectiveWeights",
    "clipped_option_loss",
    "factorized_evidence_loss",
    "masked_token_advantages",
    "pesco_objective",
    "reference_kl",
    "ExecutionReceipt",
    "OptionExecutor",
]
