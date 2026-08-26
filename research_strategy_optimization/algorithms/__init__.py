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
    objective_breakdown,
    pesco_objective,
    reference_kl,
)
from .option_executor import ExecutionReceipt, OptionExecutor
_DIFFERENTIABLE_AVAILABLE = False
try:  # PyTorch remains optional for the core Tier-0/reference loop.
    from .differentiable_strategy import (
        ACTION_SET,
        STATE_SET,
        FEATURE_DIM,
        DecisionDataset,
        DecisionExample,
        DifferentiableStrategyPolicy,
        DifferentiableStrategyTrainer,
        DifferentiableTrainerConfig,
        DifferentiableTrainingLog,
        ReversalExample,
    )
    _DIFFERENTIABLE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal stdlib installs
    pass

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
    "objective_breakdown",
    "pesco_objective",
    "reference_kl",
    "ExecutionReceipt",
    "OptionExecutor",
]

if _DIFFERENTIABLE_AVAILABLE:
    __all__.extend([
        "ACTION_SET",
        "STATE_SET",
        "FEATURE_DIM",
        "DecisionDataset",
        "DecisionExample",
        "DifferentiableStrategyPolicy",
        "DifferentiableStrategyTrainer",
        "DifferentiableTrainerConfig",
        "DifferentiableTrainingLog",
        "ReversalExample",
    ])
