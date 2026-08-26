"""Plan-aligned metrics, statistics and completion gates."""

from .evidence_metrics import evidence_macro_f1, confusion_matrix
from .strategy_regret import research_regret
from .preference_reversal import flip_accuracy, flip_counts
from .refutation_metrics import conditional_counts, refutation_acceptance, underpower_handling
from .replication_metrics import replication_counts, replication_rate, false_discovery_rate
from .novelty_metrics import validated_novel_path_rate, method_family_entropy
from .multiple_testing import benjamini_hochberg, holm_adjust, paired_permutation_pvalue
from .statistics import paired_binary_power, paired_binary_required_n, paired_bootstrap_ci, paired_sign_permutation_pvalue
from .ablations import AblationSpec, CORE_ABLATIONS, ablation_manifest, compare_metric
from .final_decision import assert_training_allowed
from .experiment_scaffolds import experiment_b_zero_shot_diagnostic, experiment_c_state_reward_diagnostic
try:  # PyTorch is optional in the standard NumPy/reference installation.
    from .tier1_differentiable_suite import (
        DEFAULT_METHODS,
        Tier1SuiteConfig,
        collect_tier1_v03_dataset,
        evaluate_differentiable_policy,
        is_invalid_local_optimization,
        run_tier1_differentiable_suite,
    )
    _DIFFERENTIABLE_SUITE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in no-torch environments
    DEFAULT_METHODS = ()
    Tier1SuiteConfig = None
    collect_tier1_v03_dataset = None
    evaluate_differentiable_policy = None
    is_invalid_local_optimization = None
    run_tier1_differentiable_suite = None
    _DIFFERENTIABLE_SUITE_AVAILABLE = False
from .legacy_certificates import (
    annotate_legacy_certificate,
    build_legacy_certificate_manifest,
    legacy_scope_payload,
)
from .tier1_v04 import (
    TRACK_ORACLE_STATE,
    TRACK_RAW_EVIDENCE,
    V04_TRACKS,
    build_tier1_v04_benchmark,
    tier1_v04_manifest,
    candidate_scenarios,
    posterior_from_evidence,
    build_candidate_action_table,
    plan_world,
)
from .tier1_v04_extended import (
    V04_EXTENDED_FAMILIES,
    V04_EXTENDED_EXPLORATION_SEEDS,
    V04_EXTENDED_CONFIRMATION_SEEDS,
    V04_FORMAL_SCHEMA,
    FORMAL_FINAL_ID_FAMILIES,
    FORMAL_FINAL_OOD_FAMILIES,
    FORMAL_SPLITS,
    V04ExtendedQuestion,
    V04ExtendedBenchmark,
    V04FormalFinalBenchmark,
    build_tier1_v04_extended_benchmark,
    build_tier1_v04_formal_final_benchmark,
    collect_tier1_v04_extended,
)

__all__ = [
    "evidence_macro_f1",
    "confusion_matrix",
    "research_regret",
    "flip_accuracy",
    "flip_counts",
    "conditional_counts",
    "refutation_acceptance",
    "underpower_handling",
    "replication_rate",
    "false_discovery_rate",
    "replication_counts",
    "validated_novel_path_rate",
    "method_family_entropy",
    "holm_adjust",
    "benjamini_hochberg",
    "paired_permutation_pvalue",
    "paired_binary_power",
    "paired_binary_required_n",
    "paired_bootstrap_ci",
    "paired_sign_permutation_pvalue",
    "AblationSpec",
    "CORE_ABLATIONS",
    "ablation_manifest",
    "compare_metric",
    "assert_training_allowed",
    "experiment_b_zero_shot_diagnostic",
    "experiment_c_state_reward_diagnostic",
    "DEFAULT_METHODS",
    "Tier1SuiteConfig",
    "collect_tier1_v03_dataset",
    "evaluate_differentiable_policy",
    "is_invalid_local_optimization",
    "run_tier1_differentiable_suite",
    "_DIFFERENTIABLE_SUITE_AVAILABLE",
    "annotate_legacy_certificate",
    "build_legacy_certificate_manifest",
    "legacy_scope_payload",
    "TRACK_ORACLE_STATE",
    "TRACK_RAW_EVIDENCE",
    "V04_TRACKS",
    "build_tier1_v04_benchmark",
    "tier1_v04_manifest",
    "candidate_scenarios",
    "posterior_from_evidence",
    "build_candidate_action_table",
    "plan_world",
    "V04_EXTENDED_FAMILIES",
    "V04_EXTENDED_EXPLORATION_SEEDS",
    "V04_EXTENDED_CONFIRMATION_SEEDS",
    "V04_FORMAL_SCHEMA",
    "FORMAL_FINAL_ID_FAMILIES",
    "FORMAL_FINAL_OOD_FAMILIES",
    "FORMAL_SPLITS",
    "V04ExtendedQuestion",
    "V04ExtendedBenchmark",
    "V04FormalFinalBenchmark",
    "build_tier1_v04_extended_benchmark",
    "build_tier1_v04_formal_final_benchmark",
    "collect_tier1_v04_extended",
]
