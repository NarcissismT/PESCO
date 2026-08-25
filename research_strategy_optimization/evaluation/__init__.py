"""Plan-aligned metrics, statistics and completion gates."""

from .evidence_metrics import evidence_macro_f1, confusion_matrix
from .strategy_regret import research_regret
from .preference_reversal import flip_accuracy
from .refutation_metrics import refutation_acceptance, underpower_handling
from .replication_metrics import replication_rate, false_discovery_rate
from .novelty_metrics import validated_novel_path_rate, method_family_entropy
from .multiple_testing import benjamini_hochberg, holm_adjust, paired_permutation_pvalue
from .statistics import paired_binary_power, paired_binary_required_n, paired_bootstrap_ci, paired_sign_permutation_pvalue
from .ablations import AblationSpec, CORE_ABLATIONS, ablation_manifest, compare_metric
from .final_decision import assert_training_allowed

__all__ = [
    "evidence_macro_f1",
    "confusion_matrix",
    "research_regret",
    "flip_accuracy",
    "refutation_acceptance",
    "underpower_handling",
    "replication_rate",
    "false_discovery_rate",
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
]
