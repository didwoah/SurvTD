"""
Evaluation layer: landmark protocol, IPCW metrics, censoring estimator, statistics.

The public surface changed with the repair. The old names promised things the code
did not do -- `compute_concordance_td` was Harrell's C on a static scalar, and both
`compute_time_dependent_auc` and `compute_integrated_brier_score` returned hardcoded
fallbacks on degenerate input. They are gone rather than renamed, so any call site
still expecting them fails loudly instead of silently producing a plausible number.
"""

from .censoring import KaplanMeierCensoring, fit_censoring_from_dataset
from .landmark import (
    DegenerateLandmarkError,
    LandmarkPredictions,
    LandmarkSpec,
    conditional_survival,
    evaluate_landmarked,
    interp_survival,
    km_marginal_reference,
    landmark_labels,
    predict_landmark,
    truncate_history,
)
from .metrics import (
    IBS_REFERENCE_LEVEL,
    assert_metric_sane,
    concordance_antolini,
    concordance_ipcw,
    cumulative_dynamic_auc_at,
    integrated_brier,
    make_structured,
)
from .stats import compute_bootstrap_ci, paired_wilcoxon_test

__all__ = [
    # censoring
    "KaplanMeierCensoring",
    "fit_censoring_from_dataset",
    # landmark protocol
    "DegenerateLandmarkError",
    "LandmarkPredictions",
    "LandmarkSpec",
    "conditional_survival",
    "evaluate_landmarked",
    "interp_survival",
    "km_marginal_reference",
    "landmark_labels",
    "predict_landmark",
    "truncate_history",
    # metrics
    "IBS_REFERENCE_LEVEL",
    "assert_metric_sane",
    "concordance_antolini",
    "concordance_ipcw",
    "cumulative_dynamic_auc_at",
    "integrated_brier",
    "make_structured",
    # statistics
    "compute_bootstrap_ci",
    "paired_wilcoxon_test",
]
