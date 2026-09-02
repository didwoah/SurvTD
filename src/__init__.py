"""
SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation
"""

from .models.backbones import GRUD, ContinuousLSTM, build_backbone
from .models.hazard_head import DiscreteHazardHead
from .models.survtd import SurvTDModel
from .models.baselines.person_period import PersonPeriodModel
from .models.baselines.dynamic_deephit import DynamicDeepHitModel
from .models.baselines.deeptcsr_clamped import DeepTCSRClampedModel

from .operators.survtd_operator import (
    categorical_projection_shift,
    compute_interval_discount,
    localized_projected_dirac,
    compute_multistep_lambda_returns,
    squared_cramer_distance_loss,
)
from .operators.ablations import (
    compute_ablated_lambda_returns,
    clamped_division_target,
    permute_patient_durations,
)

__all__ = [
    "GRUD",
    "ContinuousLSTM",
    "build_backbone",
    "DiscreteHazardHead",
    "SurvTDModel",
    "PersonPeriodModel",
    "DynamicDeepHitModel",
    "DeepTCSRClampedModel",
    "categorical_projection_shift",
    "compute_interval_discount",
    "localized_projected_dirac",
    "compute_multistep_lambda_returns",
    "squared_cramer_distance_loss",
    "compute_ablated_lambda_returns",
    "clamped_division_target",
    "permute_patient_durations",
]
