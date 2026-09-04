"""
Cohort registry: the single declared configuration for every benchmark.

Replaces the `if b_name == ...` chain in `experiments/run_track_a.py`, where the
grid (`delta_s`, `num_bins`) and the cohort size were inlined per call site and
could drift between Track A and Track B.

How the landmark grids were chosen
----------------------------------
The landmarks and prediction windows below were selected from the LABEL
DISTRIBUTION ALONE, before any model was trained, by requiring that every
(cohort, split, landmark) has enough at-risk subjects, cases and controls for the
metrics to be computable. No model output was consulted. They are fixed here so
that they cannot be adjusted after seeing a result; see deviation log A-07.

Rejected candidates, and why -- recorded so the selection is auditable:

  synthetic_icu  L=48, Delta=24  ->  0 controls in every split (L + Delta = 72 is
                                     the maximum stay, so nobody survives the window)
  cmapss FD001                   ->  a 20-unit test split gives 1 case at L=100;
                                     switched to FD002 (260 units)
  cmapss  L=50,  Delta=50        ->  0 cases: the minimum time-to-failure is 128
  cmapss  L=200, Delta=50        ->  5 controls in test
  pbc     L=365, Delta=365       ->  4 controls in test
  tumor   L=2.5, Delta=2.0       ->  9 controls in test
  tumor   lethal_threshold=2.0   ->  event rate 0.85, so the control set is
                                     exhausted by the second landmark; raised to 2.6
                                     (event rate 0.63) as a pre-run design choice

Note on C-MAPSS FD002: it spans six operating conditions rather than FD001's one,
so it is the harder and more realistic setting as well as the larger one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from src.data.cmapss_loader import load_cmapss_downsampled
from src.data.dataset import LongitudinalSurvivalDataset
from src.data.framingham_loader import load_framingham
from src.data.pbc_loader import load_pbc
from src.data.synthetic_icu_loader import load_synthetic_icu
from src.data.tumor_loader import generate_tumor_growth_cohort
from src.evaluation.landmark import LandmarkSpec


@dataclass
class CohortData:
    train: LongitudinalSurvivalDataset
    val: LongitudinalSurvivalDataset
    test: LongitudinalSurvivalDataset
    input_dim: int
    max_horizon: float
    x_mean: torch.Tensor

    def as_legacy_tuple(self):
        """For the `battle_*.py` scripts, which expect (train, test, dim, max_h)."""
        return (self.train, self.test, self.input_dim, self.max_horizon)


@dataclass(frozen=True)
class CohortSpec:
    name: str
    display_name: str
    loader: Callable[..., tuple]
    delta_s: float
    num_bins: int
    landmark_spec: LandmarkSpec
    time_unit: str
    is_real_data: bool
    person_period_grid_step: float   # Rung 1's declared regular grid

    def load(self, seed: int, **kwargs) -> CohortData:
        train, val, test, dim, max_h, x_mean = self.loader(seed=seed, **kwargs)
        return CohortData(train, val, test, dim, max_h, x_mean)

    def horizon_span(self) -> float:
        """K * delta_s: the modelled lifetime horizon. Must exceed max(Delta)."""
        return self.num_bins * self.delta_s


COHORTS: dict = {
    "synthetic_icu": CohortSpec(
        name="synthetic_icu",
        # NOT MIMIC-IV. This is a generator; the old label was wrong (X-01).
        display_name="Synthetic ICU Telemetry (simulated)",
        loader=lambda seed, **kw: load_synthetic_icu(n_patients=kw.pop("n_patients", 500),
                                                     seed=seed, **kw),
        delta_s=2.0,
        num_bins=36,                     # 72 h horizon
        landmark_spec=LandmarkSpec(landmarks=(12.0, 24.0, 36.0), horizons=(24.0,)),
        time_unit="hours",
        is_real_data=False,
        person_period_grid_step=1.0,     # the preregistered "1-hour regular grid"
    ),
    "cmapss": CohortSpec(
        name="cmapss",
        display_name="NASA C-MAPSS FD002 (Poisson-subsampled)",
        loader=lambda seed, **kw: load_cmapss_downsampled(
            dataset_id=kw.pop("dataset_id", 2), seed=seed, **kw),
        delta_s=5.0,
        num_bins=30,                     # 150 cycle horizon
        # Cap at 2 * Delta: the cap creates the censoring this cohort otherwise
        # lacks, while leaving a control set at the Delta = 50 horizon.
        landmark_spec=LandmarkSpec(landmarks=(125.0, 150.0, 175.0), horizons=(50.0,),
                                   admin_censor_at=100.0),
        time_unit="cycles",
        is_real_data=True,
        person_period_grid_step=5.0,
    ),
    "pbc": CohortSpec(
        name="pbc",
        display_name="PBC2 Clinical Trial",
        loader=lambda seed, **kw: load_pbc(seed=seed, **kw),
        delta_s=30.0,                    # residual-time median is 36 days (X-05)
        num_bins=30,                     # 900 day horizon
        landmark_spec=LandmarkSpec(landmarks=(0.0, 180.0), horizons=(365.0,)),
        time_unit="days",
        is_real_data=True,
        person_period_grid_step=30.0,
    ),
    "framingham": CohortSpec(
        name="framingham",
        display_name="Framingham Heart Study",
        loader=lambda seed, **kw: load_framingham(seed=seed, **kw),
        # A year. Framingham's residual time is measured in years, not the 36 days
        # PBC2's delta_s = 30 was sized against: median (tte - last exam) is 4,376 days.
        delta_s=365.0,
        num_bins=20,                     # 7300 day horizon > the 3650 day window
        # Only TWO landmarks, and neither is 0. The exams sit at TIME 0 / ~2174 / ~4361,
        # so 2190 and 4380 are the only points at which a subject can have more than one
        # observation -- which is what makes this a DYNAMIC prediction rather than a
        # baseline-covariate one. It also keeps every arm out of the L = 0 regime where
        # CoxSig is structurally undefined (D16).
        landmark_spec=LandmarkSpec(landmarks=(2190.0, 4380.0), horizons=(3650.0,)),
        time_unit="days",
        is_real_data=True,
        person_period_grid_step=365.0,
    ),
    "tumor": CohortSpec(
        name="tumor",
        display_name="Gompertzian Tumour Growth (simulated)",
        loader=lambda seed, **kw: generate_tumor_growth_cohort(
            n_samples=kw.pop("n_samples", 400),
            lethal_threshold=kw.pop("lethal_threshold", 2.6),
            seed=seed, **kw),
        delta_s=0.5,
        num_bins=25,                     # 12.5 time-unit horizon
        landmark_spec=LandmarkSpec(landmarks=(1.5, 2.0), horizons=(2.0,)),
        time_unit="arbitrary",
        is_real_data=False,
        person_period_grid_step=0.5,
    ),
}


def get_cohort(name: str) -> CohortSpec:
    if name not in COHORTS:
        raise KeyError(f"unknown cohort {name!r}; expected one of {sorted(COHORTS)}")
    return COHORTS[name]


def _self_check() -> None:
    """Structural invariants that do not require loading any data."""
    for spec in COHORTS.values():
        if spec.horizon_span() <= spec.landmark_spec.max_horizon():
            raise AssertionError(
                f"{spec.name}: modelled horizon K*delta_s = {spec.horizon_span()} does "
                f"not exceed the prediction window {spec.landmark_spec.max_horizon()}"
            )
        if not spec.landmark_spec.landmarks:
            raise AssertionError(f"{spec.name}: no landmarks declared")


_self_check()
