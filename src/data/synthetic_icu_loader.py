"""
Synthetic ICU Telemetry (simulated) -- irregular longitudinal survival benchmark.

NAMING. This cohort was previously labelled "MIMIC-IV Sepsis-3" in Table 1/2, the
README and preregistration sections 1/3/5/6 including Kill Criterion 3. **There is
no MIMIC-IV data in this repository and none was ever used.** It is a generator. The
honest name is used everywhere now; see deviation log X-01.

What was wrong with the old generator
-------------------------------------
1. `tte = times[-1] + U(0.1, 2.0)` for events. The event time was therefore an
   almost deterministic function of the last observation time, giving
   corr(n_obs, tte | event) = 0.778. Since `permute_patient_durations` preserves
   sum(dt), the NC-B negative control could not fail.
2. `tte = max_stay_hours` for EVERY censored subject -- one distinct censored time
   in the whole cohort. With the evaluation horizon set to the median tte, the AUC
   control set was empty, so the metric returned a hardcoded 0.5. That, not any
   model behaviour, is what produced an entire AUC column of 0.500.
3. Risk was binary (`is_deteriorating`), so C-index measured two-group separation
   rather than ranking.
4. Features were z-scored PER SUBJECT, which forced every subject's feature mean to
   0 and erased the between-subject level offsets that generate the label.

What this generator does instead
--------------------------------
    z_i     ~ N(0, 1)                                  graded latent severity
    T_i     ~ Weibull(shape=k, scale=s0 * exp(-beta * z_i / k))    event time
    C_i     ~ Uniform(c_min, max_stay)                 loss to follow-up
    tte_i   = min(T_i, C_i);  event_i = 1{T_i <= C_i}
    visits  ~ homogeneous Poisson process, rate independent of z_i and T_i

The observation process is drawn independently of both the latent severity and the
event time, so the schedule carries no information the covariates do not. Laboratory
channels are sampled at a fraction of the vitals rate, which produces a genuine,
informative missingness mask -- the first in this project, since every loader
previously passed `mask=None`.

Note what this does NOT fix: corr(n_obs, tte | event) stays mildly positive because
subjects who die sooner are observed fewer times, and that is intrinsic to
longitudinal survival data (it measures 0.843 in the real PBC2 data). Landmarking is
the fix for that, not generator surgery.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import LongitudinalSurvivalDataset
from src.data.preprocessing import (
    apply_feature_stats,
    empirical_feature_mean,
    fit_feature_stats,
    subject_level_split,
)

FEATURE_NAMES = [
    "HR", "MAP", "Temp", "RespRate", "SpO2",      # vitals, observed every visit
    "Lactate", "Creatinine", "Platelets", "WBC", "SOFA",   # labs, observed sparsely
]
N_VITALS = 5


def _simulate_subject(rng, pid, max_stay_hours, visit_rate, lab_every,
                      weibull_shape, weibull_scale0, beta, censor_min):
    """One subject. Event time and observation schedule are drawn independently."""
    z = rng.normal(0.0, 1.0)                       # graded latent severity

    # Event time from a Weibull proportional-hazards model in z. Independent of the
    # visit process by construction.
    scale = weibull_scale0 * np.exp(-beta * z / weibull_shape)
    t_event = float(scale * rng.weibull(weibull_shape))

    # Loss to follow-up, independent of z and of t_event. Spread over the stay so the
    # censoring distribution is non-degenerate and IPCW is estimable.
    t_censor = float(rng.uniform(censor_min, max_stay_hours))

    tte = min(t_event, t_censor)
    event = 1.0 if t_event <= t_censor else 0.0

    # Homogeneous Poisson observation times on (0, tte), rate independent of severity.
    times = []
    t = float(rng.exponential(1.0 / visit_rate))
    while t < tte:
        times.append(t)
        t += float(rng.exponential(1.0 / visit_rate))
    if len(times) < 2:                              # every subject needs a history
        times = list(np.linspace(0.15 * tte, 0.85 * tte, 2))
    times = np.asarray(times, dtype=np.float32)

    n = len(times)
    frac = times / max(max_stay_hours, 1e-6)

    # Physiology: level offsets scale with z (graded, not a +25/-20 step), plus a
    # trend that also scales with z.
    def chan(base, level_coef, trend_coef, noise):
        return (base + level_coef * z + trend_coef * z * frac
                + rng.normal(0.0, noise, size=n))

    feats = np.stack([
        chan(85.0, 12.0, 18.0, 5.0),        # HR
        chan(75.0, -9.0, -14.0, 4.0),       # MAP
        chan(37.0, 0.5, 0.9, 0.3),          # Temp
        chan(18.0, 3.0, 6.0, 2.0),          # RespRate
        chan(98.0, -2.0, -5.0, 1.0),        # SpO2
        chan(1.2, 1.1, 2.4, 0.4),           # Lactate
        chan(1.0, 0.6, 1.4, 0.2),           # Creatinine
        chan(220.0, -30.0, -60.0, 15.0),    # Platelets
        chan(10.0, 3.5, 8.0, 2.0),          # WBC
        chan(2.0, 1.8, 4.0, 0.8),           # SOFA
    ], axis=1).astype(np.float32)

    # Missingness: vitals at every visit, labs only every `lab_every`-th visit.
    # This is informative but not outcome-dependent, and it is what finally
    # exercises GRU-D's decayed-imputation branch.
    mask = np.ones((n, len(FEATURE_NAMES)), dtype=np.float32)
    lab_seen = np.zeros(n, dtype=bool)
    lab_seen[::max(1, int(lab_every))] = True
    mask[~lab_seen, N_VITALS:] = 0.0

    dts = np.diff(times, prepend=np.float32(0.0)).astype(np.float32)
    dts[0] = max(float(times[0]), 1e-3)

    events = np.zeros(n, dtype=np.float32)          # derived from residual times downstream

    return {
        "id": int(pid),
        "features": torch.tensor(feats, dtype=torch.float32),
        "dts": torch.tensor(dts, dtype=torch.float32),
        "times": torch.tensor(times, dtype=torch.float32),
        "events": torch.tensor(events, dtype=torch.float32),
        "mask": torch.tensor(mask, dtype=torch.float32),
        "tte": float(tte),
        "event": float(event),
        "latent_severity": float(z),                # for diagnostics only, never a feature
    }


def load_synthetic_icu(
    n_patients: int = 500,
    max_stay_hours: float = 72.0,
    visit_rate: float = 0.30,          # ~1 observation per 3.3 h
    lab_every: int = 4,                # labs at every 4th visit
    weibull_shape: float = 1.4,
    weibull_scale0: float = 95.0,      # tuned for ~35-45% events within 72 h
    beta: float = 1.1,                 # log-hazard ratio per unit latent severity
    censor_min: float = 6.0,
    seed: int = 42,
    fracs: tuple = (0.6, 0.2, 0.2),
):
    """
    Returns (train, val, test, input_dim, max_horizon, x_mean).

    Standardization statistics are fit on the TRAIN split only and applied to all
    three, at cohort level rather than per subject.
    """
    rng = np.random.default_rng(seed)
    patients = [
        _simulate_subject(rng, pid, max_stay_hours, visit_rate, lab_every,
                          weibull_shape, weibull_scale0, beta, censor_min)
        for pid in range(n_patients)
    ]

    tr_idx, va_idx, te_idx = subject_level_split(len(patients), seed=seed, fracs=fracs)
    train = [patients[i] for i in tr_idx]
    val = [patients[i] for i in va_idx]
    test = [patients[i] for i in te_idx]

    stats = fit_feature_stats(train)
    train, val, test = (apply_feature_stats(s, stats) for s in (train, val, test))
    x_mean = empirical_feature_mean(train)

    return (
        LongitudinalSurvivalDataset(train),
        LongitudinalSurvivalDataset(val),
        LongitudinalSurvivalDataset(test),
        len(FEATURE_NAMES),
        float(max_stay_hours),
        x_mean,
    )
