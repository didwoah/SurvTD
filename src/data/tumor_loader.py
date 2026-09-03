"""
Synthetic Gompertzian tumour-growth cohort with threshold-crossing survival.

Defects repaired here
---------------------
* **The latent ODE path was coupled to the observation schedule (X-06).** The old
  generator integrated `dy = r*y*log(K/y)*dt` using the VISIT GAPS as the Euler step
  and accumulated `N(0, 0.05*sqrt(dt))` noise per visit. A subject with 11 visits
  therefore had a *different latent trajectory* from one with 4 -- not a
  differently-sampled version of the same path. That is the schedule confound in its
  purest form, upstream of the event-time quantization. The path is now integrated on
  a fine grid shared by every subject, and visits merely SAMPLE it.
* **The event time was quantized to visit times.** It is now obtained by linear
  interpolation of the threshold crossing on the fine grid, so it is continuous and
  independent of when the subject happened to be observed.
* **There was no censoring at all (X-06).** Measured test-split event rate was
  1.00, so it was not a survival problem. An independent uniform loss-to-follow-up
  is now drawn, plus administrative censoring at `end_time`.
* Features were never standardized, although `dy` is heavy-tailed. Now standardized
  with train-split statistics.
* `mask` was None; it is now explicit.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src.data.dataset import LongitudinalSurvivalDataset
from src.data.preprocessing import (
    apply_feature_stats,
    empirical_feature_mean,
    fit_feature_stats,
    subject_level_split,
)

FEATURE_NAMES = ["y", "dy_dt", "sin_t", "cos_t"]


def _simulate_path(rng, end_time: float, fine_dt: float, k_cap: float):
    """
    Gompertzian growth on a fine grid shared by all subjects, so the latent path
    does not depend on the observation schedule.

    Returns (grid, y_path, r).
    """
    n = int(round(end_time / fine_dt)) + 1
    grid = np.arange(n, dtype=np.float64) * fine_dt

    r = rng.normal(0.5, 0.1)
    y = rng.uniform(0.3, 0.8)
    path = np.empty(n, dtype=np.float64)
    path[0] = y

    sqrt_dt = math.sqrt(fine_dt)
    for i in range(1, n):
        growth = r * y * max(0.01, math.log(max(1.01, k_cap / max(0.01, y)))) * fine_dt
        y = max(0.05, y + growth + rng.normal(0.0, 0.05 * sqrt_dt))
        path[i] = y

    return grid, path, float(r)


def _crossing_time(grid: np.ndarray, path: np.ndarray, threshold: float):
    """First time the path reaches `threshold`, by linear interpolation. None if never."""
    above = np.nonzero(path >= threshold)[0]
    if above.size == 0:
        return None
    i = int(above[0])
    if i == 0:
        return float(grid[0])
    y0, y1 = path[i - 1], path[i]
    if y1 == y0:
        return float(grid[i])
    frac = (threshold - y0) / (y1 - y0)
    return float(grid[i - 1] + frac * (grid[i] - grid[i - 1]))


def generate_tumor_growth_cohort(
    n_samples: int = 400,
    end_time: float = 10.0,
    lethal_threshold: float = 2.6,
    fine_dt: float = 0.01,
    visit_rate: float = 3.5,
    censor_min: float = 1.0,
    k_cap: float = 3.5,
    seed: int = 42,
    fracs: tuple = (0.6, 0.2, 0.2),
):
    """Returns (train, val, test, input_dim, max_horizon, x_mean)."""
    rng = np.random.default_rng(seed)
    patients = []

    for i in range(n_samples):
        grid, path, r = _simulate_path(rng, end_time, fine_dt, k_cap)

        t_cross = _crossing_time(grid, path, lethal_threshold)
        t_censor = float(rng.uniform(censor_min, end_time))   # independent of the path

        if t_cross is None:
            tte, event = float(end_time), 0.0                 # administrative
        elif t_cross <= t_censor:
            tte, event = float(t_cross), 1.0
        else:
            tte, event = float(t_censor), 0.0

        # Visits from a Poisson process on (0, tte), independent of the path.
        times = []
        t = float(rng.exponential(1.0 / visit_rate))
        while t < tte:
            times.append(t)
            t += float(rng.exponential(1.0 / visit_rate))
        if len(times) < 2:
            times = list(np.linspace(0.15 * tte, 0.85 * tte, 2))
        times = np.asarray(times, dtype=np.float64)

        # Sample the fine path at the visit times.
        idx = np.clip(np.searchsorted(grid, times, side="right") - 1, 0, len(grid) - 1)
        y_obs = path[idx]
        # Local derivative from the fine grid, not from the visit gaps.
        dy_obs = np.gradient(path, fine_dt)[idx]

        feats = np.stack([y_obs, dy_obs, np.sin(times), np.cos(times)], axis=1).astype(np.float32)
        dts = np.diff(times, prepend=0.0).astype(np.float32)
        dts[0] = max(float(times[0]), 1e-3)

        patients.append({
            "id": i,
            "features": torch.tensor(feats, dtype=torch.float32),
            "dts": torch.tensor(dts, dtype=torch.float32),
            "times": torch.tensor(times.astype(np.float32), dtype=torch.float32),
            "events": torch.zeros(len(times), dtype=torch.float32),
            "mask": torch.ones((len(times), len(FEATURE_NAMES)), dtype=torch.float32),
            "tte": tte,
            "event": event,
            "growth_rate": r,          # diagnostics only, never a feature
        })

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
        float(end_time),
        x_mean,
    )
