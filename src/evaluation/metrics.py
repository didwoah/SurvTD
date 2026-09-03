"""
Dynamic survival metrics, on the landmark-relative clock.

This is a full rewrite. The previous module's three headline functions were each
wrong in a way that produced reported numbers:

  * `compute_concordance_td` was plain Harrell's C on a single static scalar, not
    Antolini's time-dependent C^td that its name and the preregistration claimed.
  * `compute_time_dependent_auc` returned a hardcoded 0.5 whenever the case or
    control set was empty. Since the old generator gave every censored subject
    `tte = 72.0` exactly and the horizon was the median tte, the control set WAS
    empty -- which is the sole reason an entire AUC column read 0.500.
  * `compute_integrated_brier_score` indexed an ABSOLUTE evaluation time into a
    survival curve defined on a last-visit-RELATIVE clock, and normalized the IPCW
    sum by a count rather than by the sum of weights. Hence 0.78-0.86 against a
    ~0.25 ceiling.

Design decisions
----------------
**scikit-survival for the IPCW estimators.** Uno's C, cumulative/dynamic AUC and the
IPCW integrated Brier score are delegated rather than hand-rolled. The three defects
above are precisely the mistakes hand-rolled IPCW makes, and hand-rolling again
after failing once is not defensible; sksurv is also the reference a reviewer will
check against.

**Antolini's C^td is hand-written**, because sksurv does not provide it and it is
what the preregistration actually declares.

**No silent fallbacks.** Every function raises on a degenerate input. A metric that
cannot be computed must never print as a number.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from sksurv.metrics import (
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from src.evaluation.landmark import DegenerateLandmarkError

# Reference point, NOT an assertion threshold. Predicting a constant 0.5 everywhere
# gives a Brier score of 0.25, so a competent model should land below this. A model
# that is merely bad can legitimately exceed it: an untrained network has per-bin
# hazard ~0.5, so its survival decays like 0.5^k and it effectively predicts
# immediate failure for everyone, which scores ~1.0 on survivors.
#
# An earlier version of this module asserted IBS <= 0.30 inside integrated_brier().
# That was wrong: it conflated "the computation is broken" with "the model is bad",
# and the null-model gate immediately tripped it on untrained models. The right
# yardstick for a Brier score is the covariate-free Kaplan-Meier reference from
# landmark.km_marginal_reference, which is why that row appears in every table.
IBS_REFERENCE_LEVEL = 0.25


def make_structured(time, event) -> np.ndarray:
    """scikit-survival's structured array, dtype([('event','?'), ('time','<f8')])."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event).astype(bool)
    if t.shape != e.shape:
        raise ValueError(f"time {t.shape} and event {e.shape} must match")
    return np.array(list(zip(e, t)), dtype=[("event", "?"), ("time", "<f8")])


def _admissible_tau(train_surv, test_surv, tau: float) -> float:
    """
    sksurv requires evaluation times to lie inside the training censoring support.
    Returns a tau that satisfies that, or raises if none does.
    """
    t_max = float(min(train_surv["time"].max(), test_surv["time"].max()))
    tau = float(min(tau, t_max - 1e-6))
    if tau <= 0:
        raise DegenerateLandmarkError(
            f"no admissible evaluation horizon: train support max "
            f"{train_surv['time'].max():.4f}, test max {test_surv['time'].max():.4f}"
        )
    return tau


def concordance_ipcw(train_surv, test_surv, risk: np.ndarray, tau: float = None) -> float:
    """Uno's IPCW-weighted concordance. `risk`: higher means shorter survival."""
    n_ev = int(test_surv["event"].sum())
    if n_ev < 2:
        raise DegenerateLandmarkError(f"IPCW concordance needs >= 2 events, got {n_ev}")
    if tau is not None:
        tau = _admissible_tau(train_surv, test_surv, tau)
    value = float(
        concordance_index_ipcw(train_surv, test_surv, np.asarray(risk, float), tau=tau)[0]
    )
    assert_metric_sane("c_ipcw", value)
    return value


def concordance_antolini(
    surv: np.ndarray, grid: np.ndarray, time: np.ndarray, event: np.ndarray
) -> float:
    """
    Antolini's time-dependent concordance C^td.

    A pair (i, j) is comparable when `t_i < t_j` and subject i had the event. It is
    concordant when `S_i(t_i) < S_j(t_i)` -- both curves are read at the EARLIER
    subject's own event time, not at a common horizon. That is exactly what
    distinguishes C^td from Harrell's C on a static score, which is what the previous
    implementation computed despite its name.

    Args:
        surv: (n, G) survival curves on the shared landmark-relative clock
        grid: (G,) ascending evaluation times
        time, event: (n,) observed times and indicators on the same clock
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)
    surv = np.asarray(surv, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = time.shape[0]
    if surv.shape[0] != n or surv.shape[1] != grid.shape[0]:
        raise ValueError(f"surv {surv.shape} incompatible with n={n}, G={grid.shape[0]}")

    ev_idx = np.nonzero(event)[0]
    if ev_idx.size == 0:
        raise DegenerateLandmarkError("Antolini C^td needs at least one event")

    concordant = permissible = tied = 0
    for i in ev_idx:
        t_i = time[i]
        later = np.nonzero(time > t_i)[0]
        if later.size == 0:
            continue
        s_i = float(np.interp(t_i, grid, surv[i], left=1.0, right=surv[i][-1]))
        s_others = np.array([
            float(np.interp(t_i, grid, surv[k], left=1.0, right=surv[k][-1]))
            for k in later
        ])
        permissible += int(later.size)
        concordant += int(np.sum(s_i < s_others))
        tied += int(np.sum(s_i == s_others))

    if permissible == 0:
        raise DegenerateLandmarkError(
            "Antolini C^td has no comparable pairs: no event precedes another "
            "subject's observed time"
        )
    value = float((concordant + 0.5 * tied) / permissible)
    assert_metric_sane("c_td", value)
    return value


def cumulative_dynamic_auc_at(
    train_surv,
    test_surv,
    risk_by_horizon: dict,
    horizons: Sequence[float],
) -> dict:
    """
    Uno's cumulative/dynamic AUC at each horizon, using that horizon's own risk score.

    Raises rather than returning 0.5 when a horizon has no cases or no controls --
    the failure mode that produced the 0.500 AUC column.
    """
    out = {}
    for delta in horizons:
        d = float(delta)
        cases = (test_surv["time"] <= d) & test_surv["event"]
        controls = test_surv["time"] > d
        if cases.sum() == 0 or controls.sum() == 0:
            raise DegenerateLandmarkError(
                f"cumulative/dynamic AUC at horizon {d}: {int(cases.sum())} cases and "
                f"{int(controls.sum())} controls -- cannot be computed. The previous "
                f"implementation returned a hardcoded 0.5 here, which is how an "
                f"entire AUC column was reported as 0.500."
            )
        d_adj = _admissible_tau(train_surv, test_surv, d)
        auc, _ = cumulative_dynamic_auc(
            train_surv, test_surv, np.asarray(risk_by_horizon[d], float), times=[d_adj]
        )
        value = float(auc[0])
        assert_metric_sane("auc", value)
        out[d] = value
    return out


def integrated_brier(
    train_surv, test_surv, surv: np.ndarray, grid: np.ndarray, tau: float = None
) -> float:
    """IPCW integrated Brier score over the grid, truncated to the train support."""
    grid = np.asarray(grid, dtype=float)
    tau = _admissible_tau(train_surv, test_surv, tau if tau else float(grid[-1]))

    # sksurv requires every evaluation time to lie strictly inside the TEST
    # follow-up range, not merely below tau: brier_score rejects a grid point below
    # test_time.min() as well as one at or above test_time.max(). Clipping only the
    # upper end leaves grid points below the earliest observed time, which is how
    # this first failed on C-MAPSS (grid started at 2.5, test_time.min() was 3.0).
    lo = float(test_surv["time"].min())
    keep = (grid > lo) & (grid <= tau)
    if int(keep.sum()) < 2:
        raise DegenerateLandmarkError(
            f"integrated Brier needs >= 2 grid points inside the test follow-up "
            f"range ({lo:.4f}, {tau:.4f}], got {int(keep.sum())} from a grid spanning "
            f"[{grid[0]:.4f}, {grid[-1]:.4f}]"
        )
    value = float(integrated_brier_score(
        train_surv, test_surv, np.asarray(surv, float)[:, keep], grid[keep]
    ))
    assert_metric_sane("ibs", value)
    return value


def assert_metric_sane(name: str, value: float) -> None:
    """
    Range guard. These are bug detectors, not statistical tests: a Brier score of
    0.86 or a concordance of 1.4 cannot arise from a correct computation.
    """
    if not np.isfinite(value):
        raise AssertionError(f"{name} is not finite: {value}")
    if name in ("c_index", "c_ipcw", "c_td", "auc") and not (0.0 <= value <= 1.0):
        raise AssertionError(f"{name} outside [0, 1]: {value}")
    if name == "ibs" and not (0.0 <= value <= 1.0):
        # A Brier score is a mean squared error on probabilities, so it is bounded by
        # 1 by construction. Outside that range the computation is broken -- which is
        # the class of defect that produced the withdrawn 0.78-0.86 (an absolute time
        # indexed into a relative-clock curve, plus count-normalized IPCW weights).
        raise AssertionError(f"ibs outside [0, 1]: {value}")
