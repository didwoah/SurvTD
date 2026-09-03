"""
Landmark evaluation protocol for dynamic survival models.

Why this replaces the previous evaluation
-----------------------------------------
The shipped runners scored every model by `cdf[-1, K//2]`: the CDF at the subject's
LAST VISIT, read at an arbitrary bin. Three things are wrong with that at once.

1. **Leakage.** The last visit's timestamp correlates with the outcome. In C-MAPSS
   the final observation *is* the failure cycle; in the synthetic ICU cohort the old
   generator set `tte = times[-1] + noise`. A model that merely encodes trajectory
   length scores well.
2. **Clock mismatch.** The head predicts RESIDUAL time from the visit it is called
   at, so `cdf[-1]` lives on a per-subject clock, while `tte` is absolute. The two
   were compared directly.
3. **The dynamic claim was never measured.** Nothing in that procedure asks what the
   model predicts *at a given moment in a subject's course*, which is the entire
   premise of dynamic survival analysis.

The protocol
------------
At a landmark time L, for each subject still at risk:
  1. keep only subjects with `tte > L` and at least one observation at or before L;
  2. feed the encoder only `x[:j_L+1]` where `j_L` is the last visit at or before L,
     so the model never sees data after the landmark;
  3. convert the head's residual-time distribution to a landmark-conditional one,
     `S_i(L+u | L) = S_hat_{j_L}(gap + u) / S_hat_{j_L}(gap)` with
     `gap = L - times[j_L]`, by LINEAR INTERPOLATION on the bin grid rather than
     floor-indexing (the old code floor-indexed, which at PBC's scale threw away
     most of the resolution);
  4. score over a prediction window Delta on the shared clock `u`.

Every at-risk subject now shares one clock, and the only history the model sees is
what was available at L. This is also the load-bearing fix for the NC-B duration
permutation control: at a fixed landmark the model observes `n_obs(<= L)`, so the
intrinsic correlation between observation count and outcome -- 0.843 in the real
PBC2 data, and not something a generator fix can remove -- no longer decides the
comparison.

No silent fallbacks
-------------------
The old metrics returned a hardcoded 0.5 when the case or control set was empty,
which is how an entire AUC column of 0.500 was reported as a result. Here a
degenerate landmark raises `DegenerateLandmarkError` with the counts in the message.
A metric that cannot be computed must never print as a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch


class DegenerateLandmarkError(ValueError):
    """Raised when a landmark has too few at-risk subjects or events to score."""


@dataclass(frozen=True)
class LandmarkSpec:
    landmarks: tuple
    horizons: tuple
    brier_grid_n: int = 20
    min_at_risk: int = 20
    min_events: int = 5
    # Administrative censoring cap, as an offset from the landmark. Used for C-MAPSS,
    # where every unit runs to failure and there is otherwise no censoring at all.
    #
    # It MUST be strictly greater than max(horizons). Capping at the evaluation
    # horizon itself leaves the AUC control set (`residual > Delta`) empty by
    # construction, since every survivor has been moved to exactly Delta. That is
    # what an earlier version of this spec did, and it made all three C-MAPSS
    # landmarks degenerate.
    admin_censor_at: float = None

    def __post_init__(self):
        if self.admin_censor_at is not None and self.admin_censor_at <= self.max_horizon():
            raise ValueError(
                f"admin_censor_at={self.admin_censor_at} must exceed "
                f"max(horizons)={self.max_horizon()}, or the AUC control set is empty"
            )

    def max_horizon(self) -> float:
        return float(max(self.horizons))

    @property
    def administrative_censor(self) -> bool:
        return self.admin_censor_at is not None

    def censor_cap(self) -> float:
        return float(self.admin_censor_at) if self.admin_censor_at else float("inf")


@dataclass
class LandmarkPredictions:
    landmark: float
    patient_ids: np.ndarray            # (n,)
    residual_time: np.ndarray          # (n,)  tte - L, strictly > 0
    event: np.ndarray                  # (n,)  0/1
    surv: np.ndarray                   # (n, G) S_i(L + u | L)
    grid: np.ndarray                   # (G,)  u ascending, > 0
    risk_at: dict = field(default_factory=dict)   # Delta -> 1 - S_i(L+Delta|L)
    n_dropped_not_at_risk: int = 0

    @property
    def n(self) -> int:
        return len(self.patient_ids)

    @property
    def n_events(self) -> int:
        return int(self.event.sum())


def truncate_history(patient: dict, landmark: float):
    """
    History available at the landmark, or None if the subject is not at risk.

    At risk means `tte > L` and at least one observation at or before L. Returns a
    shallow copy with the sequence fields truncated and a `gap` entry recording
    `L - times[j_L]`.
    """
    times = patient["times"].detach().cpu().numpy()
    if float(patient["tte"]) <= landmark:
        return None

    idx = np.nonzero(times <= landmark)[0]
    if idx.size == 0:
        return None
    j_L = int(idx[-1])

    out = dict(patient)
    for key in ("features", "dts", "times", "events", "mask"):
        if patient.get(key) is not None:
            out[key] = patient[key][: j_L + 1]
    out["gap"] = float(landmark - times[j_L])
    out["j_L"] = j_L
    return out


def interp_survival(surv_bins: np.ndarray, delta_s: float, u: np.ndarray) -> np.ndarray:
    """
    Evaluate the head's survival curve at arbitrary residual times by interpolation.

    The head emits `surv_bins[k] = P(R > (k+1) * delta_s)`, so index k already refers
    to the RIGHT edge of bin k. The knots are therefore
    `[0, delta_s, 2*delta_s, ...]` with values `[1, surv_bins[0], surv_bins[1], ...]`.

    Beyond the last knot the curve is held at its terminal value, which equals the
    overflow mass p_perp -- i.e. the model's stated probability of surviving past the
    horizon. Extrapolating to 0 instead would silently assert that everyone fails by
    K * delta_s.
    """
    K = surv_bins.shape[-1]
    knots = np.concatenate([[0.0], (np.arange(K) + 1.0) * delta_s])
    values = np.concatenate([[1.0], np.asarray(surv_bins, dtype=float)])
    return np.interp(np.asarray(u, dtype=float), knots, values,
                     left=1.0, right=float(values[-1]))


def conditional_survival(
    surv_bins: np.ndarray, delta_s: float, gap: float, u: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """
    S(L + u | L) = S_hat(gap + u) / S_hat(gap), on the landmark-relative clock.

    `gap` accounts for the landmark falling between visits: the head was called at
    `times[j_L]`, so its curve must be conditioned forward by `gap` before use.
    """
    denom = max(float(interp_survival(surv_bins, delta_s, np.array([gap]))[0]), eps)
    num = interp_survival(surv_bins, delta_s, np.asarray(u, dtype=float) + float(gap))
    return np.clip(num / denom, 0.0, 1.0)


def landmark_labels(dataset, landmark: float, spec: LandmarkSpec = None) -> np.ndarray:
    """
    Structured labels for the at-risk set, on the landmark-relative clock. Requires
    no forward pass, so it can supply `survival_train` to scikit-survival metrics.

    Under `administrative_censor` subjects surviving past `L + max(horizons)` are
    censored there, which is how genuine censoring is created for C-MAPSS out of real
    data with nothing synthetic added.
    """
    times, events = [], []
    for p in dataset:
        if truncate_history(p, landmark) is None:
            continue
        r = float(p["tte"]) - landmark
        e = bool(float(p["event"]) > 0.5)
        if spec is not None and spec.administrative_censor:
            cap = spec.censor_cap()
            if r > cap:
                r, e = cap, False
        times.append(r)
        events.append(e)

    return np.array(
        list(zip(events, times)), dtype=[("event", "?"), ("time", "<f8")]
    )


@torch.no_grad()
def predict_landmark(
    model,
    dataset,
    landmark: float,
    spec: LandmarkSpec,
    delta_s: float,
    device,
) -> LandmarkPredictions:
    """Landmark-conditional survival curves for every at-risk subject."""
    model.eval()
    grid_top = min(spec.censor_cap(), spec.max_horizon())
    grid = np.linspace(grid_top / spec.brier_grid_n, grid_top, spec.brier_grid_n)

    ids, residual, events, curves = [], [], [], []
    dropped = 0

    for p in dataset:
        trunc = truncate_history(p, landmark)
        if trunc is None:
            dropped += 1
            continue

        x = trunc["features"].unsqueeze(0).to(device)
        dts = trunc["dts"].unsqueeze(0).to(device)
        mask = trunc["mask"].unsqueeze(0).to(device) if trunc.get("mask") is not None else None

        _, survival, _, _ = model(x, dts, mask)
        surv_bins = survival.squeeze(0)[-1].cpu().numpy()      # last visit AT OR BEFORE L

        r = float(p["tte"]) - landmark
        e = bool(float(p["event"]) > 0.5)
        if spec.administrative_censor and r > spec.censor_cap():
            r, e = spec.censor_cap(), False

        ids.append(p.get("id", len(ids)))
        residual.append(r)
        events.append(1.0 if e else 0.0)
        curves.append(conditional_survival(surv_bins, delta_s, trunc["gap"], grid))

    if len(ids) < spec.min_at_risk:
        raise DegenerateLandmarkError(
            f"landmark {landmark}: only {len(ids)} at-risk subjects "
            f"(min_at_risk={spec.min_at_risk}); {dropped} were not at risk"
        )
    n_ev = int(np.sum(events))
    if n_ev < spec.min_events:
        raise DegenerateLandmarkError(
            f"landmark {landmark}: only {n_ev} events among {len(ids)} at-risk "
            f"subjects (min_events={spec.min_events})"
        )

    surv = np.asarray(curves, dtype=float)
    risk_at = {}
    for delta in spec.horizons:
        s_at = np.array([
            float(np.interp(delta, grid, surv[i])) for i in range(surv.shape[0])
        ])
        risk_at[float(delta)] = 1.0 - s_at

    return LandmarkPredictions(
        landmark=float(landmark),
        patient_ids=np.asarray(ids),
        residual_time=np.asarray(residual, dtype=float),
        event=np.asarray(events, dtype=float),
        surv=surv,
        grid=grid,
        risk_at=risk_at,
        n_dropped_not_at_risk=dropped,
    )


def km_marginal_reference(
    train_dataset, landmark: float, spec: LandmarkSpec
) -> LandmarkPredictions:
    """
    A covariate-free Kaplan-Meier predictor, fit on the training split's at-risk set
    and applied identically to everyone.

    Included as a row in every results table because it costs no training and its
    expected behaviour is known: C-index ~ 0.50 and IBS in roughly [0.15, 0.25]. Any
    IBS above ~0.25 anywhere in a table is then immediately visible as a bug rather
    than a finding -- which is exactly the check that would have caught the reported
    0.78-0.86.
    """
    labels = landmark_labels(train_dataset, landmark, spec)
    if labels.size == 0:
        raise DegenerateLandmarkError(f"landmark {landmark}: empty training at-risk set")

    t = labels["time"]
    e = labels["event"].astype(float)
    grid_top = min(spec.censor_cap(), spec.max_horizon())
    grid = np.linspace(grid_top / spec.brier_grid_n, grid_top, spec.brier_grid_n)

    uniq = np.unique(t)
    surv, running = np.empty(uniq.size), 1.0
    for i, ut in enumerate(uniq):
        at_risk = int(np.sum(t >= ut))
        d = float(np.sum(e[t == ut]))
        if at_risk > 0 and d > 0:
            running *= 1.0 - d / at_risk
        surv[i] = running
    km = np.interp(grid, uniq, surv, left=1.0, right=float(surv[-1]))

    return LandmarkPredictions(
        landmark=float(landmark),
        patient_ids=np.arange(len(t)),
        residual_time=t,
        event=e,
        surv=np.tile(km, (len(t), 1)),
        grid=grid,
        risk_at={float(d): np.full(len(t), 1.0 - float(np.interp(d, grid, km)))
                 for d in spec.horizons},
        n_dropped_not_at_risk=0,
    )


def evaluate_landmarked(
    model,
    train_dataset,
    test_dataset,
    spec: LandmarkSpec,
    delta_s: float,
    device,
    strict: bool = False,
) -> dict:
    """
    Run the full landmark protocol and return one metric dict per (landmark, horizon).

    `survival_train` for the IPCW estimators is the TRAINING split's at-risk set at
    the SAME landmark, on the SAME landmark-relative clock. Anything else -- the
    whole training cohort on an absolute clock, say -- estimates the wrong censoring
    distribution.

    Degenerate landmarks are recorded as `nan` with a reason string rather than
    skipped or filled with 0.5. With `strict=True` they raise instead, which is what
    the sanity gate uses.
    """
    from src.evaluation.metrics import (
        concordance_antolini,
        concordance_ipcw,
        cumulative_dynamic_auc_at,
        integrated_brier,
        make_structured,
    )

    results = {}
    for landmark in spec.landmarks:
        try:
            train_labels = landmark_labels(train_dataset, landmark, spec)
            if train_labels.size == 0 or int(train_labels["event"].sum()) < 2:
                raise DegenerateLandmarkError(
                    f"landmark {landmark}: training at-risk set has "
                    f"{int(train_labels['event'].sum()) if train_labels.size else 0} events"
                )
            preds = predict_landmark(model, test_dataset, landmark, spec, delta_s, device)
            test_surv = make_structured(preds.residual_time, preds.event)

            c_td = concordance_antolini(preds.surv, preds.grid,
                                        preds.residual_time, preds.event)
            aucs = cumulative_dynamic_auc_at(train_labels, test_surv,
                                             preds.risk_at, spec.horizons)
            ibs = integrated_brier(train_labels, test_surv, preds.surv, preds.grid)

            for delta in spec.horizons:
                d = float(delta)
                results[(float(landmark), d)] = {
                    "c_td": c_td,
                    "c_ipcw": concordance_ipcw(train_labels, test_surv,
                                               preds.risk_at[d], tau=d),
                    "auc": aucs[d],
                    "ibs": ibs,
                    "n_at_risk": preds.n,
                    "n_events": preds.n_events,
                    "n_dropped": preds.n_dropped_not_at_risk,
                }
        except (DegenerateLandmarkError, AssertionError) as exc:
            if strict:
                raise
            for delta in spec.horizons:
                results[(float(landmark), float(delta))] = {
                    "c_td": float("nan"), "c_ipcw": float("nan"),
                    "auc": float("nan"), "ibs": float("nan"),
                    "n_at_risk": 0, "n_events": 0, "n_dropped": 0,
                    "reason": f"{type(exc).__name__}: {exc}",
                }

    return results
