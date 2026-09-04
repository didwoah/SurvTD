"""
Landmark Cox (van Houwelingen 2007) -- the classical dynamic-prediction baseline.

The method a clinician would actually reach for, and the one a new dynamic model has
to beat before any of the deep baselines matter. It is deliberately the simplest arm
in Table 1: no sequence encoder, no bootstrap, no shared representation across
landmarks.

At each landmark `L`, independently:

  1. take the subjects still at risk (`tte > L`, with at least one observation at or
     before `L` -- the same predicate `landmark.truncate_history` uses, so this arm's
     risk set is identical to every other arm's);
  2. use each subject's **most recent observation at or before L** as a fixed covariate
     vector -- this is what makes it *landmarking* rather than a joint model: the
     longitudinal process is not modelled, only its last value is carried;
  3. fit Cox proportional hazards on the **residual** clock `(tte - L, event)`.

A separate model per landmark is the method, not an implementation shortcut: it is
precisely what lets the covariate effects vary with `L` without modelling how.

Output is a `(N_test, n_landmarks, n_eval)` conditional survival matrix -- the same
contract the subprocess workers emit -- so this arm goes through
`curve_scoring.evaluate_curves` like all the others. Rows for subjects not at risk at
a landmark are filled with 1.0 and dropped by the scorer's own at-risk filter; nothing
downstream reads them.

Two declared choices
--------------------
* **Ridge penalty `alpha = 0.1`.** `CoxPHSurvivalAnalysis` is unpenalised at
  `alpha = 0` and PBC2's training risk set is ~180 subjects against 15 covariates,
  where separation makes Newton-Raphson diverge. A small ridge is applied at every
  landmark on every cohort, not only where it is needed, so the arm is one method
  rather than one method per cell.
* **Administrative censoring is applied to the training labels too** when the spec
  declares it (C-MAPSS). Fitting on uncapped residual times while being scored against
  capped ones would give this arm information no other arm has.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.landmark import truncate_history


def _last_observation(patient: dict, landmark: float) -> np.ndarray:
    """Covariates at the most recent visit at or before `landmark`."""
    hist = truncate_history(patient, landmark)
    if hist is None:
        return None
    return hist["features"].detach().cpu().numpy()[-1].astype(float)


def _design(dataset, landmark: float, spec):
    """(X, structured labels, index) for the at-risk set at one landmark."""
    xs, times, events, idx = [], [], [], []
    for i, p in enumerate(dataset):
        x = _last_observation(p, landmark)
        if x is None:
            continue
        r = float(p["tte"]) - landmark
        e = bool(float(p["event"]) > 0.5)
        if spec.administrative_censor:
            cap = spec.censor_cap()
            if r > cap:
                r, e = cap, False
        xs.append(x)
        times.append(r)
        events.append(e)
        idx.append(i)

    y = np.array(list(zip(events, times)), dtype=[("event", "?"), ("time", "<f8")])
    return np.asarray(xs, dtype=float), y, np.asarray(idx, dtype=int)


def _step_eval(x_knots: np.ndarray, y_knots: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Right-continuous step function, 1.0 before the first knot, flat after the last.

    `sksurv`'s `StepFunction.__call__` raises outside its domain, and the residual
    horizons routinely run past the last training event time. Evaluating the step
    directly keeps the same flat-left/flat-right convention as
    `landmark.interp_survival` and `_curve_export.curves_on_residual_grid`.
    """
    pos = np.searchsorted(x_knots, query, side="right") - 1
    out = np.where(pos < 0, 1.0, y_knots[np.clip(pos, 0, len(y_knots) - 1)])
    return out.astype(float)


def landmark_cox_curves(train_dataset, test_dataset, spec, eval_times,
                        alpha: float = 0.1) -> np.ndarray:
    """(N_test, n_landmarks, n_eval) conditional survival from per-landmark Cox fits.

    Args:
        spec: a `LandmarkSpec` -- supplies the landmarks and any administrative cap
        eval_times: residual horizons, shared with every other arm
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    test = list(test_dataset)
    eval_times = np.asarray(eval_times, dtype=float)
    landmarks = np.asarray(spec.landmarks, dtype=float)
    out = np.ones((len(test), len(landmarks), len(eval_times)), dtype=float)

    for j, L in enumerate(landmarks):
        x_tr, y_tr, _ = _design(train_dataset, float(L), spec)
        x_te, _, idx_te = _design(test, float(L), spec)
        if x_tr.size == 0 or x_te.size == 0 or int(y_tr["event"].sum()) < 2:
            continue                      # scorer records the degenerate landmark

        model = CoxPHSurvivalAnalysis(alpha=alpha)
        model.fit(x_tr, y_tr)

        for row, fn in zip(idx_te, model.predict_survival_function(x_te)):
            out[row, j] = _step_eval(np.asarray(fn.x), np.asarray(fn.y), eval_times)

    return np.clip(np.minimum.accumulate(out, axis=-1), 0.0, 1.0)
