"""
One scorer for every arm, whether it is a model in this process or a worker's output.

`evaluate_landmarked` couples two jobs: getting a survival curve out of a model, and
scoring it. The baselines that run as subprocesses (CoxSig, NCDE, TCSR, DeepTCSR, DDH)
can only supply the first half, and each vendored repo ships its own metric code --
CoxSig's `score()`, `lifelines.concordance_index`, and so on. Scoring each arm with
whatever its own repo happened to use would make the columns of Table 1 incomparable.

So the scoring half lives here, takes a curve matrix, and is shared:

    model in-process  --> predict_landmark  --.
                                              >-- score_landmark_predictions
    worker subprocess --> curves JSON      --'

The consequence worth stating: every arm is judged by Antolini's C^td, Uno's
cumulative-dynamic AUC and the IPCW integrated Brier score computed by the same code,
with the same train-fit censoring distribution, and the KM leak gate applies to all of
them.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.landmark import (
    DegenerateLandmarkError,
    LandmarkPredictions,
    LandmarkSpec,
    landmark_labels,
)


def score_landmark_predictions(preds: LandmarkPredictions,
                               train_labels: np.ndarray,
                               spec: LandmarkSpec) -> dict:
    """Metrics for one landmark. The scoring half of `evaluate_landmarked`.

    `train_labels` must be the TRAINING split's at-risk set at the SAME landmark on the
    SAME landmark-relative clock -- the censoring distribution is a nuisance parameter
    and estimating it from anything else answers a different question.
    """
    from src.evaluation.metrics import (
        concordance_antolini,
        concordance_ipcw,
        cumulative_dynamic_auc_at,
        integrated_brier,
        make_structured,
    )

    test_surv = make_structured(preds.residual_time, preds.event)
    c_td = concordance_antolini(preds.surv, preds.grid, preds.residual_time, preds.event)
    aucs = cumulative_dynamic_auc_at(train_labels, test_surv, preds.risk_at, spec.horizons)
    ibs = integrated_brier(train_labels, test_surv, preds.surv, preds.grid)

    out = {}
    for delta in spec.horizons:
        d = float(delta)
        out[d] = {
            "c_td": c_td,
            "c_ipcw": concordance_ipcw(train_labels, test_surv, preds.risk_at[d], tau=d),
            "auc": aucs[d],
            "ibs": ibs,
            "n_at_risk": preds.n,
            "n_events": preds.n_events,
            "n_dropped": preds.n_dropped_not_at_risk,
        }
    return out


def predictions_from_curves(curves: np.ndarray,
                            surv_labels: np.ndarray,
                            landmark: float,
                            eval_times: np.ndarray,
                            spec: LandmarkSpec) -> LandmarkPredictions:
    """Turn a worker's curve matrix at one landmark into `LandmarkPredictions`.

    Args:
        curves: (N, n_eval) conditional survival for EVERY test subject, already
            conditioned on being alive at `landmark` by the upstream model
        surv_labels: (N, 2) columns `(time, event)` on the absolute clock
        eval_times: (n_eval,) residual horizons the curves were evaluated at

    The at-risk filter lives here, not in the worker: `tte > L` is a property of the
    data, so applying it in one place keeps every arm's risk set identical by
    construction rather than by four workers agreeing.
    """
    times = np.asarray(surv_labels[:, 0], dtype=float)
    events = np.asarray(surv_labels[:, 1], dtype=float) > 0.5

    at_risk = np.nonzero(times > landmark)[0]
    dropped = int(len(times) - len(at_risk))
    if at_risk.size == 0:
        raise DegenerateLandmarkError(f"landmark {landmark}: no subject at risk")

    residual = times[at_risk] - landmark
    event = events[at_risk].astype(float)

    if spec.administrative_censor:
        cap = spec.censor_cap()
        over = residual > cap
        residual = np.where(over, cap, residual)
        event = np.where(over, 0.0, event)

    if at_risk.size < spec.min_at_risk:
        raise DegenerateLandmarkError(
            f"landmark {landmark}: only {at_risk.size} at-risk subjects "
            f"(min_at_risk={spec.min_at_risk}); {dropped} were not at risk")
    n_ev = int(event.sum())
    if n_ev < spec.min_events:
        raise DegenerateLandmarkError(
            f"landmark {landmark}: only {n_ev} events among {at_risk.size} at-risk "
            f"subjects (min_events={spec.min_events})")

    grid = np.asarray(eval_times, dtype=float)
    surv = np.clip(np.asarray(curves, dtype=float)[at_risk], 0.0, 1.0)
    surv = np.minimum.accumulate(surv, axis=-1)     # curves must not increase

    risk_at = {
        float(delta): 1.0 - np.array([float(np.interp(delta, grid, surv[i]))
                                      for i in range(surv.shape[0])])
        for delta in spec.horizons
    }

    return LandmarkPredictions(
        landmark=float(landmark),
        patient_ids=at_risk,
        residual_time=residual,
        event=event,
        surv=surv,
        grid=grid,
        risk_at=risk_at,
        n_dropped_not_at_risk=dropped,
    )


def evaluate_curves(curves: np.ndarray,
                    surv_labels: np.ndarray,
                    pred_times,
                    eval_times,
                    train_dataset,
                    spec: LandmarkSpec,
                    strict: bool = False) -> dict:
    """Score a worker's output exactly as `evaluate_landmarked` scores a model.

    Args:
        curves: (N, n_pred, n_eval) conditional survival from the worker
        surv_labels: (N, 2) `(time, event)` for the test split
        pred_times: the landmarks the worker predicted at
        train_dataset: this project's training split, for the train-fit IPCW labels
    Returns:
        `{(landmark, horizon): metrics}` -- the same shape `evaluate_landmarked` returns.
    """
    curves = np.asarray(curves, dtype=float)
    results = {}
    for j, landmark in enumerate(np.asarray(pred_times, dtype=float)):
        try:
            train_labels = landmark_labels(train_dataset, float(landmark), spec)
            if train_labels.size == 0 or int(train_labels["event"].sum()) < 2:
                raise DegenerateLandmarkError(
                    f"landmark {landmark}: training at-risk set has "
                    f"{int(train_labels['event'].sum()) if train_labels.size else 0} events")
            preds = predictions_from_curves(curves[:, j, :], surv_labels,
                                            float(landmark), eval_times, spec)
            for delta, metrics in score_landmark_predictions(preds, train_labels, spec).items():
                results[(float(landmark), delta)] = metrics
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
