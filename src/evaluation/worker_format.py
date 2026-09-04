"""
Cohort -> subprocess-worker bundle.

Every vendored baseline worker in `baselines/*/` reads the same `torch.save`'d dict:

    paths_train / paths_test        (N, T, D)  channel 0 is TIME, then features
    surv_labels_train / _test       (N, 2)     columns (time, event)
    sampling_times                  (T,)       the shared clock
    pred_times, eval_times                     landmarks and residual horizons

That format came from CoxSig's NASA loader, and writing one converter into it means
CoxSig, NCDE, TCSR, DeepTCSR and DDH all run on PBC2 / Framingham / C-MAPSS as their
authors' own code, with no porting.

The one real design choice: the shared clock
--------------------------------------------
The format needs every subject on one grid, so irregular visits have to be projected
onto it. CoxSig's own loader took the union of all observed timestamps and
`ffill().bfill()`, which on NASA left engines that failed early with >63% of their
sequence constant-filled -- the note in
`research/continuous-survival-td/notes/two_tier_benchmark_strategy_and_baseline_audit.md`
§2 calls that out as one reason their reported baselines looked weak.

This uses a **regular grid at the cohort's own `delta_s`**, forward-filled from each
subject's observations, which keeps the grid short (PBC2: ~22 points for a 666-day
horizon) and identical across arms. Two consequences to state plainly rather than bury:

* Forward-filling is a real modelling assumption -- carry-the-last-value. It is applied
  identically to every arm, so it cannot advantage one, but it *does* remove some of
  the irregular-Δt signal this project's own claim is about. That is why SurvTD is
  additionally evaluated on the raw irregular visits through
  `evaluate_landmarked`; the two protocols are reported separately, never mixed.
* A subject observed only after a grid point has nothing to carry forward there. Those
  cells are back-filled from the first observation, and `first_obs_time` is recorded so
  the count is auditable rather than invisible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class WorkerBundle:
    """What a worker needs, plus the provenance needed to interpret its output."""
    paths_train: np.ndarray
    surv_labels_train: np.ndarray
    paths_test: np.ndarray
    surv_labels_test: np.ndarray
    sampling_times: np.ndarray
    pred_times: np.ndarray
    eval_times: np.ndarray
    feature_names: list
    n_backfilled: int

    def save(self, path: str):
        torch.save({
            "paths_train": self.paths_train,
            "surv_labels_train": self.surv_labels_train,
            "paths_test": self.paths_test,
            "surv_labels_test": self.surv_labels_test,
            "sampling_times": self.sampling_times,
            "pred_times": self.pred_times,
            "eval_times": self.eval_times,
        }, path)
        return path


def _project(patients, grid):
    """(N, T, 1 + D): time channel then forward-filled features."""
    n, T = len(patients), len(grid)
    D = patients[0]["features"].shape[1]
    paths = np.zeros((n, T, 1 + D), dtype=np.float32)
    labels = np.zeros((n, 2), dtype=np.float64)
    backfilled = 0

    for i, p in enumerate(patients):
        times = np.asarray(p["times"].detach().cpu(), dtype=float)
        feats = np.asarray(p["features"].detach().cpu(), dtype=float)
        # index of the last observation at or before each grid point; -1 before the first
        idx = np.searchsorted(times, grid, side="right") - 1
        backfilled += int(np.sum(idx < 0))
        idx = np.clip(idx, 0, len(times) - 1)          # back-fill from the first visit
        paths[i, :, 0] = grid
        paths[i, :, 1:] = feats[idx]
        labels[i] = (float(p["tte"]), float(p["event"]))

    return paths, labels, backfilled


def build_worker_bundle(cohort_data, spec, pred_times=None, eval_times=None,
                        feature_names=None) -> WorkerBundle:
    """Project a `CohortData` onto the shared clock the workers expect.

    Args:
        cohort_data: from `src.data.cohorts.COHORTS[name].load(seed)`
        spec: the cohort spec, for `delta_s`, `num_bins` and `landmark_spec`
        pred_times: landmarks; defaults to the preregistered `spec.landmark_spec.landmarks`
        eval_times: residual horizons; defaults to the cohort's own bin grid, so the
            worker curve resolution matches what in-process arms are scored on
    """
    l_spec = spec.landmark_spec
    delta_s = float(spec.delta_s)
    if pred_times is None:
        pred_times = np.asarray(l_spec.landmarks, dtype=float)
    if eval_times is None:
        eval_times = (np.arange(int(spec.num_bins)) + 1.0) * delta_s

    train = list(cohort_data.train)
    test = list(cohort_data.test)

    horizon = max(
        float(np.max(pred_times)) + float(np.max(eval_times)),
        max(float(p["tte"]) for p in train + test),
    )
    grid = np.arange(0.0, horizon + delta_s, delta_s, dtype=float)

    paths_tr, labels_tr, bf_tr = _project(train, grid)
    paths_te, labels_te, bf_te = _project(test, grid)

    return WorkerBundle(
        paths_train=paths_tr, surv_labels_train=labels_tr,
        paths_test=paths_te, surv_labels_test=labels_te,
        sampling_times=grid,
        pred_times=np.asarray(pred_times, dtype=float),
        eval_times=np.asarray(eval_times, dtype=float),
        feature_names=feature_names or [f"f{i}" for i in range(paths_tr.shape[2] - 1)],
        n_backfilled=bf_tr + bf_te,
    )
