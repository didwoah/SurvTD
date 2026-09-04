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
    time_scale: float = 1.0

    def save(self, path: str):
        """Times go out DIVIDED BY `time_scale`.

        CoxSig takes level-2 path signatures, so a time channel spanning 0-1080 days
        against features standardised to ~N(0,1) contributes terms of order 1e6 and
        `np.exp(feats.dot(coefs))` overflows (`coxprox.py:108`). The symptom is silent:
        the fit returns, every subject gets the same curve, and C^td lands on exactly
        0.5000. `get_nasa_splits` carries a `time_scale` for the same reason.

        Curve VALUES are unitless probabilities and their columns still correspond to
        `eval_times[k]`, so scoring uses the original clock; only the worker sees the
        scaled one.
        """
        # `paths` go out as torch Tensors: CoxSig indexes them against a
        # `torch.FloatTensor` buffer (`src/coxsig.py:126`) and raises on a numpy array,
        # and `get_nasa_splits` -- the format's origin -- also produces Tensors.
        torch.save({
            "paths_train": self._scaled_paths(self.paths_train),
            "surv_labels_train": self._scaled_labels(self.surv_labels_train),
            "paths_test": self._scaled_paths(self.paths_test),
            "surv_labels_test": self._scaled_labels(self.surv_labels_test),
            "sampling_times": self.sampling_times / self.time_scale,
            "pred_times": self.pred_times / self.time_scale,
            "eval_times": self.eval_times / self.time_scale,
            "time_scale": self.time_scale,
        }, path)
        return path

    def _scaled_paths(self, paths):
        out = np.array(paths, dtype=np.float32, copy=True)
        out[:, :, 0] /= self.time_scale
        return torch.as_tensor(out, dtype=torch.float32)

    def _scaled_labels(self, labels):
        out = np.array(labels, dtype=np.float64, copy=True)
        out[:, 0] /= self.time_scale
        return out


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
        eval_times: residual horizons; defaults to the SAME grid `predict_landmark`
            builds, so worker arms and in-process arms are scored on one grid
    """
    l_spec = spec.landmark_spec
    delta_s = float(spec.delta_s)
    if pred_times is None:
        pred_times = np.asarray(l_spec.landmarks, dtype=float)
    if eval_times is None:
        # `linspace(top / brier_grid_n, top, brier_grid_n)` with
        # `top = min(censor_cap, max_horizon)` -- character for character what
        # `landmark.predict_landmark` uses.
        #
        # This used to default to the cohort's own BIN grid,
        # `(arange(num_bins) + 1) * delta_s`, on the reasoning that it "matches what
        # in-process arms are scored on". That reasoning was wrong: in-process arms are
        # scored on the landmark grid, not the bin grid, and on PBC2 the two are 30..900
        # days versus 18..365. The integrated Brier score is an integral OVER the grid,
        # so the bin grid was integrating 535 days past the prediction window and past
        # most of the follow-up. Measured on one SurvTD fit, the same model scored
        # IBS 0.4358 on the landmark grid and 0.6166 on the bin grid at L = 0 -- a
        # difference bigger than any between-arm gap the table is meant to show.
        top = min(l_spec.censor_cap(), l_spec.max_horizon())
        eval_times = np.linspace(top / l_spec.brier_grid_n, top, l_spec.brier_grid_n)

    train = list(cohort_data.train)
    test = list(cohort_data.test)

    horizon = max(
        float(np.max(pred_times)) + float(np.max(eval_times)),
        max(float(p["tte"]) for p in train + test),
    )
    grid = np.arange(0.0, horizon + delta_s, delta_s, dtype=float)

    # CoxSig needs at least TWO grid points at or below every subject's event time.
    # `coxsig.py:109` builds its label vector as `[0] * (n_i - 1) + [1]` for an event,
    # and Python returns `[]` for `[0] * -1` instead of raising -- so a subject who dies
    # before the second grid point contributes a label with no matching feature row, and
    # the likelihood at `coxprox.py:109` fails with a shape mismatch. On PBC2 that is 10
    # subjects with tte < 30 days; it cannot fire on NASA, where the minimum
    # time-to-failure is 128 cycles against a 5-cycle grid, which is why it was never hit.
    #
    # Rather than patch vendored author code, insert one early grid point below the
    # smallest event time. Keeps the grid short (PBC2: 38 points instead of the 360 a
    # uniformly fine grid would need) and leaves every arm on the same clock.
    min_time = min(float(p["tte"]) for p in train + test)
    if min_time <= grid[1]:
        grid = np.unique(np.concatenate([[0.0, min_time / 2.0], grid]))

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
        time_scale=float(grid[-1]) if grid[-1] > 0 else 1.0,
    )
