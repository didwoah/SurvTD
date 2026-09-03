"""
Kaplan-Meier estimator of the CENSORING distribution G(t) = P(C > t).

Deliberately hand-written rather than delegated to lifelines. Two things need to be
under explicit control and are easy to get wrong through a general-purpose wrapper:

1. **The left limit.** IPCW weights for a subject who had an event at time t use
   G(t-), not G(t). Using G(t) divides by a survival probability that has already
   been decremented by that subject's own event time, which biases the weights
   downward exactly where they matter most. The previous implementation called
   `kmf.predict(t_i)`.

2. **Fit on train, apply to test.** The censoring distribution is a nuisance
   parameter and must be estimated from the training split only, then applied to
   test subjects. Refitting per split leaks.

`lifelines` is intentionally NOT a project dependency: `src/evaluation/metrics.py`
imported it while it was not installed, which meant the shipped pipeline could not
be imported at all in this environment.
"""

from __future__ import annotations

import numpy as np


class KaplanMeierCensoring:
    """
    KM estimate of G(t) = P(C > t), i.e. the probability of remaining uncensored.

    Note the indicator flip: the "event" for this estimator is *being censored*, so
    a subject with `event == 1` (the outcome of interest occurred) is treated as
    censored here, and vice versa.
    """

    def __init__(self, floor: float = 0.05):
        # Weights are 1/G, so G must be floored. 0.05 caps any single subject's
        # weight at 20. The floor is deliberately much larger than the 0.01 the
        # previous code used, which allowed weights up to 100 -- a single subject
        # could then dominate an entire metric.
        self.floor = float(floor)
        self._times: np.ndarray | None = None
        self._surv: np.ndarray | None = None
        self._n_fit = 0

    def fit(self, times, events) -> "KaplanMeierCensoring":
        """
        Args:
            times: (n,) observed time to event or censoring
            events: (n,) 1 if the outcome of interest occurred, 0 if right-censored
        """
        t = np.asarray(times, dtype=float)
        e = np.asarray(events, dtype=float)
        if t.ndim != 1 or t.shape != e.shape:
            raise ValueError("times and events must be 1-D and the same length")
        if t.size == 0:
            raise ValueError("cannot fit the censoring distribution on an empty split")

        censored = 1.0 - (e > 0.5).astype(float)      # censoring is the "event" here

        order = np.argsort(t, kind="mergesort")
        t, censored = t[order], censored[order]

        uniq = np.unique(t)
        surv, running, n = np.empty(uniq.size), 1.0, t.size
        for i, ut in enumerate(uniq):
            at_risk = int(np.sum(t >= ut))
            d = float(np.sum(censored[t == ut]))
            if at_risk > 0 and d > 0:
                running *= 1.0 - d / at_risk
            surv[i] = running

        self._times, self._surv, self._n_fit = uniq, surv, t.size
        return self

    def predict(self, t, left_limit: bool = False) -> np.ndarray:
        """
        G(t), or G(t-) when `left_limit`. Scalar in, scalar out; array in, array out.

        Values are floored at `self.floor`, so the returned quantity is safe to
        invert. Use `clip_fraction` to report how often the floor bound.
        """
        if self._times is None:
            raise RuntimeError("call fit() before predict()")

        q = np.atleast_1d(np.asarray(t, dtype=float))
        # 'left'  -> strictly-less-than jumps only  -> G(t-)
        # 'right' -> includes the jump at t         -> G(t)
        idx = np.searchsorted(self._times, q, side="left" if left_limit else "right")
        out = np.where(idx > 0, self._surv[np.clip(idx - 1, 0, None)], 1.0)
        out = np.maximum(out, self.floor)
        return out if np.ndim(t) else float(out[0])

    def ipcw(self, t, left_limit: bool = True, max_weight: float = 10.0) -> np.ndarray:
        """
        Inverse-probability-of-censoring weights 1/G, clipped at `max_weight`.

        Preregistration section 3 declares `1 / G_hat <= 10.0` as part of the full
        method's definition. It was never actually applied: the weight argument was
        accepted by the loss and never passed a value (defect D11).
        """
        w = 1.0 / self.predict(t, left_limit=left_limit)
        return np.minimum(w, float(max_weight))

    def clip_fraction(self, t, left_limit: bool = True) -> float:
        """Fraction of queries whose G hit the floor. An unlogged clip is a silent bias."""
        g = np.atleast_1d(self.predict(t, left_limit=left_limit))
        return float(np.mean(g <= self.floor + 1e-12))

    @property
    def support_max(self) -> float:
        """Largest time the fit covers. Beyond it, G is an extrapolation."""
        if self._times is None:
            raise RuntimeError("call fit() before support_max")
        return float(self._times[-1])

    @property
    def n_fit(self) -> int:
        return self._n_fit


def fit_censoring_from_dataset(dataset, floor: float = 0.05) -> KaplanMeierCensoring:
    """Convenience: fit on a LongitudinalSurvivalDataset (or any iterable of dicts)."""
    times = np.array([float(p["tte"]) for p in dataset], dtype=float)
    events = np.array([float(p["event"]) for p in dataset], dtype=float)
    return KaplanMeierCensoring(floor=floor).fit(times, events)


# Backward-compatible alias
fit_censoring_estimator = fit_censoring_from_dataset

