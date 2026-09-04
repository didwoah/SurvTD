"""
Bridges between a faithful baseline port and this project's evaluation contract.

`src/evaluation/landmark.py:230` calls every model as

    _, survival, _, _ = model(x, dts, mask)
    surv = survival.squeeze(0)[-1]            # survival[k] == P(R > (k+1) * delta_s)

i.e. survival over **residual** time measured from each visit. The baselines do not
speak that language natively -- Dynamic-DeepHit emits a PMF over **absolute**
event-time bins, learned from the training times -- so the translation has to live
somewhere. It lives here rather than inside the ports, so a port stays a transcription
of its source and every protocol decision is visible in one file.

Two decisions, both stated because getting either wrong is silent:

1. **Conditional renormalisation.** A landmarked risk set contains only subjects alive
   at L, so the quantity being scored is `P(T > L + r | T > L)`, not `P(T > L + r)`.
   For a model on the absolute axis that is

       S_resid(r | L) = S_abs(L + r) / S_abs(L)

   Upstream's own `predict_survival` returns the *unconditional* `1 - CIF(t)`. Using
   it unchanged would score a different estimand than every other arm. The NASA Tier-1
   worker (`baselines/dynamic_deephit_pytorch/ddh_worker.py:117-125`) applies the same
   correction; this is the same protocol, reused rather than reinvented.

2. **Per-prefix prediction, not a copied last step.** Dynamic-DeepHit produces one
   prediction per sequence, from its last observed visit. Filling the other timesteps
   with a copy would be invisible to `evaluate_landmarked` (which reads only `[-1]`)
   and wrong for anything else that looked. Instead each prefix is materialised as its
   own row with the tail NaN-masked -- upstream's own padding convention, and exactly
   what its `predict_survival(..., all_step=True)` does -- and the batch is scored in
   one forward pass.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def absolute_survival_knots(cif: np.ndarray, split_time: np.ndarray):
    """Times and values at which a Dynamic-DeepHit CIF pins the survival curve.

    `discretize` builds `split_time` as `np.histogram(times, split - 1)` edges, so bin
    `j` covers `(split_time[j], split_time[j+1]]` and the mass accumulated through it
    is known at `split_time[j+1]`. Bin `split-1` is the open tail past the last edge.

    Args:
        cif: (n_bins,) cumulative sum of the head's PMF
        split_time: (n_edges,) bin edges from `discretize_absolute_times`
    Returns:
        (times, survival) knots, ascending in time, starting at `S = 1`.
    """
    n_edges = len(split_time)
    usable = min(n_edges - 1, len(cif))
    times = np.asarray(split_time[:usable + 1], dtype=float)
    surv = np.concatenate([[1.0], 1.0 - np.asarray(cif[:usable], dtype=float)])
    return times, np.clip(surv, 0.0, 1.0)


def _interp_survival(times: np.ndarray, surv: np.ndarray, query: np.ndarray):
    """Linear interpolation, held flat outside the knots.

    Flat-left at 1.0 (nothing can have happened before the first edge) and flat-right
    at the last known value, which is the truncation mass -- the same convention as
    `src/evaluation/landmark.py::interp_survival`.
    """
    return np.interp(query, times, surv, left=1.0, right=float(surv[-1]))


class ResidualSurvivalAdapter(nn.Module):
    """Wraps a port so it satisfies the `(hazard, survival, pmf, cdf)` contract.

    Args:
        model: the faithful port; must accept a NaN-padded `(B, L, D)` batch and return
            `(longitudinal_prediction, [pmf_per_risk])` on the absolute time axis
        split_time: absolute-time bin edges the model was discretised with
        delta_s, num_bins: this cohort's residual-time grid
        risk: which cause-specific head to read (1-based, as upstream indexes risks)
        eps: floor on the denominator `S_abs(L)`. Watch it: once it binds, that
            subject's curve is set by this constant rather than by the model, and the
            subjects it binds on are the highest-risk ones. `clamp_rate` reports how
            often that happened on the last forward pass.
    """

    def __init__(self, model, split_time, delta_s: float, num_bins: int,
                 risk: int = 1, eps: float = 1e-4):
        super().__init__()
        self.model = model
        self.register_buffer("split_time",
                             torch.as_tensor(np.asarray(split_time), dtype=torch.float64))
        self.delta_s = float(delta_s)
        self.K = int(num_bins)
        self.risk = int(risk)
        self.eps = float(eps)
        self.clamp_rate = 0.0

    @staticmethod
    def _prefix_batch(x_seq: torch.Tensor) -> torch.Tensor:
        """(L, D) -> (L, L, D), row i holding the prefix up to visit i, tail NaN.

        NaN is not padding invented here: it is how Dynamic-DeepHit marks unobserved
        steps, and its `inputmask` recovers the last observed index from it.
        """
        L = x_seq.shape[0]
        batch = x_seq.unsqueeze(0).repeat(L, 1, 1).clone()
        idx = torch.arange(L, device=x_seq.device)
        batch[idx.unsqueeze(1) < idx.unsqueeze(0)] = float("nan")
        return batch

    def forward(self, x, dts, mask=None):
        """`mask` is accepted and ignored -- Dynamic-DeepHit has no missingness channel.

        Returns the four-tuple on the residual-time grid; only `survival` is read by
        `evaluate_landmarked`, but all four are derived consistently so anything else
        that looks sees the same distribution.
        """
        squeeze_batch = (x.dim() == 3 and x.shape[0] == 1)
        if x.dim() == 2:
            x, dts = x.unsqueeze(0), dts.unsqueeze(0)
            squeeze_batch = False

        B, L, _ = x.shape
        split_time = self.split_time.cpu().numpy()
        grid = (np.arange(self.K) + 1.0) * self.delta_s
        out = np.empty((B, L, self.K), dtype=np.float64)
        clamped = 0

        for b in range(B):
            landmarks = torch.cumsum(dts[b], dim=0).detach().cpu().numpy()
            batch = self._prefix_batch(x[b])
            _, outcomes = self.model(batch)
            pmf = outcomes[self.risk - 1].detach().cpu().numpy()   # (L, n_bins)
            cif = np.cumsum(pmf, axis=-1)

            for i in range(L):
                times, surv = absolute_survival_knots(cif[i], split_time)
                s_at_L = float(_interp_survival(times, surv, np.array([landmarks[i]]))[0])
                if s_at_L < self.eps:
                    clamped += 1
                denom = max(s_at_L, self.eps)
                s_ahead = _interp_survival(times, surv, landmarks[i] + grid)
                row = np.clip(s_ahead / denom, 0.0, 1.0)
                out[b, i] = np.minimum.accumulate(row)   # survival cannot increase

        self.clamp_rate = clamped / float(B * L)

        survival = torch.as_tensor(out, dtype=x.dtype, device=x.device)
        ones = torch.ones_like(survival[..., :1])
        s_prev = torch.cat([ones, survival[..., :-1]], dim=-1)
        bin_mass = s_prev - survival
        hazard = bin_mass / s_prev.clamp_min(1e-12)
        pmf_out = torch.cat([bin_mass, survival[..., -1:]], dim=-1)   # overflow symbol
        cdf = torch.cumsum(bin_mass, dim=-1)

        if squeeze_batch:
            pass  # caller already passed (1, L, D); keep the batch dimension
        return hazard, survival, pmf_out, cdf
