"""
Ground-truth anchors for the SurvTD objective.

Why this module exists
---------------------
The shipped objective's only ground-truth signal was a single Dirac at the terminal
visit of an event trajectory. It propagated backwards attenuated by
lambda_j = lambda^(g_j/delta_s) ~ 0.489 on the sepsis cohort, i.e. ~0.001 after ten
visits -- and it was placed in the wrong bin (defect D9). Right-censored
trajectories, ~72% of that cohort, received a target copied from the model's own EMA
network and therefore carried no information at all (defect D-CENS). Everything else
was self-distillation against a target network, an objective with a degenerate fixed
point. That, not a weak mechanism, is why SurvTD lost to Dynamic-DeepHit, which
receives a residual-time likelihood at every visit.

The fix is an anchor term evaluated at EVERY visit, for censored and uncensored
trajectories alike:

    loss = (1 - alpha) * L_TD + alpha * L_anchor

Why censored CRPS rather than an IPCW negative log-likelihood
-------------------------------------------------------------
The Cramer distance from a predicted CDF to a Dirac at the observed residual time
*is* the CRPS, a strictly proper scoring rule for the whole distribution. Expressing
the anchor in the same metric as the TD term has four consequences that matter:

  * It is automatically scale-matched, so alpha is a genuine convex weight in [0,1]
    with no auxiliary scale hyperparameter. An NLL anchor sits around 2.0 against a
    Cramer term around 0.02 -- a 100x mismatch that would make alpha uninterpretable
    and add a tuning knob, i.e. another degree of freedom to accidentally p-hack.
  * Being strictly proper, it cannot be gamed by a degenerate distribution, which is
    exactly the failure mode being repaired.
  * It reuses the existing machinery: the event target is the projected Dirac the TD
    recursion already builds, so the two terms share a bin convention (defect D12).
  * It is where the preregistered "scalar IPCW Cramer loss weighting" finally exists.

The honest cost, stated plainly
-------------------------------
At alpha = 1 this objective is a per-visit proper scoring rule on residual time --
Dynamic-DeepHit's L1 in Cramer form. **There is a value of alpha at which SurvTD is
the baseline it claims to beat.** That reframes the contribution: SurvTD is not a TD
objective replacing the likelihood, but a duration-aware TD consistency regularizer
added to one, and C_3 must become "the TD term adds >= 0.025 over the same backbone
with the same supervision". The alpha = 1 arm is therefore a first-class condition
and a pre-declared kill criterion, not a diagnostic. See deviation log A-04.
"""

from __future__ import annotations

import math

import torch

from src.operators.survtd_operator import (
    restrict_to_interval,
    truncated_censoring_target,  # re-exported: lives with the operator, used by both
)


def dirac_cdf(residual: torch.Tensor, delta_s: float, K: int) -> torch.Tensor:
    """
    CDF of the triangular-projected Dirac at each residual time, vectorised over visits.

    Uses the same centre-based convention as `localized_projected_dirac`
    (u = s/delta_s - 0.5, split between floor(u) and floor(u)+1), so the anchor and
    the TD target agree bin for bin. Divergent conventions across the ladder are
    defect D12.

    Args:
        residual: (L,) residual times, already clamped to >= 0
        delta_s, K: grid
    Returns:
        (L, K) CDFs. All-ones when the residual falls below the first bin centre,
        all-zeros when it lands at or beyond the horizon (the mass is on perp).
    """
    device = residual.device
    ks = torch.arange(K, device=device, dtype=residual.dtype)          # (K,)

    u = residual / delta_s - 0.5
    k0 = torch.floor(u)
    frac = u - k0

    cdf = (ks.unsqueeze(0) > k0.unsqueeze(1)).to(residual.dtype)
    cdf = cdf + (ks.unsqueeze(0) == k0.unsqueeze(1)).to(residual.dtype) * (1.0 - frac).unsqueeze(1)

    # Boundary cases.
    below = (u <= 0).unsqueeze(1)
    beyond = (u >= K).unsqueeze(1)
    cdf = torch.where(below, torch.ones_like(cdf), cdf)
    cdf = torch.where(beyond, torch.zeros_like(cdf), cdf)
    return cdf


def censored_crps_anchor(
    pred_cdf: torch.Tensor,
    residual: torch.Tensor,
    event: bool,
    delta_s: float,
    K: int,
    ipcw_weight: float = 1.0,
) -> torch.Tensor:
    """
    Per-visit right-censored CRPS, in the Cramer metric.

        uncensored visit j:  delta_s * sum_k ( F_j(k) - 1{k >= k_j} )^2
        censored   visit j:  delta_s * sum_{k < k_j} F_j(k)^2  *  1/G_hat(c_j)

    For an event trajectory every visit's residual time is known exactly, so every
    visit gets the full proper score. For a censored trajectory every visit knows
    only R_j > r_j, so only the region below the censoring time is scored -- but it
    is scored at every visit, which is what turns ~72% of the cohort from
    zero-information into supervision.

    Args:
        pred_cdf: (L, K) predicted lifetime CDF at each visit
        residual: (L,) residual time to event/censoring from each visit
        event: whether the trajectory terminated in an event
        ipcw_weight: scalar IPCW weight, applied to censored trajectories only
    Returns:
        (L,) per-visit losses
    """
    r = torch.as_tensor(residual, dtype=pred_cdf.dtype, device=pred_cdf.device).clamp_min(0.0)

    if event:
        target = dirac_cdf(r, delta_s, K)
        return float(delta_s) * torch.sum((pred_cdf - target) ** 2, dim=-1)

    ks = torch.arange(K, device=pred_cdf.device, dtype=pred_cdf.dtype)
    # We know R_j > r_j, so F_j must be 0 on every bin fully below r_j.
    k_j = torch.floor(r / delta_s).unsqueeze(1)
    mask = (ks.unsqueeze(0) < k_j).to(pred_cdf.dtype)
    return float(delta_s) * torch.sum((pred_cdf ** 2) * mask, dim=-1) * float(ipcw_weight)


__all__ = [
    "dirac_cdf",
    "censored_crps_anchor",
    "truncated_censoring_target",
]
