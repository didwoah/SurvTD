"""
Ablation and Negative Control Operators:
- NC-A1 (Discount Ablation): Forces gamma_j = S(delta_s) while preserving continuous renewal shift.
- NC-A2 (Shift Ablation): Forces fixed unit grid shift Phi_{+delta_s} while preserving continuous discount.
- NC-A3 (Clamped Division): DeepTCSR unit-step continuous adaptation p / max(S(dt), 1e-3).
- NC-B (Within-Patient Duration Permutation): Shuffles inter-visit intervals within trajectory.
- NC-C (Effective Horizon Matching): Count-geometric lambda^k comparison.
"""

import math
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F

from src.operators.survtd_operator import (
    ARMS,
    categorical_projection_shift,
    compute_interval_discount,
    compute_multistep_lambda_returns,
    interval_gaps,
    localized_projected_dirac,
    residual_times
)


def compute_ablated_lambda_returns(
    target_pmfs: torch.Tensor,
    target_survivals: torch.Tensor,
    dts: torch.Tensor,
    events: torch.Tensor,
    tte: float,
    tau_event: float = None,
    ablation_mode: str = "full",
    lam: float = 0.6,
    delta_s: float = 1.0,
    K: int = 30,
    censor_ipcw_weight: float = 1.0,
    include_overflow: bool = False,
    gamma_placement: str = "bootstrap",
):
    """
    Thin delegate to `compute_lambda_returns`.

    This function used to hold a second, hand-maintained copy of the backward
    recursion, differing from the full model's only in three `if` branches. That
    duplication is why the gamma renormalization defect existed in two places and
    why an ablation arm could silently run different mathematics from the arm it is
    meant to be compared against. The arm definitions now live in the single `ARMS`
    dict in src/operators/survtd_operator.py, which experiments/run_track_b.py reads
    as well, so an arm cannot be defined one way in the operator and another way in
    the experiment script.
    """
    if ablation_mode not in ARMS:
        raise ValueError(
            "unknown ablation_mode %r; expected one of %s" % (ablation_mode, sorted(ARMS))
        )
    return compute_multistep_lambda_returns(
        target_pmfs, target_survivals, dts, events, tte, tau_event,
        lam=lam, delta_s=delta_s, K=K, censor_ipcw_weight=censor_ipcw_weight,
        include_overflow=include_overflow, arm=ablation_mode,
        gamma_placement=gamma_placement,
    )


def clamped_division_target(
    p_next: torch.Tensor,
    survival_curr: torch.Tensor,
    dt: float,
    delta_s: float,
    K: int,
    clamp_eps: float = 1e-3,
    include_overflow: bool = False,
) -> tuple:
    """
    NC-A3: Clamped continuous division baseline:
    p_target(s) = p_next(s - dt) / max(S(dt), 1e-3)

    Deliberately keeps the integer unit-step shift and the absence of a triangular
    projection: that is what the preregistration declares for Arm A3, so it must not
    be quietly upgraded to the full renewal operator.

    Returns (target, diagnostics) where diagnostics records whether the clamp bound
    and whether the uniform fallback fired. Those two rates are the direct evidence
    for or against the numerical-degeneracy claim behind C_1, and the previous
    implementation discarded them.
    """
    dt_steps = max(1, int(round(dt / delta_s)))
    S_dt = compute_interval_discount(survival_curr, dt, delta_s, K)
    S_clamped = max(S_dt, clamp_eps)

    n = p_next.shape[-1]
    target_p = torch.zeros_like(p_next)
    if dt_steps < K:
        target_p[dt_steps:K] = p_next[:K - dt_steps] / S_clamped
    if include_overflow:
        # Whatever is shifted past the last bin, plus the incoming perp mass, is
        # already beyond the horizon.
        target_p[K] = (
            p_next[K] + torch.sum(p_next[max(0, K - dt_steps):K])
        ) / S_clamped

    diagnostics = {
        "clamp_bound": bool(S_dt < clamp_eps),
        "uniform_fallback": False,
    }

    sum_mass = torch.sum(target_p)
    if sum_mass > 1e-8:
        target_p = target_p / sum_mass
    else:
        target_p = torch.ones_like(p_next) / n
        diagnostics["uniform_fallback"] = True

    return target_p.detach(), diagnostics


def permute_patient_durations(patient_dict: dict, rng: random.Random) -> dict:
    """
    NC-B: Randomly permutes interval durations dt within a patient's trajectory,
    preserving total follow-up time sum(dt), sequence length L, features, and event label.
    """
    new_patient = copy.deepcopy(patient_dict)
    dts = list(patient_dict['dts'].numpy() if isinstance(patient_dict['dts'], torch.Tensor) else patient_dict['dts'])
    if len(dts) > 1:
        rng.shuffle(dts)
        new_patient['dts'] = torch.tensor(dts, dtype=torch.float32)
    return new_patient
