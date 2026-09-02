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
    categorical_projection_shift,
    compute_interval_discount,
    localized_projected_dirac
)


def compute_ablated_lambda_returns(
    target_pmfs: torch.Tensor,
    target_survivals: torch.Tensor,
    dts: torch.Tensor,
    events: torch.Tensor,
    tte: float,
    tau_event: float,
    ablation_mode: str = "full",  # 'full', 'arm_a1_discount', 'arm_a2_shift', 'count_geometric'
    lam: float = 0.6,
    delta_s: float = 1.0,
    K: int = 30,
    censor_ipcw_weight: float = 1.0
):
    """
    Computes backward multi-step targets under designated ablation conditions.
    """
    L = target_pmfs.shape[0]
    device = target_pmfs.device
    G_targets = [None] * L
    step_weights = torch.ones(L, device=device)

    # Terminal step
    has_event_terminal = bool(events[-1].item() > 0.5)
    if has_event_terminal:
        dt_last = float(dts[-1].item())
        delta_tau = max(0.0, min(dt_last, tau_event - (tte - dt_last)))
        G_targets[-1] = localized_projected_dirac(delta_tau, delta_s, K, device=device).detach()
    else:
        G_targets[-1] = target_pmfs[-1].clone().detach()
        step_weights[-1] = min(float(censor_ipcw_weight), 10.0)

    for j in range(L - 2, -1, -1):
        dt_j = float(dts[j].item())
        event_j = bool(events[j].item() > 0.5)

        # Lambda mixing rule
        if ablation_mode == "count_geometric":
            lambda_j = float(lam)  # fixed discrete count-geometric
        else:
            lambda_j = float(lam ** (dt_j / delta_s))

        # Gamma discount rule
        if ablation_mode == "arm_a1_discount":
            # Arm A1: constant unit-step discount S(delta_s) regardless of elapsed dt
            gamma_j = compute_interval_discount(target_survivals[j], delta_s, delta_s, K)
        else:
            gamma_j = compute_interval_discount(target_survivals[j], dt_j, delta_s, K)

        # Shift rule
        if ablation_mode == "arm_a2_shift":
            # Arm A2: fixed unit grid shift Phi_{+delta_s} regardless of continuous dt
            effective_shift_dt = delta_s
        else:
            effective_shift_dt = dt_j

        if event_j:
            delta_tau = max(0.0, dt_j * 0.5)
            T_p_j = localized_projected_dirac(delta_tau, delta_s, K, device=device)
            G_targets[j] = T_p_j.detach()
        else:
            p_next_target = target_pmfs[j + 1]
            T_p_j = categorical_projection_shift(p_next_target, effective_shift_dt, delta_s, K)
            G_next_shifted = categorical_projection_shift(G_targets[j + 1], effective_shift_dt, delta_s, K)

            G_j = (1.0 - lambda_j) * T_p_j + lambda_j * gamma_j * G_next_shifted
            G_j = G_j / torch.clamp(torch.sum(G_j), min=1e-8)
            G_targets[j] = G_j.detach()

    return torch.stack(G_targets, dim=0), step_weights


def clamped_division_target(p_next: torch.Tensor, survival_curr: torch.Tensor, dt: float, delta_s: float, K: int, clamp_eps: float = 1e-3) -> torch.Tensor:
    """
    NC-A3: Clamped continuous division baseline:
    p_target(s) = p_next(s - dt) / max(S(dt), 1e-3)
    """
    # Shift p_next by integer or continuous steps
    dt_steps = max(1, int(round(dt / delta_s)))
    S_dt = compute_interval_discount(survival_curr, dt, delta_s, K)
    S_clamped = max(S_dt, clamp_eps)

    target_p = torch.zeros_like(p_next)
    if dt_steps < K:
        target_p[dt_steps:] = p_next[:K - dt_steps] / S_clamped

    # Normalization
    sum_mass = torch.sum(target_p)
    if sum_mass > 1e-8:
        target_p = target_p / sum_mass
    else:
        target_p = torch.ones_like(p_next) / K

    return target_p.detach()


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
