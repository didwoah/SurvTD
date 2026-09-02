"""
SurvTD Core Mathematical Engine:
1. Categorical Projection Pi and Continuous Renewal Shift Phi_{+Delta t}.
2. Interval Duration Discount gamma_j = S_{theta-}(Delta t_j) with sub-bin linear hazard interpolation.
3. Empirical Sample-Path Bellman Target with intra-interval projected Dirac for event transitions.
4. Compounded Multi-Step lambda-Return Recursion with duration-geometric decay lambda^(Delta t / delta_s).
5. Truncated Conditional IPCW tail weighting and Squared Cramer Distance Loss.
"""

import math
import torch
import torch.nn.functional as F


def categorical_projection_shift(p: torch.Tensor, dt: float, delta_s: float, K: int) -> torch.Tensor:
    """
    Carries distribution p backwards by renewal shift Phi_{+dt} (R_j = R_{j+1} + dt)
    and projects onto K uniform bins of width delta_s via triangular kernel Pi.

    Args:
        p: (K,) or (B, K) source probability mass function
        dt: float or scalar tensor, elapsed duration
        delta_s: float, bin width
        K: int, number of bins
    Returns:
        p_proj: (K,) or (B, K) projected PMF conserving unit probability mass exactly.
    """
    is_1d = (p.dim() == 1)
    if is_1d:
        p = p.unsqueeze(0)

    B = p.shape[0]
    device = p.device
    dtype = p.dtype

    if isinstance(dt, torch.Tensor):
        dt_val = float(dt.item())
    else:
        dt_val = float(dt)

    if dt_val <= 0.0:
        return p.squeeze(0) if is_1d else p

    # Grid centers: s_m = (m + 0.5) * delta_s
    # Shifted locations: s'_m = s_m + dt = (m + 0.5 + dt / delta_s) * delta_s
    # Target bin coordinate u_m = s'_m / delta_s - 0.5 = m + dt / delta_s
    shift_steps = dt_val / delta_s

    # Construct K x K projection matrix W where W[m, k] is the fraction of mass from bin m assigned to bin k
    W = torch.zeros(K, K, device=device, dtype=dtype)

    for m in range(K):
        u = m + shift_steps
        if u >= K - 1:
            # Entire shifted mass falls at or past the final bin center -> absorbing terminal bin
            W[m, K - 1] = 1.0
        elif u <= 0:
            # Clamped to bin 0
            W[m, 0] = 1.0
        else:
            k = int(math.floor(u))
            f = u - k  # fraction in [0, 1)
            k_next = min(k + 1, K - 1)
            if k == k_next:
                W[m, k] = 1.0
            else:
                W[m, k] = 1.0 - f
                W[m, k_next] = f

    # Matrix multiplication: p_proj = p @ W
    p_proj = torch.matmul(p, W)

    # Strict normalization guard for numerical precision
    p_proj = p_proj / torch.clamp(torch.sum(p_proj, dim=-1, keepdim=True), min=1e-8)

    return p_proj.squeeze(0) if is_1d else p_proj


def compute_interval_discount(survival_curve: torch.Tensor, dt: float, delta_s: float, K: int, h0: float = None) -> float:
    """
    Evaluates gamma_j = S_{theta-}(Delta t_j) under frozen target network.
    For sub-bin intervals (dt < delta_s), applies linear hazard interpolation: S(dt) = 1 - h_0 * dt / delta_s.
    """
    if isinstance(dt, torch.Tensor):
        dt_val = float(dt.item())
    else:
        dt_val = float(dt)

    if dt_val <= 1e-6:
        return 1.0

    # Ensure survival curve is a 1D tensor
    if survival_curve.dim() > 1:
        survival_curve = survival_curve.squeeze(0)

    if dt_val < delta_s:
        # Sub-bin interpolation
        if h0 is None:
            # Estimate first-step hazard: h0 = 1 - S(0)
            h0_val = 1.0 - float(survival_curve[0].item())
        else:
            h0_val = float(h0)
        gamma = max(0.0, min(1.0, 1.0 - h0_val * (dt_val / delta_s)))
        return gamma
    else:
        # Discrete index lookup with linear interpolation between bin edges
        step_pos = dt_val / delta_s - 0.5
        if step_pos <= 0:
            return max(0.0, min(1.0, float(survival_curve[0].item())))
        k = int(math.floor(step_pos))
        if k >= K - 1:
            return max(0.0, min(1.0, float(survival_curve[-1].item())))
        f = step_pos - k
        s_k = float(survival_curve[k].item())
        s_next = float(survival_curve[k + 1].item())
        gamma = (1.0 - f) * s_k + f * s_next
        return max(0.0, min(1.0, gamma))


def localized_projected_dirac(delta_tau: float, delta_s: float, K: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Projects intra-interval death at offset delta_tau onto K bins via triangular kernel.
    """
    if isinstance(delta_tau, torch.Tensor):
        tau_val = float(delta_tau.item())
    else:
        tau_val = float(delta_tau)

    tau_val = max(0.0, tau_val)
    target = torch.zeros(K, device=device, dtype=dtype)

    u = tau_val / delta_s - 0.5
    if u <= 0:
        target[0] = 1.0
    elif u >= K - 1:
        target[K - 1] = 1.0
    else:
        k = int(math.floor(u))
        f = u - k
        k_next = min(k + 1, K - 1)
        if k == k_next:
            target[k] = 1.0
        else:
            target[k] = 1.0 - f
            target[k_next] = f

    return target


def compute_multistep_lambda_returns(
    target_pmfs: torch.Tensor,
    target_survivals: torch.Tensor,
    dts: torch.Tensor,
    events: torch.Tensor,
    tte: float,
    tau_event: float,
    lam: float = 0.6,
    delta_s: float = 1.0,
    K: int = 30,
    censor_ipcw_weight: float = 1.0
):
    """
    Backward multi-step lambda-return recursion for a single patient trajectory:
    G_j = (1 - lambda_j) * T p_j + lambda_j * gamma_j * Pi Phi_{+dt_j} G_{j+1}
    with duration-geometric decay lambda_j = lambda^(dt_j / delta_s).

    Args:
        target_pmfs: (L, K) predictions from frozen target network theta-
        target_survivals: (L, K) survival curves from theta-
        dts: (L,) elapsed durations between visits
        events: (L,) binary event indicators for each interval
        tte: float, total time-to-event/censoring
        tau_event: float, timestamp of event if event=1
        lam: float in [0, 1]
        delta_s: float, bin width
        K: int, number of bins
        censor_ipcw_weight: scalar IPCW weight if trajectory is right-censored alive
    Returns:
        G_targets: (L, K) detached multi-step Bellman targets for each visit
        step_weights: (L,) scalar loss weights (e.g. IPCW on terminal censored step)
    """
    L = target_pmfs.shape[0]
    device = target_pmfs.device
    G_targets = [None] * L
    step_weights = torch.ones(L, device=device)

    # 1. Terminal Step L - 1 target
    has_event_terminal = bool(events[-1].item() > 0.5)
    if has_event_terminal:
        # Event occurred in terminal interval
        dt_last = float(dts[-1].item())
        delta_tau = max(0.0, min(dt_last, tau_event - (tte - dt_last)))
        G_targets[-1] = localized_projected_dirac(delta_tau, delta_s, K, device=device).detach()
    else:
        # Right-censored alive at terminal visit
        # Tail completion using target network prediction
        G_targets[-1] = target_pmfs[-1].clone().detach()
        step_weights[-1] = min(float(censor_ipcw_weight), 10.0)

    # 2. Backward Recursion from L - 2 down to 0
    for j in range(L - 2, -1, -1):
        dt_j = float(dts[j].item())
        event_j = bool(events[j].item() > 0.5)

        # Duration-geometric lambda mixing
        lambda_j = float(lam ** (dt_j / delta_s))

        # Interval duration discount gamma_j = S_{theta-}(dt_j)
        gamma_j = compute_interval_discount(target_survivals[j], dt_j, delta_s, K)

        if event_j:
            # Event occurred in interval j
            delta_tau = max(0.0, dt_j * 0.5)  # mid-interval default or localized
            T_p_j = localized_projected_dirac(delta_tau, delta_s, K, device=device)
            # When event happens in interval j, future path does not exist
            G_targets[j] = T_p_j.detach()
        else:
            # Alive transition: sample-path continuation branch
            # T p_j evaluates along Pi Phi_{+dt_j} p_{theta-, j+1}
            p_next_target = target_pmfs[j + 1]
            T_p_j = categorical_projection_shift(p_next_target, dt_j, delta_s, K)

            # Shifted and discounted future multi-step target: gamma_j * Pi Phi_{+dt_j} G_{j+1}
            G_next_shifted = categorical_projection_shift(G_targets[j + 1], dt_j, delta_s, K)

            # Multi-step convex combination
            G_j = (1.0 - lambda_j) * T_p_j + lambda_j * gamma_j * G_next_shifted

            # Mass conservation guard
            G_j = G_j / torch.clamp(torch.sum(G_j), min=1e-8)
            G_targets[j] = G_j.detach()

    return torch.stack(G_targets, dim=0), step_weights


def squared_cramer_distance_loss(pred_cdf: torch.Tensor, target_cdf: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """
    Cramér distance loss (ell_2^2 on CDFs):
    L = 1/K * sum_{k=0}^{K-1} (F_pred(k) - F_target(k))^2

    Args:
        pred_cdf: (..., K)
        target_cdf: (..., K) detached
        weights: optional scalar weights per step (...,)
    """
    diff_sq = (pred_cdf - target_cdf) ** 2
    per_step_loss = torch.mean(diff_sq, dim=-1)  # mean over K bins

    if weights is not None:
        per_step_loss = per_step_loss * weights

    return torch.mean(per_step_loss)
