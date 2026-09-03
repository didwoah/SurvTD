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


def residual_times(dts: torch.Tensor, tte: float) -> torch.Tensor:
    """
    Residual time from each visit to the event/censoring time: r_j = tte - t_j.

    The loaders store a BACKWARD difference, dts[j] = times[j] - times[j-1], so the
    absolute visit times are cumsum(dts). Values may be <= 0 when the recorded event
    time precedes the last observation.

    This is the quantity the lifetime head predicts. Confusing it with the elapsed
    interval dts[j] is the root of defects D9 (terminal Dirac placed at dt_last) and
    of the Person-Period baseline scoring below chance.
    """
    return float(tte) - torch.cumsum(dts, dim=0)


def interval_gaps(dts: torch.Tensor) -> torch.Tensor:
    """
    Forward gaps: gaps[j] = times[j+1] - times[j], for j = 0 .. L-2.

    The renewal identity for the transition j -> j+1 is
        R_j = R_{j+1} + (times[j+1] - times[j]).
    Under the loaders' backward-difference convention that elapsed duration is
    dts[j+1], NOT dts[j]. Pairing dts[j] with target_pmfs[j+1] (defect D8) shifts the
    whole duration sequence by one index, which destroys the per-transition
    correspondence between duration and transition -- precisely what the NC-B
    duration-permutation negative control does.

    Returns a tensor of length max(L - 1, 0).
    """
    return dts[1:]


def categorical_projection_shift(
    p: torch.Tensor,
    dt: float,
    delta_s: float,
    K: int,
    include_overflow: bool = False,
) -> torch.Tensor:
    """
    Carries distribution p backwards by renewal shift Phi_{+dt} (R_j = R_{j+1} + dt)
    and projects onto K uniform bins of width delta_s via triangular kernel Pi.

    Args:
        p: (n,) or (B, n) source PMF, where n = K + 1 if include_overflow else K
        dt: float or scalar tensor, elapsed duration
        delta_s: float, bin width
        K: int, number of genuine bins
        include_overflow: if True, coordinate K is an absorbing "R >= K*delta_s"
            symbol and mass shifted past the last bin lands there instead of piling
            into bin K-1. See src/models/hazard_head.py for why that distinction
            matters: without it, bin K-1 is an absorbing attractor and a
            self-referential TD target can park all its mass there.
    Returns:
        p_proj: same shape as p, conserving total probability mass exactly.
    """
    is_1d = (p.dim() == 1)
    if is_1d:
        p = p.unsqueeze(0)

    device = p.device
    dtype = p.dtype
    n = K + 1 if include_overflow else K

    if p.shape[-1] != n:
        raise ValueError(
            f"expected last dim {n} for K={K}, include_overflow={include_overflow}, "
            f"got {p.shape[-1]}"
        )

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

    W = torch.zeros(n, n, device=device, dtype=dtype)

    for m in range(K):
        u = m + shift_steps

        if include_overflow:
            if u >= K:
                W[m, K] = 1.0
                continue
            if u <= 0:
                W[m, 0] = 1.0
                continue
            k = int(math.floor(u))
            f = u - k
            W[m, k] += 1.0 - f
            W[m, min(k + 1, K)] += f     # k + 1 == K routes the fraction to perp
        else:
            # Legacy path: bin K-1 absorbs everything at or past the final bin center.
            if u >= K - 1:
                W[m, K - 1] = 1.0
            elif u <= 0:
                W[m, 0] = 1.0
            else:
                k = int(math.floor(u))
                f = u - k
                k_next = min(k + 1, K - 1)
                if k == k_next:
                    W[m, k] = 1.0
                else:
                    W[m, k] = 1.0 - f
                    W[m, k_next] = f

    if include_overflow:
        W[K, K] = 1.0                    # perp is absorbing: already beyond the horizon

    p_proj = torch.matmul(p, W)

    # Mass is conserved by construction (every row of W sums to 1). Assert instead of
    # renormalizing, so a genuine leak surfaces rather than being silently rescaled.
    if __debug__:
        total = torch.sum(p_proj, dim=-1)
        expected = torch.sum(p, dim=-1)
        if not torch.allclose(total, expected, atol=1e-5):
            raise AssertionError(
                f"projection leaked mass: {float((total - expected).abs().max()):.3e}"
            )

    return p_proj.squeeze(0) if is_1d else p_proj


def restrict_to_interval(
    p: torch.Tensor,
    dt: float,
    delta_s: float,
    K: int,
    include_overflow: bool = False,
) -> tuple:
    """
    Keeps only the mass of p at residual times s <= dt, zeroing everything beyond
    (including the overflow coordinate).

    This is the near branch of the renewal identity. Its mass is, by construction,
        P(R_j <= dt) = 1 - S_j(dt) = 1 - gamma_j,
    which is what makes the renewal mixture sum to exactly 1 with no renormalization.
    The boundary bin is split linearly, matching the linear-interpolation convention
    used by `compute_interval_discount`.

    Returns:
        (p_restricted, mass) where mass = float(p_restricted.sum())
    """
    if isinstance(dt, torch.Tensor):
        dt_val = float(dt.item())
    else:
        dt_val = float(dt)

    out = torch.zeros_like(p)
    if dt_val <= 0.0:
        return out, 0.0

    u = dt_val / delta_s
    k_full = min(int(math.floor(u)), K)
    if k_full > 0:
        out[..., :k_full] = p[..., :k_full]
    if k_full < K:
        out[..., k_full] = p[..., k_full] * (u - math.floor(u))

    return out, float(torch.sum(out).item())


def truncated_censoring_target(
    p_target_last: torch.Tensor,
    censoring_residual: float,
    delta_s: float,
    K: int,
    include_overflow: bool = False,
) -> torch.Tensor:
    """
    Censoring-aware replacement for the terminal target of a right-censored trajectory.

    The shipped code used `target_pmfs[-1]`, the target network's own output, which is
    a self-copy and carries zero information (defect D-CENS). Here the observation
    "the subject was still alive at the censoring time" is imposed as a hard
    constraint: all mass at s <= c is removed and the remainder renormalized over
    s > c (including perp).

    The shape above c still comes from theta- -- there is no data there, so that is
    unavoidable -- but the target now *removes* the probability mass the observation
    rules out, rather than reproducing the model's own belief unchanged.

    Falls back to putting all mass on the far end when the model assigns essentially
    no probability to surviving past c, which would otherwise divide by ~0.
    """
    kept = p_target_last.clone()
    near, near_mass = restrict_to_interval(
        p_target_last, censoring_residual, delta_s, K, include_overflow=include_overflow
    )
    kept = kept - near

    surviving = float(torch.sum(kept).item())
    if surviving <= 1e-6:
        # Degenerate: theta- believes the event has already happened. Put the mass at
        # the far end, which is the closest consistent statement.
        kept = torch.zeros_like(p_target_last)
        kept[-1] = 1.0
        return kept.detach()

    return (kept / surviving).detach()


def one_step_renewal_target(
    p_tgt_j: torch.Tensor,
    p_tgt_next: torch.Tensor,
    gamma_dt: float,
    shift_dt: float,
    delta_s: float,
    K: int,
    include_overflow: bool = False,
) -> tuple:
    """
    The one-step renewal target of design Step 5:

        T_hat p_j = restrict_[0, gamma_dt]( p_theta-,j )  (+)  gamma_j * Pi Phi_{+shift_dt} p_theta-,j+1

    where gamma_j = 1 - mass(near branch). Splitting `gamma_dt` from `shift_dt` is
    what lets the two preregistered ablation arms be expressed without a second copy
    of the recursion: Arm A1 fixes gamma_dt = delta_s while keeping the continuous
    shift, and Arm A2 fixes shift_dt = delta_s while keeping the continuous discount.

    Total mass is exactly (1 - gamma_j) + gamma_j * 1 = 1.

    NOTE this replaces the shipped implementation, which computed only the second
    term. With the near branch missing, gamma had nowhere to act except as a scalar
    on the bootstrap branch, where the subsequent renormalization absorbed it into
    lambda (defect D-gamma). gamma is load-bearing here because it changes the SHAPE
    of the target -- the near/far split -- not merely a scalar mixture weight.

    Returns:
        (target, gamma_j)
    """
    near, near_mass = restrict_to_interval(
        p_tgt_j, gamma_dt, delta_s, K, include_overflow=include_overflow
    )
    gamma_j = max(0.0, min(1.0, 1.0 - near_mass))
    far = categorical_projection_shift(
        p_tgt_next, shift_dt, delta_s, K, include_overflow=include_overflow
    )
    return near + gamma_j * far, gamma_j


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
        # Discrete index lookup with linear interpolation between bin edges.
        #
        # The head emits survival_curve[k] = prod_{m<=k} (1 - h_m) = P(R > (k+1)*delta_s),
        # i.e. index k already refers to the RIGHT edge of bin k. So S(dt) must be read
        # at index dt/delta_s - 1, not dt/delta_s - 0.5. Using -0.5 over-discounts by
        # half a bin (defect D10) and is also discontinuous with the sub-bin branch
        # above, which yields S(delta_s) = 1 - h0 = survival_curve[0] exactly.
        step_pos = dt_val / delta_s - 1.0
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


def localized_projected_dirac(
    delta_tau: float,
    delta_s: float,
    K: int,
    device=None,
    dtype=torch.float32,
    include_overflow: bool = False,
) -> torch.Tensor:
    """
    Projects a death at residual time delta_tau onto the bin grid via the triangular
    kernel. With include_overflow=True the returned vector has length K+1 and a death
    at or beyond K*delta_s lands on the absorbing perp coordinate rather than being
    clamped into bin K-1.
    """
    if isinstance(delta_tau, torch.Tensor):
        tau_val = float(delta_tau.item())
    else:
        tau_val = float(delta_tau)

    tau_val = max(0.0, tau_val)
    n = K + 1 if include_overflow else K
    target = torch.zeros(n, device=device, dtype=dtype)

    u = tau_val / delta_s - 0.5

    if u <= 0:
        target[0] = 1.0
        return target

    if include_overflow:
        if u >= K:
            target[K] = 1.0
            return target
        k = int(math.floor(u))
        f = u - k
        target[k] += 1.0 - f
        target[min(k + 1, K)] += f
        return target

    if u >= K - 1:
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


# Single source of truth for the preregistered arms. src/operators/ablations.py and
# experiments/run_track_b.py both read this dict, so an arm can never be defined one
# way in the operator and another way in the experiment script -- which is how the
# gamma defect came to exist in two divergent copies of the recursion.
ARMS = {
    # full proposal: continuous discount, continuous shift, duration-geometric mixing
    "full":            dict(mixing="duration", gamma_rule="continuous", shift_rule="continuous"),
    # NC-A1: discount fixed to the unit step S(delta_s), continuous shift intact
    "arm_a1_discount": dict(mixing="duration", gamma_rule="unit_step",  shift_rule="continuous"),
    # NC-A2: shift fixed to the unit grid Phi_{+delta_s}, continuous discount intact
    "arm_a2_shift":    dict(mixing="duration", gamma_rule="continuous", shift_rule="unit_step"),
    # NC-C: count-geometric mixing lambda^k instead of lambda^(dt/delta_s)
    "count_geometric": dict(mixing="count",    gamma_rule="continuous", shift_rule="continuous"),
}


def compute_lambda_returns(
    target_pmfs: torch.Tensor,
    target_survivals: torch.Tensor,
    dts: torch.Tensor,
    tte: float,
    event: bool,
    lam: float = 0.6,
    delta_s: float = 1.0,
    K: int = 30,
    censor_ipcw_weight: float = 1.0,
    include_overflow: bool = False,
    arm: str = "full",
    gamma_placement: str = "bootstrap",
    censoring_aware_target: bool = True,
):
    """
    Backward multi-step lambda-return recursion for a single patient trajectory.

    On an alive transition j -> j+1 with forward gap g_j = times[j+1] - times[j]:

        T_hat p_j = restrict_[0,g_j](p_theta-,j)  (+)  gamma_j * Pi Phi_{+g_j} p_theta-,j+1
        G_j       = (1 - lambda_j) * T_hat p_j    +    lambda_j * Pi Phi_{+g_j} G_{j+1}

    with lambda_j = lambda^(g_j / delta_s) for duration-geometric mixing.

    gamma sits in the ONE-STEP term only, never on the recursive branch. That is not
    a stylistic choice:
      * lambda = 1 then gives G_j = Pi Phi G_{j+1}, recursing to the terminal Dirac,
        i.e. the exact projected Monte-Carlo target. This is correct because the
        realized path TELLS US that no event occurred in those intervals, so no
        survival discount applies to them. gamma is the correction attached to a
        bootstrap, which is an expectation over what might have happened.
      * lambda = 0 gives G_j = T_hat p_j, the exact one-step expected renewal target.
      * Both addends have mass 1, so sum(G_j) = 1 identically, and the old
        `G_j / sum(G_j)` -- the line that made gamma equivalent to a lambda
        reparameterization -- is deleted rather than repaired.
      * The linear part of T_hat in its far argument is gamma_j * Pi Phi, whose
        Cramer-metric Lipschitz modulus is gamma_j < 1. That is what thm:1 asserts.
        Under the shipped code the linear part was lambda_eff * Pi Phi, so the
        theorem did not describe the implementation.

    `gamma_placement='compounded'` also applies the near/far split to the recursive
    branch, which is what C_2's prose ("compounds interval survival discounting")
    describes -- but then lambda = 1 no longer reduces to Monte Carlo, contradicting
    Kill Criterion 2. Both are implemented and both are reported; see deviation log
    entry A-15. Do not silently pick whichever performs better.

    Args:
        target_pmfs: (L, n) predictions from the frozen target network theta-,
            with n = K+1 when include_overflow
        target_survivals: (L, K) survival curves from theta-. No longer used to
            derive gamma: gamma now comes from the mass of the restricted near
            branch, so the mixture sums to 1 by construction rather than by
            interpolation agreement. Retained for callers that inspect it.
        dts: (L,) backward-difference durations, dts[j] = times[j] - times[j-1]
        tte: total time to event/censoring
        event: whether the trajectory ended in an event
        arm: key into ARMS
        gamma_placement: 'bootstrap' | 'compounded'
    Returns:
        G_targets: (L, n) detached targets
        step_weights: (L,) scalar loss weights
    """
    if arm not in ARMS:
        raise ValueError("unknown arm %r; expected one of %s" % (arm, sorted(ARMS)))
    cfg = ARMS[arm]
    if gamma_placement not in ("bootstrap", "compounded"):
        raise ValueError("unknown gamma_placement %r" % (gamma_placement,))

    L = target_pmfs.shape[0]
    device = target_pmfs.device
    G_targets = [None] * L
    step_weights = torch.ones(L, device=device)

    r_np = residual_times(dts, tte).detach().cpu().numpy()
    gaps_np = interval_gaps(dts).detach().cpu().numpy()

    # 1. Terminal step: a Dirac at the residual time for an event trajectory.
    if event:
        G_targets[-1] = localized_projected_dirac(
            max(0.0, float(r_np[-1])), delta_s, K, device=device,
            include_overflow=include_overflow,
        ).detach()
    elif censoring_aware_target:
        # Right-censored: impose "still alive at the censoring time" as a hard
        # constraint by removing all mass at s <= c and renormalizing over s > c.
        # The shipped code copied target_pmfs[-1] -- the target network's own
        # output -- which is a self-copy carrying zero information (defect D-CENS),
        # and ~72% of the sepsis cohort is censored.
        G_targets[-1] = truncated_censoring_target(
            target_pmfs[-1], max(0.0, float(r_np[-1])), delta_s, K,
            include_overflow=include_overflow,
        )
        step_weights[-1] = min(float(censor_ipcw_weight), 10.0)
    else:
        G_targets[-1] = target_pmfs[-1].clone().detach()
        step_weights[-1] = min(float(censor_ipcw_weight), 10.0)

    # 2. Backward recursion.
    for j in range(L - 2, -1, -1):
        gap_j = float(gaps_np[j])

        if event and r_np[j] > 0.0 >= r_np[j + 1]:
            # The event falls inside the observed interval: the ground truth is exact.
            G_targets[j] = localized_projected_dirac(
                max(0.0, float(r_np[j])), delta_s, K, device=device,
                include_overflow=include_overflow,
            ).detach()
            continue

        lambda_j = float(lam) if cfg["mixing"] == "count" else float(lam ** (gap_j / delta_s))
        gamma_dt = delta_s if cfg["gamma_rule"] == "unit_step" else gap_j
        shift_dt = delta_s if cfg["shift_rule"] == "unit_step" else gap_j

        T_p_j, gamma_j = one_step_renewal_target(
            target_pmfs[j], target_pmfs[j + 1], gamma_dt, shift_dt,
            delta_s, K, include_overflow=include_overflow,
        )

        G_next_shifted = categorical_projection_shift(
            G_targets[j + 1], shift_dt, delta_s, K, include_overflow=include_overflow
        )

        if gamma_placement == "compounded":
            near, _ = restrict_to_interval(
                target_pmfs[j], gamma_dt, delta_s, K, include_overflow=include_overflow
            )
            recursive = near + gamma_j * G_next_shifted
        else:
            recursive = G_next_shifted

        G_j = (1.0 - lambda_j) * T_p_j + lambda_j * recursive

        # Mass is exact by construction. Assert; never renormalize.
        if __debug__:
            total = float(torch.sum(G_j).item())
            if abs(total - 1.0) > 1e-4:
                raise AssertionError(
                    "lambda-return lost mass at j=%d: sum=%.6f "
                    "(arm=%s, placement=%s, lam_j=%.4f, gamma_j=%.4f)"
                    % (j, total, arm, gamma_placement, lambda_j, gamma_j)
                )

        G_targets[j] = G_j.detach()

    return torch.stack(G_targets, dim=0), step_weights


def compute_multistep_lambda_returns(
    target_pmfs: torch.Tensor,
    target_survivals: torch.Tensor,
    dts: torch.Tensor,
    events: torch.Tensor,
    tte: float,
    tau_event: float = None,
    lam: float = 0.6,
    delta_s: float = 1.0,
    K: int = 30,
    censor_ipcw_weight: float = 1.0,
    include_overflow: bool = False,
    arm: str = "full",
    gamma_placement: str = "bootstrap",
    censoring_aware_target: bool = True,
):
    """
    Back-compatible delegate to `compute_lambda_returns`.

    `events` is reduced to a trajectory-level indicator and `tau_event` is ignored:
    interval-event locations are derived from residual times. That pair is what
    produced defect D9 and is retained only so existing call sites keep working.
    """
    has_event = bool(torch.any(events > 0.5).item())
    return compute_lambda_returns(
        target_pmfs, target_survivals, dts, tte, has_event,
        lam=lam, delta_s=delta_s, K=K, censor_ipcw_weight=censor_ipcw_weight,
        include_overflow=include_overflow, arm=arm, gamma_placement=gamma_placement,
        censoring_aware_target=censoring_aware_target,
    )


def squared_cramer_distance_loss(
    pred_cdf: torch.Tensor,
    target_cdf: torch.Tensor,
    weights: torch.Tensor = None,
    delta_s: float = 1.0,
) -> torch.Tensor:
    """
    Squared Cramer distance between lifetime CDFs, as a Riemann sum over the
    lifetime axis:

        L_j = delta_s * sum_{k=0}^{K-1} (F_pred(k) - F_target(k))^2

    This is the actual Cramer distance and it carries units of time. The previous
    implementation used `mean` over K instead of `delta_s * sum` (defect D14), which
    made the loss magnitude depend on the bin count and left it dimensionless, so
    losses were not comparable across cohorts -- delta_s spans 0.5 to 100 here.
    Expressing it as an integral is also what lets the censored-CRPS anchor share
    the same scale, which is what makes the anchor weight alpha a genuine convex
    mixing weight rather than an extra tuning knob.

    Sub-unit CDFs are handled unchanged: with an explicit overflow coordinate the
    CDF legitimately terminates at 1 - p_perp.

    Args:
        pred_cdf: (..., K)
        target_cdf: (..., K) detached
        weights: optional scalar weights per step (...,), e.g. IPCW
        delta_s: bin width, in the cohort's time unit
    """
    diff_sq = (pred_cdf - target_cdf) ** 2
    per_step_loss = float(delta_s) * torch.sum(diff_sq, dim=-1)

    if weights is not None:
        per_step_loss = per_step_loss * weights

    return torch.mean(per_step_loss)
