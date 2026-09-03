"""
Discrete Hazard Head and Lifetime Support Grid:
- Discrete lifetime grid of K bins with uniform width delta_s.
- Emits conditional hazards h_j(s), survival curves S_j(s), PMF p_j(s), and CDF F_j(s).
- Optional explicit overflow coordinate for "R >= K * delta_s" (see below).

Why the overflow coordinate exists
----------------------------------
The original head forced `pmf[K-1] = S[K-2]`, so the last bin carried both "the event
happens in bin K-1" and "the event never happens within the horizon". Combined with
`categorical_projection_shift`, which routed all shifted-past-the-end mass into bin
K-1, that bin became an absorbing attractor: a self-referential TD target could park
all of its mass there and remain perfectly self-consistent. That is the degenerate
fixed point the bootstrapped objective was drifting towards.

With `include_overflow=True` the two meanings are separated. Coordinates 0..K-1 are
genuine bins and coordinate K is a distinct absorbing symbol
    perp = "R >= K * delta_s"  (survived beyond the modelled horizon).
Telescoping then gives total mass exactly 1 with no renormalization:

    sum_{k<K} (S[k-1] - S[k])  +  S[K-1]  =  (1 - S[K-1]) + S[K-1]  =  1

`cdf` stays length K in both modes, so it now terminates at `1 - p_perp <= 1`. Those
sub-unit CDFs are legitimate -- they come from horizon truncation, not from the
duration discount -- and the squared Cramer loss handles them unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteHazardHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_bins: int = 30,
        delta_s: float = 1.0,
        include_overflow: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K = num_bins
        self.delta_s = delta_s
        self.include_overflow = include_overflow

        # Grid center points: s_k = (k + 0.5) * delta_s
        bin_centers = (torch.arange(num_bins, dtype=torch.float32) + 0.5) * delta_s
        self.register_buffer("bin_centers", bin_centers)

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.K)
        )

    def forward(self, h):
        """
        Args:
            h: (B, L, hidden_dim) or (B, hidden_dim)
        Returns:
            hazard:   (..., K)     conditional hazard in (0, 1)
            survival: (..., K)     S[k] = P(R > (k+1) * delta_s); S[-1] is the perp mass
            pmf:      (..., K+1)   if include_overflow else (..., K); sums to 1 exactly
            cdf:      (..., K)     terminates at 1 - p_perp (at 1.0 without overflow)
        """
        logits = self.net(h)
        hazard = torch.sigmoid(logits)

        # Survival curve S[k] = prod_{m<=k} (1 - hazard[m]) = P(R > (k+1) * delta_s)
        survival = torch.cumprod(torch.clamp(1.0 - hazard, min=1e-6, max=1.0), dim=-1)

        # S[-1] = 1.0 shifted in, so s_prev[k] = P(R > k * delta_s)
        ones = torch.ones_like(survival[..., :1])
        s_prev = torch.cat([ones, survival[..., :-1]], dim=-1)

        # Telescoping bin masses: p[k] = hazard[k] * s_prev[k] = s_prev[k] - survival[k]
        bin_mass = hazard * s_prev

        if self.include_overflow:
            # Every bin is a genuine bin; the residual survival becomes its own symbol.
            pmf = torch.cat([bin_mass, survival[..., -1:]], dim=-1)
        else:
            # Legacy behaviour: the final bin absorbs the beyond-horizon mass, so it
            # conflates "event in bin K-1" with "no event within the horizon".
            if self.K > 1:
                pmf = torch.cat([bin_mass[..., :-1], survival[..., -2:-1]], dim=-1)
            else:
                pmf = ones

        # Mass is exact by telescoping. Assert rather than renormalize: the old
        # `pmf / pmf.sum()` is what allowed a mass leak to pass unnoticed.
        if __debug__:
            total = torch.sum(pmf, dim=-1)
            if not torch.allclose(total, torch.ones_like(total), atol=1e-4):
                raise AssertionError(
                    f"hazard head pmf does not sum to 1 (max deviation "
                    f"{float((total - 1.0).abs().max()):.3e})"
                )

        cdf = torch.cumsum(pmf[..., :self.K], dim=-1)
        return hazard, survival, pmf, cdf

    @torch.no_grad()
    def init_prior_bias(self, hazards, weight_std: float = 1e-4, eps: float = 1e-4):
        """
        Set the final projection so the head emits a given marginal hazard curve at
        step 0:  b_k = logit(h_k), with the weight shrunk to ~0.

        Why this exists (measured, not assumed): under PyTorch's default Linear init
        the logits sit at ~0, so hazard ~ 0.4999 in every bin and S(K * delta_s)
        ~ 1.4e-11. The duration discount gamma_j = S(Delta t_j) is then ~0.48 at the
        median inter-visit gap and ~0.09 at the p90 gap, against a cohort truth of
        ~0.98 / ~0.93. Since gamma is the contraction modulus of the renewal
        operator, the effective credit-assignment horizon 1/(1 - gamma) collapses
        from ~50 steps to ~2, and the terminal Dirac -- the only ground truth in the
        objective at alpha = 0 -- is annihilated before it can propagate backwards.

        Args:
            hazards: (K,) marginal per-bin hazards, fit on the TRAINING split only
            weight_std: std of the final weight matrix. Near-zero means the model
                starts at the population curve and learns deviations from it.
            eps: clamp on the hazards before the logit
        """
        h = torch.as_tensor(hazards, dtype=torch.float32).clamp(eps, 1.0 - eps)
        if h.numel() != self.K:
            raise ValueError(f"expected {self.K} hazards, got {h.numel()}")
        final = self.net[-1]
        final.bias.copy_(torch.log(h / (1.0 - h)).to(final.bias.device))
        final.weight.normal_(0.0, weight_std)

    @torch.no_grad()
    def init_constant_bias(self, b: float = -3.5, weight_std: float = 1e-4):
        """
        Data-free variant of `init_prior_bias`: a constant optimistic survival bias.
        sigmoid(-3.5) ~ 0.029 per bin, keeping gamma_j well away from 0 without
        touching the training split at all.
        """
        final = self.net[-1]
        final.bias.fill_(float(b))
        final.weight.normal_(0.0, weight_std)


def apply_hazard_prior_init(model, mode: str, train_dataset=None, delta_s: float = None):
    """
    Apply an initialization scheme to any model exposing `.head: DiscreteHazardHead`,
    and RE-SYNC the frozen target head if the model has one.

    The resync is load-bearing. SurvTD and DeepTCSR build `target_head` by
    `copy.deepcopy(self.head)` inside `__init__`, so initializing the online head
    afterwards would leave the target network holding the original collapsed bias --
    and the target network is precisely what supplies gamma_j and the bootstrapped
    target. Fixing only the online head would fix nothing.

    Args:
        model: any of SurvTD / DeepTCSR-Clamped / Dynamic-DeepHit / Person-Period
        mode: 'default' (no-op), 'optimistic' (b = -3.5), or 'km_prior'
        train_dataset: required for 'km_prior'; the TRAINING split only
        delta_s: required for 'km_prior'
    Returns:
        dict describing what was applied, for the run record.
    """
    if mode == "default":
        return {"init": "default"}

    head = getattr(model, "head", None)
    if not isinstance(head, DiscreteHazardHead):
        raise TypeError(f"{type(model).__name__} has no DiscreteHazardHead at .head")

    if mode == "optimistic":
        head.init_constant_bias(-3.5)
        record = {"init": "optimistic", "b": -3.5}
    elif mode == "km_prior":
        if train_dataset is None or delta_s is None:
            raise ValueError("km_prior needs train_dataset and delta_s")
        from src.evaluation.censoring import marginal_residual_hazards
        hazards = marginal_residual_hazards(train_dataset, head.K, delta_s)
        head.init_prior_bias(hazards)
        record = {"init": "km_prior", "eps": 1e-4,
                  "hazard_min": float(hazards.min()), "hazard_max": float(hazards.max())}
    else:
        raise ValueError(f"unknown init mode {mode!r}")

    target_head = getattr(model, "target_head", None)
    if target_head is not None:
        target_head.load_state_dict(head.state_dict())
        record["target_head_resynced"] = True

    return record
