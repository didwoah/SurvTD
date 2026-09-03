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
