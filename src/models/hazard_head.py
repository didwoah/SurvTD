"""
Discrete Hazard Head and Lifetime Support Grid:
- Discrete lifetime grid of K bins with uniform width delta_s.
- Emits conditional hazards h_j(s), survival curves S_j(s), PMF p_j(s), and CDF F_j(s).
- Uses absorbing terminal bin semantics guaranteeing exact unit probability mass: sum_{k=0}^{K-1} p_k = 1.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteHazardHead(nn.Module):
    def __init__(self, hidden_dim: int, num_bins: int = 30, delta_s: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K = num_bins
        self.delta_s = delta_s

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
            hazard: (..., K) conditional hazard in (0, 1)
            survival: (..., K) survival curve S(s)
            pmf: (..., K) probability mass function summing strictly to 1.0
            cdf: (..., K) cumulative distribution function terminating at 1.0
        """
        logits = self.net(h)
        # Hazard h(k) in (0, 1)
        hazard = torch.sigmoid(logits)

        # Survival curve S(k) = prod_{m=0}^k (1 - hazard(m))
        # Add epsilon for numerical safety
        survival = torch.cumprod(torch.clamp(1.0 - hazard, min=1e-6, max=1.0), dim=-1)

        # S(-1) = 1.0
        ones = torch.ones_like(survival[..., :1])
        s_prev = torch.cat([ones, survival[..., :-1]], dim=-1)

        # Telescoping PMF: p(k) = hazard(k) * S(k-1) = S(k-1) - S(k) for k < K - 1
        # For terminal bin K-1, absorbing semantics: p(K-1) = S(K-2)
        # Telescoping guarantees sum_{k=0}^{K-1} p(k) = 1.0 identically.
        pmf_standard = hazard * s_prev
        pmf_absorbing = survival[..., -2:-1] if self.K > 1 else ones

        if self.K > 1:
            pmf = torch.cat([pmf_standard[..., :-1], pmf_absorbing], dim=-1)
        else:
            pmf = ones

        # Ensure simplex normalization strictly
        pmf = pmf / torch.clamp(torch.sum(pmf, dim=-1, keepdim=True), min=1e-8)
        cdf = torch.cumsum(pmf, dim=-1)
        cdf = torch.clamp(cdf, min=0.0, max=1.0)

        return hazard, survival, pmf, cdf
