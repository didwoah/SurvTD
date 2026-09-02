"""
Baseline: Person-Period Expanded Discrete Hazard MLE (Rung 1 Naive)
- Expands irregular visits onto a uniform 1-hour forward-filled grid.
- Trains discrete hazard MLE via Binary Cross Entropy on interval events.
- Shares the identical GRU-D/LSTM sequence backbone for strict parity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone


class PersonPeriodModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_bins: int = 30,
        delta_s: float = 1.0,
        backbone_type: str = "grud",
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_bins
        self.delta_s = delta_s

        self.backbone = build_backbone(backbone_type, input_dim, hidden_dim, num_layers, dropout)
        # Emits K interval hazards
        self.hazard_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.K)
        )

    def forward(self, x, dts, mask=None):
        """
        Returns:
            hazard: (B, L, K)
            survival: (B, L, K)
            pmf: (B, L, K)
            cdf: (B, L, K)
        """
        h = self.backbone(x, dts, mask)
        logits = self.hazard_head(h)
        hazard = torch.sigmoid(logits)
        survival = torch.cumprod(torch.clamp(1.0 - hazard, min=1e-6, max=1.0), dim=-1)
        ones = torch.ones_like(survival[..., :1])
        s_prev = torch.cat([ones, survival[..., :-1]], dim=-1)
        pmf = hazard * s_prev
        if self.K > 1:
            pmf = torch.cat([pmf[..., :-1], survival[..., -2:-1]], dim=-1)
        pmf = pmf / torch.clamp(torch.sum(pmf, dim=-1, keepdim=True), min=1e-8)
        cdf = torch.cumsum(pmf, dim=-1)
        cdf = torch.clamp(cdf, min=0.0, max=1.0)
        return hazard, survival, pmf, cdf

    def compute_loss(self, x, dts, events, tte, mask=None):
        """
        Binary Cross Entropy on whether event occurred in each observation interval.
        """
        hazard, _, _, _ = self.forward(x.unsqueeze(0), dts.unsqueeze(0), mask.unsqueeze(0) if mask is not None else None)
        hazard = hazard.squeeze(0)  # (L, K)

        L = x.shape[0]
        loss_total = torch.tensor(0.0, device=x.device)

        for j in range(L):
            event_j = float(events[j].item())
            dt_j = float(dts[j].item())
            k_step = min(int(round(dt_j / self.delta_s)), self.K - 1)
            h_j = hazard[j, k_step]

            if event_j > 0.5:
                loss_step = -torch.log(torch.clamp(h_j, min=1e-6))
            else:
                loss_step = -torch.log(torch.clamp(1.0 - h_j, min=1e-6))
            loss_total = loss_total + loss_step

        return loss_total / max(1, L)
