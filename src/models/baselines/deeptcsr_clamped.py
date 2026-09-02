"""
Baseline: DeepTCSR with Clamped Continuous Division (Rung 2 Published / NC-A3)
- Adapts DeepTCSR (2024) to continuous time with Delta t feature concatenation.
- Enforces backward consistency by dividing by clamped interval survival probability:
    p_target(s) = p_next(s - Delta t) / max(S(Delta t), 1e-3)
- Demonstrates gradient blowup and numerical instability in high-hazard regimes.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone
from src.models.hazard_head import DiscreteHazardHead
from src.operators.ablations import clamped_division_target
from src.operators.survtd_operator import localized_projected_dirac, squared_cramer_distance_loss


class DeepTCSRClampedModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_bins: int = 30,
        delta_s: float = 1.0,
        backbone_type: str = "grud",
        num_layers: int = 2,
        dropout: float = 0.1,
        tau_ema: float = 0.95,
        clamp_eps: float = 1e-3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_bins
        self.delta_s = delta_s
        self.tau_ema = tau_ema
        self.clamp_eps = clamp_eps

        # Online network
        self.backbone = build_backbone(backbone_type, input_dim, hidden_dim, num_layers, dropout)
        self.head = DiscreteHazardHead(hidden_dim, num_bins, delta_s)

        # Target network theta-
        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_head = copy.deepcopy(self.head)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_head.parameters():
            p.requires_grad = False

    def forward(self, x, dts, mask=None):
        h = self.backbone(x, dts, mask)
        return self.head(h)

    @torch.no_grad()
    def forward_target(self, x, dts, mask=None):
        h = self.target_backbone(x, dts, mask)
        return self.target_head(h)

    @torch.no_grad()
    def update_target_network(self):
        for p_online, p_target in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            p_target.data.mul_(self.tau_ema).add_(p_online.data, alpha=1.0 - self.tau_ema)
        for p_online, p_target in zip(self.head.parameters(), self.target_head.parameters()):
            p_target.data.mul_(self.tau_ema).add_(p_online.data, alpha=1.0 - self.tau_ema)

    def compute_loss_trajectory(self, x, dts, events, tte, tau_event, mask=None):
        """
        Computes 1-step backward consistency loss via clamped division.
        """
        device = x.device
        hazard_on, surv_on, pmf_on, cdf_on = self.forward(x.unsqueeze(0), dts.unsqueeze(0), mask.unsqueeze(0) if mask is not None else None)
        cdf_on = cdf_on.squeeze(0)  # (L, K)

        with torch.no_grad():
            _, surv_tgt, pmf_tgt, _ = self.forward_target(x.unsqueeze(0), dts.unsqueeze(0), mask.unsqueeze(0) if mask is not None else None)
            surv_tgt = surv_tgt.squeeze(0)
            pmf_tgt = pmf_tgt.squeeze(0)

            L = pmf_tgt.shape[0]
            G_targets = [None] * L

            # Terminal step
            if events[-1].item() > 0.5:
                dt_last = float(dts[-1].item())
                delta_tau = max(0.0, min(dt_last, tau_event - (tte - dt_last)))
                G_targets[-1] = localized_projected_dirac(delta_tau, self.delta_s, self.K, device=device)
            else:
                G_targets[-1] = pmf_tgt[-1].clone()

            # 1-step backward recursion with clamped division
            for j in range(L - 2, -1, -1):
                dt_j = float(dts[j].item())
                event_j = bool(events[j].item() > 0.5)

                if event_j:
                    delta_tau = max(0.0, dt_j * 0.5)
                    G_targets[j] = localized_projected_dirac(delta_tau, self.delta_s, self.K, device=device)
                else:
                    p_next = pmf_tgt[j + 1]
                    s_curr = surv_tgt[j]
                    G_targets[j] = clamped_division_target(p_next, s_curr, dt_j, self.delta_s, self.K, self.clamp_eps)

            G_tensor = torch.stack(G_targets, dim=0).detach()
            G_cdf = torch.cumsum(G_tensor, dim=-1)
            G_cdf = torch.clamp(G_cdf, min=0.0, max=1.0)

        loss = squared_cramer_distance_loss(cdf_on, G_cdf)
        return loss
