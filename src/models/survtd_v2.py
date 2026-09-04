"""
SurvTD-v2 Model Architecture:
- Shared continuous-time backbone (GRU-D or ContinuousLSTM) + Discrete Hazard Head.
- Bounded Logit Anchor (BLA): Strictly bounded gradients in [-1, 1], 0% clipping saturation.
- Cumulative Hazard Huber Matching (CHHM) with Continuous Generator Discount (beta = exp(-rho * dt)).
- Martingale Terminal Censoring Likelihood (no mass dumped to perp).
- Frozen EMA target network for Lyapunov two-timescale stability.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone
from src.models.hazard_head import DiscreteHazardHead
from src.operators.survtd_operator import (
    categorical_projection_shift,
    residual_times
)


class SurvTD_v2_Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_bins: int = 40,
        delta_s: float = 0.1,
        backbone_type: str = "grud",
        num_layers: int = 1,
        dropout: float = 0.1,
        tau_ema: float = 0.95,
        rho: float = 0.5,
        alpha_anchor: float = 0.5,
        include_overflow: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_bins
        self.delta_s = delta_s
        self.rho = rho
        self.alpha_anchor = alpha_anchor
        self.tau_ema = tau_ema
        self.include_overflow = include_overflow

        # Online Network
        self.backbone = build_backbone(
            backbone_type=backbone_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.head = DiscreteHazardHead(
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            delta_s=delta_s,
            include_overflow=include_overflow
        )

        # Target Network (Frozen / EMA)
        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_head = copy.deepcopy(self.head)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_head.parameters():
            p.requires_grad = False

    def forward(self, x, dts, mask=None):
        """
        x: (B, L, D)
        dts: (B, L)
        """
        h = self.backbone(x, dts, mask=mask)
        hazard, survival, pmf, cdf = self.head(h)
        return hazard, survival, pmf, cdf

    def forward_target(self, x, dts, mask=None):
        with torch.no_grad():
            h = self.target_backbone(x, dts, mask=mask)
            hazard, survival, pmf, cdf = self.target_head(h)
            return hazard, survival, pmf, cdf

    def update_target_network(self):
        """EMA update for target network."""
        with torch.no_grad():
            for p, p_tgt in zip(self.backbone.parameters(), self.target_backbone.parameters()):
                p_tgt.data.mul_(self.tau_ema).add_(p.data, alpha=1.0 - self.tau_ema)
            for p, p_tgt in zip(self.head.parameters(), self.target_head.parameters()):
                p_tgt.data.mul_(self.tau_ema).add_(p.data, alpha=1.0 - self.tau_ema)

    def compute_loss_trajectory(
        self,
        x: torch.Tensor,
        dts: torch.Tensor,
        events: torch.Tensor,
        tte: float,
        tau_event: float,
        mask: torch.Tensor = None
    ):
        """
        Computes SurvTD-v2 loss for a single trajectory of length L.
        """
        device = x.device
        L = x.shape[0]

        # 1. Online Forward Pass
        h_on = self.backbone(x.unsqueeze(0), dts.unsqueeze(0), mask=mask.unsqueeze(0) if mask is not None else None)
        # Get raw logits from head
        logits = self.head.net(h_on).squeeze(0)  # (L, K)
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(torch.clamp(1.0 - hazards, min=1e-6, max=1.0), dim=-1)  # (L, K)

        # 2. Target Network Forward Pass
        with torch.no_grad():
            h_tgt = self.target_backbone(x.unsqueeze(0), dts.unsqueeze(0), mask=mask.unsqueeze(0) if mask is not None else None)
            hazard_tgt, surv_tgt, pmf_tgt, _ = self.target_head(h_tgt)
            surv_tgt = surv_tgt.squeeze(0)  # (L, K)
            pmf_tgt = pmf_tgt.squeeze(0)    # (L, K+1 or K)

        # 3. Bounded Logit Anchor (BLA) - Vectorized
        r = residual_times(dts, tte)  # (L,)
        # D15: `events` is an all-zero placeholder in every loader; the authoritative
        # flag arrives via tau_event (tte for an event, tte + 100 for a censored one).
        has_event = (bool(torch.any(events > 0.5).item())
                     or (tau_event is not None and float(tau_event) <= float(tte) + 1e-9))
        k_target = torch.clamp((r / self.delta_s).long(), 0, self.K - 1)  # (L,)

        grid_k = torch.arange(self.K, device=device).unsqueeze(0)  # (1, K)
        kt_col = k_target.unsqueeze(1)  # (L, 1)

        survived_mask = (grid_k < kt_col)  # (L, K)
        event_mask = (grid_k == kt_col) & has_event  # (L, K)
        active_mask = survived_mask | event_mask  # (L, K)

        target_matrix = torch.zeros((L, self.K), device=device)
        target_matrix[event_mask] = 1.0

        bce_loss = F.binary_cross_entropy_with_logits(logits, target_matrix, reduction='none')
        loss_anchor = (bce_loss * active_mask.float()).sum() / max(1.0, active_mask.float().sum())

        # If right-censored and at terminal step: Martingale log-likelihood
        if (not has_event) and (k_target[-1].item() < self.K):
            kt_last = k_target[-1].item()
            loss_anchor = loss_anchor + torch.sum(F.softplus(logits[-1, :kt_last+1])) / (kt_last + 1)

        # 4. Cumulative Hazard Huber Matching (CHHM) with Continuous Generator - Vectorized
        if L > 1:
            dt_next = float(dts[1].item())  # uniform gap across sequence
            beta_val = math.exp(-self.rho * dt_next)

            shifted_pmf = categorical_projection_shift(
                pmf_tgt[1:], dt_next, self.delta_s, self.K,
                include_overflow=self.include_overflow
            )
            shifted_surv = 1.0 - torch.cumsum(shifted_pmf[:, :self.K], dim=-1)
            shifted_surv = torch.clamp(shifted_surv, min=0.0, max=1.0)

            td_err = survival[:-1] - shifted_surv
            loss_td = F.smooth_l1_loss(td_err, torch.zeros_like(td_err), beta=0.1) * (1.0 - beta_val)
        else:
            loss_td = torch.tensor(0.0, device=device)

        alpha = self.alpha_anchor
        total_loss = (1.0 - alpha) * loss_td + alpha * loss_anchor
        return total_loss
