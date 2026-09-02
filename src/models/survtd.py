"""
SurvTD Model Architecture:
- Shared continuous-time backbone (GRU-D or ContinuousLSTM) + Discrete Hazard Head.
- Online network theta and frozen EMA target network theta-.
- Semi-gradient backward lambda-return target calculation and squared Cramer distance loss.
"""

import copy
import torch
import torch.nn as nn

from src.models.backbones import build_backbone
from src.models.hazard_head import DiscreteHazardHead
from src.operators.survtd_operator import (
    compute_multistep_lambda_returns,
    squared_cramer_distance_loss
)


class SurvTDModel(nn.Module):
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
        lam: float = 0.6
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_bins
        self.delta_s = delta_s
        self.tau_ema = tau_ema
        self.lam = lam

        # Online Network
        self.backbone = build_backbone(backbone_type, input_dim, hidden_dim, num_layers, dropout)
        self.head = DiscreteHazardHead(hidden_dim, num_bins, delta_s)

        # Target Network (theta-)
        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_head = copy.deepcopy(self.head)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_head.parameters():
            p.requires_grad = False

    def forward(self, x, dts, mask=None):
        """
        Forward pass through online network.
        Returns:
            hazard: (B, L, K)
            survival: (B, L, K)
            pmf: (B, L, K)
            cdf: (B, L, K)
        """
        h = self.backbone(x, dts, mask)
        hazard, survival, pmf, cdf = self.head(h)
        return hazard, survival, pmf, cdf

    @torch.no_grad()
    def forward_target(self, x, dts, mask=None):
        """
        Forward pass through frozen target network theta-.
        """
        h = self.target_backbone(x, dts, mask)
        hazard, survival, pmf, cdf = self.target_head(h)
        return hazard, survival, pmf, cdf

    @torch.no_grad()
    def update_target_network(self):
        """
        Polyak EMA update: theta- <- tau * theta- + (1 - tau) * theta
        """
        for p_online, p_target in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            p_target.data.mul_(self.tau_ema).add_(p_online.data, alpha=1.0 - self.tau_ema)
        for p_online, p_target in zip(self.head.parameters(), self.target_head.parameters()):
            p_target.data.mul_(self.tau_ema).add_(p_online.data, alpha=1.0 - self.tau_ema)

    def compute_loss_trajectory(
        self,
        x: torch.Tensor,
        dts: torch.Tensor,
        events: torch.Tensor,
        tte: float,
        tau_event: float,
        mask: torch.Tensor = None,
        lam: float = None,
        ipcw_weight: float = 1.0,
        ablation_mode: str = "full"
    ):
        """
        Computes Cramér loss for a single trajectory (L visits).
        """
        if lam is None:
            lam = self.lam

        # 1. Online Forward Pass
        hazard_on, surv_on, pmf_on, cdf_on = self.forward(x.unsqueeze(0), dts.unsqueeze(0), mask.unsqueeze(0) if mask is not None else None)
        cdf_on = cdf_on.squeeze(0)  # (L, K)

        # 2. Target Network Forward Pass (no_grad)
        with torch.no_grad():
            _, surv_tgt, pmf_tgt, _ = self.forward_target(x.unsqueeze(0), dts.unsqueeze(0), mask.unsqueeze(0) if mask is not None else None)
            surv_tgt = surv_tgt.squeeze(0)  # (L, K)
            pmf_tgt = pmf_tgt.squeeze(0)    # (L, K)

            if ablation_mode == "full":
                G_targets, step_weights = compute_multistep_lambda_returns(
                    pmf_tgt, surv_tgt, dts, events, tte, tau_event,
                    lam=lam, delta_s=self.delta_s, K=self.K, censor_ipcw_weight=ipcw_weight
                )
            else:
                from src.operators.ablations import compute_ablated_lambda_returns
                G_targets, step_weights = compute_ablated_lambda_returns(
                    pmf_tgt, surv_tgt, dts, events, tte, tau_event,
                    ablation_mode=ablation_mode, lam=lam, delta_s=self.delta_s, K=self.K, censor_ipcw_weight=ipcw_weight
                )

            G_cdf = torch.cumsum(G_targets, dim=-1)
            G_cdf = torch.clamp(G_cdf, min=0.0, max=1.0)

        # 3. Squared Cramér Distance Loss
        loss = squared_cramer_distance_loss(cdf_on, G_cdf, weights=step_weights)
        return loss
