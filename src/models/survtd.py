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
from src.operators.anchors import censored_crps_anchor
from src.operators.survtd_operator import (
    compute_multistep_lambda_returns,
    residual_times,
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
        lam: float = 0.6,
        include_overflow: bool = True,
        gamma_placement: str = "bootstrap",
        alpha_anchor: float = 0.5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_bins
        self.delta_s = delta_s
        self.tau_ema = tau_ema
        self.lam = lam
        self.include_overflow = include_overflow
        # 'bootstrap' keeps lambda=1 equal to Monte Carlo; 'compounded' matches C_2's
        # prose but breaks that endpoint. Both reported; see deviation log A-15.
        self.gamma_placement = gamma_placement
        # Convex weight on the ground-truth anchor. PROVISIONAL: standing rule 1
        # requires alpha to be selected exactly once on the validation split
        # (lambda=0.6, seed 42, alpha in {0, .25, .5, .75, 1}) and then frozen for
        # every lambda, arm, cohort and seed. Until that selection runs, treat this
        # default as a placeholder, not a tuned value.
        self.alpha_anchor = float(alpha_anchor)

        # Online Network
        self.backbone = build_backbone(backbone_type, input_dim, hidden_dim, num_layers, dropout)
        self.head = DiscreteHazardHead(hidden_dim, num_bins, delta_s,
                                       include_overflow=include_overflow)

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
        ablation_mode: str = "full",
        alpha_anchor: float = None,
        return_parts: bool = False
    ):
        """
        Hybrid loss for a single trajectory (L visits):

            (1 - alpha) * L_TD  +  alpha * L_anchor

        Both components are always available via `return_parts`. Logging them
        separately is not optional bookkeeping: without it there is no way to tell
        convergence from the anchor swamping the TD term, and that blindness is
        what let the self-distillation collapse persist unnoticed.
        """
        if lam is None:
            lam = self.lam
        alpha = self.alpha_anchor if alpha_anchor is None else float(alpha_anchor)

        # 1. Online Forward Pass
        hazard_on, surv_on, pmf_on, cdf_on = self.forward(x.unsqueeze(0), dts.unsqueeze(0), mask.unsqueeze(0) if mask is not None else None)
        cdf_on = cdf_on.squeeze(0)  # (L, K)

        # 2. Target Network Forward Pass (no_grad)
        with torch.no_grad():
            _, surv_tgt, pmf_tgt, _ = self.forward_target(x.unsqueeze(0), dts.unsqueeze(0), mask.unsqueeze(0) if mask is not None else None)
            surv_tgt = surv_tgt.squeeze(0)  # (L, K)
            pmf_tgt = pmf_tgt.squeeze(0)    # (L, K)

            # One recursion for every arm. The previous branch dispatched "full" and
            # the ablations to two separate implementations, which is how the gamma
            # defect ended up duplicated and why an arm could silently diverge from
            # the full model. Arms are now data (ARMS), not control flow.
            G_targets, step_weights = compute_multistep_lambda_returns(
                pmf_tgt, surv_tgt, dts, events, tte, tau_event,
                lam=lam, delta_s=self.delta_s, K=self.K, censor_ipcw_weight=ipcw_weight,
                include_overflow=self.include_overflow,
                arm=ablation_mode, gamma_placement=self.gamma_placement
            )

            # Cumulate over the K genuine bins only. With an explicit overflow
            # coordinate the CDF legitimately stops at 1 - p_perp, so the old
            # clamp(max=1.0) would now mask a real invariant rather than enforce one.
            G_cdf = torch.cumsum(G_targets[..., :self.K], dim=-1)

        # 3. TD term: squared Cramér distance to the bootstrapped target.
        loss_td = squared_cramer_distance_loss(
            cdf_on, G_cdf, weights=step_weights, delta_s=self.delta_s
        )

        # 4. Anchor term: per-visit right-censored CRPS against the observed
        #    residual time. This is the ground truth the shipped objective lacked
        #    everywhere except one (misplaced) terminal Dirac.
        has_event = bool(torch.any(events > 0.5).item())
        r = residual_times(dts, tte)
        loss_anchor = censored_crps_anchor(
            cdf_on, r, has_event, self.delta_s, self.K,
            ipcw_weight=ipcw_weight
        ).mean()

        loss = (1.0 - alpha) * loss_td + alpha * loss_anchor
        if return_parts:
            return loss, {"td": loss_td, "anchor": loss_anchor, "alpha": alpha}
        return loss
