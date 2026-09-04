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
from src.operators.anchors import censored_crps_anchor, truncated_censoring_target
from src.operators.survtd_operator import (
    interval_gaps,
    localized_projected_dirac,
    residual_times,
    squared_cramer_distance_loss,
)


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
        clamp_eps: float = 1e-3,
        include_overflow: bool = True,
        alpha_anchor: float = 0.5
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_bins
        self.delta_s = delta_s
        self.tau_ema = tau_ema
        self.clamp_eps = clamp_eps
        self.include_overflow = include_overflow
        # MUST equal SurvTD's alpha. If the clamped arm does not receive the same
        # ground-truth supervision, EXP-05 measures who got supervised rather than
        # which operator is better, and the baseline is handicapped by construction.
        self.alpha_anchor = float(alpha_anchor)
        # Rates that evidence (or refute) the numerical-degeneracy claim behind C_1.
        self.clamp_bound_count = 0
        self.uniform_fallback_count = 0
        self.target_count = 0

        # Online network
        self.backbone = build_backbone(backbone_type, input_dim, hidden_dim, num_layers, dropout)
        self.head = DiscreteHazardHead(hidden_dim, num_bins, delta_s,
                                       include_overflow=include_overflow)

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

    def compute_loss_trajectory(self, x, dts, events, tte, tau_event, mask=None,
                                alpha_anchor: float = None, return_parts: bool = False,
                                ipcw_weight: float = 1.0, event: bool = None):
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

            # D8/D9 fixes mirrored from src/operators/survtd_operator.py. NC-A3 is only
            # a fair comparison if the clamped arm receives the same correctly aligned
            # durations and the same correctly placed ground truth as the full model;
            # otherwise EXP-05 measures a bug rather than the clamped operator.
            # D15: see src/models/survtd.py -- `events` is an all-zero placeholder; the
            # authoritative flag reaches us through tau_event (tte for an event).
            has_event = (bool(event) if event is not None
                         else bool(torch.any(events > 0.5).item()))
            r_np = residual_times(dts, tte).detach().cpu().numpy()
            gaps_np = interval_gaps(dts).detach().cpu().numpy()

            # Terminal step: Dirac at the residual time tte - t_{L-1}, not at dt_last.
            if has_event:
                G_targets[-1] = localized_projected_dirac(
                    max(0.0, float(r_np[-1])), self.delta_s, self.K, device=device,
                    include_overflow=self.include_overflow
                )
            else:
                # Same censoring-aware terminal target as the full model, for the
                # same reason: a self-copy carries no information (defect D-CENS).
                G_targets[-1] = truncated_censoring_target(
                    pmf_tgt[-1], max(0.0, float(r_np[-1])), self.delta_s, self.K,
                    include_overflow=self.include_overflow
                )

            # 1-step backward recursion with clamped division
            for j in range(L - 2, -1, -1):
                gap_j = float(gaps_np[j])            # times[j+1] - times[j]
                event_in_interval = bool(has_event and r_np[j] > 0.0 >= r_np[j + 1])

                if event_in_interval:
                    G_targets[j] = localized_projected_dirac(
                        max(0.0, float(r_np[j])), self.delta_s, self.K, device=device,
                        include_overflow=self.include_overflow
                    )
                else:
                    G_targets[j], diag = clamped_division_target(
                        pmf_tgt[j + 1], surv_tgt[j], gap_j, self.delta_s, self.K,
                        self.clamp_eps, include_overflow=self.include_overflow
                    )
                    self.target_count += 1
                    self.clamp_bound_count += int(diag["clamp_bound"])
                    self.uniform_fallback_count += int(diag["uniform_fallback"])

            G_tensor = torch.stack(G_targets, dim=0).detach()
            G_cdf = torch.cumsum(G_tensor[..., :self.K], dim=-1)

        alpha = self.alpha_anchor if alpha_anchor is None else float(alpha_anchor)
        assert 0.0 <= alpha <= 1.0, f"alpha_anchor must be in [0, 1], got {alpha}"
        loss_td = squared_cramer_distance_loss(cdf_on, G_cdf, delta_s=self.delta_s)
        loss_anchor = censored_crps_anchor(
            cdf_on, residual_times(dts, tte), has_event, self.delta_s, self.K,
            ipcw_weight=ipcw_weight
        ).mean()
        loss = (1.0 - alpha) * loss_td + alpha * loss_anchor
        if return_parts:
            return loss, {"td": loss_td, "anchor": loss_anchor, "alpha": alpha}
        return loss

    def degeneracy_rates(self) -> dict:
        """EXP-05 / E5c: how often the clamp actually binds. If it never binds on a
        cohort, the numerical degeneracy this baseline is meant to exhibit is absent
        there, and a null gradient-stability result is expected rather than
        informative."""
        n = max(1, self.target_count)
        return {
            "targets": self.target_count,
            "clamp_bound_rate": self.clamp_bound_count / n,
            "uniform_fallback_rate": self.uniform_fallback_count / n,
        }
