"""
Baseline: Dynamic-DeepHit (Lee et al., 2020) in Native PyTorch
- Shares identical GRU-D/LSTM continuous sequence backbone.
- Discrete hazard output with terminal survival negative log-likelihood (L1) and pairwise ranking loss (L2).
- Zero consecutive temporal consistency constraints, reproducing bedside alarm jittering.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone
from src.models.hazard_head import DiscreteHazardHead


class DynamicDeepHitModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_bins: int = 30,
        delta_s: float = 1.0,
        backbone_type: str = "grud",
        num_layers: int = 2,
        dropout: float = 0.1,
        alpha_rank: float = 0.5,
        sigma: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_bins
        self.delta_s = delta_s
        self.alpha_rank = alpha_rank
        self.sigma = sigma

        self.backbone = build_backbone(backbone_type, input_dim, hidden_dim, num_layers, dropout)
        self.head = DiscreteHazardHead(hidden_dim, num_bins, delta_s)

    def forward(self, x, dts, mask=None):
        """
        Returns:
            hazard: (B, L, K)
            survival: (B, L, K)
            pmf: (B, L, K)
            cdf: (B, L, K)
        """
        h = self.backbone(x, dts, mask)
        return self.head(h)

    def compute_loss(self, batch_patients):
        """
        Computes L1 (terminal NLL) + alpha * L2 (ranking loss) over a batch of patient trajectories.
        """
        device = next(self.parameters()).device
        total_l1 = torch.tensor(0.0, device=device)
        total_steps = 0

        patient_evals = []

        for p in batch_patients:
            x = p['features'].unsqueeze(0).to(device)
            dts = p['dts'].unsqueeze(0).to(device)
            mask = p['mask'].unsqueeze(0).to(device) if 'mask' in p and p['mask'] is not None else None
            tte = float(p['tte'])
            event = float(p['event'])
            times = torch.cumsum(dts.squeeze(0), dim=0)

            hazard, survival, pmf, cdf = self.forward(x, dts, mask)
            pmf = pmf.squeeze(0)        # (L, K)
            survival = survival.squeeze(0)  # (L, K)
            cdf = cdf.squeeze(0)        # (L, K)

            L = pmf.shape[0]
            # Terminal NLL across all visits
            for j in range(L):
                t_j = float(times[j].item())
                rem_time = max(0.0, tte - t_j)
                k_bin = min(int(math.floor(rem_time / self.delta_s)), self.K - 1)

                if event > 0.5:
                    # Event occurred
                    p_val = torch.clamp(pmf[j, k_bin], min=1e-6)
                    loss_j = -torch.log(p_val)
                else:
                    # Right-censored
                    s_val = torch.clamp(survival[j, k_bin], min=1e-6)
                    loss_j = -torch.log(s_val)

                total_l1 = total_l1 + loss_j
                total_steps += 1

            # Save last visit cdf and event for pairwise ranking loss
            last_time = float(times[-1].item())
            patient_evals.append({
                'cdf_last': cdf[-1],
                'tte': tte,
                'last_time': last_time,
                'event': event
            })

        l1_loss = total_l1 / max(1, total_steps)

        # L2 Pairwise Dynamic Ranking Loss
        total_l2 = torch.tensor(0.0, device=device)
        rank_pairs = 0
        N = len(patient_evals)

        for i in range(N):
            for j in range(N):
                if i != j:
                    if patient_evals[i]['event'] > 0.5 and patient_evals[i]['tte'] < patient_evals[j]['tte']:
                        rem_i = max(0.0, patient_evals[i]['tte'] - patient_evals[i]['last_time'])
                        k_eval = min(int(math.floor(rem_i / self.delta_s)), self.K - 1)
                        # Patient i should have higher risk F_i(rem_i) than patient j
                        f_i = patient_evals[i]['cdf_last'][k_eval]
                        f_j = patient_evals[j]['cdf_last'][k_eval]
                        # Ranking penalty: eta(F_i - F_j)
                        penalty = torch.exp(-(f_i - f_j) / self.sigma)
                        total_l2 = total_l2 + penalty
                        rank_pairs += 1

        l2_loss = (total_l2 / max(1, rank_pairs)) if rank_pairs > 0 else torch.tensor(0.0, device=device)

        return l1_loss + self.alpha_rank * l2_loss
