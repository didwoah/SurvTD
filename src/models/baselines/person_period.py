"""
Baseline: Person-Period Expanded Discrete Hazard MLE (Rung 1 Naive)

What was wrong
--------------
The shipped implementation supervised exactly ONE hazard bin per visit,
`k_step = round(dt_j / delta_s)` -- the bin of the ELAPSED INTERVAL rather than of
the residual time to event. On the sepsis cohort that is k_step ~ 1, so bins 2..K-1
were never trained, yet evaluation read bin K//2. The reported C-index was 0.263 on
sepsis and 0.100 on C-MAPSS, both far below chance.

A below-chance Rung-1 baseline is not a weak baseline, it is a bug, and it
invalidates two things at once: the "baseline ladder parity" claim of
preregistration section 3, and every gain-retention denominator in Table 3, all of
which were normalized against this model.

What this is now
----------------
The textbook person-period expansion. Each visit j contributes one row per discrete
interval the subject is still at risk in, as seen from that visit:

    k_j            = floor(r_j / delta_s),  r_j = tte - times[j]
    risk_mask[j,k] = 1{k <= k_j}      subject is at risk in bin k
    labels[j,k]    = 1{k == k_j}      only if the trajectory ended in an event

and the loss is the masked binary cross-entropy over those rows. This is the
discrete-time hazard MLE, equivalent to the discrete survival negative
log-likelihood, and it handles right-censoring natively: a censored subject simply
contributes all-zero labels over the bins it survived.

It also shares `DiscreteHazardHead` with the rest of the ladder rather than keeping
a private copy, so the bin-index convention is identical across contenders (D12).

On the "1-hour regular grid"
----------------------------
The old docstring and preregistration section 3 Rung 1 both describe expanding the
irregular visits onto a uniform forward-filled grid, which the code never did. That
expansion is a data transform, not a model concern: see
`src.data.dataset.expand_to_regular_grid`, which the runner applies to this
baseline's inputs. Implementing it rather than dropping the claim is the
conservative choice, since it makes Rung 1 stronger.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbones import build_backbone
from src.models.hazard_head import DiscreteHazardHead
from src.operators.survtd_operator import residual_times


def person_period_targets(
    residual: torch.Tensor,
    event: bool,
    delta_s: float,
    K: int,
) -> tuple:
    """
    At-risk mask and event labels for the person-period expansion.

    Args:
        residual: (L,) residual time from each visit, r_j = tte - times[j]
        event: whether the trajectory terminated in an event
        delta_s, K: lifetime grid
    Returns:
        (risk_mask, labels), both (L, K) float tensors.

    Visits whose residual time is non-positive (the recorded event precedes the
    observation) contribute no rows: there is nothing left to predict from there.
    A residual time beyond the horizon leaves the subject at risk across all K bins
    with no event bin, which is the correct treatment of horizon truncation.
    """
    device = residual.device
    dtype = residual.dtype
    L = residual.shape[0]

    ks = torch.arange(K, device=device, dtype=dtype)                 # (K,)
    k_j = torch.floor(residual / delta_s)                            # (L,)

    valid = (residual > 0).to(dtype).unsqueeze(1)                    # (L, 1)
    risk_mask = (ks.unsqueeze(0) <= k_j.unsqueeze(1)).to(dtype) * valid

    labels = torch.zeros(L, K, device=device, dtype=dtype)
    if event:
        # The event bin is only observed when it falls inside the horizon.
        inside = (k_j >= 0) & (k_j < K) & (residual > 0)
        idx = torch.nonzero(inside, as_tuple=False).flatten()
        if idx.numel() > 0:
            labels[idx, k_j[idx].long()] = 1.0

    return risk_mask, labels


class PersonPeriodModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_bins: int = 30,
        delta_s: float = 1.0,
        backbone_type: str = "grud",
        num_layers: int = 2,
        dropout: float = 0.1,
        include_overflow: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = num_bins
        self.delta_s = delta_s
        self.include_overflow = include_overflow

        self.backbone = build_backbone(backbone_type, input_dim, hidden_dim, num_layers, dropout)
        # Shared head, not a private copy. Preregistration section 3 promises identical
        # backbones across the ladder; an identical head and an identical bin-index
        # convention are a free strengthening of that parity claim, and the private
        # copy was one source of the divergent conventions in defect D12.
        self.head = DiscreteHazardHead(hidden_dim, num_bins, delta_s,
                                       include_overflow=include_overflow)

    def forward(self, x, dts, mask=None):
        """
        Returns:
            hazard:   (B, L, K)
            survival: (B, L, K)
            pmf:      (B, L, K+1) when include_overflow
            cdf:      (B, L, K)
        """
        h = self.backbone(x, dts, mask)
        return self.head(h)

    def compute_loss(self, x, dts, events, tte, mask=None):
        """
        Masked binary cross-entropy over the at-risk person-period rows.
        """
        hazard, _, _, _ = self.forward(
            x.unsqueeze(0), dts.unsqueeze(0),
            mask.unsqueeze(0) if mask is not None else None
        )
        hazard = hazard.squeeze(0)                                   # (L, K)

        if isinstance(events, (int, float, bool)):
            has_event = bool(events > 0.5)
        elif torch.is_tensor(events) and events.numel() == 1:
            has_event = bool((events > 0.5).item())
        else:
            has_event = bool(torch.any(events > 0.5).item())
        r = residual_times(dts, tte)
        risk_mask, labels = person_period_targets(r, has_event, self.delta_s, self.K)

        n_rows = risk_mask.sum()
        if float(n_rows.item()) <= 0.0:
            # No at-risk rows (every recorded residual time is non-positive). Return a
            # differentiable zero rather than a bare constant, so the caller's backward
            # pass stays valid.
            return hazard.sum() * 0.0

        h = torch.clamp(hazard, min=1e-6, max=1.0 - 1e-6)
        bce = F.binary_cross_entropy(h, labels, reduction="none")
        return torch.sum(bce * risk_mask) / n_rows
