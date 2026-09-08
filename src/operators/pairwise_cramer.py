"""
Continuous First-Order Stochastic Dominance (FSD) Pairwise Ranking in Cramér Space.

Reference: research/continuous-survival-td/notes/pairwise_cramer_stochastic_dominance.md

Mathematical Definition:
For any comparable pair (i, j) where Subject i fails before Subject j (T_i < T_j, E_i = 1):
First-order stochastic dominance requires S_j(t) >= S_i(t) <=> F_j(t) <= F_i(t) for all t.
Any state where F_j(t) > F_i(t) is a structural ranking violation.
The Cramér hinge loss penalizes:
    L_FSD(i, j) = delta_s * sum_{k=1}^K max(0, F_j(s_k) - F_i(s_k) + margin)^2
"""

import math
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def continuous_cramer_fsd_loss(
    cdf_i: torch.Tensor,       # Shape: (K,), predicted failure CDF for shorter-lived subject
    cdf_j: torch.Tensor,       # Shape: (K,), predicted failure CDF for longer-lived subject
    delta_s: float = 0.1,
    margin: float = 0.0
) -> torch.Tensor:
    """
    Computes First-Order Stochastic Dominance violation between two CDF curves.
    Since Subject j survives longer than Subject i, F_j should be <= F_i everywhere.
    Violation: max(0, F_j(s_k) - F_i(s_k) + margin)^2.
    """
    diff = cdf_j - cdf_i + margin
    violation = F.relu(diff).pow(2)
    return delta_s * violation.sum()


def compute_batch_pairwise_cramer_loss(
    patient_evals: List[Dict[str, any]],
    delta_s: float = 0.1,
    margin: float = 0.0,
    max_pairs: int = 256,
    device: str = "cpu"
) -> Tuple[torch.Tensor, int]:
    """
    Computes average pairwise Cramér FSD loss across all comparable pairs in a batch.

    Args:
        patient_evals: List of dicts with:
            - 'cdf_last': (K,) tensor of predicted CDF at latest observation
            - 'cdf_init': (K,) tensor of predicted CDF at initial observation t=0
            - 'tte': float, total survival/censoring time
            - 'last_time': float, timestamp of latest observation
            - 'event': int/float, 1 if event occurred, 0 if censored
        delta_s: bin width
        margin: optional separation margin
        max_pairs: maximum number of pairs to evaluate per batch for efficiency
        device: torch device

    Returns:
        (loss, num_pairs)
    """
    N = len(patient_evals)
    if N < 2:
        return torch.tensor(0.0, device=device), 0

    total_fsd = torch.tensor(0.0, device=device)
    num_pairs = 0

    for i in range(N):
        p_i = patient_evals[i]
        if p_i['event'] < 0.5:
            continue  # Shorter-lived subject must be an uncensored event

        tte_i = p_i['tte']

        for j in range(N):
            if i == j:
                continue

            p_j = patient_evals[j]
            tte_j = p_j['tte']

            # Pair is valid if Subject i failed strictly before Subject j
            if tte_i < tte_j:
                # 1. Endpoint / Latest state comparison
                # At their latest recorded states, Subject j should have lower or equal failure risk
                loss_last = continuous_cramer_fsd_loss(
                    cdf_i=p_i['cdf_last'],
                    cdf_j=p_j['cdf_last'],
                    delta_s=delta_s,
                    margin=margin
                )

                # 2. Initial state (t=0) comparison if available
                # If Subject i fails earlier in ground truth, initial state risk should also satisfy dominance
                if 'cdf_init' in p_i and 'cdf_init' in p_j:
                    loss_init = continuous_cramer_fsd_loss(
                        cdf_i=p_i['cdf_init'],
                        cdf_j=p_j['cdf_init'],
                        delta_s=delta_s,
                        margin=margin
                    )
                    pair_loss = 0.5 * (loss_last + loss_init)
                else:
                    pair_loss = loss_last

                total_fsd = total_fsd + pair_loss
                num_pairs += 1

                if num_pairs >= max_pairs:
                    break
        if num_pairs >= max_pairs:
            break

    if num_pairs == 0:
        return torch.tensor(0.0, device=device), 0

    return total_fsd / num_pairs, num_pairs
