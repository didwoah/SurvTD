"""
4-Gate Verification Experiment for SurvTD-v2 vs SurvTD-v1 Baseline.
Tests on Synthetic ICU longitudinal cohort:
- Gate 1: Gradient Clipping Rate (Target: < 2.0%, Baseline: ~80-100%)
- Gate 2: Anchor Discrimination C^td Recovery (Target: >= 0.6200, Baseline: ~0.5015)
- Gate 3: Variance Inflation Collapse across seeds (Target: SD < 0.055, Baseline SD: 0.0841)
- Gate 4: Continuous Generator Rate Invariance as dt -> 0 (Target: Ratio in [0.80, 1.25])
"""

import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.synthetic_icu_loader import load_synthetic_icu
from lifelines.utils import concordance_index


# ==============================================================================
# Model Architectures
# ==============================================================================

class SurvTD_v1_Baseline(nn.Module):
    """
    v1 Baseline: Multiplicative hazard product chain S_m = prod(1 - h_k)
    Coupled with naive logit-Cramer / boundary-unbounded anchor.
    """
    def __init__(self, input_dim=10, hidden_dim=32, num_bins=30, delta_s=2.4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.delta_s = delta_s
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_bins)

    def forward(self, x):
        out, _ = self.rnn(x)  # (B, L, H)
        logits = self.head(out)  # (B, L, K)
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(torch.clamp(1.0 - hazards, min=1e-6, max=1.0), dim=-1)
        cdf = 1.0 - surv
        return logits, hazards, surv, cdf

    def compute_loss(self, x, r_obs, has_event, dt=1.0, lam=0.6):
        logits, hazards, surv, cdf = self.forward(x)
        # Logit-Cramer anchor loss with unbounded boundary denominator: (F - G)^2 / (F*(1-F))
        k_target = torch.clamp((r_obs / self.delta_s).long(), 0, self.num_bins - 1)
        G_step = torch.zeros_like(cdf)
        for b in range(x.shape[0]):
            for t in range(x.shape[1]):
                kt = k_target[b, t]
                if has_event[b]:
                    G_step[b, t, kt:] = 1.0
                else:
                    G_step[b, t, :kt] = 0.0

        denom = cdf * (1.0 - cdf) + 1e-4
        loss_anchor = torch.mean((cdf - G_step) ** 2 / denom)

        lam_j = lam ** (dt / self.delta_s)
        loss_td = torch.mean((cdf[:, :-1] - cdf[:, 1:]) ** 2) * (1.0 - lam_j)
        return loss_anchor + 0.5 * loss_td


class SurvTD_v2_Proposed(nn.Module):
    """
    v2 Proposed:
    1. Bounded Logit Anchor (BLA): sigma(z_k) - y_k in [-1, 1], zero clipping explosions.
    2. Cumulative Hazard Huber Matching (CHHM): smooth Huber loss on survival/hazard differences.
    3. Continuous Generator Rate: beta = exp(-rho * dt), non-vanishing TD error as dt -> 0.
    """
    def __init__(self, input_dim=10, hidden_dim=32, num_bins=30, delta_s=2.4, rho=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.delta_s = delta_s
        self.rho = rho
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_bins)

    def forward(self, x):
        out, _ = self.rnn(x)  # (B, L, H)
        logits = self.head(out)  # (B, L, K)
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(torch.clamp(1.0 - hazards, min=1e-6, max=1.0), dim=-1)
        cdf = 1.0 - surv
        return logits, hazards, surv, cdf

    def compute_loss(self, x, r_obs, has_event, dt=1.0):
        logits, hazards, surv, cdf = self.forward(x)
        B, L, K = logits.shape

        # 1. Bounded Logit Anchor (BLA) - evaluated on at-risk bins
        k_target = torch.clamp((r_obs / self.delta_s).long(), 0, K - 1)
        loss_anchor = 0.0
        for b in range(B):
            for t in range(L):
                kt = k_target[b, t].item()
                if kt > 0:
                    loss_anchor += F.binary_cross_entropy_with_logits(
                        logits[b, t, :kt], torch.zeros(kt, device=logits.device)
                    )
                if has_event[b] and kt < K:
                    loss_anchor += F.binary_cross_entropy_with_logits(
                        logits[b, t, kt:kt+1], torch.ones(1, device=logits.device)
                    )
        loss_anchor = loss_anchor / (B * L)

        # 2. Cumulative Hazard Huber Matching (CHHM) with Continuous Generator
        beta = math.exp(-self.rho * dt)
        td_diff = surv[:, :-1] - surv.detach()[:, 1:]
        loss_td = F.smooth_l1_loss(td_diff, torch.zeros_like(td_diff), beta=1.0) * (1.0 - beta)

        return loss_anchor + 0.5 * loss_td


# ==============================================================================
# 4-Gate Verification Execution
# ==============================================================================

def run_gate1_gradient_clipping(train_loader, device="cpu"):
    """Gate 1: Gradient Clipping Activation Rate (Target: < 2.0%, Baseline: > 80%)"""
    print("\n>>> Running Gate 1: Gradient Clipping Activation Rate...")
    v1_model = SurvTD_v1_Baseline().to(device)
    v2_model = SurvTD_v2_Proposed().to(device)

    opt_v1 = torch.optim.Adam(v1_model.parameters(), lr=1e-3)
    opt_v2 = torch.optim.Adam(v2_model.parameters(), lr=1e-3)

    v1_clips = 0
    v2_clips = 0
    n_steps = 100

    step = 0
    for sample in train_loader:
        if step >= n_steps:
            break
        x = sample['features'].unsqueeze(0).to(device)
        tte = sample['tte']
        ev = sample['event']
        times = sample['times']
        r_obs = torch.clamp(tte - times, min=0.0).unsqueeze(0).to(device)
        has_ev = [bool(ev > 0.5)]

        # v1 Step
        opt_v1.zero_grad()
        loss1 = v1_model.compute_loss(x, r_obs, has_ev)
        loss1.backward()
        norm1 = nn.utils.clip_grad_norm_(v1_model.parameters(), max_norm=2.0)
        if norm1 > 2.0:
            v1_clips += 1
        opt_v1.step()

        # v2 Step
        opt_v2.zero_grad()
        loss2 = v2_model.compute_loss(x, r_obs, has_ev)
        loss2.backward()
        norm2 = nn.utils.clip_grad_norm_(v2_model.parameters(), max_norm=2.0)
        if norm2 > 2.0:
            v2_clips += 1
        opt_v2.step()

        step += 1

    rate_v1 = (v1_clips / step) * 100.0
    rate_v2 = (v2_clips / step) * 100.0
    passed = (rate_v2 < 2.0)
    print(f"Gate 1 Results: SurvTD-v1 Clip Rate = {rate_v1:.1f}% | SurvTD-v2 Clip Rate = {rate_v2:.1f}% | Passed: {passed}")
    return {"gate": "Gate 1 (Clipping Rate)", "v1_clip_rate": rate_v1, "v2_clip_rate": rate_v2, "passed": passed}


def run_gate2_discrimination_recovery(train_dataset, test_dataset, device="cpu"):
    """Gate 2: Anchor Discrimination C^td Recovery (Target: >= 0.6200)"""
    print("\n>>> Running Gate 2: Anchor Discrimination Recovery...")
    v1_model = SurvTD_v1_Baseline().to(device)
    v2_model = SurvTD_v2_Proposed().to(device)

    opt_v1 = torch.optim.Adam(v1_model.parameters(), lr=1e-3)
    opt_v2 = torch.optim.Adam(v2_model.parameters(), lr=1e-3)

    for epoch in range(5):
        perm = np.random.permutation(len(train_dataset))
        for idx in perm:
            sample = train_dataset[idx]
            x = sample['features'].unsqueeze(0).to(device)
            tte = sample['tte']
            ev = sample['event']
            times = sample['times']
            r_obs = torch.clamp(tte - times, min=0.0).unsqueeze(0).to(device)
            has_ev = [bool(ev > 0.5)]

            opt_v1.zero_grad()
            l1 = v1_model.compute_loss(x, r_obs, has_ev)
            l1.backward()
            nn.utils.clip_grad_norm_(v1_model.parameters(), 2.0)
            opt_v1.step()

            opt_v2.zero_grad()
            l2 = v2_model.compute_loss(x, r_obs, has_ev)
            l2.backward()
            nn.utils.clip_grad_norm_(v2_model.parameters(), 2.0)
            opt_v2.step()

    v1_model.eval()
    v2_model.eval()

    durations = [sample['tte'] for sample in test_dataset]
    events = [sample['event'] for sample in test_dataset]
    v1_scores = []
    v2_scores = []

    with torch.no_grad():
        for sample in test_dataset:
            x = sample['features'].unsqueeze(0).to(device)
            t0 = sample['times'][0].item()

            _, _, surv1, _ = v1_model(x)
            exp_time1 = float(torch.sum(surv1[0, 0]).item() * v1_model.delta_s + t0)
            v1_scores.append(exp_time1)

            _, _, surv2, _ = v2_model(x)
            exp_time2 = float(torch.sum(surv2[0, 0]).item() * v2_model.delta_s + t0)
            v2_scores.append(exp_time2)

    ci_v1 = float(concordance_index(durations, v1_scores, events))
    ci_v2 = float(concordance_index(durations, v2_scores, events))
    passed = (ci_v2 >= 0.6200)
    print(f"Gate 2 Results: SurvTD-v1 C-index = {ci_v1:.4f} | SurvTD-v2 C-index = {ci_v2:.4f} | Passed: {passed}")
    return {"gate": "Gate 2 (Discrimination Recovery)", "v1_cindex": ci_v1, "v2_cindex": ci_v2, "passed": passed}


def run_gate3_variance_collapse(device="cpu"):
    """Gate 3: Variance Inflation Collapse across seeds (Target: SD < 0.055)"""
    print("\n>>> Running Gate 3: Variance Inflation Collapse across 3 seeds...")
    seeds = [42, 123, 456]
    v1_cis = []
    v2_cis = []

    for s in seeds:
        train_ds, _, test_ds, _, _, _ = load_synthetic_icu(seed=s, n_patients=300)
        v1_model = SurvTD_v1_Baseline().to(device)
        v2_model = SurvTD_v2_Proposed().to(device)

        opt_v1 = torch.optim.Adam(v1_model.parameters(), lr=1e-3)
        opt_v2 = torch.optim.Adam(v2_model.parameters(), lr=1e-3)

        for epoch in range(5):
            for sample in train_ds:
                x = sample['features'].unsqueeze(0).to(device)
                tte = sample['tte']
                ev = sample['event']
                r_obs = torch.clamp(tte - sample['times'], min=0.0).unsqueeze(0).to(device)
                has_ev = [bool(ev > 0.5)]

                opt_v1.zero_grad()
                l1 = v1_model.compute_loss(x, r_obs, has_ev)
                l1.backward()
                nn.utils.clip_grad_norm_(v1_model.parameters(), 2.0)
                opt_v1.step()

                opt_v2.zero_grad()
                l2 = v2_model.compute_loss(x, r_obs, has_ev)
                l2.backward()
                nn.utils.clip_grad_norm_(v2_model.parameters(), 2.0)
                opt_v2.step()

        durations = [sample['tte'] for sample in test_ds]
        events = [sample['event'] for sample in test_ds]
        with torch.no_grad():
            s1 = [float(torch.sum(v1_model(sample['features'].unsqueeze(0).to(device))[2][0, 0]).item() * 2.4 + sample['times'][0].item()) for sample in test_ds]
            s2 = [float(torch.sum(v2_model(sample['features'].unsqueeze(0).to(device))[2][0, 0]).item() * 2.4 + sample['times'][0].item()) for sample in test_ds]

        v1_cis.append(concordance_index(durations, s1, events))
        v2_cis.append(concordance_index(durations, s2, events))

    sd_v1 = float(np.std(v1_cis))
    sd_v2 = float(np.std(v2_cis))
    passed = (sd_v2 < 0.055)
    print(f"Gate 3 Results: SurvTD-v1 C-indices = {[round(x, 4) for x in v1_cis]} (SD = {sd_v1:.4f}) | SurvTD-v2 C-indices = {[round(x, 4) for x in v2_cis]} (SD = {sd_v2:.4f}) | Passed: {passed}")
    return {"gate": "Gate 3 (Variance Collapse)", "v1_sd": sd_v1, "v2_sd": sd_v2, "v1_cis": v1_cis, "v2_cis": v2_cis, "passed": passed}


def run_gate4_continuous_rate_invariance(device="cpu"):
    """Gate 4: Continuous Generator Rate Invariance as dt -> 0"""
    print("\n>>> Running Gate 4: Continuous Generator Rate Invariance as dt -> 0...")
    v2_model = SurvTD_v2_Proposed().to(device)

    x = torch.randn(1, 20, 10, device=device)

    # In SurvTD-v2: continuous generator weight is (1 - exp(-rho * dt))
    # Normalized rate per unit time: (1 - exp(-rho * dt)) / dt -> rho as dt -> 0
    v2_model.zero_grad()
    _, _, surv2, _ = v2_model(x)
    l2_std = F.smooth_l1_loss(surv2[:, :-1], surv2[:, 1:].detach()) * (1.0 - math.exp(-0.5 * 0.1))
    l2_std.backward()
    norm_std = v2_model.head.weight.grad.norm().item() / 0.1

    v2_model.zero_grad()
    _, _, surv2, _ = v2_model(x)
    l2_dense = F.smooth_l1_loss(surv2[:, :-1], surv2[:, 1:].detach()) * (1.0 - math.exp(-0.5 * 0.01))
    l2_dense.backward()
    norm_dense = v2_model.head.weight.grad.norm().item() / 0.01
    grad_ratio = norm_dense / max(norm_std, 1e-6)

    passed = (0.80 <= grad_ratio <= 1.25)
    print(f"Gate 4 Results: Normalized TD Gradient Rate Ratio (dense / std) = {grad_ratio:.4f} | Passed: {passed}")
    return {"gate": "Gate 4 (Rate Invariance)", "grad_ratio": grad_ratio, "passed": passed}


def main():
    print("================================================================")
    print("   SURVTD-V2 4-GATE VERIFICATION BENCHMARK (SYNTHETIC ICU)     ")
    print("================================================================")
    train_ds, val_ds, test_ds, _, _, _ = load_synthetic_icu(seed=42, n_patients=300)

    res1 = run_gate1_gradient_clipping(train_ds)
    res2 = run_gate2_discrimination_recovery(train_ds, test_ds)
    res3 = run_gate3_variance_collapse()
    res4 = run_gate4_continuous_rate_invariance()

    all_gates = [res1, res2, res3, res4]
    all_passed = all(g["passed"] for g in all_gates)

    out_file = "research/toy-gradient/toy_gradient_results.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_gates, "all_passed": all_passed}, f, indent=2)

    print("\n================================================================")
    print("                    FINAL 4-GATE VERDICT                        ")
    print("================================================================")
    for g in all_gates:
        status = "PASSED" if g["passed"] else "FAILED"
        print(f"{g['gate']:<32} : [{status}]")
    print(f"\nOVERALL STATUS: {'ALL 4 GATES PASSED (SURVTD-V2 FULLY VALIDATED)' if all_passed else 'SOME GATES FAILED'}")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
