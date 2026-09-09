"""
Replication and Extension of Figure 1 from DeepTCSR (Bleistein et al., 2024, arXiv):
"Performance of event prediction in small datasets"
Evaluates linear CoxPH models under sample size scaling N in {10, 20, 30, 50, 75, 100} across 5 seeds.
Compares:
  1. SA Init State (Baseline 1: Static Cox at t=0)
  2. SA Landmarking (Baseline 2: Unrolled landmarking Cox)
  3. TCSR (Maystre & Russo 2022: Discrete TD on linear hazard without EMA decoupling)
  4. DeepTCSR (Bleistein et al. 2024: EMA target network TD with lambda=0)
  5. SM-TCSR (Ours: Continuous Renewal Shift on CDF + Cramer L2 Contraction, NO pairwise ranking loss)
"""
import os
import sys
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index as _concordance_index

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------------------------
# 1. Metrics: Concordance Index & Integrated Brier Score
# ---------------------------------------------------------------------------
def kaplan_meier(ts, cs):
    cs = cs.astype(bool)
    max_t = int(np.max(ts)) if len(ts) > 0 else 0
    steps = np.arange(0, max_t + 1)
    ns = np.sum(ts[:, np.newaxis] >= steps, axis=0)
    ds = np.sum(ts[~cs, np.newaxis] == steps, axis=0)
    return np.cumprod(1.0 - ds / np.maximum(ns, 1))

def compute_ibs(surv_curves, ts, cs):
    """
    IPCW-weighted Integrated Brier Score (Graf et al. 1999 / DeepTCSR official protocol).
    surv_curves: (N, H)
    """
    cs = cs.astype(bool)
    km = kaplan_meier(ts - ~cs, ~cs)
    t_max = min(surv_curves.shape[1], int(np.max(ts)))
    tot = 0.0
    for h in range(1, t_max + 1):
        idx_died = (ts <= h) & ~cs
        idx_alive = (ts > h) | ((ts == h) & cs)
        w_died = np.maximum(km[np.maximum(0, ts[idx_died] - 1)], 1e-4)
        w_alive = max(km[max(0, h - 1)], 1e-4)
        tot += np.sum((1.0 / w_died) * (0.0 - surv_curves[idx_died, h - 1]) ** 2)
        tot += np.sum((1.0 / w_alive) * (1.0 - surv_curves[idx_alive, h - 1]) ** 2)
    return float(tot / (t_max * len(ts)))

def compute_ci(scores, ts, cs):
    """
    scores: higher score implies longer survival time.
    """
    cs_bool = cs.astype(bool)
    return float(_concordance_index(ts + cs_bool, scores, ~cs_bool))

# ---------------------------------------------------------------------------
# 2. Dataset Loaders & Generators
# ---------------------------------------------------------------------------
def load_pbc2_data():
    pkl_path = os.path.join(ROOT_DIR, 'baselines/deep_tcsr/data/pbc-seqs.pkl')
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    seqs = d['seqs'].astype(np.float32) # (312, 16, 15)
    cs = d['cs'].astype(bool)          # True = censored, False = event
    ts = d['ts'].astype(int)           # 0..15
    return seqs, ts, cs

def generate_small_rw(seed=0, n_samples=300, horizon=11, n_dims=20):
    rng = np.random.default_rng(seed=seed)
    thetas = rng.normal(size=n_dims).astype(np.float32)
    bias = -2.5
    sigma0 = 1.0

    seqs = np.zeros((n_samples, horizon, n_dims), dtype=np.float32)
    ts = np.zeros(n_samples, dtype=int)
    cs = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        x = rng.normal(scale=sigma0, size=n_dims).astype(np.float32)
        seqs[i, 0] = x
        died = False
        for k in range(horizon):
            logit = np.dot(thetas, x) + bias
            p = 1.0 / (1.0 + np.exp(-logit))
            if rng.uniform() < p and not died:
                ts[i] = k
                cs[i] = False
                died = True
                break
            if k + 1 < horizon:
                x = x + rng.normal(scale=0.3, size=n_dims).astype(np.float32)
                seqs[i, k + 1] = x
        if not died:
            ts[i] = horizon - 1
            cs[i] = True
            
    return seqs, ts, cs

def get_targets_and_masks(seqs, ts, cs, horizon, landmark=True):
    B = len(seqs)
    targets = np.zeros((B, horizon, horizon), dtype=np.float32)
    masks = np.zeros((B, horizon, horizon), dtype=np.float32)
    for i in range(B):
        t = ts[i]
        c = cs[i]
        if not c:
            for row in range(min(t, horizon)):
                rem_k = t - 1 - row
                if 0 <= rem_k < horizon:
                    targets[i, row, rem_k] = 1.0
            if landmark:
                for row in range(min(t, horizon)):
                    masks[i, row, :(t - row)] = 1.0
            else:
                masks[i, 0, :t] = 1.0
        else:
            if landmark:
                for row in range(min(t + 1, horizon)):
                    masks[i, row, :(t + 1 - row)] = 1.0
            else:
                masks[i, 0, :(t + 1)] = 1.0
    return torch.from_numpy(targets), torch.from_numpy(masks)

# ---------------------------------------------------------------------------
# 3. Model Architecture (Linear CoxPH matching DeepTCSR)
# ---------------------------------------------------------------------------
class LinearCoxPH(nn.Module):
    def __init__(self, n_feats, horizon):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(n_feats))
        self.alpha = nn.Parameter(torch.zeros(horizon))

    def forward(self, x):
        # x: (B, horizon, n_feats)
        logits = torch.matmul(x, self.beta).unsqueeze(-1) + self.alpha
        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1.0 - hazards.clamp(max=0.999), dim=-1)
        cdf = 1.0 - surv
        return logits, hazards, surv, cdf

# ---------------------------------------------------------------------------
# 4. Training and Evaluation Harness
# ---------------------------------------------------------------------------
def train_and_eval_model(method, train_data, test_data, n_feats, horizon,
                         epochs=80, lr=0.1, tau=0.1, seed=42):
    torch.manual_seed(seed)
    seqs_tr, ts_tr, cs_tr = train_data
    seqs_te, ts_te, cs_te = test_data

    N_tr = len(ts_tr)
    N_te = len(ts_te)

    X_tr = torch.from_numpy(seqs_tr).float()
    X_te = torch.from_numpy(seqs_te).float()

    model = LinearCoxPH(n_feats, horizon)
    target_model = copy.deepcopy(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4 if "SM-TCSR" in method else 0.0)

    # 1. SA Init State
    if method == "SA Init State":
        ys_tr, masks_tr = get_targets_and_masks(seqs_tr, ts_tr, cs_tr, horizon, landmark=False)
        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()
            _, hazards, _, _ = model(X_tr)
            loss = (F.binary_cross_entropy(hazards, ys_tr, reduction='none') * masks_tr).sum() / masks_tr.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()

    # 2. SA Landmarking
    elif method == "SA Landmarking":
        ys_tr, masks_tr = get_targets_and_masks(seqs_tr, ts_tr, cs_tr, horizon, landmark=True)
        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()
            _, hazards, _, _ = model(X_tr)
            loss = (F.binary_cross_entropy(hazards, ys_tr, reduction='none') * masks_tr).sum() / masks_tr.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()

    # 3. TCSR (Maystre 2022: Discrete TD without EMA smoothing)
    elif method == "TCSR (Maystre 2022)":
        ys_tr, masks_tr = get_targets_and_masks(seqs_tr, ts_tr, cs_tr, horizon, landmark=True)
        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()
            _, hazards, _, _ = model(X_tr)
            with torch.no_grad():
                _, tgt_hazards, tgt_surv, _ = target_model(X_tr)
                b_tgt = tgt_hazards.clone()
                h_out = torch.zeros_like(b_tgt)
                next_b = torch.zeros(N_tr, horizon)
                for t in reversed(range(horizon)):
                    h_roll = torch.roll(next_b, 1, dims=-1)
                    h_roll[:, 0] = ys_tr[:, t, 0]
                    cond = ys_tr[:, t, 0] == 1.0
                    h_roll[cond] = 0.0
                    h_roll[cond, 0] = 1.0
                    h_out[:, t] = h_roll
                    next_b = b_tgt[:, t]
                ws = torch.roll(tgt_surv, 1, dims=-1)
                ws[:, :, 0] = 1.0

            loss = (F.binary_cross_entropy(hazards, h_out, reduction='none') * ws * masks_tr).sum() / masks_tr.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()

            # Hard periodic update every 10 epochs (as in Maystre 2022 outer loop)
            if (ep + 1) % 10 == 0:
                target_model.load_state_dict(model.state_dict())

    # 4. DeepTCSR (Bleistein 2024: EMA Target Network TD)
    elif method == "DeepTCSR (Bleistein 2024)":
        ys_tr, masks_tr = get_targets_and_masks(seqs_tr, ts_tr, cs_tr, horizon, landmark=True)
        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()
            _, hazards, _, _ = model(X_tr)
            with torch.no_grad():
                _, tgt_hazards, tgt_surv, _ = target_model(X_tr)
                b_tgt = tgt_hazards.clone()
                h_out = torch.zeros_like(b_tgt)
                next_b = torch.zeros(N_tr, horizon)
                for t in reversed(range(horizon)):
                    h_roll = torch.roll(next_b, 1, dims=-1)
                    h_roll[:, 0] = ys_tr[:, t, 0]
                    cond = ys_tr[:, t, 0] == 1.0
                    h_roll[cond] = 0.0
                    h_roll[cond, 0] = 1.0
                    h_out[:, t] = h_roll
                    next_b = b_tgt[:, t]
                ws = torch.roll(tgt_surv, 1, dims=-1)
                ws[:, :, 0] = 1.0

            loss = (F.binary_cross_entropy(hazards, h_out, reduction='none') * ws * masks_tr).sum() / masks_tr.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()

            # Smooth EMA update
            with torch.no_grad():
                for p, pt in zip(model.parameters(), target_model.parameters()):
                    pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    # 5. SM-TCSR (Ours: Continuous Renewal Shift on CDF + Cramer L2 Contraction, NO pairwise ranking loss)
    elif method == "SM-TCSR (Ours)":
        ys_tr, masks_tr = get_targets_and_masks(seqs_tr, ts_tr, cs_tr, horizon, landmark=True)
        for ep in range(epochs):
            model.train()
            optimizer.zero_grad()
            _, hazards, surv, cdf = model(X_tr)
            
            with torch.no_grad():
                _, tgt_hazards, tgt_surv, tgt_cdf = target_model(X_tr)
                
                # 1) TD Hazard target
                b_tgt = tgt_hazards.clone()
                h_out = torch.zeros_like(b_tgt)
                next_b = torch.zeros(N_tr, horizon)
                for t in reversed(range(horizon)):
                    h_roll = torch.roll(next_b, 1, dims=-1)
                    h_roll[:, 0] = ys_tr[:, t, 0]
                    cond = ys_tr[:, t, 0] == 1.0
                    h_roll[cond] = 0.0
                    h_roll[cond, 0] = 1.0
                    h_out[:, t] = h_roll
                    next_b = b_tgt[:, t]
                ws = torch.roll(tgt_surv, 1, dims=-1)
                ws[:, :, 0] = 1.0

                # 2) Semi-Markov Renewal Shift target on CDF:
                # S(k | x_t) = S(1 | x_t) * S(k-1 | x_{t+1})
                # F(k | x_t) = (1 - S(1 | x_t)) + S(1 | x_t) * F(k-1 | x_{t+1})
                target_cdf_sm = torch.zeros_like(cdf)
                for t in reversed(range(horizon)):
                    if t + 1 < horizon:
                        gamma = tgt_surv[:, t, 0:1] # (N_tr, 1)
                        f_next = torch.roll(tgt_cdf[:, t + 1], 1, dims=-1)
                        f_next[:, 0] = 0.0
                        target_cdf_sm[:, t] = (1.0 - gamma) + gamma * f_next
                    else:
                        target_cdf_sm[:, t] = tgt_cdf[:, t]
                    cond = ys_tr[:, t, 0] == 1.0
                    target_cdf_sm[cond, t] = 1.0

            # Cramer L2 Distance Loss (Theorem 1 Contraction Operator)
            loss_cramer = ((cdf - target_cdf_sm) ** 2 * masks_tr).sum() / masks_tr.sum().clamp(min=1.0)
            # Hazard TD Loss
            loss_td = (F.binary_cross_entropy(hazards, h_out, reduction='none') * ws * masks_tr).sum() / masks_tr.sum().clamp(min=1.0)
            
            # Joint Continuous Semi-Markov Objective
            total_loss = loss_td + 0.6 * loss_cramer
            total_loss.backward()
            optimizer.step()

            # Target Contraction EMA
            with torch.no_grad():
                for p, pt in zip(model.parameters(), target_model.parameters()):
                    pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    # Evaluation
    model.eval()
    with torch.no_grad():
        _, _, surv_te, _ = model(X_te)
        surv_curves = surv_te[:, 0].cpu().numpy() # Survival curves from t=0
        scores = surv_te[:, 0, -1].cpu().numpy()  # Horizon survival score (DeepTCSR protocol)

    ci = compute_ci(scores, ts_te, cs_te)
    ibs = compute_ibs(surv_curves, ts_te, cs_te)
    return ci, ibs

# ---------------------------------------------------------------------------
# 5. Full Experiment Runner (Figure 1 Reproduction)
# ---------------------------------------------------------------------------
def run_figure1_experiment():
    print("==========================================================================")
    print("🔬 [DeepTCSR Figure 1 Reproduction & SM-TCSR Extension Benchmark]")
    print("   Linear CoxPH Architecture across Sample Sizes N in {10, 20, 30, 50, 75, 100}")
    print("   Methods: SA Init, SA Landmarking, TCSR, DeepTCSR, SM-TCSR (NO pairwise loss)")
    print("   Metrics: Concordance Index (CI ↑) & Integrated Brier Score (IBS ↓)")
    print("==========================================================================\n")

    sizes = [10, 20, 30, 50, 75, 100]
    seeds = [42, 123, 456, 789, 101112]
    
    methods = [
        "SA Init State",
        "SA Landmarking",
        "TCSR (Maystre 2022)",
        "DeepTCSR (Bleistein 2024)",
        "SM-TCSR (Ours)"
    ]

    datasets = {
        "PBC2": load_pbc2_data(),
        "SmallRW": generate_small_rw(seed=0, n_samples=300, horizon=11, n_dims=20)
    }

    results = {ds: {m: {"ci": np.zeros((len(seeds), len(sizes))),
                        "ibs": np.zeros((len(seeds), len(sizes)))}
                    for m in methods} for ds in datasets}

    for ds_name, (seqs, ts, cs) in datasets.items():
        print(f"\n📊 >>> Benchmarking Dataset: {ds_name} (Total N={len(ts)}, Horizon={seqs.shape[1]}, Dims={seqs.shape[2]})")
        n_feats = seqs.shape[2]
        horizon = seqs.shape[1]
        
        for s_idx, seed in enumerate(seeds):
            rng = np.random.default_rng(seed=seed)
            perm = rng.permutation(len(ts))
            
            n_test = int(0.2 * len(ts))
            test_idx = perm[:n_test]
            train_pool = perm[n_test:]
            
            test_data = (seqs[test_idx], ts[test_idx], cs[test_idx])
            
            for sz_idx, sz in enumerate(sizes):
                sub_tr_idx = train_pool[:sz]
                train_data = (seqs[sub_tr_idx], ts[sub_tr_idx], cs[sub_tr_idx])
                
                for m in methods:
                    ci, ibs = train_and_eval_model(m, train_data, test_data, n_feats, horizon,
                                                  epochs=80, lr=0.1, tau=0.1, seed=seed)
                    results[ds_name][m]["ci"][s_idx, sz_idx] = ci
                    results[ds_name][m]["ibs"][s_idx, sz_idx] = ibs
                    
            print(f"   • Seed {seed} complete (N={sizes})")

    # -------------------------------------------------------------
    # 6. Plotting Figure 1 Exact Replication
    # -------------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=300)
    
    colors = {
        "SA Init State": "#9e9e9e",
        "SA Landmarking": "#ff9800",
        "TCSR (Maystre 2022)": "#4caf50",
        "DeepTCSR (Bleistein 2024)": "#9c27b0",
        "SM-TCSR (Ours)": "#1565c0"
    }
    markers = {
        "SA Init State": "x",
        "SA Landmarking": "^",
        "TCSR (Maystre 2022)": "s",
        "DeepTCSR (Bleistein 2024)": "D",
        "SM-TCSR (Ours)": "o"
    }

    for col_idx, ds_name in enumerate(["PBC2", "SmallRW"]):
        # CI plot (Top)
        ax_ci = axes[0, col_idx]
        for m in methods:
            vals = results[ds_name][m]["ci"]
            mean = np.mean(vals, axis=0)
            stderr = np.std(vals, axis=0) / np.sqrt(len(seeds))
            lw = 2.8 if "SM-TCSR" in m else 1.6
            ax_ci.plot(sizes, mean, marker=markers[m], label=m, color=colors[m], lw=lw, markersize=6.5)
            ax_ci.fill_between(sizes, mean - stderr, mean + stderr, color=colors[m], alpha=0.15)
        ax_ci.set_title(f"{ds_name} - Concordance Index (CI ↑)", fontsize=12, fontweight='bold')
        ax_ci.set_xlabel("Nb. of sequences", fontsize=10)
        ax_ci.set_ylabel("Concordance Index", fontsize=10)
        ax_ci.grid(True, linestyle='--', alpha=0.5)
        ax_ci.legend(frameon=True, fontsize=8.5, loc='lower right')

        # IBS plot (Bottom)
        ax_ibs = axes[1, col_idx]
        for m in methods:
            vals = results[ds_name][m]["ibs"]
            mean = np.mean(vals, axis=0)
            stderr = np.std(vals, axis=0) / np.sqrt(len(seeds))
            lw = 2.8 if "SM-TCSR" in m else 1.6
            ax_ibs.plot(sizes, mean, marker=markers[m], label=m, color=colors[m], lw=lw, markersize=6.5)
            ax_ibs.fill_between(sizes, mean - stderr, mean + stderr, color=colors[m], alpha=0.15)
        ax_ibs.set_title(f"{ds_name} - Integrated Brier Score (IBS ↓)", fontsize=12, fontweight='bold')
        ax_ibs.set_xlabel("Nb. of sequences", fontsize=10)
        ax_ibs.set_ylabel("Integrated Brier Score", fontsize=10)
        ax_ibs.grid(True, linestyle='--', alpha=0.5)
        ax_ibs.legend(frameon=True, fontsize=8.5, loc='upper right')

    fig.suptitle("Performance of Event Prediction in Small Datasets (Figure 1 Replication & SM-TCSR Extension)\nLinear CoxPH Model under Sample Size Scaling across 5 Random Splits (Pure Renewal Contraction, NO Pairwise Loss)", fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_png = "figures/deeptcsr_fig1_replication_with_survtd.png"
    plt.savefig(out_png)
    plt.close()
    print(f"\n✅ [Done] High-resolution Figure saved to: {out_png}")
    print("==========================================================================")

if __name__ == "__main__":
    run_figure1_experiment()
