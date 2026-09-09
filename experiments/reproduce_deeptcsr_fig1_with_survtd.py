"""
Replication and Extension of Figure 1 from DeepTCSR (Bleistein et al., 2024, arXiv):
"Performance of event prediction in small datasets"
Evaluates linear CoxPH models under sample size scaling N in {10, 20, 30, 50, 75, 100} across 5 seeds.
Compares:
  1. SA Init State (Baseline 1: Static Cox at t=0)
  2. SA Landmarking (Baseline 2: Unrolled landmarking Cox)
  3. TCSR (Maystre & Russo 2022: Discrete TD on linear hazard)
  4. DeepTCSR (Bleistein et al. 2024: Target network EMA TD with lambda=0)
  5. SM-TCSR (Ours: Continuous Renewal Shift + Target Contraction + Cramer Loss)
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
# 1. Metric Implementations: CI and IBS
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
    IPCW-weighted Integrated Brier Score (Graf et al. 1999 / Maystre 2022).
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
    scores: expected lifetime (higher score = longer expected survival)
    ts: event/censoring time
    cs: True if censored, False if event
    """
    cs_bool = cs.astype(bool)
    return float(_concordance_index(ts + cs_bool, scores, ~cs_bool))

# ---------------------------------------------------------------------------
# 2. Data Generators & Loaders
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
    thetas = rng.normal(size=n_dims)
    bias = -3.0
    mat = 1.0 * np.eye(n_dims)
    sigma = 0.5
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
            if k < horizon - 1:
                x = np.dot(mat, x) + rng.normal(scale=sigma, size=n_dims).astype(np.float32)
                seqs[i, k + 1] = x
        if not died:
            ts[i] = horizon - 1
            cs[i] = True
    return seqs, ts, cs

# ---------------------------------------------------------------------------
# 3. Model Architecture: Linear CoxPH Hazard Model
# ---------------------------------------------------------------------------
class LinearCoxModel(nn.Module):
    """
    Standard Discrete-Time Cox Proportional Hazards Model (Prentice & Gloeckler 1978).
    logit[h_k(x)] = x^T beta + alpha_k
    """
    def __init__(self, n_feats: int, horizon: int):
        super().__init__()
        self.n_feats = n_feats
        self.horizon = horizon
        self.beta = nn.Parameter(torch.zeros(n_feats))
        self.alpha = nn.Parameter(torch.zeros(horizon))

    def forward(self, x):
        # x: (..., n_feats)
        # logits: (..., horizon)
        lin = torch.matmul(x, self.beta).unsqueeze(-1)
        logits = lin + self.alpha
        hazards = torch.sigmoid(logits)
        # S(k|x) = prod_{m=0}^k (1 - h_m)
        surv = torch.cumprod(1.0 - hazards + 1e-8, dim=-1)
        # PMF: p_k = S_{k-1} - S_k
        ones = torch.ones(*hazards.shape[:-1], 1, device=hazards.device)
        surv_aug = torch.cat([ones, surv], dim=-1)
        pmf = surv_aug[..., :-1] - surv_aug[..., 1:]
        cdf = 1.0 - surv
        return logits, hazards, surv, pmf, cdf

# ---------------------------------------------------------------------------
# 4. Training Engine: 5 Model Formulations
# ---------------------------------------------------------------------------
def train_and_eval_model(method: str, train_data, test_data, n_feats: int, horizon: int,
                         epochs: int = 100, lr: float = 0.05, tau: float = 0.1, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    seqs_tr, ts_tr, cs_tr = train_data
    seqs_te, ts_te, cs_te = test_data

    model = LinearCoxModel(n_feats, horizon)
    target_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N_tr = seqs_tr.shape[0]

    # Pre-extract active subsequences for landmarking / unrolling
    unrolled_tr = []
    for i in range(N_tr):
        T_i = ts_tr[i]
        c_i = cs_tr[i]
        # visits up to event/censoring
        max_vis = min(T_i + 1, horizon)
        for ell in range(max_vis):
            unrolled_tr.append({
                'x': torch.from_numpy(seqs_tr[i, ell]).float(),
                'ell': ell,
                't_remain': T_i - ell,
                'event': not c_i and (ell == T_i or ell == max_vis - 1),
                'censored': c_i,
                'full_seq': torch.from_numpy(seqs_tr[i]).float(),
                'T_i': T_i
            })

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        if method == "SA Init State":
            # Supervised MLE on t=0 only
            x0 = torch.from_numpy(seqs_tr[:, 0]).float()
            _, hazards, surv, pmf, cdf = model(x0)
            
            # Binary Cross Entropy on survival steps
            loss = 0.0
            for i in range(N_tr):
                T_i = ts_tr[i]
                c_i = cs_tr[i]
                # for steps k < T_i: lived (target=0)
                if T_i > 0:
                    loss += F.binary_cross_entropy(hazards[i, :T_i], torch.zeros(T_i))
                # at step T_i: died if not censored (target=1)
                if T_i < horizon and not c_i:
                    loss += F.binary_cross_entropy(hazards[i, T_i], torch.tensor(1.0))
            loss = loss / N_tr

        elif method == "SA Landmarking":
            # Supervised MLE on all unrolled landmark steps
            loss = 0.0
            for item in unrolled_tr:
                x = item['x']
                rem = item['t_remain']
                c = item['censored']
                _, hazards, _, _, _ = model(x)
                if rem > 0:
                    loss += F.binary_cross_entropy(hazards[:rem], torch.zeros(rem))
                if rem < horizon and not c:
                    loss += F.binary_cross_entropy(hazards[rem], torch.tensor(1.0))
            loss = loss / len(unrolled_tr)

        elif method == "TCSR (Maystre 2022)":
            # Discrete TD consistency (Maystre 2022): soft target = roll(h, 1)
            loss = 0.0
            for item in unrolled_tr:
                x = item['x']
                ell = item['ell']
                rem = item['t_remain']
                c = item['censored']
                seq = item['full_seq']
                
                _, hazards, _, _, _ = model(x)
                
                if ell < item['T_i'] and ell + 1 < horizon:
                    # Transition to next state: soft target from roll(1)
                    with torch.no_grad():
                        _, h_next, _, _, _ = model(seq[ell + 1])
                        tgt_h = torch.roll(h_next, shifts=1)
                        tgt_h[0] = 0.0
                    loss += F.mse_loss(hazards[1:], tgt_h[1:])
                else:
                    # Terminal state
                    if not c and rem < horizon:
                        loss += F.binary_cross_entropy(hazards[rem], torch.tensor(1.0))
            loss = loss / len(unrolled_tr)

        elif method == "DeepTCSR (Bleistein 2024)":
            # EMA Target Network with lambda=0 (Bleistein 2024)
            loss = 0.0
            for item in unrolled_tr:
                x = item['x']
                ell = item['ell']
                rem = item['t_remain']
                c = item['censored']
                seq = item['full_seq']
                
                _, hazards, _, _, _ = model(x)
                
                if ell < item['T_i'] and ell + 1 < horizon:
                    with torch.no_grad():
                        _, h_tgt, _, _, _ = target_model(seq[ell + 1])
                        soft_tgt = torch.roll(h_tgt, shifts=1)
                        soft_tgt[0] = 0.0
                    loss += F.binary_cross_entropy(hazards.clamp(1e-4, 1-1e-4), soft_tgt.clamp(1e-4, 1-1e-4))
                else:
                    if not c and rem < horizon:
                        loss += F.binary_cross_entropy(hazards[rem], torch.tensor(1.0))
            loss = loss / len(unrolled_tr)

        elif method == "SM-TCSR (Ours)":
            # Continuous Renewal Shift + Cramer Loss + Frozen Contraction Target
            loss = 0.0
            for item in unrolled_tr:
                x = item['x']
                ell = item['ell']
                rem = item['t_remain']
                c = item['censored']
                seq = item['full_seq']
                
                _, _, surv_on, pmf_on, cdf_on = model(x)
                
                if ell < item['T_i'] and ell + 1 < horizon:
                    # Target renewal shift Phi_{+1} under target network
                    with torch.no_grad():
                        _, _, surv_t, pmf_t, cdf_t = target_model(seq[ell + 1])
                        # interval discount gamma = S_target(1)
                        gamma = float(surv_t[0].item())
                        # renewal shift Phi_{+1}: mass shifts right by 1 bin
                        shifted_pmf = torch.zeros(horizon)
                        shifted_pmf[1:] = pmf_t[:-1]
                        # target: (1 - gamma) mu_death + gamma Phi p_tgt
                        target_pmf = (1.0 - gamma) * torch.eye(horizon)[0] + gamma * shifted_pmf
                        target_cdf = torch.cumsum(target_pmf, dim=-1)
                        
                    loss_td = torch.sum((cdf_on - target_cdf) ** 2)
                    loss += loss_td
                else:
                    # Terminal state: Dirac target or Censoring target
                    target_pmf = torch.zeros(horizon)
                    if not c and rem < horizon:
                        target_pmf[rem] = 1.0
                    else:
                        target_pmf[-1] = 1.0
                    target_cdf = torch.cumsum(target_pmf, dim=-1)
                    loss += torch.sum((cdf_on - target_cdf) ** 2)
                    
            loss = loss / len(unrolled_tr)
        else:
            raise ValueError(f"Unknown method {method}")

        loss.backward()
        optimizer.step()

        # Update EMA target network for DeepTCSR and SM-TCSR
        if method in ["DeepTCSR (Bleistein 2024)", "SM-TCSR (Ours)"]:
            with torch.no_grad():
                for p_t, p_o in zip(target_model.parameters(), model.parameters()):
                    p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)

    # -------------------------------------------------------------
    # Evaluation on Test Set: CI and IBS
    # -------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        x0_te = torch.from_numpy(seqs_te[:, 0]).float()
        _, _, surv_te, pmf_te, _ = model(x0_te)
        # Expected lifetime score = sum(S(k))
        scores = torch.sum(surv_te, dim=-1).cpu().numpy()
        surv_curves = surv_te.cpu().numpy()

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
            
            # 80/20 train/test pool
            n_test = int(0.2 * len(ts))
            test_idx = perm[:n_test]
            train_pool = perm[n_test:]
            
            test_data = (seqs[test_idx], ts[test_idx], cs[test_idx])
            
            for sz_idx, sz in enumerate(sizes):
                sub_tr_idx = train_pool[:sz]
                train_data = (seqs[sub_tr_idx], ts[sub_tr_idx], cs[sub_tr_idx])
                
                for m in methods:
                    ci, ibs = train_and_eval_model(m, train_data, test_data, n_feats, horizon,
                                                  epochs=60, lr=0.05, tau=0.1, seed=seed)
                    results[ds_name][m]["ci"][s_idx, sz_idx] = ci
                    results[ds_name][m]["ibs"][s_idx, sz_idx] = ibs
                    
            print(f"   • Seed {seed} complete (N={sizes})")

    # -------------------------------------------------------------
    # 6. Plotting Figure 1 Exact Replication
    # -------------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    
    colors = {
        "SA Init State": "#757575",
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

    # Row 0: CI, Row 1: IBS
    for col_idx, ds_name in enumerate(["PBC2", "SmallRW"]):
        # CI plot
        ax_ci = axes[0, col_idx]
        for m in methods:
            vals = results[ds_name][m]["ci"]
            mean = np.mean(vals, axis=0)
            stderr = np.std(vals, axis=0) / np.sqrt(len(seeds))
            lw = 2.5 if "SM-TCSR" in m else 1.5
            ax_ci.plot(sizes, mean, marker=markers[m], label=m, color=colors[m], lw=lw, markersize=6)
            ax_ci.fill_between(sizes, mean - stderr, mean + stderr, color=colors[m], alpha=0.15)
        ax_ci.set_title(f"{ds_name} - Concordance Index (CI ↑)", fontsize=12, fontweight='bold')
        ax_ci.set_xlabel("Nb. of sequences", fontsize=10)
        ax_ci.set_ylabel("Concordance Index", fontsize=10)
        ax_ci.grid(True, linestyle='--', alpha=0.5)
        ax_ci.legend(frameon=True, fontsize=8, loc='lower right')

        # IBS plot
        ax_ibs = axes[1, col_idx]
        for m in methods:
            vals = results[ds_name][m]["ibs"]
            mean = np.mean(vals, axis=0)
            stderr = np.std(vals, axis=0) / np.sqrt(len(seeds))
            lw = 2.5 if "SM-TCSR" in m else 1.5
            ax_ibs.plot(sizes, mean, marker=markers[m], label=m, color=colors[m], lw=lw, markersize=6)
            ax_ibs.fill_between(sizes, mean - stderr, mean + stderr, color=colors[m], alpha=0.15)
        ax_ibs.set_title(f"{ds_name} - Integrated Brier Score (IBS ↓)", fontsize=12, fontweight='bold')
        ax_ibs.set_xlabel("Nb. of sequences", fontsize=10)
        ax_ibs.set_ylabel("Integrated Brier Score", fontsize=10)
        ax_ibs.grid(True, linestyle='--', alpha=0.5)
        ax_ibs.legend(frameon=True, fontsize=8, loc='upper right')

    fig.suptitle("Performance of Event Prediction in Small Datasets (Figure 1 Replication & Extension)\nLinear CoxPH Model under Sample Size Scaling across 5 Random Splits", fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_png = "figures/deeptcsr_fig1_replication_with_survtd.png"
    plt.savefig(out_png)
    plt.close()
    print(f"\n✅ [Done] High-resolution Figure saved to: {out_png}")
    print("==========================================================================")

if __name__ == "__main__":
    run_figure1_experiment()
