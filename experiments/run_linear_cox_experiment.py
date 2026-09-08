import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.cohorts import get_cohort
from src.data.nasa_protocol_loader import get_nasa_splits
from src.operators.anchors import censored_crps_anchor
from src.operators.survtd_operator import (
    categorical_projection_shift,
    compute_multistep_lambda_returns,
    squared_cramer_distance_loss
)

# -------------------------------------------------------------
# 1. Pure Linear / Cox Proportional Hazards Model
# -------------------------------------------------------------
class LinearCoxModel(nn.Module):
    """
    Pure Linear CoxPH Hazard Model (Maystre 2022 / Prentice & Gloeckler 1978).
    logit_k(x) = beta_k^T x + b_k
    Strictly linear in features: No RNN, No LSTM, No MLP!
    """
    def __init__(self, input_dim: int, num_bins: int = 30, delta_s: float = 1.0, include_overflow: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.K = num_bins
        self.delta_s = delta_s
        self.include_overflow = include_overflow
        self.linear = nn.Linear(input_dim, num_bins)
        
    def forward(self, x):
        # x: (L, D)
        logits = self.linear(x)
        hazards = torch.sigmoid(logits)
        # S(s_m) = cumprod(1 - h)
        surv = torch.cumprod(1.0 - hazards, dim=-1)
        ones = torch.ones(*hazards.shape[:-1], 1, device=hazards.device)
        surv_curve = torch.cat([ones, surv], dim=-1)
        pmf_bins = surv_curve[..., :-1] - surv_curve[..., 1:]
        if self.include_overflow:
            overflow = surv_curve[..., -1:]
            pmf = torch.cat([pmf_bins, overflow], dim=-1)
        else:
            pmf = pmf_bins
            pmf = pmf / (pmf.sum(dim=-1, keepdim=True) + 1e-8)
        cdf = 1.0 - surv
        return hazards, surv_curve[..., 1:], pmf, cdf

# Simple concordance index calculation
def compute_cindex(risk_scores, times, events):
    """Harrell's concordance index."""
    n = len(times)
    concordant, permissible = 0.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if times[i] == times[j]:
                continue
            if times[i] < times[j] and events[i] == 1:
                permissible += 1.0
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1.0
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5
            elif times[j] < times[i] and events[j] == 1:
                permissible += 1.0
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1.0
                elif risk_scores[j] == risk_scores[i]:
                    concordant += 0.5
    return concordant / (permissible + 1e-8) if permissible > 0 else 0.5

# -------------------------------------------------------------
# 2. Unified Training Routine for Cox Variants
# -------------------------------------------------------------
def train_and_eval_cox(method: str, train_samples, test_samples, input_dim, K, delta_s, epochs=15, lr=3e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    online_model = LinearCoxModel(input_dim, num_bins=K, delta_s=delta_s, include_overflow=True)
    target_model = copy.deepcopy(online_model)
    optimizer = torch.optim.Adam(online_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        perm = np.random.permutation(len(train_samples))
        for idx in perm:
            s = train_samples[idx]
            x = s['x']
            dts = s['dts']
            tte = s['tte']
            event = s['event']
            L = x.shape[0]
            
            optimizer.zero_grad()
            haz_on, surv_on, pmf_on, cdf_on = online_model(x)
            
            # Anchor loss (Supervised CRPS on residual times)
            times_arr = s.get('times', None)
            if times_arr is None:
                times_arr = np.cumsum(dts.numpy())
            res_times = np.maximum(0.0, tte - times_arr)
            res_t = torch.from_numpy(res_times).float()
            is_censored = 1.0 - event
            
            # Loss computation based on method
            if method == "Classic CoxPH (Supervised Only)":
                # Pure Supervised Anchor (no TD)
                loss = censored_crps_anchor(cdf_on, res_t, is_censored, delta_s, K=K).mean()
                
            elif method == "Linear TCSR (Maystre 2022 Discrete Roll)":
                # Discrete integer shift (roll 1 step, ignore continuous dt)
                with torch.no_grad():
                    _, _, pmf_tgt, _ = target_model(x)
                    # Discrete unit roll
                    rolled_target = torch.roll(pmf_tgt, shifts=1, dims=-1)
                    rolled_target[..., 0] = 0.0 # zero out wrapped mass
                
                loss_td = squared_cramer_distance_loss(pmf_on[:-1], rolled_target[1:])
                loss_anchor = censored_crps_anchor(cdf_on, res_t, is_censored, delta_s, K=K).mean()
                loss = 0.5 * loss_td + 0.5 * loss_anchor
                
            elif method == "Linear SurvTD (Continuous Renewal + Projection)":
                # Continuous Renewal Shift + Projection + Duration Discount
                events_seq = torch.zeros(L)
                if event == 1:
                    events_seq[-1] = 1.0
                    
                with torch.no_grad():
                    _, surv_tgt, pmf_tgt, _ = target_model(x)
                    td_targets, _ = compute_multistep_lambda_returns(
                        pmf_tgt, surv_tgt, dts, events_seq, tte=tte,
                        delta_s=delta_s, K=K, lam=0.6, include_overflow=True
                    )
                
                loss_td = squared_cramer_distance_loss(pmf_on, td_targets)
                loss_anchor = censored_crps_anchor(cdf_on, res_t, is_censored, delta_s, K=K).mean()
                loss = 0.5 * loss_td + 0.5 * loss_anchor
            else:
                raise ValueError(f"Unknown method {method}")
                
            loss.backward()
            optimizer.step()
            
        # Target network EMA
        with torch.no_grad():
            for p_t, p_o in zip(target_model.parameters(), online_model.parameters()):
                p_t.data.mul_(0.90).add_(p_o.data, alpha=0.10)
                
    # Evaluation on test set
    online_model.eval()
    test_risks = []
    test_ttes = []
    test_events = []
    
    with torch.no_grad():
        for s in test_samples:
            x = s['x']
            tte = s['tte']
            event = s['event']
            # Risk score = cumulative hazard or 1 - survival at median horizon
            _, surv_pred, _, _ = online_model(x)
            # Take prediction at last observed visit
            risk = 1.0 - surv_pred[-1, K // 2].item()
            test_risks.append(risk)
            test_ttes.append(tte)
            test_events.append(event)
            
    cindex = compute_cindex(np.array(test_risks), np.array(test_ttes), np.array(test_events))
    return cindex

# -------------------------------------------------------------
# 3. Main Benchmark Runner
# -------------------------------------------------------------
if __name__ == "__main__":
    print("===============================================================")
    print("🔬 [CoxPH 비교 실험] 선형 모델(CoxPH) 하에서의 TCSR vs SurvTD")
    print("   모든 모델은 완전히 동일한 Linear Hazard Layer를 사용합니다.")
    print("===============================================================\n")
    
    # ---------------------------------------------------------
    # Benchmark 1: Synthetic ICU (Continuous Poisson arrival)
    # ---------------------------------------------------------
    print("📍 [Benchmark 1] Synthetic ICU Telemetry (Continuous Irregular Hours)")
    spec_icu = get_cohort("synthetic_icu")
    data_icu = spec_icu.load(seed=42)
    
    def convert_dataset(ds):
        return [{"x": p["features"], "dts": p["dts"], "times": p["times"].numpy(), "tte": float(p["tte"]), "event": float(p["event"])} for p in ds]
        
    icu_train = convert_dataset(data_icu.train)
    icu_test = convert_dataset(data_icu.test)
    
    methods = [
        "Classic CoxPH (Supervised Only)",
        "Linear TCSR (Maystre 2022 Discrete Roll)",
        "Linear SurvTD (Continuous Renewal + Projection)"
    ]
    
    results_icu = {}
    for m in methods:
        c = train_and_eval_cox(m, icu_train, icu_test, input_dim=data_icu.input_dim, K=36, delta_s=2.0, epochs=12, seed=42)
        results_icu[m] = c
        print(f"  • {m:48s}: C-index = {c:.4f}")
        
    # ---------------------------------------------------------
    # Benchmark 2: NASA C-MAPSS FD001 (Authentic Turbofan Degradation)
    # ---------------------------------------------------------
    print("\n📍 [Benchmark 2] NASA C-MAPSS FD001 (Turbofan Engine Degradation)")
    splits_nasa = get_nasa_splits(seed=42)
    
    feat_cols = splits_nasa['feat_cols']
    df_train = splits_nasa['df_train']
    df_test = splits_nasa['df_test']
    
    def get_nasa_trajs(df, units):
        samples = []
        for u in units:
            sub = df[df['id'] == u].sort_values('times')
            times = sub['times_scaled'].values.astype(np.float32)
            features = sub[feat_cols].values.astype(np.float32)
            tte = float(sub['tte_scaled'].iloc[0])
            event = int(sub['event'].iloc[0])
            dts = np.zeros_like(times)
            dts[1:] = times[1:] - times[:-1]
            dts[0] = times[0] if times[0] > 0 else 0.01
            samples.append({"x": torch.from_numpy(features), "dts": torch.from_numpy(dts), "times": times, "tte": tte, "event": event})
        return samples

    nasa_train = get_nasa_trajs(df_train, splits_nasa['train_units'])
    nasa_test = get_nasa_trajs(df_test, splits_nasa['test_units'])
    
    results_nasa = {}
    for m in methods:
        c = train_and_eval_cox(m, nasa_train, nasa_test, input_dim=len(feat_cols), K=30, delta_s=0.1, epochs=12, seed=42)
        results_nasa[m] = c
        print(f"  • {m:48s}: C-index = {c:.4f}")

    print("\n===============================================================")
    print("📊 [최종 결과 요약: CoxPH 선형 모델 하에서의 C-index 비교]")
    print(f"{'Method':<48s} | {'Synthetic ICU':<14s} | {'NASA FD001':<12s}")
    print("-" * 80)
    for m in methods:
        print(f"{m:<48s} | {results_icu[m]:<14.4f} | {results_nasa[m]:<12.4f}")
    print("===============================================================")
