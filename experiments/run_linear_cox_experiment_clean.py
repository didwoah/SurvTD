import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.cohorts import get_cohort
from src.data.nasa_protocol_loader import get_nasa_splits
from src.operators.anchors import censored_crps_anchor
from src.operators.survtd_operator import (
    compute_multistep_lambda_returns,
    squared_cramer_distance_loss
)

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
        logits = self.linear(x)
        hazards = torch.sigmoid(logits)
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

def train_and_eval_linear_cox(method: str, train_samples, test_samples, input_dim, K, delta_s, epochs=15, lr=2e-3, seed=42):
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
            
            times_arr = s.get('times', None)
            if times_arr is None:
                times_arr = np.cumsum(dts.numpy())
            res_times = np.maximum(0.0, tte - times_arr)
            res_t = torch.from_numpy(res_times).float()
            is_censored = 1.0 - event
            
            if method == "Classic CoxPH (Supervised Only)":
                loss = censored_crps_anchor(cdf_on, res_t, is_censored, delta_s, K=K).mean()
                
            elif method == "Linear TCSR (Maystre 2022 Discrete Roll)":
                with torch.no_grad():
                    _, _, pmf_tgt, _ = target_model(x)
                    rolled_target = torch.roll(pmf_tgt, shifts=1, dims=-1)
                    rolled_target[..., 0] = 0.0
                
                loss_td = squared_cramer_distance_loss(pmf_on[:-1], rolled_target[1:])
                loss_anchor = censored_crps_anchor(cdf_on, res_t, is_censored, delta_s, K=K).mean()
                loss = 0.5 * loss_td + 0.5 * loss_anchor
                
            elif method == "Linear SurvTD (Continuous Renewal + Projection)":
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
            
        with torch.no_grad():
            for p_t, p_o in zip(target_model.parameters(), online_model.parameters()):
                p_t.data.mul_(0.90).add_(p_o.data, alpha=0.10)
                
    # Evaluation: Concordance index of expected lifetime
    online_model.eval()
    test_ttes = np.array([s['tte'] for s in test_samples])
    test_events = np.array([s['event'] for s in test_samples])
    
    # 1. Static t=0 C-index (expected lifetime predicted at initial observation)
    exp_lifetimes_t0 = []
    # 2. Dynamic C-index (average over trajectory visits)
    dynamic_concordances = []
    
    with torch.no_grad():
        for s in test_samples:
            x = s['x']
            _, surv_pred, _, _ = online_model(x)
            # Area under survival curve = expected residual lifetime
            exp_res = torch.sum(surv_pred, dim=-1) * delta_s # (L,)
            exp_lifetimes_t0.append(exp_res[0].item())
            
        c_t0 = concordance_index(test_ttes, np.array(exp_lifetimes_t0), test_events)
        
    return float(c_t0)

if __name__ == "__main__":
    print("===============================================================")
    print("🔬 [CoxPH 비교 실험] 선형 모델(CoxPH) 하에서의 TCSR vs SurvTD")
    print("   평가 지표: 공식 lifelines Concordance Index (C-index)")
    print("===============================================================\n")
    
    # 1. Synthetic ICU
    print("📍 [Benchmark 1] Synthetic ICU Telemetry (Continuous Irregular)")
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
    
    res_icu = {}
    for m in methods:
        c = train_and_eval_linear_cox(m, icu_train, icu_test, input_dim=data_icu.input_dim, K=36, delta_s=2.0, epochs=15, seed=42)
        res_icu[m] = c
        print(f"  • {m:48s}: C-index = {c:.4f}")
        
    # 2. NASA C-MAPSS FD001
    print("\n📍 [Benchmark 2] NASA C-MAPSS FD001 (Authentic Turbofan)")
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
    
    res_nasa = {}
    for m in methods:
        c = train_and_eval_linear_cox(m, nasa_train, nasa_test, input_dim=len(feat_cols), K=30, delta_s=0.1, epochs=15, seed=42)
        res_nasa[m] = c
        print(f"  • {m:48s}: C-index = {c:.4f}")

    print("\n===============================================================")
    print("📊 [공식 C-index 최종 비교: CoxPH 선형 모델 하에서의 성능]")
    print(f"{'Method':<48s} | {'Synthetic ICU':<14s} | {'NASA FD001':<12s}")
    print("-" * 80)
    for m in methods:
        print(f"{m:<48s} | {res_icu[m]:<14.4f} | {res_nasa[m]:<12.4f}")
    print("===============================================================")
