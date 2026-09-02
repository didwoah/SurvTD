"""
================================================================================
SurvTD vs CoxSig (ICML 2024): 논문 공식 3대 벤치마크 전체 1:1 배틀
================================================================================
- 벤치마크 1: OU Process (Ornstein-Uhlenbeck 연속 확률 미분방정식 경로)
- 벤치마크 2: Tumor Growth (종양 증식 바이오피지컬 연속 ODE 프로세스)
- 벤치마크 3: NASA C-MAPSS FD001 (산업용 제트엔진 실측 마모 열화)
"""

import os
import sys
import copy
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

base_dir = "/Users/yangjaemo/Desktop/SurvTD/baselines/signature_survival"
sys.path.append(base_dir)
os.chdir(base_dir)

from data_loader import load_OU, load_Tumor_Growth, load_NASA
from src.coxsig import CoxSignature
from src.utils import score

torch.manual_seed(0)
np.random.seed(0)

# ==========================================
# SurvTD 신경망 아키텍처 및 손실 함수
# ==========================================
class SurvTD_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_buckets=30):
        super().__init__()
        self.K = num_buckets
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.K)
        )
        
    def forward(self, x):
        out, _ = self.encoder(x)
        logits = self.head(out)
        hazard = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazard + 1e-7, dim=-1)
        s_prev = torch.cat([torch.ones_like(survival[..., :1]), survival[..., :-1]], dim=-1)
        pmf = hazard * s_prev
        cdf = torch.cumsum(pmf, dim=-1)
        return hazard, survival, pmf, cdf

def compute_survival_bellman_target(p_next, S_curr, dt_steps, event, K=30):
    target_pmf = torch.zeros(K, device=p_next.device)
    dt_int = max(1, min(int(dt_steps), K - 1))
    
    if event > 0.5:
        target_pmf[min(dt_int - 1, K - 1)] = 1.0
        return target_pmf
        
    S_dt = S_curr[min(dt_int - 1, K - 1)]
    if dt_int < K:
        shifted_p_next = p_next[:K - dt_int] * S_dt
        target_pmf[dt_int:] = shifted_p_next.detach()
    return target_pmf


def run_benchmark(dataset_name, paths, surv_labels):
    print("\n" + "=" * 75)
    print(f"  🥊 [{dataset_name}] 배틀 시작")
    print("=" * 75)
    paths = paths.clone()
    n_samples, n_sampling_times, feat_dim = paths.shape
    print(f"• 데이터 형상: {n_samples}개 샘플, {n_sampling_times}개 시점, {feat_dim}개 피처")

    # 80/20 Train/Test Split (Seed 0)
    train_test_share = 0.8
    n_train_samples = int(train_test_share * n_samples)
    train_index = np.random.default_rng(0).choice(n_samples, n_train_samples, replace=False)
    test_index = [i for i in np.arange(n_samples) if i not in train_index]

    paths_train = paths[train_index, :, :]
    surv_labels_train = surv_labels[train_index, :]

    paths_test = paths[test_index, :, :]
    surv_labels_test = surv_labels[test_index, :]

    # 평가 기준 시점 (Quantile 10%, 20%, 40%)
    sampling_times = np.array(paths[0, :, 0])
    tte = surv_labels[surv_labels[:, 1] == 1][:, 0]
    quantile_pred_times = np.array([0.1, 0.2, 0.4])
    pred_times = np.quantile(np.array(tte), quantile_pred_times)
    n_eval_times = 3
    eval_times = []
    for k in range(n_eval_times):
        eval_times.append(max(np.quantile(np.array(tte), quantile_pred_times + (k+1) * 0.05) - pred_times))
    eval_times = np.array(eval_times)

    # 1. CoxSig 실행
    print(f"  [1/2] CoxSig (ICML 2024) 훈련 및 평가...")
    start_cs = time()
    coxsig = CoxSignature(sig_level=2, alphas=1e-3 if "OU" in dataset_name else 1e-5, max_iter=100, plot_loss=False)
    coxsig.train(paths_train, surv_labels_train)
    cs_time = time() - start_cs

    coxsig_bs = coxsig.score(paths_test, surv_labels_test, pred_times, eval_times, "bs")
    coxsig_cindex = coxsig.score(paths_test, surv_labels_test, pred_times, eval_times, 'c_index')
    coxsig_auc = coxsig.score(paths_test, surv_labels_test, pred_times, eval_times, 'auc')

    res_cs = {
        'bs': np.nanmean(coxsig_bs),
        'cindex': np.nanmean(coxsig_cindex),
        'auc': np.nanmean(coxsig_auc),
        'time': cs_time
    }
    print(f"  ✅ CoxSig 완료 (시간: {cs_time:.1f}s, C-index: {res_cs['cindex']:.4f}, AUC: {res_cs['auc']:.4f})")

    # 2. SurvTD 실행
    print(f"  [2/2] SurvTD (제안 모델) 훈련 및 평가...")
    num_buckets = 30
    dt_res = (np.max(tte) - np.min(pred_times)) / num_buckets
    if dt_res <= 0:
        dt_res = 0.05

    model = SurvTD_Model(input_dim=feat_dim, hidden_dim=64, num_buckets=num_buckets)
    target_net = copy.deepcopy(model)
    target_net.eval()

    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    tau = 0.90
    epochs = 20

    surv_times_train = surv_labels_train[:, 0]
    surv_inds_train = surv_labels_train[:, 1]
    train_sequences = []
    for i in range(n_train_samples):
        valid_mask = sampling_times <= surv_times_train[i]
        seq = paths_train[i, valid_mask, :]
        train_sequences.append({
            'path': seq,
            'tte': surv_times_train[i],
            'event': surv_inds_train[i],
            'times': sampling_times[valid_mask]
        })

    start_std = time()
    for epoch in range(1, epochs + 1):
        for item in train_sequences:
            seq = item['path'].unsqueeze(0)
            times_arr = item['times']
            seq_len = seq.shape[1]
            if seq_len < 3:
                continue
                
            hazard, survival, pmf, cdf = model(seq)
            with torch.no_grad():
                _, _, target_pmf_all, _ = target_net(seq)
                
            loss_seq = 0.0
            step_stride = max(1, seq_len // 25)
            for t in range(0, seq_len - 1, step_stride):
                p_next = target_pmf_all[0, t + 1]
                S_curr = survival[0, t]
                dt = (times_arr[t + 1] - times_arr[t])
                dt_steps = max(1, int(round(dt / dt_res)))
                
                is_terminal = (t >= seq_len - 2) and (item['event'] == 1.0)
                event_flag = 1.0 if is_terminal else 0.0
                
                target_p = compute_survival_bellman_target(p_next, S_curr, dt_steps, event_flag, K=num_buckets)
                target_F = torch.cumsum(target_p, dim=-1)
                pred_F = cdf[0, t]
                
                cramer_loss = torch.mean((pred_F - target_F) ** 2)
                loss_seq += cramer_loss
                
            optimizer.zero_grad()
            loss_seq.backward()
            optimizer.step()
            
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target_net.parameters()):
                    tp.data.mul_(tau).add_(p.data, alpha=(1.0 - tau))

    std_time = time() - start_std

    # SurvTD 평가
    model.eval()
    surv_times_test = surv_labels_test[:, 0]
    surv_inds_test = surv_labels_test[:, 1]
    survtd_bs = np.zeros((len(pred_times), n_eval_times))
    survtd_cindex = np.zeros((len(pred_times), n_eval_times))
    survtd_auc = np.zeros((len(pred_times), n_eval_times))

    eps = 1e-4
    for j, pred_time in enumerate(pred_times):
        mask = sampling_times <= pred_time + eps
        alive_mask = surv_times_test >= pred_time
        test_indices_alive = np.where(alive_mask)[0]
        
        cond_surv_preds_j = np.zeros((len(test_indices_alive), n_eval_times))
        for idx_local, idx_global in enumerate(test_indices_alive):
            seq_test = paths_test[idx_global, mask, :].unsqueeze(0)
            with torch.no_grad():
                _, survival, _, _ = model(seq_test)
                surv_curve = survival[0, -1].numpy()
                
            for k, ev_t in enumerate(eval_times):
                k_bucket = max(0, min(int(round(ev_t / dt_res)), num_buckets - 1))
                cond_surv_preds_j[idx_local, k] = surv_curve[k_bucket]

        surv_times_ = surv_times_test[alive_mask] - pred_time
        surv_inds_ = surv_inds_test[alive_mask]
        surv_labels_ = np.array([surv_times_, surv_inds_]).T

        survtd_bs[j] = score("bs", surv_labels_, surv_labels_, cond_surv_preds_j, eval_times)
        survtd_cindex[j] = score("c_index", surv_labels_, surv_labels_, cond_surv_preds_j, eval_times)
        survtd_auc[j] = score("auc", surv_labels_, surv_labels_, cond_surv_preds_j, eval_times)

    res_std = {
        'bs': np.nanmean(survtd_bs),
        'cindex': np.nanmean(survtd_cindex),
        'auc': np.nanmean(survtd_auc),
        'time': std_time
    }
    print(f"  ✅ SurvTD 완료 (시간: {std_time:.1f}s, C-index: {res_std['cindex']:.4f}, AUC: {res_std['auc']:.4f})")

    return res_cs, res_std


def main():
    print("=" * 80)
    print("  🏆 [CoxSig 논문 공식 3대 벤치마크] SurvTD vs CoxSig (ICML 2024) 배틀")
    print("=" * 80)

    # 1. OU Process
    print("\n[데이터 로딩 1/3] Ornstein-Uhlenbeck (OU) 생성...")
    paths_ou, surv_labels_ou, _ = load_OU.load(n_samples=250, n_sampling_times=100)
    cs_ou, std_ou = run_benchmark("OU Process (Stochastic CDE)", paths_ou, surv_labels_ou)

    # 2. Tumor Growth
    print("\n[데이터 로딩 2/3] Tumor Growth 생성...")
    paths_tg, surv_labels_tg, _ = load_Tumor_Growth.load(n_samples=250, n_sampling_times=100)
    cs_tg, std_tg = run_benchmark("Tumor Growth (Biophysical ODE)", paths_tg, surv_labels_tg)

    # 3. NASA C-MAPSS FD001
    print("\n[데이터 로딩 3/3] NASA C-MAPSS FD001 로딩...")
    paths_nasa, surv_labels_nasa, _ = load_NASA.load(1)
    cs_nasa, std_nasa = run_benchmark("NASA Turbofan (FD001)", paths_nasa, surv_labels_nasa)

    # -------------------------------------------------------------
    # 종합 결과표
    # -------------------------------------------------------------
    all_benchmarks = [
        ("OU Process (SDE)", cs_ou, std_ou),
        ("Tumor Growth (ODE)", cs_tg, std_tg),
        ("NASA C-MAPSS (FD001)", cs_nasa, std_nasa)
    ]

    print("\n" + "=" * 80)
    print("  📊 [최종 공식 벤치마크 결과표 (Table 1)] CoxSig 논문 전 벤치마크 비교")
    print("=" * 80)
    print(f"{'Benchmark Dataset':<24} | {'Metric':<12} | {'CoxSig (ICML 2024)':<20} | {'SurvTD (Ours)':<18} | {'승자':<10}")
    print("-" * 80)

    for name, cs, std in all_benchmarks:
        ci_winner = "SurvTD 🏆" if std['cindex'] > cs['cindex'] else "CoxSig"
        auc_winner = "SurvTD 🏆" if std['auc'] > cs['auc'] else "CoxSig"
        bs_winner = "SurvTD 🏆" if std['bs'] < cs['bs'] else "CoxSig"

        print(f"{name:<24} | {'C-index':<12} | {cs['cindex']:<20.4f} | {std['cindex']:<18.4f} | {ci_winner}")
        print(f"{'':<24} | {'Dynamic AUC':<12} | {cs['auc']:<20.4f} | {std['auc']:<18.4f} | {auc_winner}")
        print(f"{'':<24} | {'Brier Score':<12} | {cs['bs']:<20.4f} | {std['bs']:<18.4f} | {bs_winner}")
        print("-" * 80)

    # -------------------------------------------------------------
    # 차트 생성
    # -------------------------------------------------------------
    ds_labels = ["OU Process", "Tumor Growth", "NASA FD001"]
    x = np.arange(len(ds_labels))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    # C-index
    ax = axes[0]
    ax.bar(x - width/2, [b[1]['cindex'] for b in all_benchmarks], width, label='CoxSig (ICML 2024)', color='#7f7f7f')
    ax.bar(x + width/2, [b[2]['cindex'] for b in all_benchmarks], width, label='SurvTD (Ours)', color='#2ca02c')
    ax.set_title("C-index across Paper Benchmarks (Higher is Better)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("C-index")
    ax.set_ylim(0.40, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # AUC
    ax = axes[1]
    ax.bar(x - width/2, [b[1]['auc'] for b in all_benchmarks], width, label='CoxSig (ICML 2024)', color='#7f7f7f')
    ax.bar(x + width/2, [b[2]['auc'] for b in all_benchmarks], width, label='SurvTD (Ours)', color='#d62728')
    ax.set_title("Dynamic AUC across Benchmarks (Higher is Better)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.40, 1.05)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # Brier
    ax = axes[2]
    ax.bar(x - width/2, [b[1]['bs'] for b in all_benchmarks], width, label='CoxSig (ICML 2024)', color='#7f7f7f')
    ax.bar(x + width/2, [b[2]['bs'] for b in all_benchmarks], width, label='SurvTD (Ours)', color='#1f77b4')
    ax.set_title("Brier Score across Benchmarks (Lower is Better)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("Brier Score")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = "/Users/yangjaemo/Desktop/SurvTD/coxsig_paper_benchmarks.png"
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    print(f"\n📊 논문 공식 벤치마크 전체 비교 차트 저장 완료: {chart_path}")

if __name__ == '__main__':
    main()
