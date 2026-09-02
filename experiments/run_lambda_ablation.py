"""
================================================================================
SurvTD(λ) 핵심 이론 검증: λ-스펙트럼 어블레이션 연구 (Ablation Study)
================================================================================
- 목적: TD(0) [1-step 부트스트래핑]부터 TD(1) [순수 몬테카를로/NLL]까지
        λ ∈ [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] 스펙트럼에서 편향-분산 트레이드오프 실증
- 데이터셋: NASA C-MAPSS FD001 (실측 제트엔진 시계열)
- 평가지표: Time-dependent C-index, Dynamic AUC, Brier Score
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

from data_loader import load_NASA
from src.utils import score

torch.manual_seed(0)
np.random.seed(0)

# ==========================================
# 1. SurvTD 모델 정의
# ==========================================
class SurvTD_Model(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=64, num_buckets=30):
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

# =========================================================
# 2. SurvTD(λ) λ-Return 계산 함수 (기하학적 혼합 재귀식)
# =========================================================
def compute_lambda_returns(target_pmfs, survivals, dt_list, is_failure, lam, K=30, dt_res=0.05):
    """
    시퀀스 전체에 대해 SurvTD(λ)의 기하학적 λ-타깃 시퀀스를 후진 재귀(Backward Recursion)로 계산
    G^\lambda_j(s) = I_j + S_j(\Delta t) * [ (1 - \lambda) * p_{j+1}(s - \Delta t) + \lambda * G^\lambda_{j+1}(s - \Delta t) ]
    """
    seq_len = target_pmfs.shape[0]
    G_lambda = [None] * seq_len
    
    # 마지막 시점(종단)의 타깃 초기화
    G_lambda[-1] = target_pmfs[-1].clone().detach()
    
    # t = seq_len - 2 부터 0까지 거꾸로 재귀 계산 (Backward View)
    for t in range(seq_len - 2, -1, -1):
        dt = dt_list[t]
        dt_steps = max(1, min(int(round(dt / dt_res)), K - 1))
        
        is_terminal = (t == seq_len - 2) and is_failure
        target_p = torch.zeros(K, device=target_pmfs.device)
        
        if is_terminal:
            # 최종 고장 시점: 즉시 보상 I = 1
            target_p[min(dt_steps - 1, K - 1)] = 1.0
            G_lambda[t] = target_p
        else:
            S_dt = survivals[t, min(dt_steps - 1, K - 1)]
            
            # 1-step 부트스트랩 예측 p_{t+1}(s - dt)
            p_next_shifted = torch.zeros(K, device=target_pmfs.device)
            if dt_steps < K:
                p_next_shifted[dt_steps:] = target_pmfs[t + 1, :K - dt_steps] * S_dt
                
            # 다단계 λ-리턴 G^\lambda_{t+1}(s - dt)
            G_next_shifted = torch.zeros(K, device=target_pmfs.device)
            if dt_steps < K:
                G_next_shifted[dt_steps:] = G_lambda[t + 1][:K - dt_steps] * S_dt
                
            # λ에 따른 기하학적 보간: (1 - λ) * 1-Step + λ * 다단계 누적
            G_lambda[t] = ((1.0 - lam) * p_next_shifted + lam * G_next_shifted).detach()
            
    return G_lambda


def train_and_eval_survtd_lambda(lam, train_sequences, paths_test, surv_labels_test, sampling_times, pred_times, eval_times, feat_dim=17):
    """
    특정 λ 값에 대해 SurvTD(λ) 모델을 훈련하고 C-index, AUC, Brier Score 평가
    """
    print(f"\n--- [실행 중] λ = {lam:.2f} ---")
    torch.manual_seed(0)
    np.random.seed(0)
    
    num_buckets = 30
    dt_res = 0.05
    model = SurvTD_Model(input_dim=feat_dim, hidden_dim=64, num_buckets=num_buckets)
    target_net = copy.deepcopy(model)
    target_net.eval()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    tau = 0.90
    epochs = 20
    
    start_time = time()
    for epoch in range(1, epochs + 1):
        for item in train_sequences:
            seq = item['path'].unsqueeze(0) # (1, Seq_Len, 17)
            times_arr = item['times']
            seq_len = seq.shape[1]
            if seq_len < 3:
                continue
                
            hazard, survival, pmf, cdf = model(seq)
            with torch.no_grad():
                _, _, target_pmf_all, _ = target_net(seq)
                
            # 시퀀스 내 각 전이의 dt 계산
            dt_list = [times_arr[i+1] - times_arr[i] for i in range(seq_len - 1)]
            is_failure = (item['event'] == 1.0)
            
            # SurvTD(λ)의 λ-Return 타깃 시퀀스 생성
            G_lambda_seq = compute_lambda_returns(
                target_pmf_all[0], survival[0], dt_list, is_failure, lam=lam, K=num_buckets, dt_res=dt_res
            )
            
            loss_seq = 0.0
            step_stride = max(1, seq_len // 25)
            for t in range(0, seq_len - 1, step_stride):
                target_p = G_lambda_seq[t]
                target_F = torch.cumsum(target_p, dim=-1)
                pred_F = cdf[0, t]
                
                # Cramér Loss
                cramer_loss = torch.mean((pred_F - target_F) ** 2)
                loss_seq += cramer_loss
                
            optimizer.zero_grad()
            loss_seq.backward()
            optimizer.step()
            
            # EMA 타깃 망 업데이트
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target_net.parameters()):
                    tp.data.mul_(tau).add_(p.data, alpha=(1.0 - tau))
                    
    elapsed = time() - start_time
    
    # 테스트셋 평가
    model.eval()
    surv_times_test = surv_labels_test[:, 0]
    surv_inds_test = surv_labels_test[:, 1]
    n_eval_times = len(eval_times)
    
    bs_arr = np.zeros((len(pred_times), n_eval_times))
    cindex_arr = np.zeros((len(pred_times), n_eval_times))
    auc_arr = np.zeros((len(pred_times), n_eval_times))
    
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

        bs_arr[j] = score("bs", surv_labels_, surv_labels_, cond_surv_preds_j, eval_times)
        cindex_arr[j] = score("c_index", surv_labels_, surv_labels_, cond_surv_preds_j, eval_times)
        auc_arr[j] = score("auc", surv_labels_, surv_labels_, cond_surv_preds_j, eval_times)
        
    res = {
        'lambda': lam,
        'cindex': np.nanmean(cindex_arr),
        'auc': np.nanmean(auc_arr),
        'bs': np.nanmean(bs_arr),
        'time': elapsed
    }
    print(f"  결과 (λ={lam:.2f}) -> C-index: {res['cindex']:.4f} | AUC: {res['auc']:.4f} | Brier: {res['bs']:.4f} ({elapsed:.1f}s)")
    return res


def main():
    print("=" * 80)
    print("  🔬 SurvTD(λ) λ-스펙트럼 어블레이션 연구 (NASA C-MAPSS FD001)")
    print("=" * 80)
    
    # 데이터 로드
    paths, surv_labels, _ = load_NASA.load(1)
    paths = paths.clone()
    n_samples, n_sampling_times, feat_dim = paths.shape
    
    train_test_share = 0.8
    n_train_samples = int(train_test_share * n_samples)
    train_index = np.random.default_rng(0).choice(n_samples, n_train_samples, replace=False)
    test_index = [i for i in np.arange(n_samples) if i not in train_index]

    paths_train = paths[train_index, :, :]
    surv_labels_train = surv_labels[train_index, :]
    paths_test = paths[test_index, :, :]
    surv_labels_test = surv_labels[test_index, :]

    sampling_times = np.array(paths[0, :, 0])
    tte = surv_labels[surv_labels[:, 1] == 1][:, 0]
    quantile_pred_times = np.array([0.1, 0.2, 0.4])
    pred_times = np.quantile(np.array(tte), quantile_pred_times)
    n_eval_times = 3
    eval_times = []
    for k in range(n_eval_times):
        eval_times.append(max(np.quantile(np.array(tte), quantile_pred_times + (k+1) * 0.05) - pred_times))
    eval_times = np.array(eval_times)

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

    # λ 스펙트럼 스윕: 0.0 (Pure 1-Step TD) 부터 1.0 (Pure Monte Carlo / NLL)
    lambda_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []
    
    for lam in lambda_list:
        res = train_and_eval_survtd_lambda(
            lam, train_sequences, paths_test, surv_labels_test, sampling_times, pred_times, eval_times, feat_dim=feat_dim
        )
        results.append(res)
        
    # -------------------------------------------------------------
    # 종합 표 출력
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  📊 [λ-스펙트럼 어블레이션 종합 결과표] Bias-Variance Tradeoff")
    print("=" * 80)
    print(f"{'λ 값':<10} | {'학습 패러다임':<22} | {'C-index ⬆️':<14} | {'Dynamic AUC ⬆️':<16} | {'Brier Score ⬇️':<14}")
    print("-" * 80)
    
    for r in results:
        lam = r['lambda']
        if lam == 0.0:
            paradigm = "Pure 1-Step TD"
        elif lam == 1.0:
            paradigm = "Pure Monte Carlo / NLL"
        else:
            paradigm = f"SurvTD(λ={lam:.1f}) Blend"
            
        print(f"{lam:<10.1f} | {paradigm:<22} | {r['cindex']:<14.4f} | {r['auc']:<16.4f} | {r['bs']:<14.4f}")
    print("=" * 80)
    
    # -------------------------------------------------------------
    # 시각화 그래프 생성
    # -------------------------------------------------------------
    lams = [r['lambda'] for r in results]
    cindices = [r['cindex'] for r in results]
    aucs = [r['auc'] for r in results]
    bss = [r['bs'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    
    # 1. C-index vs λ
    ax = axes[0]
    ax.plot(lams, cindices, 'o-', color='#2ca02c', linewidth=2.5, markersize=8)
    best_idx = np.argmax(cindices)
    ax.plot(lams[best_idx], cindices[best_idx], '*', color='red', markersize=16, label=f'Best λ={lams[best_idx]} ({cindices[best_idx]:.4f})')
    ax.set_title("C-index vs λ (Discrimination)", fontweight='bold')
    ax.set_xlabel("λ (0.0: TD -> 1.0: Monte Carlo)")
    ax.set_ylabel("C-index")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # 2. Dynamic AUC vs λ
    ax = axes[1]
    ax.plot(lams, aucs, 's-', color='#d62728', linewidth=2.5, markersize=8)
    best_auc_idx = np.argmax(aucs)
    ax.plot(lams[best_auc_idx], aucs[best_auc_idx], '*', color='blue', markersize=16, label=f'Best λ={lams[best_auc_idx]} ({aucs[best_auc_idx]:.4f})')
    ax.set_title("Dynamic AUC vs λ (Early Alarm)", fontweight='bold')
    ax.set_xlabel("λ (0.0: TD -> 1.0: Monte Carlo)")
    ax.set_ylabel("AUC")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # 3. Brier Score vs λ
    ax = axes[2]
    ax.plot(lams, bss, '^-', color='#1f77b4', linewidth=2.5, markersize=8)
    best_bs_idx = np.argmin(bss)
    ax.plot(lams[best_bs_idx], bss[best_bs_idx], '*', color='gold', markersize=16, label=f'Best λ={lams[best_bs_idx]} ({bss[best_bs_idx]:.4f})')
    ax.set_title("Brier Score vs λ (Calibration Error)", fontweight='bold')
    ax.set_xlabel("λ (0.0: TD -> 1.0: Monte Carlo)")
    ax.set_ylabel("Brier Score (Lower is Better)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    plt.tight_layout()
    chart_path = "/Users/yangjaemo/Desktop/SurvTD/lambda_ablation_study.png"
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    print(f"\n📊 λ-어블레이션 연구 차트 저장 완료: {chart_path}")

if __name__ == '__main__':
    main()
