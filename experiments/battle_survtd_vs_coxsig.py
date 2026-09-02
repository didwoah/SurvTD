"""
================================================================================
SurvTD vs CoxSig (ICML 2024) 공식 벤치마크 배틀: NASA C-MAPSS (FD001)
================================================================================
- 데이터셋: NASA Turbofan Engine Degradation Simulation (FD001)
- 베이스라인: CoxSig (Linus Bleistein et al., ICML 2024 공식 구현체)
- 제안모델: SurvTD(λ) (Survival Bellman Operator + Cramér Distance Loss)
- 평가 메트릭: Brier Score (BS), C-index, Cumulative Dynamic AUC
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

# 1. CoxSig 경로 연동
base_dir = "/Users/yangjaemo/Desktop/SurvTD/baselines/signature_survival"
sys.path.append(base_dir)
os.chdir(base_dir)

from data_loader import load_NASA
from src.coxsig import CoxSignature
from src.utils import score

# 시드 고정
torch.manual_seed(0)
np.random.seed(0)

# ==========================================
# 2. SurvTD 신경망 아키텍처 및 손실 함수 정의
# ==========================================
class SurvTD_NASA(nn.Module):
    """
    NASA 터보팬 센서 시계열용 SurvTD 모델
    - Backbone: 2-layer GRU (시계열 센서 인코더)
    - Head: Discrete Hazard Head (미래 잔여 수명 RUL 버킷별 위험도 추정)
    """
    def __init__(self, input_dim=16, hidden_dim=64, num_buckets=30):
        super().__init__()
        self.K = num_buckets
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.K)
        )
        
    def forward(self, x):
        """
        x: (Batch, Seq_Len, input_dim)
        반환: hazard, survival, pmf, cdf
        """
        out, _ = self.encoder(x)
        logits = self.head(out)
        hazard = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazard + 1e-7, dim=-1)
        s_prev = torch.cat([torch.ones_like(survival[..., :1]), survival[..., :-1]], dim=-1)
        pmf = hazard * s_prev
        cdf = torch.cumsum(pmf, dim=-1)
        return hazard, survival, pmf, cdf

def compute_survival_bellman_target(p_next, S_curr, dt_steps, event, K=30):
    """
    연속시간/가변간격 Survival Bellman Operator:
    T p_j(s) = 1[Event in dt] + S_j(dt) * p_{j+1}(s - dt)
    """
    target_pmf = torch.zeros(K, device=p_next.device)
    dt_int = max(1, min(int(dt_steps), K - 1))
    
    if event > 0.5:
        # 고장(Event) 발생 시 즉시 관측 사실 주입
        target_pmf[min(dt_int - 1, K - 1)] = 1.0
        return target_pmf
        
    # 생존 시: 할인율 S(dt) * 다음 시점 예측 좌측 시프트
    S_dt = S_curr[min(dt_int - 1, K - 1)]
    if dt_int < K:
        shifted_p_next = p_next[:K - dt_int] * S_dt
        target_pmf[dt_int:] = shifted_p_next.detach()
    return target_pmf


def main():
    print("=" * 75)
    print("  🥊 SurvTD vs CoxSig (ICML 2024) 공식 벤치마크 배틀 (NASA FD001)")
    print("=" * 75)

    # 1. 데이터 로드 및 전처리
    paths, surv_labels, _ = load_NASA.load()
    paths = paths.clone()
    n_samples, n_sampling_times, feat_dim = paths.shape
    print(f"• 데이터 규모: 총 {n_samples}개 엔진, {n_sampling_times}개 시점, 피처 차원: {feat_dim}")

    # Train / Test Split (80% / 20%, CoxSig 논문과 완벽히 동일한 시드)
    train_test_share = 0.8
    n_train_samples = int(train_test_share * n_samples)
    train_index = np.random.default_rng(0).choice(n_samples, n_train_samples, replace=False)
    test_index = [i for i in np.arange(n_samples) if i not in train_index]

    paths_train = paths[train_index, :, :]
    surv_labels_train = surv_labels[train_index, :]

    paths_test = paths[test_index, :, :]
    surv_labels_test = surv_labels[test_index, :]

    # 평가 기준 시점 (Quantile 10%, 20%, 40% 고장 시점)
    sampling_times = np.array(paths[0, :, 0])
    tte = surv_labels[surv_labels[:, 1] == 1][:, 0]
    quantile_pred_times = np.array([0.1, 0.2, 0.4])
    pred_times = np.quantile(np.array(tte), quantile_pred_times)
    n_eval_times = 3
    eval_times = []
    for k in range(n_eval_times):
        eval_times.append(max(np.quantile(np.array(tte), quantile_pred_times + (k+1) * 0.05) - pred_times))
    eval_times = np.array(eval_times)

    print(f"• 기준 예측 시점(pred_times): {np.round(pred_times, 3)}")
    print(f"• 미래 평가 지평(eval_times): {np.round(eval_times, 3)}")

    # -------------------------------------------------------------
    # 2. 베이스라인 1: CoxSig (ICML 2024) 실행
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("  [1/2] CoxSig (ICML 2024) 베이스라인 훈련 및 평가")
    print("-" * 75)
    start_cs = time()
    coxsig = CoxSignature(sig_level=2, alphas=1e-5, max_iter=100, plot_loss=False)
    coxsig.train(paths_train, surv_labels_train)
    coxsig_time = time() - start_cs
    print(f"  • CoxSig 훈련 완료 ({coxsig_time:.2f}초)")

    coxsig_bs = coxsig.score(paths_test, surv_labels_test, pred_times, eval_times, "bs")
    coxsig_cindex = coxsig.score(paths_test, surv_labels_test, pred_times, eval_times, 'c_index')
    coxsig_auc = coxsig.score(paths_test, surv_labels_test, pred_times, eval_times, 'auc')

    mean_cs_bs = np.nanmean(coxsig_bs)
    mean_cs_cindex = np.nanmean(coxsig_cindex)
    mean_cs_auc = np.nanmean(coxsig_auc)

    print(f"  • Brier Score: {mean_cs_bs:.4f}  |  C-index: {mean_cs_cindex:.4f}  |  AUC: {mean_cs_auc:.4f}")

    # -------------------------------------------------------------
    # 3. 제안 모델: SurvTD(λ) 훈련
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("  [2/2] SurvTD(λ) 제안 모델 훈련 및 평가")
    print("-" * 75)

    # 미래 예측 시간 버킷 설정 (최대 잔여수명 1.5단위 = 150사이클, 버킷당 dt_res = 0.05 = 5사이클)
    num_buckets = 30
    dt_res = 0.05
    model = SurvTD_NASA(input_dim=feat_dim, hidden_dim=64, num_buckets=num_buckets)
    target_net = copy.deepcopy(model)
    target_net.eval()

    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    tau = 0.90
    epochs = 25

    start_survtd = time()
    # 훈련 데이터셋 구성
    surv_times_train = surv_labels_train[:, 0]
    surv_inds_train = surv_labels_train[:, 1]

    # 각 엔진별 실제 관측 시퀀스 추출
    train_sequences = []
    for i in range(n_train_samples):
        # tte 이전의 유효한 시점들만 선택
        valid_mask = sampling_times <= surv_times_train[i]
        seq = paths_train[i, valid_mask, :] # (Seq_Len, 16)
        train_sequences.append({
            'path': seq,
            'tte': surv_times_train[i],
            'event': surv_inds_train[i],
            'times': sampling_times[valid_mask]
        })

    loss_history = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        step_count = 0
        
        # 엔진별 루프
        for item in train_sequences:
            seq = item['path'].unsqueeze(0) # (1, Seq_Len, 16)
            times_arr = item['times']
            seq_len = seq.shape[1]
            if seq_len < 3:
                continue
                
            hazard, survival, pmf, cdf = model(seq)
            with torch.no_grad():
                _, _, target_pmf_all, _ = target_net(seq)
                
            loss_seq = 0.0
            # 인접 시점 간격마다 Survival Bellman Cramér Loss
            for t in range(seq_len - 1):
                p_next = target_pmf_all[0, t + 1]
                S_curr = survival[0, t]
                dt = (times_arr[t + 1] - times_arr[t])
                dt_steps = max(1, int(round(dt / dt_res)))
                
                # 마지막 전이 구간이고 실제 고장인 경우 event=1
                is_terminal = (t == seq_len - 2) and (item['event'] == 1.0)
                event_flag = 1.0 if is_terminal else 0.0
                
                target_p = compute_survival_bellman_target(p_next, S_curr, dt_steps, event_flag, K=num_buckets)
                target_F = torch.cumsum(target_p, dim=-1)
                pred_F = cdf[0, t]
                
                # Cramér Loss (CDF L2 Loss)
                cramer_loss = torch.mean((pred_F - target_F) ** 2)
                loss_seq += cramer_loss
                step_count += 1
                
            optimizer.zero_grad()
            loss_seq.backward()
            optimizer.step()
            total_loss += loss_seq.item()
            
            # EMA Target Net 업데이트
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target_net.parameters()):
                    tp.data.mul_(tau).add_(p.data, alpha=(1.0 - tau))
                    
        avg_loss = total_loss / max(step_count, 1)
        loss_history.append(avg_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs}]  |  SurvTD Cramér Loss: {avg_loss:.6f}")

    survtd_time = time() - start_survtd
    print(f"  • SurvTD 훈련 완료 ({survtd_time:.2f}초)")

    # -------------------------------------------------------------
    # 4. SurvTD 테스트셋 조건부 생존 확률 예측 및 평가
    # -------------------------------------------------------------
    model.eval()
    n_test = len(test_index)
    surv_times_test = surv_labels_test[:, 0]
    surv_inds_test = surv_labels_test[:, 1]

    # CoxSig의 score 함수와 정확히 동일한 방식으로 평가
    survtd_bs = np.zeros((len(pred_times), n_eval_times))
    survtd_cindex = np.zeros((len(pred_times), n_eval_times))
    survtd_auc = np.zeros((len(pred_times), n_eval_times))

    eps = 1e-4
    for j, pred_time in enumerate(pred_times):
        # pred_time 시점까지의 경로를 입력으로 전달
        mask = sampling_times <= pred_time + eps
        
        # pred_time 시점에 살아있는 엔진들만 평가 대상
        alive_mask = surv_times_test >= pred_time
        test_indices_alive = np.where(alive_mask)[0]
        
        # SurvTD 예측 계산
        cond_surv_preds_j = np.zeros((len(test_indices_alive), n_eval_times))
        
        for idx_local, idx_global in enumerate(test_indices_alive):
            seq_test = paths_test[idx_global, mask, :].unsqueeze(0) # (1, T_pred, 16)
            with torch.no_grad():
                _, survival, _, _ = model(seq_test)
                # 마지막 관측 시점(pred_time)에서의 생존 곡선 S_t(s)
                surv_curve = survival[0, -1].numpy() # (K,)
                
            # eval_times 각각에 대한 생존 확률 보간 (s = eval_time)
            for k, ev_t in enumerate(eval_times):
                k_bucket = int(round(ev_t / dt_res))
                k_bucket = max(0, min(k_bucket, num_buckets - 1))
                cond_surv_preds_j[idx_local, k] = surv_curve[k_bucket]

        surv_times_ = surv_times_test[alive_mask] - pred_time
        surv_inds_ = surv_inds_test[alive_mask]
        surv_labels_ = np.array([surv_times_, surv_inds_]).T

        survtd_bs[j] = score("bs", surv_labels_, surv_labels_, cond_surv_preds_j, eval_times)
        survtd_cindex[j] = score("c_index", surv_labels_, surv_labels_, cond_surv_preds_j, eval_times)
        survtd_auc[j] = score("auc", surv_labels_, surv_labels_, cond_surv_preds_j, eval_times)

    mean_std_bs = np.nanmean(survtd_bs)
    mean_std_cindex = np.nanmean(survtd_cindex)
    mean_std_auc = np.nanmean(survtd_auc)

    # -------------------------------------------------------------
    # 5. 최종 결과 비교 표 출력
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  🏆 [최종 배틀 결과] SurvTD vs CoxSig (ICML 2024) 종합 비교")
    print("=" * 75)
    print(f"{'평가 지표':<25} | {'CoxSig (ICML 2024)':<20} | {'SurvTD (제안 모델)':<20} | {'승자':<10}")
    print("-" * 75)
    
    bs_winner = "SurvTD 🏆" if mean_std_bs < mean_cs_bs else "CoxSig"
    ci_winner = "SurvTD 🏆" if mean_std_cindex > mean_cs_cindex else "CoxSig"
    auc_winner = "SurvTD 🏆" if mean_std_auc > mean_cs_auc else "CoxSig"
    time_winner = "SurvTD 🏆" if survtd_time < coxsig_time else "CoxSig"

    print(f"{'Brier Score (낮을수록 우수)':<22} | {mean_cs_bs:<20.4f} | {mean_std_bs:<20.4f} | {bs_winner}")
    print(f"{'C-index     (높을수록 우수)':<22} | {mean_cs_cindex:<20.4f} | {mean_std_cindex:<20.4f} | {ci_winner}")
    print(f"{'Dynamic AUC (높을수록 우수)':<22} | {mean_cs_auc:<20.4f} | {mean_std_auc:<20.4f} | {auc_winner}")
    print(f"{'훈련 소요 시간':<24} | {f'{coxsig_time:.2f}초':<20} | {f'{survtd_time:.2f}초':<20} | {time_winner}")
    print("=" * 75)

    # -------------------------------------------------------------
    # 6. 배틀 결과 시각화 차트 생성
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    pred_labels = [f"Q1 (t={pred_times[0]:.2f})", f"Q2 (t={pred_times[1]:.2f})", f"Q3 (t={pred_times[2]:.2f})"]
    x = np.arange(len(pred_labels))
    width = 0.35

    # 1. Brier Score 바 차트
    ax = axes[0]
    ax.bar(x - width/2, np.nanmean(coxsig_bs, axis=1), width, label='CoxSig (ICML 2024)', color='#7f7f7f')
    ax.bar(x + width/2, np.nanmean(survtd_bs, axis=1), width, label='SurvTD (Ours)', color='#1f77b4')
    ax.set_title("Brier Score by Quantile (Lower is Better)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pred_labels)
    ax.set_ylabel("Brier Score")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # 2. C-index 바 차트
    ax = axes[1]
    ax.bar(x - width/2, np.nanmean(coxsig_cindex, axis=1), width, label='CoxSig (ICML 2024)', color='#7f7f7f')
    ax.bar(x + width/2, np.nanmean(survtd_cindex, axis=1), width, label='SurvTD (Ours)', color='#2ca02c')
    ax.set_title("C-index by Quantile (Higher is Better)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pred_labels)
    ax.set_ylabel("C-index")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # 3. Dynamic AUC 바 차트
    ax = axes[2]
    ax.bar(x - width/2, np.nanmean(coxsig_auc, axis=1), width, label='CoxSig (ICML 2024)', color='#7f7f7f')
    ax.bar(x + width/2, np.nanmean(survtd_auc, axis=1), width, label='SurvTD (Ours)', color='#d62728')
    ax.set_title("Dynamic AUC by Quantile (Higher is Better)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pred_labels)
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_path = "/Users/yangjaemo/Desktop/SurvTD/nasa_survtd_vs_coxsig.png"
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    print(f"\n📊 배틀 결과 차트 저장 완료: {chart_path}")

if __name__ == '__main__':
    main()
