"""
SurvTD 학습 결과 고품질 시각화 스크립트 (버그 수정 및 완벽 렌더링)
================================================================
- Terminal Event가 [t, t+dt) 전이 구간에 올바르게 주입되어 위험 신호가 시간 축을 타고 역전파되는 과정 시각화
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. 모델 정의
# ==========================================
class SurvTDModel(nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=64, num_buckets=24):
        super().__init__()
        self.K = num_buckets
        self.encoder = nn.GRU(input_size=feature_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
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

def compute_survival_bellman_target(p_next, S_curr, dt, event, K=24):
    target_pmf = torch.zeros(K, device=p_next.device)
    dt_int = max(1, min(int(dt), K - 1))
    
    if event > 0.5:
        # [t, t+dt) 구간 내 실제 이벤트 발생 (즉시 관측 Fact 주입)
        target_pmf[dt_int - 1] = 1.0
        return target_pmf
        
    # 생존 시: 할인율 S(dt) * 다음 시점 예측 좌측 시프트
    S_dt = S_curr[dt_int - 1]
    if dt_int < K:
        shifted_p_next = p_next[:K - dt_int] * S_dt
        target_pmf[dt_int:] = shifted_p_next.detach()
    return target_pmf

# ==========================================
# 2. 임상 데이터 생성기 (전이 구간에 이벤트 매핑)
# ==========================================
def generate_clinical_patients(num_patients=100, num_buckets=24):
    data = []
    for pid in range(num_patients):
        seq_len = np.random.randint(4, 7)
        is_high_risk = (pid % 2 == 0)
        
        dts = np.random.randint(1, 4, size=seq_len).astype(np.float32)
        
        features = []
        val = 2.0 if is_high_risk else 0.4
        for t in range(seq_len):
            trend = 0.4 if is_high_risk else -0.05
            val += np.random.randn() * 0.1 + trend
            features.append([val, val * 1.2, np.sin(t * 0.5), np.cos(t * 0.5)])
            
        # transition_events[t] : t -> t+1 전이 구간에서 이벤트 발생 여부
        transition_events = np.zeros(seq_len - 1, dtype=np.float32)
        if is_high_risk:
            transition_events[-1] = 1.0 # 마지막 전이 구간에서 이벤트 발생!
            
        data.append({
            'features': torch.tensor(np.array(features, dtype=np.float32)),
            'dts': torch.tensor(dts),
            'transition_events': torch.tensor(transition_events),
            'is_high_risk': is_high_risk
        })
    return data

# ==========================================
# 3. 학습 및 시각화 실행
# ==========================================
def main():
    K = 24
    dataset = generate_clinical_patients(num_patients=100, num_buckets=K)
    model = SurvTDModel(feature_dim=4, hidden_dim=64, num_buckets=K)
    target_net = copy.deepcopy(model)
    target_net.eval()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    tau = 0.90
    epochs = 40
    loss_history = []
    
    print("🚀 SurvTD 훈련 중...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0
        for patient in dataset:
            features = patient['features'].unsqueeze(0)
            dts = patient['dts']
            trans_events = patient['transition_events']
            seq_len = features.shape[1]
            
            hazard, survival, pmf, cdf = model(features)
            with torch.no_grad():
                _, _, target_pmf_all, _ = target_net(features)
                
            loss_patient = 0.0
            for t in range(seq_len - 1):
                p_next = target_pmf_all[0, t + 1]
                S_curr = survival[0, t]
                dt = dts[t].item()
                event = trans_events[t].item()
                
                target_p = compute_survival_bellman_target(p_next, S_curr, dt, event, K=K)
                target_F = torch.cumsum(target_p, dim=-1)
                pred_F = cdf[0, t]
                
                cramer_loss = torch.mean((pred_F - target_F) ** 2)
                loss_patient += cramer_loss
                steps += 1
                
            optimizer.zero_grad()
            loss_patient.backward()
            optimizer.step()
            total_loss += loss_patient.item()
            
            with torch.no_grad():
                for p, tp in zip(model.parameters(), target_net.parameters()):
                    tp.data.mul_(tau).add_(p.data, alpha=(1.0 - tau))
                    
        loss_history.append(total_loss / max(steps, 1))
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs}] - Cramér Loss: {loss_history[-1]:.6f}")
            
    # ==========================================
    # 4. 고품질 플롯 렌더링
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)
    time_horizons = np.arange(1, K + 1)
    
    # ----------------------------------------------------
    # [Panel 1] Loss Convergence Curve
    # ----------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(range(1, epochs + 1), loss_history, color='#1f77b4', lw=2.8, marker='o', markersize=4)
    ax1.set_title("1. Cramér TD Loss Convergence (Training)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Training Epoch", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Cramér Loss (CDF L2)", fontsize=11, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.annotate(f"Final Loss: {loss_history[-1]:.4f}", 
                 xy=(epochs, loss_history[-1]), xytext=(epochs - 14, loss_history[-1] * 2.5),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=2),
                 fontsize=11, fontweight='bold', color='#d62728')
    
    # ----------------------------------------------------
    # [Panel 2] Dynamic Risk Progression (Deteriorating Patient)
    # ----------------------------------------------------
    ax2 = axes[0, 1]
    high_risk_patient = [p for p in dataset if p['is_high_risk']][0]
    with torch.no_grad():
        _, _, _, hr_cdf = model(high_risk_patient['features'].unsqueeze(0))
        
    seq_len = high_risk_patient['features'].shape[0]
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.95, seq_len))
    
    for t in range(seq_len):
        risk_curve = hr_cdf[0, t].numpy()
        label = f"Visit t={t} (Pre-Event Alert!)" if t == seq_len - 2 else f"Visit t={t} (Baseline)" if t == 0 else f"Visit t={t}"
        ax2.plot(time_horizons, risk_curve, label=label, color=colors[t], lw=2.5)
        
    ax2.axhline(0.5, color='gray', linestyle=':', label='Clinical Alarm Threshold (50%)')
    ax2.set_title("2. Dynamic Risk Progression F(s) (Lead-Time Early Alert)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Prediction Horizon s (Hours Ahead)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Cumulative Risk Probability F(s)", fontsize=11, fontweight='bold')
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(loc='upper left', frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # ----------------------------------------------------
    # [Panel 3] High-Risk vs Low-Risk Patient Comparison (Survival S(s))
    # ----------------------------------------------------
    ax3 = axes[1, 0]
    low_risk_patient = [p for p in dataset if not p['is_high_risk']][0]
    with torch.no_grad():
        _, hr_surv, _, _ = model(high_risk_patient['features'].unsqueeze(0))
        _, lr_surv, _, _ = model(low_risk_patient['features'].unsqueeze(0))
        
    hr_s = hr_surv[0, -2].numpy() # 고위험군 이벤트 직전 생존곡선
    lr_s = lr_surv[0, -2].numpy() # 저위험군 동 시점 생존곡선
    
    ax3.plot(time_horizons, hr_s, label="High-Risk Patient (Sepsis / AKI)", color='#d62728', lw=2.8, linestyle='-')
    ax3.plot(time_horizons, lr_s, label="Low-Risk Patient (Stable / Recovering)", color='#2ca02c', lw=2.8, linestyle='--')
    ax3.fill_between(time_horizons, hr_s, lr_s, color='#ff7f0e', alpha=0.15, label="Risk Margin (Discrimination Gap)")
    
    ax3.set_title("3. Survival Curves S(s): High-Risk vs Low-Risk", fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel("Time Horizon s (Hours Ahead)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Survival Probability S(s)", fontsize=11, fontweight='bold')
    ax3.set_ylim(-0.02, 1.05)
    ax3.legend(loc='lower left', frameon=True, fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # ----------------------------------------------------
    # [Panel 4] Survival Bellman Alignment (t vs shifted t+1 target)
    # ----------------------------------------------------
    ax4 = axes[1, 1]
    with torch.no_grad():
        _, S_all, p_all, cdf_all = model(high_risk_patient['features'].unsqueeze(0))
        t_eval = max(0, seq_len - 3)
        dt = int(high_risk_patient['dts'][t_eval].item())
        p_next = p_all[0, t_eval + 1]
        S_curr = S_all[0, t_eval]
        event = high_risk_patient['transition_events'][t_eval].item()
        
        target_p = compute_survival_bellman_target(p_next, S_curr, dt, event, K=K)
        target_F = torch.cumsum(target_p, dim=-1).numpy()
        pred_F = cdf_all[0, t_eval].numpy()
        
    ax4.plot(time_horizons, pred_F, label=f"Model Prediction F_t(s) at t={t_eval}", color='#1f77b4', lw=2.8)
    ax4.plot(time_horizons, target_F, label=f"Survival Bellman Target G_t(s) (Shifted by Δt={dt}h)", 
             color='#ff7f0e', lw=2.8, linestyle='--')
    ax4.set_title(f"4. Survival Bellman Target Alignment (Δt = {dt}h Interval)", fontsize=13, fontweight='bold', pad=10)
    ax4.set_xlabel("Time Horizon s (Hours Ahead)", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Cumulative Risk F(s)", fontsize=11, fontweight='bold')
    ax4.set_ylim(-0.02, 1.05)
    ax4.legend(loc='upper left', frameon=True, fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    out_path_workspace = "/Users/yangjaemo/Desktop/SurvTD/survtd_results.png"
    out_path_artifact = "/Users/yangjaemo/.gemini/antigravity-cli/brain/86a1170d-d67d-4fbf-8da3-aef411fae24d/survtd_results.png"
    
    plt.savefig(out_path_workspace, dpi=200, bbox_inches='tight')
    plt.savefig(out_path_artifact, dpi=200, bbox_inches='tight')
    print(f"✅ 차트 저장 완료: {out_path_workspace}")

if __name__ == '__main__':
    main()
