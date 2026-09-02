"""
SurvTD 모델 아키텍처: 순차 인코더 및 이산 해저드 예측 헤드
"""
import torch
import torch.nn as nn

class SurvTD_Model(nn.Module):
    """
    SurvTD(λ) 시계열 생존분석 모델
    - Backbone: 2-layer GRU (시계열 종단 특징을 잠재 상태 h_t로 인코딩)
    - Hazard Head: K개 미래 시간 버킷에 대한 조건부 해저드 h_t(s) 출력
    """
    def __init__(self, input_dim=17, hidden_dim=64, num_buckets=30):
        super().__init__()
        self.K = num_buckets
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.K)
        )
        
    def forward(self, x):
        """
        입력:
          x: (Batch, Seq_Len, input_dim)
        반환:
          hazard: (Batch, Seq_Len, K)
          survival: (Batch, Seq_Len, K)
          pmf: (Batch, Seq_Len, K)
          cdf: (Batch, Seq_Len, K)
        """
        out, _ = self.encoder(x)
        logits = self.head(out)
        hazard = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazard + 1e-7, dim=-1)
        s_prev = torch.cat([torch.ones_like(survival[..., :1]), survival[..., :-1]], dim=-1)
        pmf = hazard * s_prev
        cdf = torch.cumsum(pmf, dim=-1)
        return hazard, survival, pmf, cdf
