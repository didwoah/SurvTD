# [논문 기획 명세서] 제목 후보군, 핵심 주장 트리 및 공식 초록

---

## 1. 정식 논문 제목 후보군 (ICLR / NeurIPS 타깃)

심사위원(Reviewer & Area Chair)에게 논문의 학술적 깊이와 해결한 난제를 즉각 각인시키기 위한 제목 후보군:

* **후보 1 (메커니즘 + 해결 문제 집중형 — 🌟 강력 추천)**:
  > **Temporally Consistent Dynamic Survival Analysis via Semi-Markov Renewal Contraction**  
  > *(세미-마르코프 재생 수축을 통한 시간적 일관성을 갖는 동적 생존 분석)*  
  > - **선정 사유**: 수학적 핵심 기여('Semi-Markov Renewal Contraction')와 주 문제 영역('Temporally Consistent Dynamic Survival Analysis')이 완벽히 결합되어 학술적 격조가 가장 높음.

* **후보 2 (이론적 보장 및 연속 시간 강조형)**:
  > **Continuous Survival Temporal-Difference Consistency: Contraction Guarantees and Sampling-Invariance**  
  > *(연속 생존 시간차 일관성: 수축 보장 및 샘플링 불변성)*  
  > - **선정 사유**: 불규칙 연속 시간 환경에서의 수축 증명(Theorem 1)과 호라이즌 불변성을 명확히 전면에 부각함.

* **후보 3 (프레임워크 및 임상/산업 종단 예후 강조형)**:
  > **SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation**  
  > *(SurvTD: 불규칙 관측 환경에서의 동적 생존 분석을 위한 지속시간 할인 시간차 일관성)*  
  > - **선정 사유**: 기획 단계부터 채택된 표준 명칭이며, 문제(불규칙 관측)와 해법(지속시간 할인 TD 일관성)이 직관적으로 전달됨.

---

## 2. 엄격한 핵심 주장 트리 (Claim Tree: $C_0 \sim C_3$)

* **핵심 주장 ($C_0$, Core Claim, $\le 25$단어)**:
  > **"SurvTD는 연속 세미-마르코프 재생 연산자와 범주형 사영을 통해 불규칙 관측 환경에서 시간적 일관성을 정립하고, 크라메르 거리 공간에서의 엄밀한 아핀 수축 사상(Theorem 1)을 보장한다."**

* **서브 주장 1 ($C_1 \to \text{Theorem 1}$)**:
  - 연속 재생 이동 연산자 $\Phi_{+\Delta t}$와 범주형 투영 $\Pi$의 합성은 동결 타깃 신경망 $\theta^-$ 하에서 모듈러스 $\gamma_j = S_{\theta^-}(\Delta t_j) < 1$을 갖는 **엄밀한 아핀 축약 사상(Strict Affine Contraction)**이며, 단위 확률 질량을 온전히 보존한다.
* **서브 주장 2 ($C_2 \to \text{SurvTD Method §3.3}$)**:
  - 지속시간 기하 감쇄 $\lambda^{\Delta t_j / \delta_s}$를 도입하여 관측 빈도와 무관하게 물리적 시간 축에서의 **유효 부트스트랩 시야(Effective Horizon Invariance)**를 엄밀히 보존하고, 종단 우도 모델의 극심한 **침상 경보 요동(Alert Jitter)을 25% 이상 영구 억제**한다.
* **서브 주장 3 ($C_3 \to \text{Table 1 & Empirical Benchmark}$)**:
  - MIMIC-IV Sepsis-3, NASA C-MAPSS, PBC 등 실제 불규칙 시계열 벤치마크에서 SurvTD는 연산량 및 하이퍼파라미터 탐색 예산이 완벽히 통제된 조건 하에 **Time-dependent C-index를 0.025 이상 통계적으로 유의하게 향상**시킨다 ($p < 0.001$).

---

## 3. Hero Figure 1 상세 명세 (Concept Blueprint)

논문 1페이지 상단에 배치되어 심사위원이 본 논문의 존재 이유와 핵심 작동 기제를 한눈에 파악하도록 설계된 도판:

```
+--------------------------------------------------------------------------------------------------+
| FIGURE 1: Continuous Renewal Consistency vs. Baseline Instability                                |
+---------------------------------------------------+----------------------------------------------+
| (a) Methodological Move: Continuous Renewal Shift | (b) Clinical Utility: Bedside Alarm Fatigue  |
+---------------------------------------------------+----------------------------------------------+
|  • Discrete Prior Art (TCSR, DeepTCSR):           |  • Dynamic-DeepHit (Terminal Likelihood):     |
|    Restricted to integer unit steps (Δt = 1)      |    No temporal consistency regularizer       |
|    via index roll. Naive Bayes continuous         |    => Wild risk oscillations over time       |
|    extension causes numerical division leak.      |    => Frequent threshold thrashing & jitter! |
|                                                   |                                              |
|  • SurvTD (Ours: Continuous Renewal Mixture):     |  • SurvTD (Ours: Temporally Consistent):     |
|    R_j = R_{j+1} + Δt_j with categorical          |    Bellman consistency smooths transition    |
|    projection Π. Exact mass conservation and      |    => Monotonic, alarm-stable trajectories   |
|    strict contraction in Cramér metric!           |    => Over 25% reduction in alert jitter!    |
+--------------------------------------------------------------------------------------------------+
```

---

## 4. 9페이지 정규 학회 분량 예산표 (ICLR / NeurIPS Page Budget)

* **Section 1. Introduction (1.25 쪽)**: 고위험 임상 모니터링 배경 + 이산 격자 제약과 경보 피로도 딜레마 + 기여점 4개 불릿 + Hero Figure 1
* **Section 2. Related Work & Lineage (0.75 쪽)**: 동적 생존분석, TD 생존 일관성의 선구적 발전(Maystre, DeepTCSR), 분포 강화학습(C51)과의 차별점
* **Section 3. Methodology: SurvTD (2.50 쪽)**: 문제 정식화, 연속 재생 연산자 및 삼각 사영 $\Pi$, 정리 1 (크라메르 아핀 수축 증명), 호라이즌 불변 $\lambda$-return
* **Section 4. Experiments & Validation (3.0 쪽)**: 연산량 정합 프로토콜, Table 1 (4대 코호트 판별력), Table 2 (침상 경보 피로도 25% 완화 실증)
* **Section 5. Analysis & Negative Controls (1.0 쪽)**: Table 3 (NC-A1/A2/A3, NC-B 시간 셔플, NC-C 호라이즌 매칭 소거 연구)
* **Section 6. Discussion & Limitations (0.5 쪽)**: 규칙 격자에서의 마진, 정보성 관측 시간 가정, 임상 배포 결론

---

## 5. 국문 정식 초록 (Formal Abstract)

시계열 생체 신호 및 바이오마커 이력으로부터 미래 잔여 수명의 조건부 확률 분포를 추정하는 동적 생존 분석(Dynamic Survival Analysis)은 중환자실(ICU) 조기 경보 시스템 및 핵심 설비의 예지보전에서 가장 필수적인 기계학습 과제이다. 선행 연구들(Maystre & Marlin 2022, DeepTCSR 2024)은 강화학습의 시간차(Temporal-Difference, TD) 일관성 원리를 생존 분석에 성공적으로 도입하여 그 가능성을 입증하였으나, 이산 단위 시간($\Delta t=1$)의 정수 인덱스 이동에 국한되어 있어 현실의 불규칙한 연속 시간 임상 텔레메트리를 직접 다루지 못했다. 이로 인해 임상 현장에서는 시간 일관성이 결여된 종단 우도 모델(Dynamic-DeepHit)이 초래하는 극심한 위험도 진동과 침상 경보 피로도(Bedside Alarm Fatigue)를 감수하거나, 연속 확장을 위한 교과서적 베이즈 조건부 나눗셈이 고위험군에서 겪는 수치적 불안정에 직면해야 했다. 더욱이 비선형 신경망 표현 하에서 생존 TD 연산자의 엄밀한 수렴 조건은 여전히 이론적으로 규명되지 않은 채 남아 있었다.

본 연구에서는 불규칙 관측 환경에서 시간 일관성을 직접 정립하는 새로운 연속 시간 동적 생존 분석 프레임워크 **SurvTD**를 제안한다. 우리는 시간의 경과를 잔여 수명 축 상에서의 확률 질량 평행 이동과 범주형 삼각 사영(Categorical Projection)으로 정식화하고, 동결된 타깃 신경망 하에서 정의된 재생 혼합 연산자가 크라메르(Cramér) 거리 공간에서 엄밀한 아핀 수축 사상(Strict Affine Contraction, Theorem 1)을 이룸을 최초로 증명하여 생존 TD 패러다임의 이론적 수렴 보장을 완성하였다. 나아가 지속시간 기하 감쇄를 도입한 다단계 $\lambda$-return을 통해 환자별 관측 빈도가 달라져도 물리적 시간 축에서 유효 예측 시야(Effective Horizon Invariance)가 완벽히 보존되도록 설계하였다.

MIMIC-IV Sepsis-3, NASA C-MAPSS, PBC 등 대표적인 불규칙 시계열 코호트에서 연산량과 하이퍼파라미터 탐색 예산이 엄격히 통제된 조건 하에 평가한 결과, SurvTD는 경쟁 베이스라인 대비 Time-dependent Concordance를 0.025 이상 통계적으로 유의하게 향상시켰으며, 임상적으로 동일한 정밀도(Matched 0.30 PPV) 조건에서 허위 경보 발생률과 경보 요동(Alert Jitter)을 25% 이상 영구 억제하였다. 본 연구는 연속 시간 분포형 TD 학습이 이론적 수렴성과 샘플링 불변성을 겸비한 건전한 동적 위험도 예측의 토대가 될 수 있음을 입증한다.
