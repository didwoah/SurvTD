# 학습 기록 0008: SMDP 강화학습과 SurvTD의 수학적 계보

## 학습 일시
2026-09-09

## 핵심 학습 내용
1. **Semi-Markov Decision Process (SMDP)와 연속 시간 TD**:
   - Bradtke & Duff (1994, NeurIPS)의 연속 시간 SMDP TD 학습: 체류 시간(Sojourn time) $\tau \in \mathbb{R}^+$에 따른 지수 할인 $\gamma(\tau) = e^{-\beta \tau}$.
   - 점프 시점(Jump epochs)에서만 마르코프 성질이 성립하는 구조.

2. **SurvTD와의 1:1 대응**:
   - 체류 시간 $\tau \longleftrightarrow$ 환자 방문 간격 $\Delta t_j = t_{j+1} - t_j \in \mathbb{R}^+$
   - 지속시간 할인율 $e^{-\beta \tau} \longleftrightarrow$ 구간 생존 확률 $\gamma_j = S(\Delta t_j)$
   - 구간 내 보상 적분 $\int_0^\tau e^{-\beta t} r_t dt \longleftrightarrow$ 사망 위험 질량 $(1 - \gamma_j) \mu_{\text{death}}$
   - 미래 가치 반영 $\longleftrightarrow$ 재생 평행이동 및 사영 $\gamma_j \Pi \Phi_{+\Delta t_j} p_{\theta^-}$

3. **생존분석만의 3대 고유 난제와 극복**:
   - **내생적 할인율 (Endogenous Discount)**: $\gamma_j = S_\theta(\Delta t_j)$가 신경망 출력인 문제 $\to$ 타깃 네트워크 동결 ($\theta^-$)로 아핀 상수화하여 Theorem 1 수축 증명.
   - **분포 재생 평행이동 (Renewal Shift)**: 스칼라 덧셈이 아닌 잔여 시간 단축 ($R_j = R_{j+1} + \Delta t_j$) $\to$ 재생 이동 연산자 $\Phi_{+\Delta t}$와 범주형 삼각 사영 $\Pi$.
   - **우측 중도절단 (Right-Censoring)**: 미관측 꼬리 분포 $\to$ 타깃망 기반 꼬리 완성 및 Truncated IPCW 가중치 부여.

4. **논문 포지셔닝 강화**:
   - Related Work 및 Method 섹션에서 30년 역사의 정통 SMDP TD 이론의 '분포형 생존분석 확장(Distributional Semi-Markov TD)'으로 위상 격상.
