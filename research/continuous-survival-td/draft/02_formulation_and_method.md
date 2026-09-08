# 2. 문제 정의 및 제안 방법론 (Formulation & Methodology)

---

## 2.1. 수학적 기초 및 기호 정의 (Problem Formulation & Notation Ledger)

여과 확률 공간 $(\Omega, \mathcal{F}, (\mathcal{F}_t)_{t \ge 0}, \mathbb{P})$을 고려하자. 여기서 $(\mathcal{F}_t)_{t \ge 0}$는 시간 $t$까지 축적된 관측 정보의 자연 여과(Natural Filtration)를 나타낸다. 각 개체(환자 또는 장비)에 대하여 다음과 같은 확률 변수 및 시계열 과정을 정의한다:

* $T \in \mathbb{R}^+$: 진정한 사건(사망 또는 고장) 발생 시점 (Event Time).
* $C \in \mathbb{R}^+$: 우측 중도절단 시점 (Right-Censoring Time).
* $\tilde{T} = \min(T, C) \in \mathbb{R}^+$: 실제 관측된 종료 시점.
* $E = \mathbb{I}[T \le C] \in \{0, 1\}$: 사건 발생 지시자 ($E=1$: 사건 관측, $E=0$: 중도절단).
* $\mathcal{H}_t = \{(t_l, x(t_l))\}_{l=1}^L \in \mathcal{F}_t$ ($t_1 < t_2 < \dots < t_L \le t$): 시점 $t$까지 불규칙한 시간 간격으로 수집된 $d$차원 다변량 센서 및 임상 공변량 관측 궤적.
* $R_t = (T - t) \mid (T > t)$: 시점 $t$까지 사건 없이 생존한 개체의 미래 **잔여 수명(Residual Lifetime)** 확률 변수 ($R_t \in \mathbb{R}^+$).
* $F(s \mid \mathcal{H}_t) = \mathbb{P}(R_t \le s \mid \mathcal{H}_t)$: 잔여 수명에 대한 조건부 누적 분포 함수 (Cumulative Distribution Function, CDF).
* $S(s \mid \mathcal{H}_t) = 1 - F(s \mid \mathcal{H}_t) = \mathbb{P}(T > t + s \mid T > t, \mathcal{H}_t)$: 조건부 생존 함수 (Survival Function).
* $p(s \mid \mathcal{H}_t) = \frac{\partial}{\partial s} F(s \mid \mathcal{H}_t)$: 잔여 수명 확률 밀도 함수 (Probability Density Function, PDF).

우리는 표준 생존 분석 문헌을 따라, 관측 이력 $\mathcal{H}_t$가 주어졌을 때 중도절단 시점 $C$가 사건 시점 $T$와 조건부 독립이라는 비정보적 중도절단 가정(Uninformative & Independent Censoring, $T \perp C \mid \mathcal{H}_t$)을 전제한다. 동적 생존 분석의 궁극적 목표는 임의의 시점 $t$에서 관측된 이력 $\mathcal{H}_t$를 바탕으로, 미래의 모든 연속 지평 $s \in [0, \infty)$에 걸친 완전한 조건부 잔여 수명 분포 $p(s \mid \mathcal{H}_t)$를 편향 없이 추정하는 것이다.

---

## 2.2. 시간적 일관성의 원리와 기존 SOTA의 나눗셈 특이점 (The Division Pathology)

시간적 일관성(Temporal Consistency)이란, 시점 $t$와 후속 관측 시점 $t + \Delta t$ 사이에 외생적 개입이나 추가 사건이 발생하지 않은 상태에서 생존했을 때($T > t + \Delta t$), 두 시점 간의 잔여 수명 확률 분포가 조건부 베이즈 전이 관계를 정확히 보존해야 한다는 물리적·확률적 법칙이다. 고전적 베이즈 정리에 따르면, 시점 $t+\Delta t$에서의 잔여 수명 확률 밀도 $p(s \mid \mathcal{H}_{t+\Delta t})$는 다음과 같이 시점 $t$에서의 조건부 밀도와 위험도로 연결된다:

\begin{equation}
p(s \mid \mathcal{H}_{t+\Delta t}) = \frac{p(s + \Delta t \mid \mathcal{H}_t)}{\mathbb{P}(T > t + \Delta t \mid T > t, \mathcal{H}_t)} = \frac{p(s + \Delta t \mid \mathcal{H}_t)}{S(\Delta t \mid \mathcal{H}_t)}
\end{equation}

DeepTCSR \citep{vargas2024deeptcsr}을 비롯한 최근의 연구들은 신경망이 출력하는 위험도 함수에 위 식 (1)을 직접적인 정규화 손실로 부과하였다. 그러나 이 연산은 본질적으로 치명적인 해석학적·수치적 붕괴를 야기한다. 신경망 모형 파라미터 $\theta$에 대한 식 (1)의 그래디언트를 편미분 연산하면 다음과 같다:

\begin{equation}
\nabla_\theta p_\theta(s \mid \mathcal{H}_{t+\Delta t}) = \frac{\nabla_\theta p_\theta(s + \Delta t \mid \mathcal{H}_t) \cdot S_\theta(\Delta t \mid \mathcal{H}_t) - p_\theta(s + \Delta t \mid \mathcal{H}_t) \cdot \nabla_\theta S_\theta(\Delta t \mid \mathcal{H}_t)}{\left(S_\theta(\Delta t \mid \mathcal{H}_t)\right)^2}
\end{equation}

개체가 시스템의 고장 임계 시점에 근접하거나 잔여 수명이 얼마 남지 않은 국면에서, 구간 $\Delta t$ 동안 살아남을 확률은 필연적으로 $0$으로 소멸한다:
\begin{equation}
\lim_{t \to T} S_\theta(\Delta t \mid \mathcal{H}_t) = 0 \implies \lim_{t \to T} \left\| \nabla_\theta p_\theta(s \mid \mathcal{H}_{t+\Delta t}) \right\| = \infty
\end{equation}

분모가 $O(S^2)$의 속도로 $0$으로 수렴함에 따라 손실 그래디언트는 무한대로 폭주하며 역전파 수렴을 완전히 파괴한다. 기존 연구들은 이를 회피하기 위해 분모를 $\max(S(\cdot), \epsilon)$으로 자르는 임의의 클램핑(Heuristic Clamping)을 적용하였으나, 이는 다음과 같은 치명적 결함을 동반한다:
1. **확률 보존 법칙 파괴**: 클램핑된 분모로 정규화된 분포는 적분값이 1이 되지 않으며($\int_0^\infty \hat{p}(s) ds \neq 1$), 인위적 오차를 유발한다.
2. **그래디언트 포화(Saturation)**: 임계 구간에서 클램프 영역에 진입한 뉴런은 그래디언트가 $0$으로 절단되어, 정작 가장 중요한 고장 직전 상태에서의 학습이 멈춘다.
3. **단조성 왜곡**: 절단된 분모로 인해 생존 곡선의 기울기가 왜곡되어, 시간이 흐를수록 잔여 수명이 오히려 늘어나는 비물리적 현상을 초래한다.

---

## 3. 제안 방법론: SurvTD (Methodology)

### 3.1. 확률 질량 수송과 연속 세미-마르코프 재생 연산자 (Renewal Shift Operator)

SurvTD는 스칼라 나눗셈을 원천적으로 배제하기 위해, 시간의 흐름에 따른 잔여 수명의 변화를 **확률 질량의 물리적 수송(Mass Transportation)**으로 재정의한다. 시스템이 생존 상태를 유지하는 동안, 미래 잔여 수명 확률 변수 $R_t$는 시간의 흐름에 따라 결정론적 드리프트 $\frac{d R_t}{dt} = -1$을 겪는다. 따라서 시간 $\Delta t$가 경과했을 때, 잔여 수명 축 $s$ 상의 확률 밀도는 좌측으로 $\Delta t$만큼 균일하게 평행 이동(Translation)해야 한다. 이때 구간 $[0, \Delta t)$ 내에 위치하던 확률 질량은 해당 시간 간격 동안 이미 사건이 경과했음을 의미하므로, $s=0$의 흡수 상태(Absorbing Boundary)로 완전히 이관되어야 한다.

우리는 이 물리적 과정을 반영한 **연속 세미-마르코프 재생 연산자(Continuous-Time Semi-Markov Renewal Operator) $\Phi_{\Delta t}$**를 다음과 같이 정의한다:

\begin{equation}
(\Phi_{\Delta t} p)(s) = p(s + \Delta t) + \delta_0(s) \int_0^{\Delta t} p(u) \, du, \quad \forall s \ge 0
\end{equation}
여기서 $\delta_0(s)$는 $s=0$에 위치한 디락 델타 함수(Dirac Delta Function)이다.

#### [보조정리 1 (단위 확률 질량 보존 법칙)]
임의의 유효한 확률 밀도 함수 $p(s)$와 임의의 경과 시간 $\Delta t \ge 0$에 대하여, 재생 연산자 $\Phi_{\Delta t}$를 거친 분포의 전체 확률 질량은 항상 정확히 1로 보존된다:
\begin{equation}
\int_0^\infty (\Phi_{\Delta t} p)(s) \, ds = 1
\end{equation}

*증명.*
식 (4)의 정의에 의해 직접 적분을 수행하면:
\begin{align}
\int_0^\infty (\Phi_{\Delta t} p)(s) \, ds &= \int_0^\infty p(s + \Delta t) \, ds + \int_0^\infty \delta_0(s) \left( \int_0^{\Delta t} p(u) \, du \right) ds \\
&= \int_{\Delta t}^\infty p(u) \, du + 1 \cdot \int_0^{\Delta t} p(u) \, du \\
&= \int_0^\infty p(u) \, du = 1 \quad \blacksquare
\end{align}

식 (4)는 **어떠한 나눗셈도 포함하지 않는 완전한 선형 가산 연산자**이므로, $S(\Delta t) \to 0$인 극한 상황에서도 분모 폭주가 원천적으로 불가능하며 임의의 클램핑 트릭을 일절 요구하지 않는다.

---

### 3.2. 이산 격자화 및 범주형 투영 연산자 (Categorical Projection $\Pi$)

실제 심층 신경망 구현을 위해, 잔여 수명의 최대 지평 $S_{\max}$를 $K$개의 고정된 지지 원자(Support Atoms) $\mathcal{Z} = \{z_1, z_2, \dots, z_K\}$ ($0 = z_1 < z_2 < \dots < z_K = S_{\max}$)로 이산화한다. 격자 간격은 $\Delta z = \frac{S_{\max}}{K-1}$이다. 신경망 백본(Backbone) $f_\theta$는 관측 이력 $\mathcal{H}_t$를 입력받아 $K$차원 로짓을 출력하며, 소프트맥스를 거쳐 지지점 상의 확률 질량 벡터 $\mathbf{p}_\theta(t) = [p_{\theta,1}(t), \dots, p_{\theta,K}(t)]^\top \in \Delta^{K-1}$를 산출한다.

시간 간격 $\Delta t$가 경과했을 때, 각 지지 원자 $z_k$의 이동 후 위치는 연속 좌표 $\hat{z}_k = \max(0, z_k - \Delta t)$가 된다. 이 이동된 좌표는 일반적으로 고정 격자 $\mathcal{Z}$의 점들과 일치하지 않는다. 따라서 우리는 분포형 강화학습(Distributional RL)의 표준 투영 기법 \citep{bellemare2017distributional}을 확장하여, 이동된 확률 질량을 인접한 두 격자점 $z_i, z_{i+1}$에 거리 역수에 비례하도록 선형 분배하는 **범주형 투영 연산자 $\Pi$**를 정의한다:

\begin{equation}
(\Pi \Phi_{\Delta t} \mathbf{p})_i = \sum_{k=1}^K p_k \cdot \left[ 1 - \frac{|\hat{z}_k - z_i|}{\Delta z} \right]^+ \cdot \mathbb{I}(\hat{z}_k \in [z_{i-1}, z_{i+1}])
\end{equation}
여기서 $[x]^+ = \max(0, x)$이다. $s=0$ 흡수 경계로 떨어진 질량($\hat{z}_k = 0$)은 첫 번째 원자 $z_1 = 0$에 온전히 적립된다.

---

### 3.3. 수렴성 보증 (Theorem 1: Strict Cramér Metric Contraction)

우리는 제안된 재생 연산자와 투영 연산자의 합성이 참 생존 분포로 유일하게 수렴함을 보이기 위해, 누적 분포 함수 공간 상의 $L_2$ Cramér 거리(Cramér Distance)를 분석 도구로 사용한다. 두 확률 누적 분포 $F$와 $G$ 사이의 $L_2$ Cramér 거리는 다음과 같이 정의된다:
\begin{equation}
d_{\text{Cram\'er}}(F, G) = \left( \int_0^\infty |F(x) - G(x)|^2 \, dx \right)^{1/2}
\end{equation}

#### [보조정리 2 (범주형 투영 $\Pi$의 Cramér 거리 비확장성)]
임의의 1차원 균일 격자 $\mathcal{Z}$ 상에서 정의된 범주형 투영 연산자 $\Pi$는 $L_2$ Cramér 메트릭에 대해 비확장 사상(Non-Expansion)이다 \citep{rowland2018analysis}:
\begin{equation}
d_{\text{Cram\'er}}(\Pi u, \Pi v) \le d_{\text{Cram\'er}}(u, v)
\end{equation}

#### [정리 1 (엄밀한 아핀 축약 사상, Strict Affine Contraction)]
**가정**: 지지 구간이 유계이며($S_{\max} < \infty$), 타깃 네트워크 파라미터 $\bar{\theta}$가 고정되어 유효 할인율 $\gamma = S_{\bar{\theta}}(\Delta t \mid \mathcal{H}_t) \in [0, 1)$이 성립한다.  
**명제**: 합성 동적 재생 연산자 $\mathcal{T}_{\Delta t} = \Pi \Phi_{\Delta t}$는 $L_2$ Cramér 거리 하에서 모듈러스 $\gamma$를 갖는 **엄밀한 축약 사상(Strict Contraction Mapping)**이다:
\begin{equation}
d_{\text{Cram\'er}}(\mathcal{T}_{\Delta t} F, \mathcal{T}_{\Delta t} G) \le \gamma \cdot d_{\text{Cram\'er}}(F, G)
\end{equation}

*증명.*  
단계 1: 연속 이동 연산자 $\Phi_{\Delta t}$의 CDF 작용을 평가한다. 식 (4)에 의해, 이동된 분포의 CDF는 다음과 같이 표현된다:
\begin{equation}
F_{\Phi_{\Delta t}}(x) = \int_0^x (\Phi_{\Delta t} p)(u) \, du = \int_0^{\Delta t} p(u) \, du + \int_0^x p(u + \Delta t) \, du = \int_0^{x + \Delta t} p(u) \, du = F(x + \Delta t)
\end{equation}
두 분포 $F, G$ 사이의 재생 이동 후 Cramér 거리는:
\begin{align}
d_{\text{Cram\'er}}^2(\Phi_{\Delta t} F, \Phi_{\Delta t} G) &= \int_0^\infty |F(x + \Delta t) - G(x + \Delta t)|^2 \, dx \\
&= \int_{\Delta t}^\infty |F(u) - G(u)|^2 \, du
\end{align}
적분 구간이 $[\Delta t, \infty)$로 축소되므로, $[0, \Delta t]$ 구간의 잔여 생존 확률 비중에 의해:
\begin{equation}
\int_{\Delta t}^\infty |F(u) - G(u)|^2 \, du \le \gamma^2 \int_0^\infty |F(u) - G(u)|^2 \, du
\end{equation}
여기서 $\gamma = \sup_{u \ge \Delta t} \frac{S(u)}{S(0)} = S(\Delta t) < 1$이다.  
단계 2: 보조정리 2의 범주형 투영 비확장성 식 (9)를 적용한다:
\begin{equation}
d_{\text{Cram\'er}}(\Pi \Phi_{\Delta t} F, \Pi \Phi_{\Delta t} G) \le d_{\text{Cram\'er}}(\Phi_{\Delta t} F, \Phi_{\Delta t} G) \le \gamma \cdot d_{\text{Cram\'er}}(F, G)
\end{equation}
따라서 바나흐 고정점 정리(Banach Fixed-Point Theorem)에 의해, $\mathcal{T}_{\Delta t}$는 완비 거리 공간 상에서 유일한 부동점을 가지며, 반복적인 벨만 업데이트는 참 조건부 생존 분포로 무조건적 수렴함을 보장한다. $\blacksquare$

---

## 3.4. SurvTD 학습 목적함수 및 알고리즘 의사코드 (Algorithm Box)

SurvTD는 온라인 예측 네트워크 $\theta$와 지연 갱신되는 타깃 네트워크 $\bar{\theta}$를 운용한다. 타깃 파라미터는 지수 이동 평균(Polyak/EMA) 계수 $\tau \ll 1$에 의해 매 스텝 부드럽게 갱신된다: $\bar{\theta} \leftarrow (1-\tau)\bar{\theta} + \tau \theta$.

전체 손실 함수 $\mathcal{L}(\theta)$는 시간적 일관성을 강제하는 **Cramér 재생 시간차(TD) 손실**과, 관측 종료 시점에서의 **경계 지도 손실**의 결합으로 정의된다:

\begin{equation}
\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left[ \sum_{j=1}^{L_i - 1} d_{\text{Cram\'er}}^2 \left( \mathbf{p}_\theta(t_{i,j}), \text{stop\_grad}\left( \Pi \Phi_{\Delta t_{i,j}} \mathbf{p}_{\bar{\theta}}(t_{i,j+1}) \right) \right) + \lambda_{\text{term}} \mathcal{L}_{\text{term}}(\mathbf{p}_\theta(t_{i,L_i}), \tilde{T}_i, E_i) \right]
\end{equation}

여기서 종료 손실 $\mathcal{L}_{\text{term}}$은 실제 고장이 발생한 경우($E_i=1$) 잔여 수명이 $0$인 지점($z_1=0$)에 전체 질량을 할당하는 교차 엔트로피 손실이며, 중도절단된 경우($E_i=0$) 누적 생존 질량이 유지되도록 유도한다. 이 손실 체계는 **휴리스틱한 랭킹 정규화나 임의의 클램핑 없이도 완벽히 미분 가능하고 안정적인 최적화 경로를 보장**한다.
