# 불규칙 ICU EHR를 위한 Semi-Markov TCSR 연구 아이디어와 설계

작성일: 2026-09-09  
문서 목적: 연구 배경을 모르는 독자가 핵심 아이디어를 이해하고, 구현 담당자가 같은 가정과 목표로 첫 실험을 시작할 수 있도록 정리한다.

## 1. 연구 아이디어를 한 문장으로

**현재까지의 ICU EHR로 사건까지 남은 시간의 분포를 예측하고, 다음 관측에서의 예측분포를 실제 경과시간만큼 이동시켜 현재 모델의 학습 target으로 사용한다.**

우리는 이를 관측시점 기반 semi-Markov 구조 아래의 distributional temporal-difference learning으로 정식화한다. 잠정적으로 SM-TCSR라고 부른다. TCSR는 기반 논문의 알고리즘 명칭이며, SM-TCSR는 이 문서에서 제안하는 확장이다.

가장 중요한 설계 결정은 다음과 같다.

- 예측하는 시간은 방문 횟수가 아니라 실제 시간이다.
- 다음 상태와 다음 상태까지의 소요시간을 함께 다루는 Markov renewal 가정을 둔다.
- 검열된 환자도 관측된 transition은 학습에 사용한다. 마지막 관측 상태에서도 미래 분포를 bootstrap할 수 있다.
- 행정적 검열을 사망이나 생존퇴실로 바꾸지 않는다.
- 첫 구현의 중심은 하나의 분포형 TD loss다. 별도 survival likelihood의 상시 결합은 필수가 아니다.
- 검열이 전이 중간에 발생한 경우와 관측된 전이의 선택 편향은 추가 분석 대상으로 남긴다.

아직 성능이나 수렴이 검증된 방법은 아니다. 연구 가설과 구현 후보를 아래에 명확하게 구분한다.

## 2. 어떤 문제를 해결하려는가

ICU에서는 혈압·검사 결과·처치 기록이 일정하지 않은 간격으로 들어온다. 어느 환자는 30분 뒤 새 관측이 있고, 다른 환자는 5시간 뒤에 다음 기록이 생긴다. 이 둘을 모두 한 step으로 처리하면 실제 시간 차이가 사라진다.

우리는 새 기록이 들어올 때마다 “현재부터 6시간, 24시간, 72시간 이내 사건이 발생할 확률”을 갱신하고 싶다. 이 시간들은 예시이며, 실제 평가 horizon은 데이터의 추적 범위와 임상 목적에 맞게 정한다.

직접 지도학습은 현재 기록과 최종 사건 결과를 연결한다. TD는 여기에 다른 학습 방식을 제공한다. 다음 상태에서 더 구체적으로 예측할 수 있는 미래를 현재 상태의 학습 신호로 활용한다. 이것이 데이터 효율성을 개선할 수 있다는 것이 연구 가설이다. Bootstrap은 오차를 전파할 수도 있으므로 개선을 미리 가정하지 않는다.

### 직관적인 예시

오전 9시에 현재 상태를 관측하고, 오후 1시에 다음 상태를 관측했다고 하자. 환자는 그때까지 살아 있고 ICU에 재원 중이다.

오후 1시 상태에서 “앞으로 8시간 이내 사망”에 대한 예측은 오전 9시 상태에서 “앞으로 12시간 이내 사망”을 학습하는 target이 된다. 이미 지나간 4시간을 더하기 때문이다.

이는 한 샘플의 target이다. 오전 9시의 실제 사망확률이 4시간 동안 반드시 0이라는 주장은 아니다. 비슷한 초기 상태에서 일찍 사망한 환자들의 transition도 함께 학습해야 현재의 조건부 위험을 추정할 수 있다.

## 3. 예측 목표와 용어

### 첫 임상 실험의 목표

첫 임상 구현에서는 **현재 ICU 입원 중 사망**을 예측 대상으로 둔다. 종료 사건은 사망과 생존퇴실 두 종류로 구성한다. ICU 퇴실 이후의 사망까지 예측하는 문제는 별도의 목표이며, 이 설계와 혼합하지 않는다.

전원, 병원 내 이동, 재입실, 같은 시각의 사건·측정 처리 규칙은 cohort 생성 전에 고정한다. 데이터 수집 종료처럼 결과가 확인되지 않은 경우는 행정적 검열로 둔다. 기술 검증의 첫 합성 실험은 사망 하나만 있는 단순 모델로 시작해도 된다.

### 모델이 출력할 확률

현재 관측시각을 기준으로 첫 종료 사건까지 남은 시간을 $Z$, 사건 종류를 $J$라고 한다. $c$는 사망 또는 생존퇴실이다.

$$
F_c(\tau\mid s)=P(Z\leq\tau,\ J=c\mid s,\text{현재까지 종료 사건 없음}).
$$

$F_c$는 사건별 누적발생확률(CIF)이다. 단일 사건 문제에서는 일반적인 사건시간 CDF가 된다. 아직 어느 종료 사건도 발생하지 않을 확률은 다음과 같다.

$$
S(\tau\mid s)=1-\sum_c F_c(\tau\mid s).
$$

| 용어 | 이 설계에서의 의미 |
|---|---|
| 상태 $s_n$ | 현재까지의 EHR을 요약한 환자 표현 |
| 전이시간 $D_n$ | 다음 관측 또는 종료 사건까지의 실제 시간 |
| 예측시간 $\tau$ | 지금부터 몇 시간 뒤를 예측하는지 |
| Horizon $H$ | TD loss를 평가하는 미래 시간 범위 |
| Target model | 다음 상태의 예측을 제공하는 지연된 모델 |
| Bootstrap | 모델의 다음 상태 예측을 학습 target에 활용하는 것 |

## 4. Semi-Markov 가정을 어떻게 정의하는가

### 상태와 전이

환자의 관측 기록은 $(t_0,x_0),(t_1,x_1),\ldots$이다. 현재까지의 이력을 $\mathcal H_{t_n}$이라 하고, 상태를 다음처럼 정의한다.

$$
s_n=\operatorname{Encoder}(\mathcal H_{t_n}).
$$

상태는 현재 측정값뿐 아니라 이전 기록, 결측 여부, 마지막 측정 이후 시간, ICU 입실 후 경과시간을 포함할 수 있다. 현재 측정값만 사용하는 것도 가능하지만, 그것만으로 미래를 예측하기에 충분하다는 더 강한 가정을 하게 된다.

다음 EHR 관측, 사망, 생존퇴실 중 먼저 발생하는 시점까지를 하나의 전이로 정의한다. 사망·퇴실은 별도의 absorbing state로 표시한다. 행정적 검열은 임상 전이가 아니라 관측이 중단된 것으로 처리한다.

### 핵심 가정

$$
P(s_{n+1}\in B,\ D_n\leq u\mid\mathcal H_{t_n})
=Q(B,u\mid s_n).
$$

말로 풀면, **현재 상태 표현을 알면 다음 상태와 거기까지 걸리는 시간의 결합분포를 예측하는 데 충분하다**는 뜻이다. 다음 상태와 경과시간은 독립일 필요가 없다. 경과시간은 지수분포로 제한하지 않는다.

관측시점의 상태·시간 쌍은 Markov renewal 구조를 이룬다. 관측 사이에 마지막 상태 표현을 유지하는 과정으로 확장하면 semi-Markov 표현으로 연결된다. 관측 사건이 무한히 짧은 시간 안에 누적되지 않도록 전이시간과 동시 기록 처리 규칙도 정한다.

**측정 간격과 실제 임상 상태의 지속시간은 다르다.** 이 모델은 관측 상태 사이의 시간을 사용한다. 환자의 몸이 다음 관측까지 변하지 않는다고 가정하지 않으며, 잠재 임상 상태에 진입한 정확한 시각을 추정한다고 주장하지 않는다.

또한 불규칙한 관측 자체가 semi-Markov의 증거는 아니다. 위 구조를 연구의 모델링 가정으로 명시하는 것이다. RNN이나 Transformer를 사용해도 상태의 충분성이 자동으로 보장되지는 않는다.

### Model-free라는 의미

결합분포 $Q$를 직접 추정하는 전이모델은 필수가 아니다. 관측한 $(s_n,D_n,s_{n+1})$로부터 사건시간 분포를 바로 학습한다. 이것이 model-free semi-Markov TD 접근이다. 가변 시간 전이를 value-based TD에 반영하는 의료 연구로 Fatemi et al. [3]를 참고할 수 있다.

## 5. TCSR에서 무엇을 계승하는가

원래 TCSR은 terminal state까지의 이산 hitting-time 분포를 학습한다. 대표적인 모델 head는 다음과 같은 이산 logistic hazard이다 [1].

$$
h_\theta(k\mid x)=\operatorname{sigmoid}(\alpha_k+\beta^\top\phi(x)).
$$

이에 대응하는 CDF는 다음과 같다.

$$
F_\theta(k\mid x)=1-\prod_{j=1}^{k}\{1-h_\theta(j\mid x)\}.
$$

원문은 이를 discrete-time Cox PH라고 부른다. 일반적인 연속시간 Cox partial-likelihood 모델과 같은 구현으로 취급하지 않는다. 원문의 TCSR는 hazard pseudo-label과 생존 가중치를 이용한 cross-entropy로 학습한다.

우리 설계는 **다음 상태의 분포를 이동시켜 학습하는 원리와 검열된 trajectory의 관측 transition을 활용하는 방식**을 계승한다. 시간 이동을 1 step에서 실제 $D_n$으로 바꾸고, 연속시간 CIF head 및 CIF 제곱오차를 구현 후보로 둔다. 이는 원문의 loss를 그대로 사용한다는 뜻은 아니다.

## 6. TD target과 하나의 기본 loss

### 살아 있는 다음 관측

다음 관측이 $D_n$시간 뒤에 있고 그때까지 종료 사건이 없었다면, 사건별 target은 다음과 같다.

$$
\widetilde F_{n,c}(\tau)=
\begin{cases}
0, & 0\leq\tau<D_n,\\
F_{\bar\theta,c}(\tau-D_n\mid s_{n+1}), & \tau\geq D_n.
\end{cases}
$$

$\bar\theta$는 target model의 파라미터다. 연속시간 head에서는 $F_c(0)=0$이므로 경계에서도 일관된다.

### 사망 또는 생존퇴실

종류 $c^*$의 종료 사건이 $D_n$시간 뒤에 관측되었다면 다음 hard target을 사용한다.

$$
\widetilde F_{n,c}(\tau)
=\mathbf 1\{c=c^*\}\mathbf 1\{\tau\geq D_n\}.
$$

모든 가능한 관측·종료 전이를 포함하고 현재 상태가 충분하다는 가정 아래, 참 모델의 관계는 다음처럼 조건부 기대값으로 표현된다.

$$
F_c^*(\tau\mid s)
=\mathbb E[\widetilde F_{n,c}^*(\tau)\mid s_n=s].
$$

하나의 다음 상태가 모든 미래를 대표한다는 뜻이 아니다. 여러 환자와 전이에서 얻은 target을 통해 조건부 평균을 학습한다. 검열로 선택된 전이 분포가 이 기대값을 보존하는지는 별도의 검토 대상이다.

### CDF 제곱오차

관측 완료 transition에 대해 다음을 기본 목적함수로 둔다.

$$
\mathcal L_{\mathrm{TD}}
=\mathbb E_n\left[
\sum_c\frac{1}{H}\int_0^H
\left(F_{\theta,c}(\tau\mid s_n)
-\operatorname{sg}[\widetilde F_{n,c}(\tau)]\right)^2d\tau
\right].
$$

$\operatorname{sg}$는 target 쪽 gradient를 차단한다는 뜻이다. 단일 사건의 전체 시간축에서는 squared Cramér distance의 형태이며, 여기서는 유한 horizon과 competing-risk CIF에 적용한 제곱오차이다. 전체 분포에 대한 거리의 성질이나 수렴 정리가 자동으로 따라오는 것은 아니다.

NFDRL [5]도 연속 분포와 Bellman target을 비교하지만, 그 논문의 geometry-aware surrogate는 위 적분식과 다르다. 우리는 CDF/CIF를 직접 평가할 수 있으므로 우선 직접 제곱오차를 사용한다. Normalizing flow와 KDE는 필수 구성요소가 아니다.

### 긴 관측 간격과 수치적분

구현에서는 $\tau_m\sim\operatorname{Uniform}(0,H)$를 샘플링하여 적분을 근사한다.

$$
\widehat{\mathcal L}_{\mathrm{TD}}
=\frac{1}{BM}\sum_{n=1}^{B}\sum_{m=1}^{M}\sum_c
\left(F_{\theta,c}(\tau_m\mid s_n)
-\operatorname{sg}[\widetilde F_{n,c}(\tau_m)]\right)^2.
$$

$B$는 batch 크기, $M$은 적분 평가점 수다. 이 평가점은 모델의 출력 bin이 아니다. Shift는 항상 반올림하지 않은 실수 $D_n$으로 계산한다.

$D_n\geq H$이고 그때까지 사건 없이 재원한 것이 확인되었다면, horizon 안에서는 모든 사건 CIF target이 0이다. Bootstrap이 horizon 밖에 있더라도 사건 없는 구간의 학습은 가능하다. 다만 $H$ 이후 tail의 정확성을 이 loss만으로 보장하지는 않는다.

## 7. 검열 처리에 대한 현재 결론

### TCSR처럼 관측된 전이를 사용한다

기록이 $x_0\to x_1\to x_2$에서 끝나고 $x_2$가 살아 있는 마지막 상태라면, TCSR는 $(x_0,x_1)$과 $(x_1,x_2)$를 사용한다. $x_2$를 사망으로 바꾸거나 관측되지 않은 $x_3$를 만들지 않는다. 마지막 도착 상태 $x_2$의 예측은 이전 상태의 bootstrap target에 사용한다 [1, Algorithm 1].

우리도 같은 원칙을 따른다. 3.7시간과 8시간에 상태가 관측되고 8시간에서 추적이 종료되었다면, 두 상태 사이의 4.3시간 transition을 그대로 학습한다. **검열 이후의 실제 결과를 아는 것과, 마지막 상태에서 모델이 미래를 예측하는 것은 다르다.** 후자는 사용할 수 있다.

따라서 검열된 환자를 전부 제거하거나, 검열 환자의 마지막 관측으로 들어오는 전이를 제외하지 않는다. 다음 상태가 없는 마지막 관측에서 출발하는 가짜 전이는 만들지 않는다. 마지막 상태의 미래 예측은 공유 파라미터와 다른 환자의 전이에서 정보를 얻지만, 데이터가 없는 영역에서 정확성을 보장할 수는 없다.

### 관측 사이에서 검열된 경우를 구분한다

| 데이터 상황 | 기본 처리 |
|---|---|
| 다음 EHR 상태를 관측했고 이후 추적이 종료됨 | 해당 관측 완료 전이를 TD에 사용 |
| 사망 시각을 관측함 | 사망 hard target 사용 |
| 생존퇴실 시각을 관측함 | 퇴실 hard target 사용 |
| 마지막 EHR 이후 생존만 추가 확인되고 검열됨 | 이전 전이는 사용하고, 추가 구간은 부분 관측으로 별도 보존 |
| 다음 상태나 종료 사건을 모름 | 해당 미완료 전이의 전체 TD target을 만들지 않음 |

예를 들어 마지막 EHR은 8시간이고 11시간까지 재원·생존만 확인된다면, 추가 3시간은 사건 없는 구간이지만 11시간의 새 EHR 상태는 없다. 8시간 값을 복사해 새로운 상태로 간주하는 것은 추가 가정이다.

첫 기본 TD 구현은 이 부분 구간을 별도 테이블에 보존한다. 추가 생존 정보를 사용하려면 $-\log S(3\mid s_8)$ 같은 검열 likelihood를 보조항 또는 비교 실험으로 고려할 수 있다. **별도 likelihood의 상시 결합은 현재 핵심 설계의 필수 조건이 아니다.** 보조항을 사용한 버전은 단일 TD loss 버전과 구분하여 보고한다.

### 검열을 처리하는 것과 편향을 해결하는 것은 다르다

행정적 종료 전에 완료된 전이만 모으면 긴 전이가 덜 관측될 수 있다. 따라서 검열이 외부에서 발생하더라도, 완료된 가변 길이 전이의 표본분포가 원래 결합분포와 같다고 단정할 수 없다. 검열 이전에 다음 전이를 관측할 확률이 소요시간에 따라 달라지기 때문이다.

이를 확인하기 위해 같은 합성 trajectory에 서로 다른 검열 규칙을 적용해 분포 복원 편향을 측정한다. 필요하면 전이 관측확률 보정이나 부분 관측 전이를 활용하는 방법을 도출한다. 원래 TCSR의 생존 가중치는 IPCW가 아니며, 이를 그대로 복사하는 것만으로 이 문제를 해결하지는 않는다.

## 8. 첫 구현의 구체적인 구성

### Encoder

첫 모델은 causal GRU로 시작한다. 입력은 측정값, mask, 변수별 마지막 측정 이후 시간, 입실 후 경과시간, 정적 특성이다. 미래 측정으로 보간하거나 전체 환자 이력을 미리 요약하는 방식은 사용하지 않는다. 정규화 통계는 train split에서만 계산한다.

측정값 하나마다 전이를 만들지, 동일 시각의 기록을 묶을지 정한다. 같은 시각에 여러 기록이 있는 경우 하나의 관측으로 묶는 것이 초기 구현에 단순하다. 기록 시각과 실제 임상 이용 가능 시각이 다르면 누출 여부를 점검한다.

### 연속시간 head

초기 구현 후보는 사건별 Weibull mixture이다. 상태별 사건 확률 $\pi_c$와 사건별 mixture 가중치 $w_{cm}$에는 softmax를 사용한다. Scale $a_{cm}$와 shape $b_{cm}$는 양수로 제한한다.

$$
F_c(\tau\mid s)
=\pi_c(s)\sum_{m=1}^{M_{\mathrm{mix}}}w_{cm}(s)
\left[1-\exp\left\{-\left(\frac{\tau}{a_{cm}(s)}\right)^{b_{cm}(s)}\right\}\right].
$$

이 표현은 사건별 CIF의 단조성과 합이 1 이하라는 조건을 보장한다. 충분히 오래 추적하면 두 사건 중 하나로 종료된다는 분포 가정이 있다. 장기 tail과 사건 종류 확률은 검열이 많으면 약하게 식별될 수 있다. Mixture head는 실용적인 시작점이며 핵심 신규 기여는 아니다.

단일 사건에서는 $\pi=1$로 두면 된다. 일반 Cox PH처럼 hazard에서 CDF를 만드는 head도 가능하지만, 원문의 이산 logistic head를 그대로 사용하면 임의의 실수 시간 평가에는 추가 설계가 필요하다.

### 학습 데이터 구조

| 필드 | 내용 |
|---|---|
| patient_id / episode_id | 환자 및 ICU episode 식별자 |
| history_prefix | 현재 시각까지의 기록 |
| next_history_prefix | 다음 관측까지의 기록 또는 종료 표식 |
| delta_hours | 실제 전이시간 |
| transition_kind | observation / death / discharge |
| partial_censor_duration | 마지막 관측 이후 추가로 알려진 무사건 시간 |

상태 embedding은 학습 중 encoder로 계산한다. 고정된 전처리 embedding과 학습 가능한 encoder를 혼동하지 않는다. Target model은 자신의 encoder로 다음 이력을 처리한다.

### 학습 절차

1. 환자 단위로 train, validation, test를 분리한다. 같은 환자의 episode가 split을 넘지 않게 한다.
2. 관측 완료 전이와 미완료 검열 구간을 별도로 구성한다. 검열된 환자도 완료 전이는 남긴다.
3. Online model을 초기화하고 target model에 복사한다. Survival likelihood 사전학습은 안정화 옵션이며 필수는 아니다. 사용 여부를 보고하고 ablation으로 비교한다.
4. 환자와 그 환자의 관측시점을 샘플링한다. 빈번히 측정된 환자가 과도하게 지배하지 않도록 샘플링 목표를 고정한다.
5. Target model로 다음 상태의 CIF를 계산한다. 실제 경과시간 shift 또는 관측된 종료 사건으로 target을 만든다.
6. 하나의 TD 제곱오차로 online model을 업데이트한다. Target에는 gradient를 보내지 않는다.
7. Target model을 이동평균 또는 주기적 복사로 갱신한다. Validation 예측 지표로 모델을 선택한다.

고정된 짧은 history window나 학습된 상태 표현이 Markov 가정을 충족하는지는 별도의 가설이다. 환자별 샘플링도 informative observation을 자동으로 보정하는 것은 아니다.

### 구현 의사코드

```python
for batch in loader:
    tau = sample_uniform_times(0, horizon_hours)
    pred = online.cif(batch.history_prefix, tau)

    with no_grad():
        # shape: batch x evaluation_times x causes
        target_cif = zeros_like(pred)
        alive = batch.transition_kind == "observation"
        shifted_tau = clamp(tau - batch.delta_hours, min=0)

        # 다음 관측이 있는 행에만 next-state model을 호출한다.
        target_cif[alive] = target.cif(
            batch.next_history_prefix[alive], shifted_tau[alive]
        )

        ended = ~alive  # loader에는 완료된 임상 전이만 포함
        target_cif[ended] = event_indicator(
            tau, batch.delta_hours[ended], batch.event_cause[ended]
        )

    loss = ((pred - target_cif) ** 2).sum(dim="cause").mean()
    update_online(loss)
    update_target_by_ema()
```

이 코드는 의미를 전달하는 의사코드다. 실제 구현에서는 시간 tensor broadcasting, 종료 행의 인덱싱, $F_c(0)=0$, target encoder의 gradient 차단을 명시적으로 확인한다. 추론 시에는 현재까지의 기록과 online model만 필요하다.

## 9. 검증 실험과 성공 기준

### 먼저 작은 합성 데이터에서 확인한다

참 사건시간 분포를 계산하거나 많은 시뮬레이션으로 얻을 수 있는 환경을 만든다. 처음에는 완전 관측과 단일 사건으로 구현을 점검하고, 이어 competing risks와 검열을 추가한다.

- 동일한 임상 경로에서 관측 빈도만 바꿨을 때 결과가 어떻게 변하는가.
- 고정 간격, 불규칙 간격, 상태 의존 관측 간격에서 분포를 복원하는가.
- Exponential holding time과 비지수 holding time에서 성능이 달라지는가.
- 검열을 마지막 관측에 맞춘 경우와 전이 중간에 발생시킨 경우가 어떻게 다른가.
- $D_n\geq H$인 전이가 많을 때 무사건 구간 학습과 tail 예측은 어떻게 변하는가.

모델의 성능은 가정에 따라 달라질 수 있다. 특정 상황의 성공만으로 일반적인 informative censoring을 해결했다고 주장하지 않는다.

### 동일 모델에서 학습 방법을 비교한다

| 방법 | 비교 목적 |
|---|---|
| 동일 encoder와 head의 직접 survival likelihood | TD가 실제로 추가 이득을 주는지 |
| 실제 경과시간 TD | 제안 학습 방법의 효과 |
| 실제 경과시간을 고정값으로 바꾼 TD ablation | 시간 정보가 필요한지 확인하는 의도적 대조군 |
| Target network 및 초기화 ablation | 안정화 장치의 효과 |
| Head 변경 ablation | 효과가 분포 표현력에 의존하는지 |

시간 간격을 고정값으로 바꾼 ablation은 정확한 TCSR 재현과 동일하지 않다. 원래 TCSR·DeepTCSR를 비교할 때는 이산 시간축으로 변환하는 방법을 명시하고, 서로 다른 시간 단위의 점수를 바로 비교하지 않는다.

이후 Dynamic-DeepHit, SurvLatent ODE 등 방법별 아키텍처를 허용하는 외부 비교를 수행한다. 데이터 분할, 입력 가능한 정보, 사건 정의, tuning budget, 예측시점과 horizon은 맞춘다.

### 평가 기준

주요 지표는 사건별 Brier score와 calibration이다. 검열이 있으면 사용한 평가 보정과 가정을 함께 보고한다. 사건별 discrimination과 학습 데이터 크기별 성능을 보조적으로 평가한다. TD training loss 감소만으로 예측이 좋아졌다고 판단하지 않는다.

MIMIC-IV에서 개발하고 eICU에서 외부 검증하는 방향을 고려하되, 변수·코호트·결과 정의가 일치하는지 먼저 확인한다. 외부 검증의 성능 개선은 기대 결과이며 아직 확인된 사실이 아니다.

## 10. 연구 기여와 아직 주장할 수 없는 것

제안하는 기여는 **관측시점 기반 semi-Markov 구조 아래 실제 경과시간을 보존하는 사건시간 분포 TD를 설계하고, TCSR의 검열 trajectory 활용 원칙을 계승하여 그 학습 이득과 한계를 검증하는 것**이다.

다음 항목은 이미 관련 선행연구가 있다.

- 이산 사건시간 분포의 temporal-consistency 학습: TCSR [1].
- Target network를 이용한 end-to-end survival 학습: DeepTCSR [2].
- 불규칙 ICU EHR에서 semi-MRP 기반 TD 사망위험 예측: Frost et al. [4].
- 연속 분포와 distributional Bellman target 및 Cramér 계열 loss: NFDRL [5].
- 연속시간 homogeneous Markov chain의 temporal-consistent hitting-time 추정: Escobar-Bach et al. [6].

따라서 단순한 실제 시간 shift, target network, flow 사용만으로 신규성을 주장하지 않는다. TCSR 자체도 연속시간 확장 가능성을 논의한다 [1, §5]. 우리 구성 전체의 신규성은 추가 문헌 검토와 결과에 따라 판단한다.

시간을 누적하는 target은 discount가 없는 형태이므로 NFDRL의 discount 기반 contraction 보장을 그대로 적용할 수 없다. Target network도 일반적인 수렴 증명이 아니다. 관측 시점의 선택이 충분히 반영되지 않거나 상태 표현이 과거를 놓치면 TD의 bootstrap 편향이 커질 수 있다.

Cox partial likelihood의 결합과 normalizing flow head는 초기 핵심 범위에서 제외한다. 우선 실제 시간 TD, 검열된 전이 활용, 동일 모델의 직접 지도학습 비교를 완성한다. 추가 생존 정보용 likelihood나 검열 보정은 그 효과와 가정을 분리해 평가한다.

## 11. 참고문헌

1. **Maystre & Russo (2022). Temporally-Consistent Survival Analysis.** NeurIPS. 이산 survival TD, Algorithm 1의 검열 trajectory 처리, 연속시간 확장 논의의 출발점. [논문](https://papers.nips.cc/paper_files/paper/2022/file/455e1e30edf721bd7fa334fffabdcad8-Paper-Conference.pdf) · [코드](https://github.com/spotify-research/tdsurv)

2. **Vargas Vieyra & Frossard (2024). Deep End-to-End Survival Analysis with Temporal Consistency.** DeepTCSR의 target network와 end-to-end 학습. [논문](https://arxiv.org/abs/2410.06786)

3. **Fatemi et al. (2022). Semi-Markov Offline Reinforcement Learning for Healthcare.** CHIL. 가변 전이시간을 반영한 value-based offline RL. 치료 정책 학습이며 생존분포 예측 자체는 아니다. [논문](https://proceedings.mlr.press/v174/fatemi22a.html)

4. **Frost, Li & Harris (2024). Robust Real-Time Mortality Prediction in the Intensive Care Unit using Temporal Difference Learning.** ML4H. 불규칙 ICU 데이터와 semi-MRP 기반 TD의 직접적인 관련 연구이며, scalar mortality risk를 예측한다. [논문](https://arxiv.org/abs/2411.04285) · [코드](https://github.com/tdgfrost/td-icu-mortality)

5. **Alami C. et al. (v2, 2026). Parameter-Efficient Distributional RL via Normalizing Flows and a Geometry-Aware Cramér Surrogate.** 최초 게시 2025. Continuous distributional RL, exact Cramér 및 surrogate 비교. [논문 v2](https://arxiv.org/pdf/2505.04310v2)

6. **Escobar-Bach, Popier & Sahin (2026). Multi-state model with temporal-consistent survival analysis for homogeneous Markov chains.** Markov hitting-time 분포의 추정과 이론. [논문](https://arxiv.org/abs/2605.18358)

7. **Moon, Groha & Gusev (2022). SurvLatent ODE.** MLHC. 불규칙 longitudinal EHR와 competing risks의 neural survival 비교 모델. [논문](https://proceedings.mlr.press/v182/moon22a.html)

8. **Alaa & van der Schaar (2018). A Hidden Absorbing Semi-Markov Model for Informatively Censored Temporal Data: Learning and Inference.** JMLR. 실제 잠재 임상 상태와 체류시간을 모델링하는 대안이며, 관측시점 기반인 본 설계와 구분된다. [논문](https://jmlr.org/papers/v19/16-656.html)














논문 흐름은 “평균 시간 환산의 한계 → 실제 전이시간을 반영한 TD → 원인 분리 실험 → 실제 EHR 검증”으로 잡으면 좋아.

Introduction
목표: 불규칙하게 관측되는 EHR로 실제시간 생존분포 예측.
도입 그림: 방문 수와 평균 간격이 같아도 사망시간 분포는 다르다.
제안: 방문 전이 기반 학습을 유지하면서 실제 \(D\)를 반영하는 SM-TCSR.
검증된 contribution을 요약.
Background & Problem Setup
TCSR의 방문축 TD와 실제시간 예측의 차이.
상태 \(s_n\), 전이시간 \(D_n\), 잔여 사건시간 \(R_n\), 검열 정의.
Semi-Markov 가정과 현재 상태가 충분하다는 조건.
예측 시 사용 가능한 정보 명시.
Method: SM-TCSR
실제 \(D_n\)만큼 이동하는 분포 TD target.
사망·생존 전이·검열별 학습 처리.
분포 표현과 목적함수.
학습 알고리즘과 시간 복잡도.
CDF와 categorical PMF는 연결된 표현이므로, 하나를 본문 기본 구현으로 두고 다른 표현은 변형으로 비교.
Theoretical Analysis
Semi-Markov 가정하에서 temporal-consistency 관계 유도.
일정한 간격에서 방문축 TD로 환원되는 성질.
식별 가능한 조건과 검열 처리의 조건.
가능하면 수렴 분석. 수축사상 증명은 성립하는 조건을 확인한 뒤 포함.
Synthetic Experiments
메인 예제: 같은 전이 구조·같은 평균·다른 시간 분포.
Global/State mean clock과 실제 \(D\) 사용의 차이.
같은 모델 구조에서 TCSR·DeepTCSR·SM-TCSR 비교.
표본 수, 시간 변동, 검열에 따른 영향 분석.
Real-world EHR Experiments
데이터셋, 사건 정의, 예측 시점, 평가 지표.
동일 encoder/head 비교: TD 설계 자체의 효과 확인.
기존 모델별 아키텍처 비교: 전체 예측 성능 평가.
실제시간 격자로 재구성한 TCSR와 직접 생존학습 기준선 포함.
Calibration·예측 오차·계산 비용 평가.
Discussion & Conclusion
언제 실제 시간 분포를 반영하는 것이 유리한가?
상태 표현, 관측 과정, 검열 가정의 한계.
실험과 이론이 뒷받침하는 범위에서 결론.
