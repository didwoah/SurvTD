# 미션: SurvTD (시계열 TD 일관성 기반 연속 동적 생존분석)

## 학습 목적 (Why)
SurvTD의 차별화된 수학적 이론, 실제 코드베이스 구현체, 벤치마크 및 음성 대조군 실험 설계, 그리고 NeurIPS/ICML 최상위 학회 제출 수준의 논문 서사 구조를 완벽히 마스터하여, 독자적으로 연구를 주도하고 실험을 실행 및 집필할 수 있는 역량을 갖춘다.

## 성공의 기준 (Success looks like)
- 나눗셈 발산을 소멸시키는 연속 재생 이동 연산자 $\Pi \Phi_{+\Delta t}$와 Theorem 1(Cramér 거리 하의 아핀 엄밀 수축 사상)의 기하학적 증명 뼈대를 명쾌하게 설명할 수 있다.
- `src/operators/survtd_operator.py`, `src/models/survtd.py`, `src/models/hazard_head.py`의 핵심 텐서 연산과 최근 해결된 A-16 `km_prior` 초기화 및 타깃 네트워크 재동기화 로직을 자유자재로 다룬다.
- Track A/B 벤치마크 스크립트를 구동하고, Bedside Alarm Fatigue 지표 및 5대 사전등록 킬 스위치(Kill Criteria 0~5)의 판정 기준을 분석할 수 있다.
- 9페이지 분량 제한 내에서 Title, Abstract, Figure 1(Teaser), Table 1~3, 관련 연구 대조표 및 적대적 리뷰어 방어 논리를 완벽히 구조화하여 서술할 수 있다.

## 제약 사항 (Constraints)
- 학습자는 이미 CoxPH(위험비, 부분우도, 랜드마크 기법)와 TCSR/DeepTCSR(이산 $\Delta t=1$ 가정, 베이즈 $p/S$ 나눗셈, 1-step TD 손실)의 기초를 완벽히 이해하고 있으므로 기초 설명은 생략한다.
- 건조한 델타-엡실론 증명 나열보다는 기하학적 직관과 핵심 보조정리(Lemma) 논리 뼈대 중심의 설명 방식을 선호한다.

## 제외 범위 (Out of scope)
- 초급 통계학 생존분석(카플란-마이어 기본 유도, 단순 콕스 회귀 가정 등).
- 일반 강화학습 기초(기본 Q-러닝, 표준 MDP 가치반복법 등).
