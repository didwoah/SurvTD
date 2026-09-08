# SurvTD 학습 참고 자료 (RESOURCES.md)

## 지식 기반 (Knowledge - 1차 문헌)

- [논문: _Temporally-Consistent Survival Analysis (TCSR)_ by Maystre & Marlin (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9a85a49ae1e6efd2ec2cfa02462e08a6-Abstract-Conference.html)
  생존분석에 TD 학습을 최초로 도입한 선행 논문. 활용처: 이산 단위시간 벨만 일관성 공식 및 $p / S(\Delta t)$ 나눗셈 갱신의 기원 파악.
- [논문: _Deep End-to-End Survival Analysis with Temporal Consistency (DeepTCSR)_ by Vargas Vieyra & Frossard (EPFL, arXiv 2024)](https://arxiv.org/abs/2410.06786)
  타깃 네트워크를 도입하여 TCSR을 딥러닝으로 확장한 직전 연구. 활용처: C-MAPSS 및 PBC 벤치마크 수치 대조 및 연속시간 나눗셈 불안정성 분석.
- [논문: _Dynamic-DeepHit: A Deep Learning Approach for Dynamic Survival Analysis With Competing Risks Based on Longitudinal Data_ by Lee et al. (IEEE TBME 2019)](https://ieeexplore.ieee.org/document/8894451)
  대표적인 터미널 우도 기반 동적 생존분석 베이스라인. 활용처: 최종 랭킹 손실과 연속 일관성의 대조, 알람 피로(Alarm Fatigue) 유발 원인 분석.
- [논문: _A Distributional Perspective on Reinforcement Learning (C51)_ by Bellemare, Dabney, & Munos (ICML 2017)](https://proceedings.mlr.press/v70/bellemare17a.html)
  확률 분포 사영의 이론적 기초. 활용처: Cramér/Wasserstein 거리 하에서의 비확장성(Non-expansiveness) 및 삼각 사영($\Pi$) 원리 이해.
- [논문: _Continuous-Time Markov Decision Processes_ by Bradtke & Duff (NeurIPS 1994)](https://proceedings.neurips.cc/paper/1994/hash/c9f0f895fb98ab9159f51fd0297e236d-Abstract.html)
  준-마르코프 결정 과정(SMDP) 이론. 활용처: 연속 시간 경과에 따른 지수 할인율 $\gamma(\Delta t)$ 및 연속 전이 연산자 이해.

## 커뮤니티 및 전문가 지혜 (Wisdom)

- [NeurIPS / ICML / ICLR Machine Learning for Health (ML4H) 워크샵](https://ml4health.github.io)
  연속 시계열 생존분석 및 의료 텔레메트리 최상위 전문가 커뮤니티. 활용처: 논문 포지셔닝 및 임상 평가 프로토콜 피드백.
- [scikit-survival / lifelines 오픈소스 개발 생태계](https://github.com/sebp/scikit-survival)
  생존분석 통계 라이브러리 개발자 그룹. 활용처: 역확률 검열 가중치(IPCW) 표준 구현 및 벤치마크 표준 검증.
