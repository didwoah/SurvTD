# Top-Tier ML Paper Writing Principles
## ICLR / ICML / NeurIPS 논문 작성 바이블: 세계적 석학 5인의 핵심 원칙 요약

> 본 문서는 ICLR, ICML, NeurIPS 등 최고 권위 기계학습 학회 논문 작성 시 항상 곁에 두고 점검하는 범용 가이드라인입니다. 특정 연구 주제에 종속되지 않고 컴퓨터 비전, 자연어 처리, 강화학습, 시계열, 생성 모델 등 모든 기계학습 분야에 보편적으로 적용할 수 있도록 정리되었습니다.

---

### 1. Zachary Lipton & Jacob Steinhardt (CMU / Stanford)
**출처**: *"Troubling Trends in Machine Learning Scholarship"* (ICML 2018 Debates / CACM 2019)  
**핵심 가르침**: **"심사위원이 가장 경멸하는 4대 허세와 악습을 박멸하라"**

* **1. 설명(Explanation) vs 추측(Speculation) 엄격 분리**:
  * 수학적 증명이나 엄밀한 실험으로 입증된 사실과 저자의 직관적 뇌피셜을 절대로 섞어 쓰지 않는다.
  * 추측을 쓸 때는 반드시 *"We hypothesize that..."* 처럼 가설임을 분명히 밝힌다.
* **2. 성능 향상 원인의 명확한 귀인 (Ablation 필수)**:
  * "우리 모델이 베이스라인을 이겼다"로 끝내지 마라.
  * 성능 향상이 **하이퍼파라미터 튜닝 덕분인지, 단순 정규화 트릭인지, 핵심 메커니즘 때문인지** 엄격한 어블레이션으로 분리 입증하라.
* **3. 허세 수학(Mathiness) 퇴출**:
  * 아이디어를 명확히 전달하기 위한 도구로만 수학을 사용하라.
  * 사소한 아이디어를 권위적으로 포장하거나 비판을 회피하기 위해 난해한 수식을 도배하는 행위를 엄단한다.
* **4. 의인화·과장 어휘 금지 (Misuse of Language)**:
  * 모델이 무언가를 *"이해한다(understands)"*, *"추론한다(reasons)"* 같은 모호한 인간적 표현을 금지한다.
  * 최적화 목적함수, 기하학적 사상, 확률적 연산 등 구체적 과학 용어로 기술하라.

---

### 2. Bill Freeman (MIT CSAIL / Google Research)
**출처**: *"How to Write a Good Paper / CVPR Submission"* (Ted Adelson 공식)  
**핵심 가르침**: **"1페이지에서 승부를 내라 (The 5-Minute Coffee Test)"**

* **1. 커피 테스트 (The Coffee Test)**:
  * 바쁜 심사위원은 커피 한 잔을 마시는 5분 동안 Abstract, Figure 1, Introduction만 훑어보고(Scan) 심리적 점수(Accept vs Reject)의 80%를 결정한다.
* **2. Introduction 4단계 깔대기 구조**:
  1. **문제 정의 (Problem)**: 이 문제가 왜 중요하며 풀기 어려운가?
  2. **기존 해법의 결함 (Existing Flaws)**: 기존 SOTA 모델들이 왜 불만족스럽고 실패하는가?
  3. **우리의 해법 (Our Solution)**: 우리는 어떤 새로운 통찰로 이 병목을 해결했는가?
  4. **관련 연구와의 경계 (Context)**: 유사 기법들과의 차별점이 무엇인가?
* **3. Hero Figure (Figure 1의 법칙)**:
  * Figure 1은 복잡한 신경망 레이어 블록도(Conv-Linear)를 그리는 자리가 아니다.
  * 글을 전혀 안 읽고 그림만 봐도 **"기존 방식의 치명적 한계와 우리 방식의 핵심 원리 및 우월한 결과"**를 한눈에 직관적으로 이해할 수 있어야 한다.

---

### 3. Simon Peyton Jones (Microsoft Research / Cambridge)
**출처**: *"How to Write a Great Research Paper"* (CS 연구자들의 필독 바이블)  
**핵심 가르침**: **"독자를 긴장시키지 마라 (Don't keep the reader in suspense)"**

* **1. Put your key idea on page 1**:
  * 논문은 추리 소설이 아니다. 핵심 결론을 4~5페이지 뒤에 숨기지 말고, **1페이지 첫 머리에서 즉시 폭로하라.**
* **2. Nail your contributions**:
  * 논문의 기여점을 독자가 본문에서 추측하게 만들지 마라.
  * 1페이지 말미에 **반증 가능하고(falsifiable) 구체적인 3개의 불릿 포인트**로 명확히 못을 박아라.
* **3. Tell a story**:
  * 논문은 시간순 연구 일기가 아니다.
  * "우리가 이것도 해보고 저것도 해봤다"가 아니라, 하나의 명쾌한 중심 가설을 증명해가는 단단한 서사(Narrative)여야 한다.

---

### 4. Joelle Pineau (McGill / Mila / Meta AI VP)
**출처**: *"The Machine Learning Reproducibility Checklist"* (NeurIPS, ICLR, ICML 공식 규격)  
**핵심 가르침**: **"체리피킹을 근절하고 엄격한 재현성을 입증하라"**

* **1. 다중 시드 오차 범위 필수 (Multi-Seed Error Bars)**:
  * 가장 잘 나온 1개 시드 결과만 제시하는 행위는 탈락 사유다.
  * 동일한 조건에서 최소 3~5개 이상의 랜덤 시드로 수행한 **평균과 표준편차($\mu \pm \sigma$)**를 반드시 보고하라.
* **2. 튜닝 예산의 공정성 (Tuning Parity)**:
  * 내 모델만 하이퍼파라미터 탐색을 수백 번 돌려 최적화하고, 비교 대상 베이스라인은 기본값(Default)으로 대충 돌리는 "부당한 비교"를 엄단한다.
* **3. 데이터 누수(Data Leakage) 완전 차단**:
  * 전처리(스케일러, 임퓨터) 파라미터가 테스트셋에서 계산되어 들어가지 않도록 훈련셋만으로 엄격히 고립시켜라.
* **4. 가정의 사전 명시 (Assumption Ledger)**:
  * 정리(Theorem)가 등장하기 전에, 성립 조건과 수학적 가정을 누락 없이 명시하라.

---

### 5. Devi Parikh & Dhruv Batra (Georgia Tech / FAIR)
**출처**: *"How We Write Papers / Planning Paper Writing"*  
**핵심 가르침**: **"골격을 먼저 짜고, 문장은 깎아서 밀도를 높여라"**

* **1. Coarse-to-Fine (골격 우선 작성)**:
  * 처음부터 줄글 문장을 쓰지 마라.
  * 단락별 **1문장 요약(Intro Skeleton)**을 먼저 작성하여 논리적 인과 흐름이 완벽할 때만 세부 문장을 채운다.
* **2. Tighten, Don't Just Delete (압축과 밀도)**:
  * 분량 초과 시 섹션을 통째로 날리거나 부록으로 유배 보내지 마라.
  * 같은 의미를 더 적고 정밀한 단어로 압축(Tightening)하면 글의 긴장감과 완성도가 극대화된다.

---

## 📋 Top-Tier ML 논문 집필 실전 체크리스트

| 점검 항목 | 기준 대가 | 실전 점검 기준 | 완료 여부 |
|:---|:---:|:---|:---:|
| **1페이지 승부** | Bill Freeman | 1페이지 끝까지 읽었을 때 핵심 아이디어와 기여점 3개가 완전히 이해되는가? | [ ] |
| **도판의 직관성** | Bill Freeman | Figure 1만 보고도 제안 방법과 기존 SOTA의 차이 및 핵심 메커니즘을 알 수 있는가? | [ ] |
| **설명 vs 추측** | Zachary Lipton | 증명되지 않은 직관을 사실인 것처럼 단정적으로 쓰지 않고 가설로 분리했는가? | [ ] |
| **AI 어휘 박멸** | Zachary Lipton | `delve into`, `pivotal role`, `testament` 등 상투적 수식어가 제거되었는가? | [ ] |
| **시드 분산 표기** | Joelle Pineau | 표의 모든 비교 수치에 다중 시드(3~5 seeds) 표준편차($\pm \sigma$)가 표기되어 있는가? | [ ] |
| **튜닝 공정성** | Joelle Pineau | 비교 대상 베이스라인들과 공정한 탐색 예산(Tuning Parity) 조건에서 비교되었는가? | [ ] |
| **기여점 불릿** | Peyton Jones | 기여점이 3개의 명확하고 반증 가능한(falsifiable) 불릿으로 정리되었는가? | [ ] |
| **표 세로선 제거** | Michael Black | LaTeX 표에 세로선(`|`)이 없고 `booktabs` 표준을 준수하는가? | [ ] |
