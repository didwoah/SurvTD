# Paper Outline & Page Budget: SurvTD

**Target Venue**: NeurIPS / ICML / ICLR (Main Track, 9-page limit excluding references)  
**Spine Alignment**: Grounded in `claim-tree.json` (`C0 ~ C4`) and audited against 5-member critic committee.

---

## 1. Section Budget & Page Allocation (Total: 9.0 Pages)

| Section | Title | Target Budget | Bound Anchors | Associated Claims |
| :---: | :--- | :---: | :---: | :---: |
| **§ 1** | **Introduction** | **1.25 Pages** | `fig:1` | `C0` |
| **§ 2** | **Related Work & Lineage** | **0.75 Pages** | — | — |
| **§ 3** | **Continuous-Time Semi-Markov Survival Bellman Operator** | **2.50 Pages** | `thm:1`, `eq:1`–`eq:8` | `C1`, `C2` |
| **§ 4** | **Empirical Evaluation** | **3.00 Pages** | `fig:2`, `tab:1`, `tab:2` | `C2`, `C3`, `C4` |
| **§ 5** | **Discussion & Limitations** | **1.00 Pages** | — | — |
| **§ 6** | **Conclusion** | **0.50 Pages** | — | — |

---

## 2. Section-by-Section Architectural Blueprint

### § 1. Introduction (1.25 Pages)
- **1.1 The Clinical & Cyber-Physical Setting**: Proactive early warning from irregularly sampled longitudinal time-series subject to right-censoring.
- **1.2 The Dilemma of Prior Paradigms**:
  - *Terminal MC NLL* (`Lee2019`, `Bleistein2024`): High variance, alarm jittering, ignores intermediate consistency.
  - *Discrete TCSR* (`Maystre2022`, `DeepTCSR2024`): Uniform step assumption ($\Delta=1$) and catastrophic division explosion ($p / S(\Delta t)$).
- **1.3 The Proposed Solution (SurvTD)**: Continuous-time Semi-Markov Bellman Operator with rightward temporal renewal and non-expansive Cramér projection.
- **1.4 Figure 1 Spec (`fig:1`)**: Renders `C0` directly—contrasts TCSR's explosive $\div S$ curve and MC NLL's high-variance jitter against SurvTD's stable, contractive hazard trajectory.
- **1.5 Summary of Falsifiable Contributions**: 4 concise bullets mapped to `thm:1`, `fig:1`, `fig:2`, `tab:1`.

---

### § 2. Related Work & Lineage (0.75 Pages)
- **2.1 Dynamic Survival Analysis**: Joint models, Landmarking, DeepSurv, Dynamic-DeepHit (`Lee2019`), and CoxSig (`Bleistein2024`).
- **2.2 Temporal Consistency in Survival Prediction**: Maystre & Russo (`Maystre2022`, TCSR), DeepTCSR (`2024`). Structural delta: continuous intervals, non-divergent operator.
- **2.3 Continuous-Time Reinforcement Learning & SMDPs**: Bradtke & Duff (`Bradtke1994`), Continuous-time Bellman equations, Bellemare et al. (`Bellemare2017`, Distributional RL / C51).

---

### § 3. Continuous-Time Semi-Markov Survival Bellman Operator (2.50 Pages)
- **3.1 Problem Setup & Absorbing SMDP Formulation**:
  - Longitudinal history $\mathcal{H}_{t_j}$, irregular gaps $\Delta t_j = t_{j+1} - t_j$.
  - State space, absorbing failure state, and discrete future hazard bins $\{s_1, \dots, s_K\}$.
- **3.2 Rightward Support Shift & Non-Expansive Categorical Projection ($\Pi$)**:
  - Renewal remaining lifetime $R_j = R_{j+1} + \Delta t_j$.
  - Formalizing shift operator $\Phi_{+\Delta t_j}$ and categorical projection kernel $\Pi$.
  - **Theorem 1 (`thm:1`)**: Proof of Cramér metric non-expansiveness: $\|\Pi \Phi_{+\Delta t_j} F_1 - \Pi \Phi_{+\Delta t_j} F_2\|_2 \le \|F_1 - F_2\|_2$.
- **3.3 Probability-Conserving Survival Bellman Target**:
  - Formulation: $T_{\theta^-} p_j(s) = \mathbb{I}[E_j \in \Delta t_j] \cdot \delta_{\tau_j - t_j}(s) + \mathbb{I}[E_j \notin \Delta t_j] \cdot (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j(s)$.
  - Conservation lemma: $\sum_k T_{\theta^-} p_j(s_k) = 1.0$, preventing Cramér loss divergence.
  - Decoupling endogenous discount via parameter-frozen EMA Target Network $\theta^-$.
- **3.4 Terminal Right-Censoring IPCW Tail Completion**:
  - Handling alive discharges ($E_M = 0$) at $t_M$ via Kaplan-Meier inverse censoring weights.
- **3.5 Continuous-Time Multi-Step $\lambda$-Return Recursion**:
  - Formulation of $G^\lambda_j(s)$ with continuous temporal decay, uniting 1-step bootstrapping with terminal returns.

---

### § 4. Experimental Evaluation (3.00 Pages)
- **4.1 Benchmark Ladder & Experimental Protocol**:
  - *Ladder*: (1) Tumor Growth ODE, (2) NASA C-MAPSS FD001–FD004, (3) Real clinical cohorts (MIMIC-IV Sepsis-3, PBC).
  - *Tuning Parity*: 50-trial hyperparameter optimization for all baselines (CoxSig, TCSR with $\epsilon$-clamping, Dynamic-DeepHit) under patient-level 5-fold cross-validation.
- **4.2 Comparative Discrimination & Calibration (Table 1 `tab:1` & Table 2 `tab:2`)**:
  - Time-dependent C-index, Uno's Dynamic AUC, Integrated Brier Score (IBS).
  - Demonstrating up to 24.2%p concordance gain and 84.4% IBS reduction over CoxSig.
- **4.3 Bias-Variance $\lambda$-Spectrum & Mechanistic Negative Controls (Figure 2 `fig:2`)**:
  - Sweeping $\lambda \in [0.0, 1.0]$. Proof of Sutton & Barto's inverted-U curve (peak at $\lambda=0.6$, collapse at $\lambda=1.0$).
  - Discrete uniform ablation ($\Delta t = 1.0$) confirming the active role of continuous interval discounting.
- **4.4 Clinical Alarm Stability & Alarm Fatigue Reduction**:
  - Alarm Jitter Count (transition frequency) and False Alarm Rate (FAR) per bed-day.
  - Demonstrating that SurvTD prevents high-frequency alarm flicker while preserving early warning lead time.

---

### § 5. Discussion, Scope Boundaries, & Ethical Impact (1.00 Page)
- **5.1 Where the Gain Lives (Scope Boundary)**: Highly non-linear degradation and irregular sampling regimes. In linear stationary processes (OU process), simple signature/Cox models remain sufficient.
- **5.2 Clinical Deployment Considerations**: Bedside inference latency ($< 2 \text{ ms}$), Decision Curve Analysis (Net Clinical Benefit).
- **5.3 Ethical Impact**: Eliminating alarm fatigue to protect nursing staff from alert desensitization.

---

### § 6. Conclusion (0.50 Page)
- Summary of core theoretical findings and the foundational role of Semi-Markov Bellman operators in continuous-time dynamic survival analysis.
