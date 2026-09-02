# Paper Architecture & Section Budget — SurvTD

**Target Venue**: NeurIPS / ICML (Main Track, 9-Page Limit excluding references)  
**Spine Document**: `research/continuous-survival-td/claim-tree.json`  
**Abstract Reference**: `research/continuous-survival-td/abstract.md`  

---

## 1. Page Budget Allocation (9.0 Pages Total)

| Section | Title | Page Target | Carried Claims | Primary Exhibits |
| :--- | :--- | :---: | :---: | :---: |
| **§1** | **Introduction** | **1.25 pp** | $C_0$ | **`fig:1`** (Teaser: Divergence vs. Renewal, Alarm Stability) |
| **§2** | **Related Work & Lineage** | **0.75 pp** | $C_0$ | Comparative Lineage Table |
| **§3** | **The SurvTD Framework** | **2.50 pp** | $C_1, C_2$ | **`thm:1`** (Contraction), **`fig:2`** (Horizon Invariance / Variance) |
| **§4** | **Experimental Evaluation** | **3.00 pp** | $C_3, C_4$ | **`tab:1`** (Benchmark Discrimination), **`tab:2`** (Alarm Fatigue) |
| **§5** | **Analysis & Negative Controls** | **1.00 pp** | $C_0, C_1, C_2$ | **`tab:3`** (Ablation Ladder: NC-A1/A2/A3, NC-B, NC-C) |
| **§6** | **Discussion & Limitations** | **0.50 pp** | — | Scope Boundaries & Informative Sampling Discussion |
| **Total** | | **9.00 pp** | | |

---

## 2. Paragraph-by-Paragraph Blueprint with Claim Binding

### §1. Introduction (1.25 Pages)
- **¶1 (Setting & High Stakes)**: Continuous dynamic survival analysis in intensive care telemetry (MIMIC-IV) and industrial predictive maintenance. The necessity of real-time lifetime distribution updates under irregular sampling.
- **¶2 (The Core Dilemma)**: 
  - *Branch A (Terminal Likelihood)*: Dynamic-DeepHit optimizes final outcomes; predictions oscillate wildly across consecutive hours, causing threshold thrashing and bedside alarm fatigue.
  - *Branch B (Discrete Consistency)*: TCSR/DeepTCSR enforces consistency but assumes unit steps ($\Delta t = 1$). Under continuous time, renormalizing via division by survival probability $p / S(\Delta t)$ explodes to $\infty$ as hazard escalates ($S \to 0$), collapsing predictions into an uninformative uniform distribution.
- **¶3 (The SurvTD Insight & Figure 1 Reference)**: Presenting `fig:1`. Reframing survival transition from multiplicative division into an additive renewal shift ($\Phi_{+\Delta t}$) and categorical projection ($\Pi$) with duration-discounted mixture weighting ($\gamma_j$). Core claim $C_0$ stated in 21 words.
- **¶4 (Contributions Summary)**: The 4 falsifiable contribution bullets from `claim-tree.json`.

---

### §2. Related Work & Lineage (0.75 Pages)
- **¶1 (Dynamic Survival Analysis)**: Landmark models (Landmarking, Dynamic-DeepHit, CoxSig). The limitation: lack of temporal consistency operators.
- **¶2 (Temporal Consistency in Survival)**: TCSR (Maystre & Marlin 2022), DeepTCSR (2024). Lineage analysis: why unit-step division was inherited and why it fails under continuous irregular time.
- **¶3 (Distributional RL & Continuous-Time MDPs)**: C51 (Bellemare et al. 2017) and Bradtke & Duff (1994, SMDP). Clear demarcation: survival renewal is additive translation in remaining time without additive reward, discounted by endogenous survival probability under right-censoring.

---

### §3. The SurvTD Framework (2.50 Pages)
- **§3.1 Problem Formulation & Step-by-Step Architecture (1.00 p)**:
  - Sequence encoding under conditionally unconfounded observation timing ($T \perp\!\!\perp T^{\text{obs}}_{j+1} \mid h_j$).
  - Lifetime support discretization on $K$ uniform bins $\delta_s$.
  - Interval duration discount $\gamma_j = S_{\theta^-}(\Delta t_j)$ under frozen EMA target network $\theta^-$, with sub-bin linear hazard interpolation for $\Delta t < \delta_s$.
  - Composed transition operator $\Pi \Phi_{+\Delta t_j}$: renewal rightward shift with triangular categorical projection. Absorbing terminal bin ($\ge s_K$) semantics.
- **§3.2 Empirical Sample Bellman Target & Operator Contraction (0.75 p)**:
  - **Sample vs. Expected Target**: Empirical sample update evaluates continuation branch $\Pi \Phi_{+\Delta t_j} p_{\theta^-}$ on living transitions and projected Dirac $\delta_{\tau - t_j}$ on terminal deaths.
  - **Theorem 1 (`thm:1`)**: Proof that under a frozen target network $\theta^-$, the expected renewal mixture target map $\mathcal{T} p = (1-\gamma_j)\mu + \gamma_j \Pi \Phi p$ is an affine strict contraction in the squared Cramér metric ($\ell_2^2$ on CDFs) with modulus $\gamma_j < 1$. Mass conservation $\sum p_k = 1.0$. [Carries $C_1$].
- **§3.3 Censoring Completion & Multi-Step Bootstrapping (0.75 p)**:
  - **Truncated Conditional IPCW**: Right-censored tail completed with $\theta^-$, applying $1/\hat{G}(t_M \mid X) \le 10.0$ as a scalar sample weight in the squared Cramér loss.
  - **Multi-Step Recursion**: $G_j = (1 - \lambda_j) \mathcal{T} p_j + \lambda_j \gamma_j \Pi \Phi_{+\Delta t_j} G_{j+1}$ with duration-geometric mixing $\lambda_j = \lambda^{\Delta t_j / \delta_s}$. 
  - **Horizon Invariance & Variance Diffusion (`fig:2`)**: Demonstrating effective bootstrapping horizon invariance across irregular sampling rates, while bounding per-step projection diffusion by $\delta_s^2 / 6$. [Carries $C_2$].

---

### §4. Experimental Evaluation (3.00 Pages)
- **§4.1 Experimental Protocol & Benchmark Setup (0.75 p)**:
  - Four cohorts: MIMIC-IV Sepsis-3 (clinical), Poisson-downsampled NASA C-MAPSS FD001–FD004 (industrial degradation with 50% cycle drop), PBC (clinical trial), and simulated ODE tumor growth.
  - Four compute-matched baselines on identical continuous GRU-D/LSTM backbones: SurvTD, DeepTCSR (with continuous $\Delta t$ feature and clamped division $p / \max(S, 10^{-3})$), Dynamic-DeepHit, and Person-period discrete hazard on 1h grid.
  - 120 GPU-hour compute budget protocol: 20-trial Bayesian HPO per method + 5 evaluation seeds with 1,000-sample test bootstrap intervals.
- **§4.2 Discrimination & Calibration Results (1.25 p)**:
  - **Table 1 (`tab:1`)**: Time-dependent AUC, Concordance index (C-index), and Integrated Brier Score (IBS).
  - Demonstrates $\ge 0.025$ improvement in AUC and C-index over DeepTCSR and Dynamic-DeepHit on MIMIC-IV and Poisson-downsampled C-MAPSS (grounded against Bleistein2024 anchor 0.8791). [Carries $C_3$].
- **§4.3 Clinical Utility & Alarm Stability (1.00 p)**:
  - **Table 2 (`tab:2`)**: False alert episode rate per patient-day, alarm jitter count, and Decision Curve Analysis (DCA) at matched 0.30 PPV.
  - Demonstrates $\ge 25\%$ reduction in alert jitter and false alert episodes compared to unregularized Dynamic-DeepHit and EMA smoothing heuristics. [Carries $C_4$].

---

### §5. Analysis & Negative Controls (1.00 Page)
- **Table 3 (`tab:3`)**: Complete Factorial Ablation Matrix.
  - **NC-A1 (Discount Ablation)**: Setting $\gamma_j = S(\delta_s)$ destroys duration scaling $\to$ performance drops by 0.028 in AUC.
  - **NC-A2 (Shift Ablation)**: Setting $\Phi_{+\delta_s}$ destroys continuous renewal alignment $\to$ performance drops by 0.031 in AUC.
  - **NC-A3 (Clamped Division Comparison)**: Clamped division $p / \max(S, 10^{-3})$ produces gradient explosion and instability in high-risk strata.
  - **NC-B (Within-Patient Permutation)**: Scrambling $\Delta t_j$ within patient trajectories eliminates performance gains, proving sensitivity to continuous temporal alignment.
  - **NC-C (Horizon Matching Sweep)**: Duration-geometric $\lambda^{\Delta t / \delta_s}$ maintains stable performance across 2x subsampling, while count-geometric $\lambda^k$ degrades.

---

### §6. Discussion, Scope Limitations, & Conclusion (0.50 Page)
- Explicit scope declarations: Performance gain diminishes on regularly sampled cohorts ($\Delta t \equiv 1$); model relies on conditional unconfoundedness ($T \perp\!\!\perp T^{\text{obs}} \mid h$).
- Final takeaway: Unifying continuous renewal temporal difference learning with deep survival models unlocks robust, alarm-stable dynamic risk monitoring for high-stakes telemetry.

---

## 3. The 4-Element Reader-Path Check

Reviewers read in this exact sequence: **Title $\to$ Abstract $\to$ Figure 1 $\to$ Table 1**.

```
[Title]
"SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation"
  │
  ▼
[Abstract]
Identifies the division divergence problem ($p/S \to \infty$) -> states renewal shift + categorical projection -> reports $\ge 0.025$ AUC gain & $\ge 25\%$ alert jitter reduction -> notes scope boundary.
  │
  ▼
[Figure 1]
Renders C0 visually:
- Left: Mathematical mechanism (Divergence of $p/S$ vs. Contractive renewal mixture).
- Right: Downstream clinical impact (Threshold thrashing & alarm fatigue in DeepTCSR vs. Stable monotonic hazard in SurvTD).
  │
  ▼
[Table 1]
Comprehensive benchmark comparison across MIMIC-IV, C-MAPSS, PBC:
- Proves C3: Statistically significant $\ge 0.025$ gain over DeepTCSR, Dynamic-DeepHit, and Person-Period baselines under 1,000 bootstrap test samples.
```

**Audit Verdict**: The core claim ($C_0$), the mathematical move ($C_1$), and the empirical proof ($C_3, C_4$) are 100% delivered within the first 4 elements, before the reviewer reads Section 2.
