# Empirical Skeptic & Benchmark Auditor Report: SurvTD

**Reviewer Identity**: Empirical Skeptic & Benchmark Auditor (ICLR / NeurIPS Benchmark & Empirical Track Standard)  
**Target Idea**: `SurvTD: Continuous-Time Semi-Markov Temporal Difference Learning for Irregular Longitudinal Dynamic Survival Analysis`  
**Target Document**: `/Users/yangjaemo/Desktop/SurvTD/research/continuous-survival-td/phase2/candidate.json`  
**Audit Context**: `phase1/bottleneck.md`, `phase0/lit_table.md`, `phase3/quality_gauntlet.md`

---

## Executive Summary & Final Verdict

| Axis | Score (1–5) | Primary Empirical Concern |
| :--- | :---: | :--- |
| **A — Problem Position** | **3** | **Baseline Caricature**: Over-simplification of CoxSig's non-linear signature representation and TCSR's numerical stabilization. |
| **B — Method Quality** | **3** | **Censoring Blindspot**: Terminal recursion of backward $\lambda$-returns under right-censored trajectories is undefined; event point-mass assignment is dimensionally ambiguous. |
| **C — Problem-Fit** | **3** | **The C-MAPSS Trap**: Primary benchmark (NASA C-MAPSS) has 0% right-censoring, evading the fundamental statistical bottleneck of survival analysis. |
| **D — Falsifiability & Rigor** | **2** | **Confounded Negative Control & Calibration Amnesia**: Confounding $\lambda$ and $\Delta t$ in a single ablation; metric chasing on C-index while ignoring Brier score calibration under IPCW; zero tuning parity protocol. |

### Final Verdict: `REVISE` (Major Empirical Overhaul Required)

> **Auditor Summary**:  
> While the theoretical committee praised the paper's contraction algebra and Cramér projection proofs, this auditor refuses to be dazzled by blackboard mathematics that dissolves upon contact with real clinical data. Dynamic survival analysis is defined by **right-censoring**, **temporal leakage risks**, and **calibration accuracy**. The current candidate proposal commits three critical empirical sins: (1) it proposes evaluating a survival model primarily on an uncensored run-to-failure engineering dataset (NASA C-MAPSS), (2) its backward $\lambda$-return recursion completely omits how right-censored terminal trajectories are handled, and (3) its negative control is hopelessly confounded by bundling two distinct hypotheses ($\lambda=1.0$ and $\Delta t=1.0$) into a single test. The idea cannot advance to full execution until these empirical vulnerabilities are strictly remediated.

---

## 1. Four-Axis Deep Empirical Audit

### Axis A — Problem Position: [Score: 3/5]

#### Direct Quotations from `candidate.json`:
> *"CoxSig relies on linear signature regression under proportional hazards and terminal Monte Carlo NLL; SurvTD introduces recursive Semi-Markov Bellman temporal difference learning to capture non-linear degradation dynamics."* (`differentiation_from_lit[1]`)
>
> *"TCSR is restricted to discrete uniform steps (Delta=1) and divides by survival probability causing divergence; SurvTD operates over continuous irregular intervals (Delta t) using contractive multiplication and non-expansive Cramér projection."* (`differentiation_from_lit[0]`)
>
> *"Dynamic-DeepHit uses terminal Monte Carlo NLL with an ad-hoc pairwise ranking loss, lacking inter-step Bellman consistency across adjacent observations..."* (`differentiation_from_lit[2]`)

#### Empirical Auditor Critique:
1. **Strawmanning Baseline Non-Linearity (CoxSig)**:  
   The authors claim CoxSig (`Bleistein2024`) "relies on linear signature regression under proportional hazards" and therefore fails on non-linear systems. This is an unfair characterization. Rough path signatures map continuous, irregular trajectories into a graded tensor algebra that forms a universal basis for non-linear path functionals (via Chen's identity and Stone-Weierstrass). When combined with linear regression, signatures approximate *non-linear dynamical functionals* of the underlying stream. If SurvTD is benchmarked against CoxSig with a low signature truncation level ($m \le 2$), SurvTD's superiority would be an artifact of an under-parameterized baseline rather than a true architectural triumph.
2. **Artificial TCSR Division Explosion**:  
   TCSR (`Maystre2022`) divergence ($p / S$) occurs in high-risk tails when $S \to 0$. However, any competent practitioner implements numerical clamping ($S_{\epsilon} = \max(S, \epsilon)$) or log-space hazard updates. Claiming TCSR fails purely because of division without testing regularized TCSR is a classic empirical strawman.
3. **Dynamic-DeepHit Ranking Loss Utility**:  
   Dynamic-DeepHit's pairwise ranking loss directly optimizes the concordance index. By dismissing it as "ad-hoc," the authors ignore why DDH remains a dominant clinical benchmark: ranking losses are empirically robust against uncalibrated hazard scales.

---

### Axis B — Method Quality: [Score: 3/5]

#### Direct Quotations from `candidate.json`:
> *"Step 1: Encode irregular longitudinal observation sequence H_{t_j} into Markovian latent representation h_j via a recurrent sequential encoder."*
>
> *"Step 5: Formulate Survival Bellman target T_{\theta^-} p_j(s) = I[Event in \Delta t_j] + S_{\theta^-}(\Delta t_j) * (\Pi \Phi p_{\theta^-})_j(s)."*
>
> *"Step 6: Construct multi-step geometric lambda-return G^lambda_j(s) via backward trajectory recursion to unify 1-step Bellman bootstrapping (lambda=0) with terminal empirical returns (lambda=1)."*

#### Empirical Auditor Critique:
1. **The Censoring Void in Backward $\lambda$-Return**:  
   In standard reinforcement learning, episodes terminate at terminal states with unambiguous rewards. In survival analysis, patients do NOT conveniently experience events at the end of their recorded trajectories; **50% to 85% of clinical patients are right-censored** (discharged, transferred, or study ends without event).  
   When patient $i$ is censored at $t_M$ ($E_M = 0$), what is the terminal return $G^\lambda_M(s)$ in Step 6?
   - If $G^\lambda_M(s) = 0$, you artificially force censored patients to look like eternal survivors, causing severe underestimation of hazard.
   - If $G^\lambda_M(s) = (\Pi \Phi p_{\theta^-})_M(s)$, you bootstrap entirely from your own unverified model predictions, compounding bias across the trajectory without empirical ground truth.
   - If you do not apply Inverse Probability of Censoring Weighting (IPCW) or conditional survival completion at $t_M$, backward recursion produces biased targets that violate the true survival distribution. Step 6 completely ignores this statistical reality.
2. **Dimensional Ambiguity in Step 5**:  
   In Step 5, the target is written as:
   $$T_{\theta^-} p_j(s) = \mathbb{I}[\text{Event in } \Delta t_j] + S_{\theta^-}(\Delta t_j) \cdot (\Pi \Phi p_{\theta^-})_j(s)$$
   Here, $p_j(s)$ is a probability vector over $K$ discrete future time bins. The indicator $\mathbb{I}[\text{Event in } \Delta t_j]$ is a scalar! If an event occurs at continuous time $t_e \in [t_j, t_{j+1})$, the relative elapsed time $\tau = t_e - t_j$ must be mapped to a specific support bin $k \in \{1, \dots, K\}$ via a Dirac point mass or projection kernel. Adding a scalar indicator directly to a probability vector is mathematically ill-defined in implementation.
3. **Sequential Encoder Time-Delta Confounder**:  
   Step 1 specifies "a recurrent sequential encoder". If SurvTD uses a continuous-time recurrent architecture (e.g., GRU-ODE, Neural CDE, or explicit $\Delta t$ concatenation) while baselines are fed raw sequential observations without timestamps, the performance delta stems from the encoder's inductive bias, not the Bellman TD operator.

---

### Axis C — Problem-Fit: [Score: 3/5]

#### Direct Quotations from `candidate.json`:
> *"Reframes irregular dynamic survival prediction into policy evaluation over an absorbing Semi-Markov Decision Process, decoupling endogenous discounts via an EMA target network and achieving the optimal bias-variance Pareto frontier via multi-step geometric lambda-returns."* (`gap_closure[1]`)
>
> *"1x Apple M-series GPU or 1x NVIDIA RTX 3090, training time 20 minutes for 20 epochs across NASA C-MAPSS and simulated continuous benchmarks."* (`compute_budget`)

#### Empirical Auditor Critique:
1. **The NASA C-MAPSS Trap (0% Right-Censoring)**:  
   NASA C-MAPSS is an industrial turbofan degradation benchmark where **all engines are run to catastrophic failure**. There is **zero natural right-censoring**.
   Evaluating a *survival analysis* algorithm primarily on C-MAPSS turns survival analysis into simple Remaining Useful Life (RUL) regression disguised in survival notation. Survival models excel precisely because they handle incomplete observations (censoring) via hazard likelihoods. Benchmarking on C-MAPSS allows the authors to dodge the primary challenge of survival analysis. A valid survival benchmark MUST include real-world clinical cohorts (e.g., MIMIC-IV Sepsis/AKI, Primary Biliary Cirrhosis [PBC], or Rotterdam/GBSG breast cancer) where censoring rates exceed 50%.
2. **Mismatch with Clinical Observation Regimes**:  
   In clinical intensive care, sampling is irregular because clinicians order tests when patients destabilize (informative observation). While the authors rightfully stepped back from claiming a theoretical cure for informative sampling bias, their SMDP assumption assumes transitions depend only on $(h_j, \Delta t_j)$. If clinical sampling frequency changes the state representation itself, the semi-Markov property is conditioned on the encoder capturing the history of visit intensities.

---

### Axis D — Falsifiability & Experimental Rigor: [Score: 2/5]

#### Direct Quotations from `candidate.json`:
> *"When multi-step bootstrapping is ablated by setting lambda to 1.0 (collapsing to terminal Monte Carlo NLL), ranking concordance drops significantly below the intermediate bootstrap baseline (measured in Bleistein2024 as 0.8791 on NASA turbofan degradation) due to unmitigated trajectory variance."* (`falsification_prediction`)
>
> *"Ablate lambda_return_mixture_parameter by sweeping lambda to 1.0 (pure Monte Carlo NLL) and fixing Delta t to 1.0 (discrete uniform discretization), which disables continuous temporal difference bootstrapping and causes time-dependent AUC and C-index to degrade to baseline performance."* (`negative_control`)
>
> *"lambda_return_mixture_parameter"* (`load_bearing_variable`)

#### Empirical Auditor Critique:
1. **Confounded Negative Control (Fatal Experimental Flaw)**:  
   Look at the negative control: *"sweeping lambda to 1.0 (pure Monte Carlo NLL) AND fixing Delta t to 1.0 (discrete uniform discretization)"*.  
   This bundles **two separate independent variables** into a single ablation!  
   If performance degrades, an auditor cannot tell whether the degradation was caused by:
   - Disabling TD bootstrapping ($\lambda = 1.0$), OR
   - Distorting the time horizon by fixing $\Delta t = 1.0$.  
   This violates basic scientific control principles. An ablation must vary ONE load-bearing factor at a time.
2. **Metric Gaming: The C-Index Fallacy & Calibration Amnesia**:  
   The candidate exclusively measures success via "ranking concordance" (C-index) and "time-dependent AUC".  
   In clinical risk modeling, **C-index is notoriously easy to game and hides dangerous miscalibration**. A survival model can output wildly inaccurate absolute risk probabilities (e.g. predicting 99% mortality for a patient whose true risk is 10%) while maintaining a high C-index as long as the relative ranking is preserved.  
   SurvTD's explicit loss is the **Cramér distance on cumulative distributions** ($\mathcal{L}_{\text{TD}}$), which directly penalizes distribution mismatch. Therefore, the primary metric of superiority MUST be **probabilistic calibration**:
   - Time-Dependent Brier Score (with IPCW weighting),
   - Integrated Brier Score (IBS),
   - D-calibration / 1-calibration tests.  
   Excluding calibration metrics from the primary falsification criteria is unacceptable for a distributional survival paper.
3. **Absence of Tuning Parity Protocol**:  
   There is no mention of hyperparameter tuning budgets for baselines.
   - Dynamic-DeepHit requires tuning its ranking weight $\alpha \in [0, 1]$, hidden dimensions, and dropout.
   - CoxSig requires tuning signature depth $m \in \{2, 3, 4\}$, path augmentations (lead-lag, time-augmentation), and ridge penalty.
   - TCSR requires tuning bin discretization count $K$ and $\epsilon$-clipping.  
   If SurvTD is extensively tuned on validation splits while baselines are run with arbitrary default parameters, any claimed improvement is empirically meaningless.
4. **Data Leakage & Landmark Evaluation Vulnerability**:  
   Dynamic survival evaluation requires landmarking (evaluating predictions at fixed landmark times $t_{\text{land}}$ using only data $\mathcal{H}_{t_{\text{land}}}$ to predict events in $[t_{\text{land}}, t_{\text{land}} + \Delta\tau]$).  
   In Step 6, backward recursion uses future trajectory data $t_{j+1}, \dots, t_M$ to construct training targets. The authors must explicitly state that:
   - Backward recursion is **strictly confined to training**;
   - At inference/evaluation time, the model executes a pure feed-forward rollout without access to future observations or target networks;
   - All dataset splits are strictly **patient-level (subject-level)**, guaranteeing that zero trajectory slices of test patients leak into training or validation folds.

---

## 2. Quantitative Metric Aggregation

$$\text{Overall Score} = \text{round}\left(100 \times \frac{A + B + C - 3}{12}\right) = \text{round}\left(100 \times \frac{3 + 3 + 3 - 3}{12}\right) = \mathbf{50} \quad (\text{Band: } \mathbf{borderline})$$

### Gate Evaluation:
- $A = 3$, $B = 3$, $C = 3$.
- **$D = 2 \le 2$ $\implies$ GATE FIRED: CRITICAL FALSIFIABILITY & EXPERIMENTAL FLAW**.
- **Rule**: Any axis $\le 2$ mandates an immediate `REVISE` or `ABANDON`.

---

## 3. Mandatory Empirical Revisions (Actionable Protocol)

Before SurvTD can proceed to experiment execution, the authors must submit an updated candidate specification satisfying the following 5 requirements:

```
[MANDATORY REVISION CHECKLIST]
├── 1. Disentangle Negative Controls into 3 Orthogonal Ablations:
│    ├── NC-1 (Bootstrap ablation): Sweep lambda in {0.0, 0.3, 0.7, 1.0} with true Delta t.
│    ├── NC-2 (Temporal discretization ablation): Continuous Cramer projection vs Rounding Delta t to 1.0 with fixed optimal lambda.
│    └── NC-3 (Target network ablation): Frozen EMA target network (tau=0.01) vs Instant online update (tau=1.0).
├── 2. Incorporate a True Right-Censored Clinical Benchmark:
│    └── Add MIMIC-IV (Sepsis-3 or AKI continuous ICU trajectory) or PBC longitudinal dataset (censoring rate > 40%)
│        alongside NASA C-MAPSS.
├── 3. Formulate Right-Censored Terminal Backward Return:
│    └── Explicitly define G^lambda_M(s) when E_M = 0 (IPCW-weighted survival tail or bounded Kaplan-Meier completion).
├── 4. Mandate Calibration & Distributional Metrics:
│    └── Add Time-Dependent Brier Score (IBS) and D-calibration alongside Uno's time-dependent AUC / C-index.
└── 5. Establish Strict Tuning Parity & Leakage Prevention:
│    ├── 50-iteration Random/Bayesian search budget per baseline (CoxSig, TCSR, DDH) on validation folds.
│    └── Explicit Subject-Level (Patient-Level) 5-fold cross-validation with zero landmark leakage.
```

---

## 4. Final Verdict: `REVISE`

The theoretical framework of SurvTD has merit, but its empirical architecture currently resembles a sandbox toy designed to win on an uncensored dataset (C-MAPSS) with a confounded negative control.  
**SurvTD is halted at Phase 3 with a verdict of `REVISE`.** The authors must update `candidate.json` to resolve these 5 empirical mandates before Phase 4 experiment execution can be authorized.
