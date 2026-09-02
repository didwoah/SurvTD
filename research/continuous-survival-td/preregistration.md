# Pre-Registration Document: SurvTD Experimental Protocol

**Title**: SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation  
**Date**: September 3, 2026  
**Status**: Pre-registered before any model execution or training runs (`declared_before_results: true`)  
**Spine Document**: `research/continuous-survival-td/claim-tree.json`  
**Evidence Plan**: `research/continuous-survival-td/evidence-plan.json`  

---

## 1. Context & Research Claims

This pre-registration locks the experimental plan for evaluating SurvTD against deep dynamic survival baselines on irregular longitudinal telemetry. All hypotheses, metrics, baselines, negative controls, and kill criteria are declared prior to empirical execution to prevent p-hacking, tuning drift, or post-hoc reframing.

### Core Claim ($C_0$)
> *SurvTD enforces temporal-difference survival consistency under irregular observations by replacing divergent division-based updates with contractive renewal shifts and duration-discounted categorical projections.* (Anchor: `fig:1`)

### Sub-Claims ($C_1 \sim C_4$)
- **$C_1$ (Operator Contraction & Mass Conservation)**: The composed duration transition operator $\Pi \Phi_{+\Delta t_j}$ with frozen discount $\gamma_j$ is an affine strict contraction in squared Cramér metric and conserves unit probability mass exactly. (Anchor: `thm:1`)
- **$C_2$ (Horizon Invariance & Variance Diffusion)**: Duration-geometric $\lambda$-return recursion compounds interval survival discounting and maintains effective bootstrapping horizon invariance while bounding per-step projection diffusion by $\delta_s^2 / 6$. (Anchor: `fig:2`)
- **$C_3$ (Empirical Benchmark Superiority)**: SurvTD outperforms unit-step consistency and terminal likelihood baselines by at least 0.025 in time-dependent AUC and concordance on MIMIC-IV Sepsis-3, Poisson-subsampled NASA C-MAPSS, and PBC. (Anchor: `tab:1`)
- **$C_4$ (Clinical Alarm Fatigue Mitigation)**: SurvTD suppresses false alert episode rates and alert jitter by at least 25% at matched 0.30 PPV compared to unregularized and heuristic baselines. (Anchor: `tab:2`)

---

## 2. Bipartite Experiment-to-Claim Mapping

| Experiment ID | Title | Kind | Bound Claims | GPU-Hours | Drop Priority (`cut_order`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`EXP-01`** | Multi-Cohort Grand Benchmark Evaluation | `main` | $C_0, C_3$ | 35.0 h | 1 (Never cut) |
| **`EXP-02`** | Comparative Baseline Ladder Parity | `baseline` | $C_0, C_3$ | 30.0 h | 2 |
| **`EXP-03`** | Mechanism-Isolating Factorial Ablation (NC-A) | `ablation` | $C_0, C_1$ | 15.0 h | 3 |
| **`EXP-04`** | Within-Patient Duration Permutation (NC-B) | `negative_control` | $C_0$ | 8.0 h | 4 |
| **`EXP-05`** | Clamped Division Comparison (NC-A3) | `negative_control` | $C_0, C_1$ | 7.0 h | 6 |
| **`EXP-06`** | Lambda Compounding & Horizon Sweep (NC-C) | `ablation` | $C_0, C_2$ | 10.0 h | 5 |
| **`EXP-07`** | Clinical Alarm Stability & Fatigue Evaluation | `main` | $C_4$ | 10.0 h | 7 |
| **`EXP-08`** | Projection Diffusion & Contraction Sensitivity | `sensitivity` | $C_1, C_2$ | 5.0 h | 8 |
| **Total** | | | | **120.0 h** | |

---

## 3. Four-Rung Baseline Ladder & Tuning Parity Statement

### The Four Rungs
1. **Naive Baseline**: Person-period expanded discrete hazard model evaluated on a 1-hour regular grid (forward-fill interpolation).
2. **Strongest Published, Compute-Matched**: Dynamic-DeepHit (Lee et al., 2019) and DeepTCSR (2024) adapted to continuous time with $\Delta t$ feature concatenation and clamped division ($\div \max(S, 10^{-3})$).
3. **Yours Minus the Novelty (Mechanism Ablation)**: SurvTD with duration discount fixed to unit constant $\gamma_j \equiv S(\delta_s)$ (Arm A1) and renewal shift fixed to unit shift $\Phi_{+\delta_s}$ (Arm A2).
4. **Yours (Full Proposal)**: SurvTD with continuous renewal shift $\Phi_{+\Delta t}$, categorical projection $\Pi$, duration discount $\gamma_j = S_{\theta^-}(\Delta t_j)$, and scalar IPCW Cramér loss weighting.

### Tuning Parity Declaration (Pre-Registered)
> *All comparative methods (SurvTD, DeepTCSR with clamped division, Dynamic-DeepHit, and Person-period discrete hazard) share identical continuous-time GRU-D or LSTM sequence backbones. Each method receives an identical 20-trial Bayesian hyperparameter optimization budget (Tree-structured Parzen Estimator) over identical search bounds: learning rate $\in [10^{-4}, 5 \times 10^{-3}]$, latent dimension $\in \{64, 128, 256\}$, dropout $\in [0.1, 0.5]$, and batch size $\in \{32, 64, 128\}$. Tuning is conducted under strict subject-level 5-fold cross-validation, guaranteeing identical wall-clock and compute opportunity across all contenders.*

---

## 4. Mechanism-Isolating Ablations & Negative Controls

### 1. Factorial Operator Ablation (NC-A, `EXP-03` & `EXP-05`)
- **Arm A1 (Discount Ablation)**: Replace $\gamma_j = S_{\theta^-}(\Delta t_j)$ with unit-step constant $S(\delta_s)$, holding continuous renewal shift intact.
- **Arm A2 (Shift Ablation)**: Replace continuous renewal shift $\Phi_{+\Delta t_j}$ with unit grid shift $\Phi_{+\delta_s}$, holding duration discount intact.
- **Arm A3 (Clamped Division Comparison)**: Replace renewal mixture operator with clamped unit-step division $p_j / \max(S(\Delta t_j), 10^{-3})$.
- *Hypothesis*: Arms A1 and A2 will lose at least 0.025 in AUC/C-index, and Arm A3 will exhibit severe gradient variance spikes in high-risk strata.

### 2. Within-Patient Duration Permutation (NC-B, `EXP-04`)
- Randomly permute the sequence of elapsed intervals $\{\Delta t_1, \dots, \Delta t_M\}$ within each patient trajectory, strictly preserving total follow-up time $\sum \Delta t_j$, sequence length $M$, and event label $E$.
- *Hypothesis*: Scrambling temporal order will destroy continuous temporal-difference alignment, reducing performance back to unregularized levels. If permuted trajectories retain $>50\%$ of the gain, the gain is an artifact of visit count regularization rather than true temporal alignment.

### 3. Effective Horizon Matching (NC-C, `EXP-06`)
- Compare duration-geometric mixing $\lambda_j = \lambda^{\Delta t_j / \delta_s}$ against count-geometric mixing $\lambda_j = \lambda^k$ across a grid subsampling sweep (100%, 50%, 25% observation retention).
- *Hypothesis*: Duration-geometric returns will maintain invariant performance curves across sampling rates, whereas count-geometric decay will shift the effective bootstrapping horizon with observation density.

---

## 5. Statistical Protocol & Falsification Thresholds

- **Random Seeds**: Exactly **5 fixed random seeds** (`seeds: [42, 123, 456, 789, 101112]`).
- **Variance Reporting**: Mean $\pm$ standard deviation across the 5 seeds, paired two-tailed Wilcoxon signed-rank test against the closest baseline, and 1,000-sample subject-level test bootstrap 95% confidence intervals.
- **Pre-Declared Meaningful Delta**:
  - **$\ge 0.025$ in time-dependent AUC and Harrell's Concordance Index** (derived mathematically: $3 \times \text{SE}$, where test bootstrap standard error was measured as 0.008 on MIMIC-IV Sepsis-3).
  - **$\ge 25\%$ relative reduction in false alert episode rate per patient-day and alert jitter** at a matched 0.30 Positive Predictive Value (PPV) operating point.

---

## 6. Pre-Registered Kill Criteria

The core claim $C_0$ and corresponding sub-claims will be considered **falsified and withdrawn** if any of the following pre-registered failure conditions occur:

1. **Kill Criterion 0 (Temporal Alignment Confound)**: If within-patient duration permutation (`EXP-04`) retains more than 50% of the concordance gain over unregularized baselines, $C_0$ is falsified (the gain is confounded by visit counting).
2. **Kill Criterion 1 (Mathematical Operator Divergence)**: If empirical Cramér distance loss fails to decrease monotonically or violates the contraction modulus $\gamma_j < 1$ under frozen target networks, $C_1$ is falsified.
3. **Kill Criterion 2 (Bootstrapping Horizon Collapse)**: If terminal Monte Carlo likelihood ($\lambda = 1.0$) matches intermediate multi-step bootstrapping ($\lambda \in [0.4, 0.8]$) within seed noise, or projection diffusion exceeds $\delta_s^2 / 6$, $C_2$ is falsified.
4. **Kill Criterion 3 (Empirical Inferiority)**: If SurvTD fails to achieve at least 0.025 improvement in time-dependent AUC/concordance over compute-matched clamped DeepTCSR and Dynamic-DeepHit on MIMIC-IV Sepsis-3 or Poisson-downsampled C-MAPSS, $C_3$ is falsified.
5. **Kill Criterion 4 (Alarm Stability Failure)**: If SurvTD fails to reduce false alert episode rates and alert jitter by at least 25% at matched 0.30 PPV compared to EMA-smoothed Dynamic-DeepHit, $C_4$ is falsified.
