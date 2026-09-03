# Pre-Registration Document: SurvTD Experimental Protocol
*(Amended 2026-09-03, Tagged `decided-before-results`)*

**Title**: SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation  
**Date**: September 3, 2026  
**Status**: Pre-registered before any reportable model execution (`declared_before_results: true`)  
**Spine Document**: `research/continuous-survival-td/claim-tree.json`  
**Evidence Plan**: `research/continuous-survival-td/evidence-plan.json`  
**Audit & Deviation Log**: `research/continuous-survival-td/deviation_log.md`

---

## 1. Context & Research Claims

This pre-registration locks the experimental protocol for evaluating SurvTD against deep dynamic survival baselines on irregular longitudinal telemetry. Following the experiment repair effort on 2026-09-03 (documented in `HANDOVER.md` and `deviation_log.md`), all previous results were withdrawn due to operator defects, evaluation leakage, and broken baselines. This document incorporates Amendments A-01 through A-15, locked prior to re-running benchmarks.

### Core Claim ($C_0$)
> *SurvTD enforces temporal-difference survival consistency under irregular observations by adding a duration-aware Bellman consistency regularizer to per-visit survival likelihood, replacing divergent division-based updates with contractive renewal shifts and duration-discounted categorical projections.* (Anchor: `fig:1`)

### Sub-Claims ($C_1 \sim C_4$)
- **$C_1$ (Operator Contraction & Mass Conservation)**: The composed duration transition operator $\Pi \Phi_{+\Delta t_j}$ with frozen discount $\gamma_j$ is an affine strict contraction in squared Cramér metric with modulus $\gamma_j < 1$, and conserves unit probability mass exactly without renormalisation. (Anchor: `thm:1`)
- **$C_2$ (Horizon Invariance & Variance Diffusion)**: Duration-geometric $\lambda$-return recursion maintains effective bootstrapping horizon invariance across irregular sampling rates, while per-step categorical projection diffusion is bounded in expectation by $\mathbb{E}[f(1-f)]\delta_s^2 = \delta_s^2 / 6$ (worst-case $0.25\delta_s^2$). (Anchor: `fig:2`)
- **$C_3$ (Empirical Benchmark Superiority)**: SurvTD outperforms compute-matched baselines (Person-Period, Dynamic-DeepHit, DeepTCSR Clamped) and its own anchor-only likelihood ($\alpha=1$) by at least 0.025 in landmarked time-dependent AUC / concordance on Synthetic ICU, Poisson-downsampled NASA C-MAPSS, PBC, and Tumor Growth ODE cohorts. (Anchor: `tab:1`)
- **$C_4$ (Clinical Alarm Fatigue Mitigation)**: SurvTD suppresses false alert episode rates and alert jitter by at least 25% at matched 0.30 PPV (calibrated strictly on validation splits) compared to unregularized and heuristic baselines. (Anchor: `tab:2`)

---

## 2. Pre-Registered Amendments Ledger (2026-09-03, Decided-Before-Results)

Every amendment below was locked before executing new benchmark runs:

1. **A-01 (Dataset Renaming)**: The cohort previously labelled "MIMIC-IV Sepsis-3" was a synthetic telemetry generator; it is honestly renamed to `synthetic_icu`.
2. **A-02 (Renewal Mixture Implementation)**: Step 5 renewal mixture implemented. Near branch carries $(1-\lambda)(1-\gamma)$, far branch carries $(1-\lambda)\gamma + \lambda$. Unit mass is conserved analytically.
3. **A-03 (Diffusion Bound Restated)**: Projection variance diffusion bound $\delta_s^2/6$ holds in expectation over the fractional offset $f \sim \mathcal{U}(0,1)$; worst-case bound is $0.25\delta_s^2$. EXP-08 demoted to unit test; replaced by target sharpness (E8b) and empirical contraction modulus (E8c).
4. **A-04 (Anchor Term & Reframe)**: Loss is $\mathcal{L} = (1-\alpha)\mathcal{L}_{\text{TD}} + \alpha\mathcal{L}_{\text{anchor}}$ with a per-visit censored-CRPS anchor and IPCW. An $\alpha=1$ (anchor-only) arm is added, and **Kill Criterion 5** is established. $\alpha$ is tuned once on seed 42 validation split and frozen.
5. **A-05 (HPO Budget Realism)**: 12-trial Bayesian/random search under subject-level 3-fold CV (6 epochs/trial) run once per (cohort, method) at seed 42; parameter bounds unchanged.
6. **A-06 (Standard Error Claim Withdrawal)**: Fabricated 0.008 SE claim withdrawn. Fixed delta $\ge 0.025$ retained, evaluated via 1,000-sample paired bootstrap CI.
7. **A-07 (Landmarked Primary Endpoint)**: Evaluated at pre-declared $(L, \Delta)$ grids using Antolini $C^{td}$ and Uno cumulative-dynamic AUC with train-fit IPCW.
8. **A-08 (Adjudicating Statistic)**: Replaced 5-seed Wilcoxon significance (minimum achievable exact $p=0.0625$) with 1,000-sample subject-level paired bootstrap 95% CI of the metric difference.
9. **A-09 (EXP-05 Split)**: Split into E5a (discrimination), E5b (gradient stability), and E5c (numerical clamp activation rate).
10. **A-10 (EXP-06 Subsampling Sweep)**: Sweeps $\{\text{duration}, \text{count}\} \times \{1.0, 0.5, 0.25\}$ across 5 seeds; requires mean $|\text{slope}_{\text{duration}}| \le 0.5 \times \text{mean} |\text{slope}_{\text{count}}|$. Analytic check E6a run first.
11. **A-11 (Paired Absolute Delta)**: Primary ablation metric is paired absolute delta $C_{\text{full}} - C_{\text{arm}}$ against 0.025.
12. **A-12 (EXP-04 Positive Control)**: Cross-patient duration shuffling added as noise floor.
13. **A-13 (Validation Calibration)**: 0.30 PPV operating threshold calibrated on validation split only.
14. **A-14 (Alert Jitter Relabelling)**: Jitter metric defined as unstable 6h windows per subject-day.
15. **A-15 (Gamma Placement)**: Default `'bootstrap'` and alternative `'compounded'` both reported.

---

## 3. Four-Rung Baseline Ladder & Parity

### Baseline Ladder
1. **KM-Marginal Reference**: Non-parametric Kaplan-Meier landmark baseline (verifies absence of test leakage, $C^{td} \equiv 0.500$).
2. **Naive Discrete Baseline**: Person-Period model expanded onto regular grid with validation-selected step.
3. **Strongest Published Baselines**: Dynamic-DeepHit (per-visit likelihood + ranking loss) and DeepTCSR-Clamped (continuous-time division with clamped survival).
4. **Anchor-Only Control ($\alpha=1$)**: SurvTD backbone supervised solely by per-visit censored-CRPS anchor without TD regularization.
5. **Full Proposal**: SurvTD with continuous renewal shift, categorical projection, duration discount, and multi-step TD consistency.

---

## 4. Evaluation Protocol & Pre-Declared Landmarks

All evaluations use `src.evaluation.landmark.evaluate_landmarked` with administrative censoring at $L + \Delta$:

| Cohort | Time Unit | $\delta_s$ | $K$ | Landmarks $L$ | Horizon $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Synthetic ICU** | Hours | 2.0 | 36 | $\{12, 24, 36\}$ | 24 |
| **NASA C-MAPSS (50%)** | Cycles | 5.0 | 30 | $\{125, 150, 175\}$ | 50 |
| **PBC Trial** | Days | 30.0 | 30 | $\{0, 180\}$ | 365 |
| **Tumor Growth ODE** | Months | 0.5 | 25 | $\{1.5, 2.0\}$ | 2.0 |

---

## 5. Statistical Rigor Protocol

- **Fixed Random Seeds**: Exactly 5 seeds: `[42, 123, 456, 789, 101112]`.
- **Reported Uncertainty**: Mean $\pm$ SD across seeds; 1,000-sample subject-level test bootstrap 95% CI; paired Wilcoxon $p$-values.
- **Pre-Declared Significance**: Primary adjudication requires the 95% bootstrap CI lower bound of the paired difference $(C_{\text{SurvTD}} - C_{\text{baseline}})$ to exceed $0.0$, and mean delta $\ge 0.025$.

---

## 6. Pre-Registered Kill Criteria

The proposal will be declared **falsified and withdrawn** if any of the following occur:

1. **Kill Criterion 0 (Temporal Alignment Confound)**: If within-patient duration permutation (`EXP-04`) retains more than 50% of the gain over the cross-patient permuted floor, $C_0$ is falsified.
2. **Kill Criterion 1 (Mathematical Operator Divergence)**: If empirical Cramér distance loss fails to decrease monotonically or violates contraction modulus $\gamma_j < 1$ under frozen target networks, $C_1$ is falsified.
3. **Kill Criterion 2 (Bootstrapping Horizon Collapse)**: If terminal Monte Carlo ($\lambda=1.0$) matches intermediate bootstrapping ($\lambda=0.6$) within seed noise, or analytic effective horizon fails to remain invariant in E6a, $C_2$ is falsified.
4. **Kill Criterion 3 (Empirical Inferiority)**: If SurvTD fails to achieve at least 0.025 improvement in time-dependent AUC / concordance over compute-matched clamped DeepTCSR and Dynamic-DeepHit, $C_3$ is falsified.
5. **Kill Criterion 4 (Alarm Stability Failure)**: If SurvTD fails to reduce false alert episode rates and alert jitter by at least 25% at validation-matched 0.30 PPV, $C_4$ is falsified.
6. **Kill Criterion 5 (Anchor-Equivalence Kill)**: If full SurvTD fails to outperform the anchor-only arm ($\alpha=1$, identical backbone without TD consistency) by at least 0.015 in C-index, the temporal-difference consistency mechanism carries no empirical value, and claims $C_0$ and $C_3$ are falsified.

---

## 7. Post-Result Amendments — NOT Pre-Registered

Everything in §1–§6 was locked before any reportable execution. This section is
different in kind and is quarantined from the ledger in §2 deliberately.

### A-16 (`decided-after-results`, 2026-09-04)

**Trigger.** Kill Criterion 3 fired on Cohort 1 (Synthetic ICU, 5 seeds,
$\alpha = 0.0$, default initialization): SurvTD $C^{td} = 0.5536 \pm 0.0573$
against Dynamic-DeepHit $0.6461 \pm 0.0605$ ($\Delta = -0.0925$) and
Person-Period $0.6367 \pm 0.0759$ ($\Delta = -0.0831$). **$C_3$ is falsified on
this cohort and that verdict stands as the primary outcome.**

**Correction to the §6 KC3 adjudication.** The margin over DeepTCSR-Clamped
($\Delta = +0.0421$) was initially read as a PASS. It is not: §5 requires the
paired 95% CI lower bound to exceed 0, and the seed-level CI is
$[-0.013, +0.097]$. The correct verdict is **undetermined, not PASS**. Separately,
DeepTCSR was run at SurvTD's selected $\alpha = 0.0$ and collapsed
(IBS $0.506 \pm 0.156$, versus $0.115 \pm 0.021$ for Dynamic-DeepHit), so that arm
is not a valid baseline in this run regardless of the statistic. KC3's text also
omits Person-Period, which $C_3$ explicitly names; $C_3$ is falsified against
**2 of 3** named baselines.

**Deviations in the primary run, recorded rather than corrected retroactively.**
(i) A-05's 12-trial HPO was **not executed** — every method ran at fixed
`lr=1e-3, hidden=64, batch=16`; uniform across arms, but declared and not
performed. (ii) The arms were **not compute-matched**: SurvTD received 335–952 s
per seed against DeepTCSR's 119–189 s and Dynamic-DeepHit's 188–286 s, i.e. 3–6×.
The falsification is therefore conservative — SurvTD lost with more compute.

**Amendment.** A measured initialization defect (default init gives per-bin hazard
0.4999 against a cohort truth of 0.010–0.039, collapsing the duration discount
$\gamma_j = S(\Delta t_j)$ from ~0.98 to 0.481 at the median inter-visit gap and
to 0.091 at the p90 gap) licenses a **secondary, exploratory** re-run under a
Kaplan-Meier prior bias initialization, applied **identically to all four neural
arms**. Full statement, evidence table and the pre-declared proceed/stop gate are
in `deviation_log.md` §4 A-16. $\alpha = 0.0$ is void as a frozen value and must
be re-selected over 3 seeds after the fix.

**Reporting rule.** Results under A-16 appear in a separate table block labelled
*secondary / exploratory* and never replace §5's primary outcome. The primary
run's raw record is preserved at
`experiments/results/preregistered_primary_2026-09-04/`.
