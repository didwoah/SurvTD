# Experimental Pre-Registration Document: SurvTD

**Date**: 2026-09-02  
**Status**: Pre-Registered & Locked (Pre-Results Verification)  
**Target Submission**: NeurIPS / ICML / ICLR  
**Associated Claim Tree**: `research/continuous-survival-td/claim-tree.json`  
**Structured Evidence Plan**: `research/continuous-survival-td/evidence-plan.json`

---

## 1. Research Hypotheses & Bipartite Mapping

This pre-registration locks the experimental protocol designed to critically test and potentially falsify the core claims of **SurvTD** (*Continuous-Time Semi-Markov Temporal Difference Learning for Irregular Longitudinal Dynamic Survival Analysis*).

### Bipartite Claim-to-Experiment Matrix

| Claim ID | Claim Summary | Evidence Anchor | Assigned Experiments | Falsification Trigger |
| :---: | :--- | :---: | :---: | :--- |
| **`C0`** | Continuous Semi-Markov Bellman contraction recovers dynamic consistency under irregular sampling where discrete uniform models fail. | `fig:1` | `EXP-01`, `EXP-02`, `EXP-04`, `EXP-NC1`, `EXP-NC2` | Negative control retains $>50\%$ gain, or continuous projection fails to beat uniform discretization. |
| **`C1`** | Rightward temporal shift and categorical projection $\Pi$ form a probability-conserving non-expansive operator under Cramér metric. | `thm:1` | `EXP-02`, `EXP-03`, `EXP-NC2` | Cramér loss diverges or fails to decrease monotonically under continuous interval variations. |
| **`C2`** | Geometric $\lambda$-return recursion achieves an inverted-U curve, outperforming both 1-step TD ($\lambda=0$) and terminal MC NLL ($\lambda=1$). | `fig:2` | `EXP-04` | $\lambda=1.0$ matches $\lambda=0.6$ within seed noise, or curve fails to show inverted-U structure. |
| **`C3`** | On continuous ODE tumor growth and industrial degradation, SurvTD significantly improves C-index and AUC over linear signatures and discrete TD. | `tab:1` | `EXP-01`, `EXP-05` | SurvTD fails to beat compute-matched CoxSig and TCSR by at least $0.03$ C-index. |
| **`C4`** | On real right-censored clinical cohorts, SurvTD reduces Integrated Brier Score and suppresses high-frequency alarm jittering. | `tab:2` | `EXP-06` | SurvTD shows worse IBS or higher alarm jitter frequency than Dynamic-DeepHit on MIMIC-IV Sepsis-3. |

---

## 2. Benchmark Ladder & Data Regimes

To avoid the "C-MAPSS 0% Censoring Trap" surfaced by the 5-member critic committee, experiments span four diverse rungs:

1. **Rung 1 (Real Right-Censored Clinical Telemetry)**:
   - **MIMIC-IV Sepsis-3**: Longitudinal ICU vitals and labs with $>70\%$ right-censoring rate, highly irregular sampling, and realistic clinical shocks.
   - **Primary Biliary Cirrhosis (PBC)**: Longitudinal clinical trial benchmark with $>55\%$ right-censoring rate.
2. **Rung 2 (Continuous Non-Linear Biophysical Dynamics)**:
   - **Tumor Growth ODE**: Non-linear Gompertzian progression under irregular observation schedules.
3. **Rung 3 (Industrial Cyber-Physical Systems)**:
   - **NASA C-MAPSS (FD001–FD004)**: Complex multi-sensor mechanical degradation under variable operating regimes.
4. **Rung 4 (Stationary Linear Baseline Control)**:
   - **Ornstein-Uhlenbeck (OU) SDE**: Controlled linear drift process representing the scope boundary where linear signature regression is competitive.

---

## 3. Declared Tuning Parity & Baseline Controls

To prevent the classic "undertuned baseline" confound:
- **Equal Search Budget**: All baselines (**CoxSig**, **TCSR** with $\epsilon$-regularization, **Dynamic-DeepHit**, and **SurvTD**) receive an identical budget of **50 trials of Bayesian Hyperparameter Optimization** evaluated on validation folds.
- **Search Space Parity**:
  - Learning rate: $\log\text{Uniform}(10^{-4}, 10^{-2})$
  - Latent/hidden representation size: $\{32, 64, 128\}$
  - Weight decay: $\log\text{Uniform}(10^{-6}, 10^{-2})$
  - Dropout: $\text{Uniform}(0.0, 0.4)$
- **Cross-Validation**: Strict subject-level (patient-level / engine-level) 5-fold cross-validation. Zero cross-temporal contamination between train and test splits.

---

## 4. Orthogonal Negative Controls (Kill Switches)

1. **`EXP-NC1` (Interval Scrambling Control)**:
   - *Intervention*: Randomize the observation intervals $\Delta t_j$ across independent patient trajectories while preserving input feature values.
   - *Expected Downstream Outcome*: Destroys the temporal alignment of the Semi-Markov Bellman operator.
   - *Kill Rule*: If this control retains $>50\%$ of the discrimination gain over baseline, the gain is attributed to data noise/regularization rather than the temporal Bellman contraction, and claim `C0` is withdrawn.
2. **`EXP-NC2` (Target Network Ablation Control)**:
   - *Intervention*: Set target network update rate $\tau = 1.0$ (instant online updates), removing the parameter-frozen decoupling.
   - *Expected Downstream Outcome*: Triggers Deadly Triad instability and gradient oscillation.
   - *Kill Rule*: If instant updates achieve equal stability and performance without divergence, the quasi-linear contraction hypothesis is falsified.

---

## 5. Statistical Rigor Plan

- **Seeds**: 5 independent runs per condition (Seeds: 42, 101, 2024, 777, 999).
- **Reported Uncertainty**: All tables report $\text{mean} \pm \text{std}$ across the 5 seeds, accompanied by paired Wilcoxon signed-rank test $p$-values and 95% bootstrap confidence intervals.
- **Pre-Declared Meaningful Delta**:
  - Minimum Concordance Delta: $\Delta\text{C-index} \ge +0.030$
  - Minimum Dynamic AUC Delta: $\Delta\text{AUC} \ge +0.030$
  - Minimum Calibration Delta: Relative reduction in Integrated Brier Score $\ge 20.0\%$
  - Minimum Alarm Stability Delta: Reduction in Alarm Jitter Count $\ge 30.0\%$ at fixed lead time.
- **Pre-Registration Lock**: `declared_before_results: true`. Any metrics or baselines introduced post-hoc must be labeled exploratory in the final paper.

---

## 6. Pre-Registered Kill Criteria

1. **Core Mechanism Kill (`C0`)**:
   If `EXP-NC1` retains more than 50% of the concordance gain, or continuous projection in `EXP-02` fails to beat discrete uniform rounding ($\Delta t = 1.0$) by at least $0.03$ C-index, **claim `C0` is abandoned**.
2. **Multi-Step $\lambda$ Return Kill (`C2`)**:
   If pure Monte Carlo NLL ($\lambda=1.0$) matches intermediate bootstrapping ($\lambda=0.6$) within seed variance ($\pm 1\sigma$), **the bias-variance variance-reduction claim `C2` is abandoned**.
3. **Clinical Calibration Kill (`C4`)**:
   If the Integrated Brier Score on MIMIC-IV Sepsis-3 fails to improve over the unregularized Dynamic-DeepHit baseline by at least $0.02$, **the clinical calibration claim `C4` is abandoned**.

---

## 7. Compute Budget & Cut Order

- **Total Planned Budget**: 17.5 GPU-Hours (Apple M-series or NVIDIA RTX 3090).
- **Cut Order in Case of Halved Budget**:
  - Priority 1 (Never cut): `EXP-01` (Main battle), `EXP-02` (Mechanism-isolating ablation), `EXP-04` ($\lambda$-sweep).
  - Priority 2: `EXP-05` (Baseline tuning ladder), `EXP-06` (Clinical MIMIC-IV).
  - Priority 3 (Drop first if compute constrained): `EXP-NC2` (Target network control), `EXP-NC1` (Interval scrambling), `EXP-03` (Projection sensitivity).
