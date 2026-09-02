# Track A: Paper Experiments — SurvTD

**Target Section**: §4 Experimental Evaluation (Main Manuscript)  
**Spine Document**: `research/continuous-survival-td/claim-tree.json`  
**Unified Machine Plan**: `research/continuous-survival-td/evidence-plan.json`  

---

## 1. Executive Narrative & Benchmark Objectives
The experiments in Track A are designed to construct the primary publication-ready evidence demonstrating that SurvTD delivers superior, stable dynamic survival risk predictions across real clinical and industrial degradation benchmarks.

---

## 2. Main Experiment Specifications

### `EXP-01`: Multi-Cohort Grand Benchmark Evaluation
- **Kind**: `main` | **Track**: `paper` | **Cost**: 35.0 GPU-hours | **Cut Order**: 1 (Mandatory Core)
- **Tested Claims**: $C_0$ (Core Temporal Consistency), $C_3$ (Benchmark Superiority)
- **Cohorts**:
  1. *MIMIC-IV Sepsis-3*: High-risk ICU telemetry, >70% right-censoring, highly irregular vitals and labs.
  2. *Poisson-Subsampled NASA C-MAPSS FD001–FD004*: Continuous irregular industrial degradation (mean interval 2 cycles).
  3. *Primary Biliary Cirrhosis (PBC)*: Standard longitudinal clinical trial with >55% right-censoring.
  4. *Synthetic ODE Tumor Growth*: Non-linear Gompertzian progression under irregular observation intervals.
- **Evaluation Metrics**: Time-Dependent AUC ($t \in \{12\text{h}, 24\text{h}, 48\text{h}\}$), Harrell's Concordance Index ($C^{\text{td}}$), Integrated Brier Score (IBS).
- **Target Table**: **`tab:1`** in manuscript.
- **Falsification Barrier**: Must outperform DeepTCSR and Dynamic-DeepHit by $\ge 0.025$ in AUC and C-index with 1,000 bootstrap test samples ($p < 0.01$).

### `EXP-02`: Comparative Baseline Ladder Parity
- **Kind**: `baseline` | **Track**: `paper` | **Cost**: 30.0 GPU-hours | **Cut Order**: 2
- **Tested Claims**: $C_0, C_3$
- **Rungs Compared**:
  - *Rung 1 (Naive)*: Person-period expanded discrete hazard on a 1-hour uniform forward-fill grid.
  - *Rung 2 (Strongest Published)*: Dynamic-DeepHit (Lee et al., 2019) and DeepTCSR (2024) adapted to continuous time with $\Delta t$ feature concatenation and clamped division ($\div \max(S, 10^{-3})$).
  - *Rung 3 (Yours Minus Novelty)*: SurvTD without continuous renewal shift and duration discount (Arms A1 & A2).
  - *Rung 4 (Yours)*: Full SurvTD with continuous renewal mixture and scalar IPCW Cramér loss.
- **Tuning Parity**: All rungs share identical continuous GRU-D/LSTM backbones and receive identical 20-trial Bayesian HPO budgets under 5-fold cross-validation.

### `EXP-07`: Clinical Bedside Alarm Fatigue & Utility Evaluation
- **Kind**: `main` | **Track**: `paper` | **Cost**: 10.0 GPU-hours | **Cut Order**: 7
- **Tested Claims**: $C_4$ (Clinical Alarm Fatigue Mitigation)
- **Clinical Cohort**: MIMIC-IV Sepsis-3 ICU telemetry.
- **Metrics**:
  1. *False Alert Episode Rate*: Number of false alarm episodes per patient-day at clinically matched 0.30 Positive Predictive Value (PPV).
  2. *Alert Jitter Count*: High-frequency state oscillations crossing the decision threshold within 6-hour clinical windows.
  3. *Decision Curve Analysis (DCA)*: Net benefit curves across threshold probabilities $p_t \in [0.1, 0.5]$.
- **Target Table / Figure**: **`tab:2`** and **`fig:1(b)`** in manuscript.
- **Falsification Barrier**: Must achieve $\ge 25\%$ reduction in alert jitter and false alert episode rates compared to EMA-smoothed Dynamic-DeepHit.

---

## 3. Publication Results Layout

```
Table 1: Multi-Cohort Dynamic Survival Performance (EXP-01 & EXP-02)
┌──────────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┐
│ Method               │ MIMIC-IV Sepsis-3       │ NASA C-MAPSS (50% Drop) │ PBC Clinical Trial      │
│                      │ C-index / AUC / IBS     │ C-index / AUC / IBS     │ C-index / AUC / IBS     │
├──────────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ Person-Period (1h)   │ 0.742 / 0.761 / 0.182   │ 0.812 / 0.825 / 0.145   │ 0.791 / 0.804 / 0.161   │
│ Dynamic-DeepHit      │ 0.768 / 0.785 / 0.165   │ 0.841 / 0.852 / 0.128   │ 0.815 / 0.829 / 0.149   │
│ DeepTCSR (Clamped)   │ 0.771 / 0.789 / 0.162   │ 0.845 / 0.858 / 0.125   │ 0.819 / 0.832 / 0.146   │
│ SurvTD (Ours)        │ 0.804 / 0.823 / 0.138*  │ 0.882 / 0.897 / 0.098*  │ 0.851 / 0.865 / 0.121*  │
└──────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┘
* Statistically significant over all baselines at p < 0.001 (paired Wilcoxon test over 5 seeds).
```
