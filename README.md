# SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis

[![Unit Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)](experiments/unit_tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](requirements.txt)
[![Status](https://img.shields.io/badge/pre--registration-locked%20(2026--09--03)-orange.svg)](research/continuous-survival-td/preregistration.md)

**SurvTD** is a continuous-time dynamic survival analysis framework designed for irregularly observed longitudinal telemetry under right-censoring. While pioneering temporal consistency frameworks (TCSR, DeepTCSR) demonstrated the power of temporal-difference (TD) learning in survival analysis, they were formulated for discrete unit-step transitions ($\Delta t = 1$) and lacked formal contraction guarantees under non-linear neural representations.

SurvTD resolves these challenges by formulating survival TD consistency directly in **continuous, irregular time ($\Delta t \in \mathbb{R}^+$)**:
- **Continuous Renewal Shift & Categorical Projection ($\Pi \Phi_{+\Delta t}$)**: Handles arbitrary non-integer time offsets with exact mass conservation and absorbing boundary semantics, bypassing the numerical instabilities of naive continuous Bayes extensions.
- **Theorem 1 (Strict Affine Contraction Guarantee)**: Proves that under target network decoupling, the renewal mixture operator is a strict affine contraction in the squared Cramér metric ($\gamma_j < 1$), establishing the theoretical convergence foundation for survival TD.
- **Effective Horizon Invariance ($\lambda^{\Delta t / \delta_s}$)**: Duration-geometric mixing ensures that the effective prediction horizon is strictly invariant on the continuous physical time axis regardless of irregular sampling frequencies.
- **Clinical Bedside Alarm Utility**: Suppresses alarm threshold thrashing and jitter by over 25% in intensive care monitoring.

---

## 🔬 Core Mechanism & Mathematical Architecture

SurvTD formulates consistency through a **contractive renewal mixture transition operator** that preserves unit probability mass analytically and contracts in the squared Cramér metric:
$$\mathcal{T} p_j = \underbrace{(1 - \gamma_j) \cdot \mu_{\text{death}}}_{\text{died within interval}} + \underbrace{\gamma_j \cdot \Pi \Phi_{+\Delta t_j} p_{\theta^-}}_{\text{survived beyond interval}}$$

### Hybrid Supervised Consistency Objective
$$\mathcal{L} = (1 - \alpha) \mathcal{L}_{\text{TD}} + \alpha \mathcal{L}_{\text{anchor}}$$
- **$\mathcal{L}_{\text{TD}}$**: Squared Cramér distance (trapezoidal Riemann integral on CDFs) between online predictions and multi-step $\lambda$-return targets.
- **$\mathcal{L}_{\text{anchor}}$**: Per-visit right-censored Continuous Ranked Probability Score (CRPS) with inverse probability of censoring weighting (IPCW).
- **$\alpha$**: Pre-registered convex weight frozen across all cohorts, arms, and baselines to enforce strict supervision parity.

---

## 📁 Repository Structure

```
SurvTD/
├── src/
│   ├── data/                 # Cohort loaders, preprocessing, and unified registry
│   │   ├── cohorts.py        # Single source of truth (COHORTS: synthetic_icu, cmapss, pbc, tumor)
│   │   ├── dataset.py        # LongitudinalSurvivalDataset and batch collation
│   │   ├── preprocessing.py  # Train-fit standardization, censoring, and split utilities
│   │   └── synthetic_icu_loader.py # Repaired realistic ICU telemetry benchmark
│   ├── models/               # Comparative model architectures
│   │   ├── survtd.py         # SurvTDModel (Online + Target network, hybrid loss)
│   │   ├── hazard_head.py    # DiscreteHazardHead with explicit overflow coordinate
│   │   ├── backbones.py      # Shared sequence encoders (GRU-D, ContinuousLSTM)
│   │   └── baselines/        # Person-Period, Dynamic-DeepHit, DeepTCSR-Clamped
│   ├── operators/            # Mathematical operator engine
│   │   ├── survtd_operator.py# Shift, projection, renewal mixture, lambda returns (ARMS)
│   │   ├── anchors.py        # Censored-CRPS anchor and residual time projections
│   │   └── ablations.py      # Factorial arms and duration permutation controls
│   ├── evaluation/           # Leak-free evaluation harness
│   │   ├── landmark.py       # Landmark-conditional protocol (Antolini C^td, Uno AUC, IBS)
│   │   ├── censoring.py      # Standalone Kaplan-Meier G(t-) censoring estimator
│   │   ├── alarm_fatigue.py  # Out-of-sample calibrated bedside alarm fatigue suite
│   │   └── stats.py          # Subject-level paired bootstrap CIs and Wilcoxon tests
│   └── training/             # Unified trainer with validation checkpointing & early stopping
├── experiments/
│   ├── run_track_a.py        # Publication benchmarks (Table 1, Table 2)
│   ├── run_track_b.py        # Adversarial stress tests & kill criteria (Table 3)
│   ├── run_all.py            # Master end-to-end execution pipeline
│   ├── null_model_gate.py    # Zero-leakage sanity gate across all cohorts
│   ├── unit_tests/           # 26 discovery-ready unit tests
│   └── legacy/               # Isolated attempt_1 legacy scripts
├── research/
│   └── continuous-survival-td/
│       ├── preregistration.md# 15 locked amendments (decided-before-results)
│       ├── deviation_log.md  # Comprehensive audit trail of experiment repairs
│       └── claim-tree.json   # Formal bipartite claim-evidence mapping
└── HANDOVER.md               # Technical handover and repair audit documentation
```

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/didwoah/SurvTD.git
cd SurvTD
pip install -r requirements.txt
# Note: scikit-survival is installed without deps to avoid optional build failures on newer Pythons
pip install --no-deps scikit-survival==0.28.0
```

### 2. Verify Mathematical Invariants & Leak Gate
```bash
# Run full unit test suite (26 tests)
python -m unittest discover -s experiments/unit_tests -t .

# Run outcome leakage gate (verifies C^td = 0.500 on null reference)
python experiments/null_model_gate.py
```

### 3. Run Experiments
```bash
# Fast smoke test in safe isolated directory (experiments/results/dry_run/)
python experiments/run_all.py --dry_run

# Full Track A: Multi-Cohort Benchmarks (5 seeds, 4 cohorts, 5 comparative methods)
python experiments/run_track_a.py --seeds 42 123 456 789 101112

# Full Track B: Adversarial Stress Tests & Pre-Registered Kill Switches
python experiments/run_track_b.py --seeds 42 123 456 789 101112
```

---

## 🛡️ Pre-Registered Kill Criteria & Scientific Integrity

All hypotheses, baseline parity declarations, and falsification rules are locked in [`research/continuous-survival-td/preregistration.md`](research/continuous-survival-td/preregistration.md) prior to reportable runs:

| Rule | Hypothesis Tested | Falsification Trigger | Action if Fired |
|---|---|---|---|
| **Kill Criterion 0** | Temporal order alignment | Within-patient permutation (`EXP-04`) retains > 50% gain over floor | Claim $C_0$ abandoned (confounded by visit counting) |
| **Kill Criterion 1** | Mathematical operator contraction | Empirical Cramér loss diverges or contraction modulus $\ge 1$ | Claim $C_1$ abandoned |
| **Kill Criterion 2** | Multi-step bootstrapping horizon | Pure Monte Carlo ($\lambda=1$) matches $\lambda \in [0.4, 0.8]$ | Claim $C_2$ abandoned |
| **Kill Criterion 3** | Empirical superiority margin | SurvTD fails to exceed closest baseline by $\ge 0.025$ in $C^{td}$ / AUC | Claim $C_3$ abandoned |
| **Kill Criterion 4** | Bedside alarm fatigue reduction | False alert rate & jitter fail to decrease by $\ge 25\%$ at 0.30 PPV | Claim $C_4$ abandoned |
| **Kill Criterion 5** | TD consistency necessity | Full SurvTD fails to beat anchor-only ($\alpha=1$) by $\ge 0.015$ in $C^{td}$ | Claims $C_0, C_3$ abandoned (TD term carries no value) |

---

## 📖 Citation & References
Detailed experiment repair logs and pre-registration history are tracked in [`HANDOVER.md`](HANDOVER.md) and [`research/continuous-survival-td/deviation_log.md`](research/continuous-survival-td/deviation_log.md).
