# Invalidated Results (Withdrawn 2026-09-03)

The benchmark tables (Table 1–3) and `falsification_report.json` in this directory are **withdrawn and invalidated, not superseded**.

## Root Causes of Invalidation

1. **Operator Defects (D-gamma, D8, D9, D10, D14)**:
   - **D-gamma**: The Step 5 renewal mixture was missing, rendering duration discount $\gamma$ algebraically identical to $\lambda$. The reported 93.9% gain retention in EXP-03 / NC-A1 was an artifact of this identity, not an empirical falsification of $C_1$.
   - **D8**: Durations were indexed by `dts[j]` instead of `dts[j+1]`, shifting temporal alignment across all transitions.
   - **D9**: Ground-truth target Dirac was placed at the previous interval length instead of the residual time.
   - **D14**: Cramér loss was discrete Euclidean summation instead of continuous trapezoidal Riemann integration.

2. **Degenerate Data Generators**:
   - The cohort labelled "MIMIC-IV Sepsis-3" was a synthetic generator where `tte = times[-1] + U(0.1, 2.0)` and all censored subjects were fixed at 72.0h. AUC case/control at median horizon was 12/1, forcing AUC to 0.500.

3. **Evaluation Harness Leaks & Baseline Defects**:
   - Risk prediction leaked outcome information from the final visit (`cdf[-1, K//2]`).
   - Person-Period baseline had a critical grid indexing bug causing below-chance C-index (0.263), corrupting all relative gain-retention denominators in Table 3.
   - Dynamic-DeepHit had supervision advantages that were uncalibrated against SurvTD's missing anchor.

See `HANDOVER.md` and `research/continuous-survival-td/deviation_log.md` for the full technical autopsy and audit trail.
