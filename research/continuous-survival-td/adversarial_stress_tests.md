# Track B: Adversarial Stress Tests & Hypothesis Destroyer — SurvTD

**Target Section**: §5 Analysis, Ablations, & Appendix B (Defensive Armor)  
**Spine Document**: `research/continuous-survival-td/claim-tree.json`  
**Unified Machine Plan**: `research/continuous-survival-td/evidence-plan.json`  

---

## 1. Objective of the Adversarial Track
This document contains the unsparing, hostile stress tests designed specifically to break SurvTD's core hypothesis ($C_0$). Rather than seeking positive results, these experiments systematically interrogate unstated confounds, representation leakage, heuristic regularisation artifacts, and numerical boundary collapses.

**If any of these tests succeed in reproducing SurvTD's gains through a confounded mechanism, the hypothesis is declared falsified, and `falsification-pivot` is immediately triggered.**

---

## 2. Adversarial Stress Test Specifications

### `EXP-04`: Negative Control NC-B — Within-Patient Duration Permutation
- **Kind**: `negative_control` | **Track**: `adversarial` | **Cost**: 8.0 GPU-hours | **Cut Order**: 4
- **Tested Claims**: $C_0$ (Core Mechanism Attribution)
- **The Attack**:
  - *Hostile Question*: *"Does SurvTD actually exploit continuous temporal alignment, or does it merely regularise the sequence encoder by injecting random noise / visit counts?"*
  - *Intervention*: Randomly permute the sequence of interval durations $\{\Delta t_1, \dots, \Delta t_M\}$ within each patient trajectory. This strictly preserves sequence length $M_i$, total patient follow-up time $\sum \Delta t_j = T_i$, and marginal event labels $E_i$, but completely scrambles the temporal duration between consecutive feature measurements.
- **Hypothesis Destroyer Kill Condition**:
  > **If permuted trajectories retain $> 50\%$ of SurvTD's concordance gain over unregularized baselines, $C_0$ is FALSIFIED.**
  The gain is proven to be an artifact of visit count regularisation, and the core claim must be withdrawn.

### `EXP-03`: Factorial Operator Ablations NC-A1 & NC-A2
- **Kind**: `ablation` | **Track**: `adversarial` | **Cost**: 15.0 GPU-hours | **Cut Order**: 3
- **Tested Claims**: $C_0, C_1$
- **The Attack**:
  - *Arm A1 (Discount Ablation)*: Force $\gamma_j \equiv S(\delta_s)$ (unit grid constant) while keeping continuous renewal shift $\Phi_{+\Delta t_j}$.
  - *Arm A2 (Shift Ablation)*: Force fixed unit grid shift $\Phi_{+\delta_s}$ while keeping continuous duration discount $\gamma_j = S(\Delta t_j)$.
- **Hypothesis Destroyer Kill Condition**:
  > **If either Arm A1 or Arm A2 retains $> 50\%$ of the gain, $C_0$ is FALSIFIED.**
  This isolates whether the two operators are jointly load-bearing or whether one is cosmetic packaging around the other.

### `EXP-05`: Negative Control NC-A3 — Clamped Continuous Division Comparison
- **Kind**: `negative_control` | **Track**: `adversarial` | **Cost**: 7.0 GPU-hours | **Cut Order**: 6
- **Tested Claims**: $C_0, C_1$
- **The Attack**:
  - *Hostile Question*: *"Why invent a renewal mixture and Cramér projection when you could simply clamp DeepTCSR's division: $p / \max(S(\Delta t), 10^{-3})$?"*
  - *Intervention*: Train the naive clamped division patch under identical continuous GRU-D encoders. Monitor gradient norms $\|\nabla_\theta \mathcal{L}\|$ and calibration specifically in the highest-risk decile ($S < 0.05$).
- **Hypothesis Destroyer Kill Condition**:
  > **If clamped division achieves equal training stability, bounded gradient variance, and equal high-risk calibration without divergence, $C_0$ is FALSIFIED.**
  The renewal mixture operator substitution is rendered an unnecessary complexity.

### `EXP-06`: Effective Horizon Matching & Subsampling Sweep NC-C
- **Kind**: `ablation` | **Track**: `adversarial` | **Cost**: 10.0 GPU-hours | **Cut Order**: 5
- **Tested Claims**: $C_0, C_2$
- **The Attack**:
  - Evaluate models across artificial observation downsampling rates (100%, 50%, 25% observation retention).
  - Compare duration-geometric mixing $\lambda_j = \lambda^{\Delta t_j / \delta_s}$ against discrete count-geometric $\lambda^k$.
- **Hypothesis Destroyer Kill Condition**:
  > **If duration-geometric decay fails to maintain invariant performance across observation densities, or if pure terminal Monte Carlo NLL ($\lambda = 1.0$) matches intermediate bootstrapping within seed noise, $C_2$ is FALSIFIED.**

### `EXP-08`: Projection Variance Diffusion $O(\sqrt{n})$ & Boundary Shock
- **Kind**: `sensitivity` | **Track**: `adversarial` | **Cost**: 5.0 GPU-hours | **Cut Order**: 8
- **Tested Claims**: $C_1, C_2$
- **The Attack**:
  - *Hostile Question*: *"In ultra-high-frequency telemetry ($\Delta t \to 0$), does repeated categorical projection act as a numerical heat-diffusion filter, blurring out sharp risk peaks?"*
  - *Intervention*: Measure per-step projection variance under synthetic uniform offsets and evaluate survival curve sharpness under 100-step continuous chains.
- **Hypothesis Destroyer Kill Condition**:
  > **If cumulative projection variance exceeds the theoretical bound of $\delta_s^2 / 6$ per step or Cramér loss fails to contract with modulus $\gamma_j$, $C_1$ is FALSIFIED.**

---

## 3. Falsification Kill Protocol Summary

```
Table 3: Adversarial Falsification Matrix (§5 & Appendix B)
┌──────────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────────┐
│ Stress Test ID       │ Hostile Threat Interrogated  │ Kill Threshold (Falsifier)   │ Action on Failure    │
├──────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────┤
│ EXP-04 (NC-B)        │ Visit count confounding      │ Permuted data retains > 50%  │ Trigger Pivot Loop   │
│ EXP-03 (NC-A1/A2)    │ Component bundling           │ Arm A1 or A2 retains > 50%   │ Trigger Pivot Loop   │
│ EXP-05 (NC-A3)       │ Naive clamped division       │ Clamped division stable      │ Withdraw Claim C0    │
│ EXP-06 (NC-C)        │ Bootstrapping horizon drift  │ Lambda = 1.0 matches optimal │ Withdraw Claim C2    │
│ EXP-08 (Diffusion)   │ Projection variance blowup   │ Variance > delta_s^2 / 6     │ Withdraw Claim C1    │
└──────────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────┘
```
