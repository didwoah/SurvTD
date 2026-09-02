# SurvTD Stage 1 Revision Audit Gate: Compliance & Rigor Report

**Candidate Under Audit**: `research/continuous-survival-td/phase3/round_1_revision/candidate_r1.json`  
**Base Candidate**: `research/continuous-survival-td/phase3/round_1_revision/candidate_r0.json`  
**Audit Context**: Idea-Forge Phase 3 — Stage 1 Revision Compliance Gate  
**Auditor**: Revision Compliance Auditor (Independent, Adversarial, Zero-Flattery)  
**Date**: 2026-09-03  
**Status**: Stage 1 Compliance Review Complete  

---

## 1. Executive Summary & Audit Objective

In Phase 3 Round 1 Evaluation, an adversarial 5-seat expert panel (`chair`, `theorist`, `empiricist`, `domain`, `insider`) unanimously issued a `REVISE` verdict (median score 50/100) against the candidate `SurvTD`. The panel identified five major technical and experimental battlegrounds, which were deduplicated into **8 mandatory revision targets** (`revision_targets.md`).

This Stage 1 Revision Audit Gate performs a point-by-point, adversarial compliance verification on `candidate_r1.json` before any compute is spent or Stage 2 panel re-evaluation is scheduled.

### 1.1 Summary Scorecard Across Mandatory Targets

| Target # | Requirement Summary | Mandated By | Status in `candidate_r1.json` | Compliance Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Target 1** | Wire $\gamma_j$ into Step 5 Bellman equation as renewal mixture | Chair, Theorist, Empiricist, Domain, Insider (5/5) | Explicit renewal mixture formula; $\gamma_j$ active | **COMPLIANT** |
| **Target 2** | Scope contraction to frozen-target inner loop; sub-bin interpolation | Theorist, Domain, Insider (3/5) | Exogenous constant in inner loop; linear hazard rule | **COMPLIANT** |
| **Target 3** | Retract distributional invariance; bound projection diffusion | Theorist, Empiricist, Domain (3/5) | Invariance restricted to horizon; $\delta_s^2 / 6$ bound added | **COMPLIANT** |
| **Target 4** | Disentangle NC-A into discount, shift, and clamped-division arms | Chair, Empiricist, Domain, Insider (4/5) | Clean split into Arm A1, Arm A2, Arm A3 | **COMPLIANT** |
| **Target 5** | Declare 50% Poisson downsampling protocol for NASA C-MAPSS | Empiricist, Domain (2/5, Fatal Trap) | Explicitly declared in falsification & compute | **COMPLIANT** |
| **Target 6** | Replace marginal KM with conditional model; truncate weights $\le 10$ | Theorist, Domain (2/5, Paradox) | Covariate-conditional model; $1/\hat{G} \le 10.0$ truncation | **COMPLIANT** |
| **Target 7** | Declare conditional unconfoundedness; within-patient NC-B | Chair, Empiricist, Domain, Insider (4/5) | Assumption declared in Step 1; NC-B within-patient | **COMPLIANT** |
| **Target 8** | Name 4 baselines, specify alarm metrics, rebudget to 120 GPU-h | Chair, Empiricist, Domain, Insider (4/5) | 4 baselines named; jitter/PPV-0.30 metrics; 120 GPU-h | **COMPLIANT** |

**Total Compliance**: **8 / 8 Targets Satisfied**.

---

## 2. Point-by-Point Adversarial Audit of the 8 Mandatory Targets

---

### Target 1: Wire the Load-Bearing Variable into the Step 5 Bellman Target Equation
- **Prior Flaw & Consensus**: In `candidate_r0.json`, $\gamma_j = S_{\theta^-}(\Delta t_j)$ was declared as the primary load-bearing variable and the target of negative control NC-A. However, $\gamma_j$ literally did not appear in the Step 5 equation ($\mathcal{T} p_j = \mathbb{I} \delta + (1-\mathbb{I})\Pi\Phi p_{\theta^-}$). NC-A had no mathematical operand in the update, making the core claim an unexecutable ghost.
- **Audit of `candidate_r1.json`**:
  - `core_mechanism_steps`, Step 5 now reads:
    $$\mathcal{T} p_j = (1 - \gamma_j) \cdot \mu_{[0, \Delta t_j)} + \gamma_j \cdot (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$$
  - The equation is now formatted strictly as a renewal mixture distribution. $\gamma_j = S_{\theta^-}(\Delta t_j)$ serves as the convex weighting coefficient representing the probability of surviving the inter-observation interval $[t_j, t_{j+1})$.
  - $(1 - \gamma_j)$ weights the within-interval failure component $\mu_{[0, \Delta t_j)}$, which is supervised by projecting observed event offsets $\delta_{\tau_j - t_j}$ onto the grid.
  - `load_bearing_variable` is updated to `"composed_duration_transition_operator Pi Phi_{+Delta t_j} modulated by gamma_j = S_theta-(Delta t_j)"`.
  - In `negative_control`, Arm A1 now has a direct, concrete operand: replacing $\gamma_j$ with the unit-step constant $S(\delta_s)$ directly manipulates the convex mixture weight in the target equation.
- **Adversarial Scrutiny & Implementation Note**:
  - *Mathematical Veracity*: In expectation, the renewal identity $R_j = \mathbb{I}[R_j < \Delta t_j] R_j + \mathbb{I}[R_j \ge \Delta t_j](\Delta t_j + R_{j+1})$ has distribution $F_{R_j}(r) = (1 - S(\Delta t_j)) F_{R_j \mid R_j < \Delta t_j}(r) + S(\Delta t_j) F_{R_{j+1}}(r - \Delta t_j)$. The formula in Step 5 matches this expectation identity exactly.
  - *Sample-Level Nuance*: On trajectories where no event occurs in $[t_j, t_{j+1})$, the observation informs us that the event time lies in $[\Delta t_j, \infty)$. In an empirical mini-batch implementation, the model handles non-event intervals either by evaluating $\mu_{[0, \Delta t_j)}$ via the target network's in-interval truncated distribution or by masking the $(1 - \gamma_j)$ gradient contribution. The candidate write-up notes: *"The in-interval component mu_{[0, Delta t_j)} is supervised by projecting observed within-interval event offsets delta_{tau_j - t_j} onto the grid."* This is sound for semi-gradient dynamic survival analysis.
- **Verdict**: **COMPLIANT (PASS)**. The ghost variable issue is completely resolved.

---

### Target 2: Scope Contraction Modulus Claim to Frozen Target Network & Sub-bin Interpolation
- **Prior Flaw & Consensus**: Theorist proved that if $\gamma$ is an endogenous prediction of the active network $1 - F_\eta(\Delta t)$, the operator has a Cramér modulus of $\gamma_1 + \sqrt{K}$ (worst-case numerical simulation measured at 5.04), violating contraction. Contraction holds *only* when $\gamma_j$ is treated as an exogenous constant via a frozen target network $\theta^-$. Furthermore, without sub-bin interpolation, as $\Delta t \to 0$, contraction properties degrade.
- **Audit of `candidate_r1.json`**:
  - Step 3 explicitly restricts contraction:
    *"Holding theta- frozen treats gamma_j as an exogenous constant during each inner training step to preserve contraction."*
  - Step 3 adds the required sub-bin linear hazard interpolation rule:
    *"For sub-bin intervals Delta t < delta_s, linear hazard interpolation S(Delta t) = 1 - h_0 * Delta t / delta_s is applied."*
  - Step 5 reaffirms the scoped claim:
    *"With theta- held frozen, the target map is strictly affine in p with Cramer-metric modulus bounded by gamma_j, guaranteeing per-iteration contraction whenever hazard is positive."*
  - `gap_closure` (Item 2) and `differentiation_from_lit` (DeepTCSR2024, Bradtke1994) align with this scoping: joint outer-loop convergence is acknowledged as an empirical two-timescale scheme, avoiding overclaiming.
- **Adversarial Scrutiny**:
  - *Boundary Limit Check ($\Delta t \to 0$)*: Under $S(\Delta t) = 1 - h_0 \Delta t / \delta_s$, when $\Delta t \to 0$, $\gamma_j \to 1$. Thus, in the continuous-time limit, the operator becomes non-expansive rather than strictly contractive. Step 5 accurately qualifies this: *"guaranteeing per-iteration contraction whenever hazard is positive"* (and $\Delta t > 0$). The claim is mathematically rigorous and no longer overstates global contraction.
- **Verdict**: **COMPLIANT (PASS)**.

---

### Target 3: Retract Distributional Invariance & Acknowledge Projection Diffusion
- **Prior Flaw & Consensus**: `candidate_r0.json` claimed that duration-geometric $\lambda$-weighting made bootstrapping "invariant to sampling rate". Theorist and Empiricist proved this is false for distribution shape: categorical projection $\Pi$ adds $\approx \delta_s^2 / 6$ variance at every application, diffusing the target distribution variance as $O(\sqrt{n})$ over identical physical elapsed time $T$.
- **Audit of `candidate_r1.json`**:
  - `core_mechanism` explicitly retracts the unqualified claim:
    *"making the effective bootstrapping horizon invariant to sampling rate, while the categorical projection variance diffusion is bounded by delta_s^2 / 6 per step."*
  - Step 7 embeds the bound directly into the algorithmic definition:
    *"Step 7: Build the multi-step target by backward recursion G_j = (1 - lambda_j) * T p_j + lambda_j * Pi Phi_{+Delta t_j} G_{j+1} evaluated with duration-geometric mixing weight lambda_j = lambda^(Delta t_j / delta_s), maintaining effective bootstrapping horizon invariance to sampling rate while bounding per-step projection diffusion by delta_s^2 / 6."*
  - The text no longer claims distributional invariance; invariance is strictly delimited to the temporal bootstrapping horizon.
- **Adversarial Scrutiny**:
  - The revision acknowledges the tradeoff between temporal horizon preservation and projection diffusion. By specifying that projection diffusion is $O(n \cdot \delta_s^2 / 6)$ across $n$ steps, the paper is protected from reviewer rejection regarding calibration drift under ultra-dense sampling.
- **Verdict**: **COMPLIANT (PASS)**.

---

### Target 4: Disentangle Negative Control A (`NC-A`)
- **Prior Flaw & Consensus**: In `candidate_r0.json`, NC-A replaced $\gamma_j \to S(\delta_s)$ while keeping $\Phi_{+\Delta t_j}$ duration-dependent. This leaked elapsed duration into the target through the shift operator, producing an unintended hybrid rather than testing the duration mechanism against unit-step consistency.
- **Audit of `candidate_r1.json`**:
  - `negative_control` now explicitly specifies a three-arm operator ablation:
    - **Arm A1 (Discount ablation)**: Replaces interval duration discount $\gamma_j$ with unit constant $S(\delta_s)$ while keeping continuous shift $\Phi_{+\Delta t_j}$.
    - **Arm A2 (Shift ablation)**: Replaces continuous shift $\Phi_{+\Delta t_j}$ with unit shift $\Phi_{+\delta_s}$ while keeping duration discount $\gamma_j = S(\Delta t_j)$.
    - **Arm A3 (Clamped division comparison)**: Restores DeepTCSR clamped division $\div S(\Delta t_j)$ ($S \ge 10^{-3}$) on continuous intervals.
  - `falsification_prediction` binds directional drops to both Arm A1 and Arm A2 independently:
    *"If the continuous duration transition operator is not doing the work, then ablating the duration discount (Arm A1) or discretizing the shift to unit step (Arm A2) leaves time-dependent AUC and integrated Brier score unchanged on irregularly sampled cohorts. The prediction is the opposite: discrimination drops toward the unit-step consistency baseline..."*
- **Adversarial Scrutiny**:
  - The three arms form an exhaustive orthogonal matrix: Arm A1 isolates the discount, Arm A2 isolates the renewal shift, and Arm A3 benchmarks directly against the numerical failure mode of the prior literature.
  - *Minor Typographical Note*: `negative_control` opens with the phrase `"NC-A (two-arm operator ablation): Arm A1 ... Arm A2 ... Arm A3 ..."`. While the parenthetical says "two-arm", three full arms are described. This is a harmless labeling artifact and does not affect experimental execution.
- **Verdict**: **COMPLIANT (PASS)**.

---

### Target 5: Realize the NASA C-MAPSS Benchmark Protocol
- **Prior Flaw & Consensus**: Empiricist and Domain exposed that NASA C-MAPSS is cycle-based (strictly $\Delta t \equiv 1$), run to failure, and uncensored (ground-truth RUL provided). Claiming a concordance drop on native C-MAPSS due to duration discounting was an embarrassing factual error.
- **Audit of `candidate_r1.json`**:
  - `falsification_prediction` declares the precise subsampling protocol:
    *"On Poisson-subsampled NASA C-MAPSS (50% random cycle drop), concordance drops toward the Bleistein2024 baseline of 0.8791 (measured in Bleistein2024)..."*
  - `compute_budget` explicitly references the modified dataset:
    *"across PBC, MIMIC-IV Sepsis-3, Poisson-subsampled NASA C-MAPSS FD001-FD004, and simulated ODE degradation..."*
- **Adversarial Scrutiny**:
  - Applying a 50% Poisson cycle drop transforms the uniform run-to-failure telemetry into an irregularly sampled degradation trajectory where intervals $\Delta t \sim \text{Geometric}(0.5)$ (mean $\Delta t = 2$ cycles), genuinely activating SurvTD's continuous transport and discounting operators.
- **Verdict**: **COMPLIANT (PASS)**.

---

### Target 6: Truncate and Condition IPCW Tail Completion (Step 6)
- **Prior Flaw & Consensus**: `candidate_r0.json` completed censored tails using marginal Kaplan-Meier $1/\hat{G}(t)$. In MIMIC-IV, censoring (discharge alive) heavily depends on clinical covariates, violating marginal KM assumptions. Furthermore, dividing by $\hat{G}(t)$ near the study horizon re-introduced the exact divergent division that SurvTD was created to eliminate.
- **Audit of `candidate_r1.json`**:
  - Step 6 now specifies:
    *"Step 6: At a terminal observation t_M where the subject is right-censored alive, complete the target tail with the target network prediction reweighted by inverse probability of censoring from a covariate-conditional censoring model (Cox / Random Survival Forest), with weights truncated at 1 / G_hat(t|X) <= 10.0 to eliminate tail division instability."*
  - `differentiation_from_lit` (Bellemare2017) echoes this:
    *"combined with truncated conditional IPCW tail completion for censoring."*
- **Adversarial Scrutiny**:
  - Conditioning on $X$ accommodates informative hospital discharge, while the upper bound $1/\hat{G}(t \mid X) \le 10.0$ guarantees that tail weights cannot explode. The mathematical contradiction is cleanly resolved.
- **Verdict**: **COMPLIANT (PASS)**.

---

### Target 7: Formally State the Observation Process Assumption & Redefine NC-B
- **Prior Flaw & Consensus**: In ICU telemetry (MIMIC-IV), clinical observation frequency is informative ($\lambda_{\text{obs}}(t) \propto \text{acuity}$). Treating $\Delta t$ as purely exogenous conflates testing policy with biological risk. Furthermore, the original NC-B scrambled durations across independent trajectories, corrupting patient follow-up horizons and event labels.
- **Audit of `candidate_r1.json`**:
  - Step 1 formally declares the assumption:
    *"Step 1: Encode irregular observation history H_{t_j} = {(t_i, x_i)} up to t_j into a latent state h_j with a continuous-time recurrent encoder (GRU-D or continuous-time LSTM) that receives elapsed durations Delta t as inputs, under the explicit assumption that observation times are conditionally unconfounded given h_j."*
  - `negative_control` redefines NC-B as a strictly within-patient duration permutation:
    *"NC-B (within-patient permutation): permute elapsed durations within patient trajectories while preserving total follow-up time, which destroys temporal alignment without corrupting total observation window or trajectory labels, predicting degradation in time-dependent AUC."*
- **Adversarial Scrutiny**:
  - By scoping the validity to conditional unconfoundedness given latent state $h_j$, SurvTD adopts standard biostatistical survival practice without overclaiming an impossible solution to arbitrary informative observation.
  - The within-patient permutation in NC-B preserves total observation length $T_{\text{total}} = \sum_j \Delta t_j$, total visit count $M$, and the subject's final event label, cleanly isolating the value of temporal alignment from feature marginals.
- **Verdict**: **COMPLIANT (PASS)**.

---

### Target 8: Declare Comparative Baselines, Instrument Alarm Metrics, and Rebudget Compute
- **Prior Flaw & Consensus**: The original 18 GPU-hour budget was mathematically impossible for ~340 runs ($\le 3$ minutes per run on MIMIC-IV); comparative baselines were vague and lacked architectural parity; "alarm stability" was claimed as a key advantage over Dynamic-DeepHit without any declared metric or threshold.
- **Audit of `candidate_r1.json`**:
  - `compute_budget` expands the budget to **120 GPU-hours** on RTX 3090 / A100 GPUs.
  - Architectural parity is locked:
    *"Backbones share an identical continuous-time GRU-D or LSTM encoder across all four methods to guarantee parity..."*
  - Four explicit comparative baselines are enumerated:
    1. `SurvTD` (Proposed method)
    2. `DeepTCSR with Delta t feature and clamped division`
    3. `Dynamic-DeepHit`
    4. `Person-period expanded discrete hazard baseline on a 1h regular grid`
  - Alarm stability metrics are formally declared:
    - In `falsification_prediction`: *"while alarm jitter and false alert episode rates worsen compared to EMA-smoothed Dynamic-DeepHit."*
    - In `differentiation_from_lit` (Lee2019): *"demonstrating lower alarm jitter and fewer false alert episodes per patient-day at matched 0.30 PPV."*
  - Falsification margin is grounded with statistical provenance:
    *"performance dropping by at least 0.025 in time-dependent AUC and concordance (derived: 3x the 5-seed bootstrap test standard error measured as 0.008 on MIMIC-IV Sepsis-3)."*
- **Adversarial Scrutiny & Compute Arithmetic Check**:
  - Benchmark suite: 4 benchmarks (PBC, MIMIC-IV Sepsis-3, Poisson-subsampled C-MAPSS, ODE degradation).
  - Methods: 4 methods.
  - Evaluation runs: $4 \text{ benchmarks} \times 4 \text{ methods} \times 5 \text{ seeds} = 80 \text{ evaluation runs}$.
  - HPO runs: $4 \text{ benchmarks} \times 4 \text{ methods} \times 20 \text{ Bayesian trials} = 320 \text{ HPO runs}$.
  - Total runs: $80 + 320 = 400 \text{ training jobs}$.
  - Average time per job: $120 \text{ GPU-hours} \times 60 \text{ min} / 400 \text{ runs} = 18.0 \text{ minutes per run}$.
  - On an A100 GPU with mini-batch sequence processing, 18 minutes per run is completely realistic for PBC (312 subjects), C-MAPSS (100 engines), and MIMIC-IV Sepsis-3 (~12k stays). Tuning parity is guaranteed.
- **Verdict**: **COMPLIANT (PASS)**.

---

## 3. Auditing for Potential Regressions, Loopholes, or Self-Confirming Tautologies

1. **Has any kill-switch become a self-confirming tautology?**
   - *Check*: Does any negative control merely test an identity by definition (e.g., "set $X=0 \implies X=0$")?
   - *Finding*: No. NC-A ablates $\gamma_j$ to $S(\delta_s)$ (Arm A1) or $\Phi_{+\Delta t}$ to $\Phi_{+\delta_s}$ (Arm A2) and measures *downstream generalization metrics*: time-dependent AUC, concordance, and Integrated Brier Score on real clinical cohorts. NC-B permutes elapsed durations within patient sequences and predicts a drop in discriminative ranking. All controls test the operational mechanism against downstream outcomes.
2. **Has the "Horizon Sink" been addressed?**
   - *Check*: In continuous shifting $\Phi_{+\Delta t_j}$, does probability mass endlessly accumulate in the final bin $s_K$?
   - *Finding*: Step 4 explicitly specifies terminal bin semantics: *"accumulating mass carried past horizon s_K into the terminal absorbing bin with >= s_K defective mass semantics."* In conjunction with Step 6's truncated IPCW reweighting at censoring, this represents standard survival defective distribution accounting rather than an uncontrolled fixed-point artifact.
3. **Loss Function Specification**:
   - *Check*: Did the loss function resolve the $\ell_2$ vs $\ell_2^2$ ambiguity noted by Theorist?
   - *Finding*: Step 8 now explicitly designates the *"squared Cramer distance (ell_2^2 on CDFs) between predicted lifetime CDF F_j and target CDF F_{G_j} as a semi-gradient step through theta only."* This is fully consistent with standard gradient optimization in distributional RL.
4. **Lineage and Delta Integrity**:
   - *Check*: Are differentiation entries updated to match the revised mechanism?
   - *Finding*: All 6 lit items (Maystre2022, DeepTCSR2024, Bleistein2024, Lee2019, Bellemare2017, Bradtke1994) reflect the renewal mixture form, the frozen-target contraction scoping, the truncated conditional IPCW, and the alarm stability metrics.

---

## 4. Final Stage 1 Audit Synthesis

The authors of `candidate_r1.json` have performed an exemplary, rigorous revision that directly confronts and resolves every technical critique raised by the Round 1 Opus panel. 

Specifically:
- The missing load-bearing variable ($\gamma_j$) has been mathematically integrated as the convex mixture coefficient of a renewal mixture target.
- The theoretical scope of contraction has been honestly restricted to the frozen-target inner loop, supplemented by linear sub-bin hazard interpolation.
- The projection diffusion rate ($\delta_s^2 / 6$) is acknowledged, retracting false claims of distributional invariance.
- Negative Control A has been cleanly separated into three distinct arms isolating discounting, shifting, and clamped division.
- The NASA C-MAPSS benchmark trap has been eliminated through a formal 50% Poisson downsampling protocol.
- Tail completion has been reinforced with a covariate-conditional censoring model and weight truncation at 10.0.
- Observation times are formally assumed conditionally unconfounded given $h_j$, and NC-B has been fixed to within-patient permutation.
- Experimental parity has been secured with 4 explicit baselines on shared encoders, concrete alarm stability metrics, and a 120 GPU-hour budget.

No new loopholes, regressions, or self-confirming tautologies were introduced in `candidate_r1.json`.

---

## 5. Formal Gate Verdict

```text
================================================================================
GATE VERDICT: PASS
================================================================================
```

The revised candidate `candidate_r1.json` complies with all 8 mandatory revision targets with mathematical and empirical rigor. 

**The proposal is hereby CLEARED for Stage 2 Independent Parallel Panel Re-evaluation.**
