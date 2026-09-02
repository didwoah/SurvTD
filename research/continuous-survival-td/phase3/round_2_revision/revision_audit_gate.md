# SurvTD Stage 1 Revision Compliance Audit Gate: Round 2 Revision Report

**Candidate Under Audit**: `research/continuous-survival-td/phase3/round_2_revision/candidate_r2.json`  
**Base Candidate (Round 1 Snapshot)**: `research/continuous-survival-td/phase3/round_1_revision/candidate_r1.json`  
**Audit Context**: Idea-Forge Phase 3 — Stage 1 Revision Compliance Gate (Round 2)  
**Auditor**: Revision Compliance Auditor (Independent, Adversarial, Zero-Flattery)  
**Date**: 2026-09-03  
**Status**: Stage 1 Compliance Review Complete  

---

## 1. Executive Summary & Audit Context

In Phase 3 Round 2 Evaluation (`round_2_eval/`), the 5-seat expert panel evaluated `candidate_r1.json`. Four of the five seats (`chair`, `insider`, `empiricist`, `domain`) voted to `ADVANCE` with scores of 75–83/100. However, the **Theoretician (`theorist`)** filed a dissenting `REVISE` verdict (score 67/100) documenting three fundamental theoretical and sample-estimator defects (`critic_report_theorist.md`).

Under the strict rules of `idea-forge`, **advancing to Phase 4 requires an unequivocal 5/5 unanimous advance**. Any dissenting `REVISE` triggers the Two-Stage Re-Evaluation Protocol:
1. Formulate mandatory revision targets (`revision_targets.md`).
2. Implement fixes into `candidate_r2.json` and document the diff (`candidate_diff_summary.md`).
3. **Stage 1 Revision Compliance Audit Gate** (this audit): An independent, zero-flattery compliance audit checking whether all mandatory targets are rigorously satisfied without introducing regressions, loopholes, or kill-switch tampering.

### 1.1 Summary Scorecard Across Mandatory Targets

| Target # | Requirement Summary | Mandated By | Status in `candidate_r2.json` | Compliance Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Target 1** | Explicitly branch Step 5 into continuation sample updates on living visits and Dirac updates on terminal death visits; distinguish from expected mixture | Theorist (`critic_report_theorist.md`) | Step 5 branches living $(\Pi \Phi p_{\theta^-})$ from terminal Dirac $\delta_{\tau_j - t_j}$; frames $(1-\gamma_j)\mu + \gamma_j \Pi \Phi p_{\theta^-}$ as conditional expectation | **COMPLIANT** |
| **Target 2** | Relocate IPCW weight $1/\hat{G}(t_M \mid X) \le 10.0$ to scalar squared Cramér loss weight in Step 6; preserve unit probability mass | Theorist (`critic_report_theorist.md`) | Step 6 applies $1/\hat{G}(t_M \mid X) \le 10.0$ as scalar sample weight in squared Cramér loss; unscaled probabilities preserve unit mass | **COMPLIANT** |
| **Target 3** | Compound continuation discount $\gamma_j$ in Step 7 backward recursion ($\lambda_j \gamma_j \Pi \Phi_{+\Delta t_j} G_{j+1}$) | Theorist (`critic_report_theorist.md`) | Step 7 recursion explicitly compounds $\lambda_j * \gamma_j * \Pi \Phi_{+\Delta t_j} G_{j+1}$ along multi-step chains | **COMPLIANT** |

**Total Compliance**: **3 / 3 Mandatory Targets Fully Satisfied**.

---

## 2. Point-by-Point Adversarial Audit of Mandatory Targets

---

### Target 1: Distinguish Empirical Sample Realization from Expected Renewal Mixture in Step 5

- **Prior Flaw & Theoretical Attack**:
  In `candidate_r1.json`, Step 5 asserted:
  > *"Form the one-step target in renewal mixture form: $\mathcal{T} p_j = (1 - \gamma_j) \cdot \mu_{[0, \Delta t_j)} + \gamma_j \cdot (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$, where $\gamma_j$ explicitly weights interval survival. The in-interval component $\mu_{[0, \Delta t_j)}$ is supervised by projecting observed within-interval event offsets $\delta_{\tau_j - t_j}$ onto the grid..."*

  The Theoretician exposed a fatal sample-estimator conflation: In clinical observation trajectories, any interval $[t_j, t_{j+1})$ between two consecutively observed patient visits is *by definition* an interval in which the patient survived without an event ($E_j = 0$). Forcing an empirical sample update on surviving visits to evaluate an in-interval failure component $\mu_{[0, \Delta t_j)}$ supervised by a non-existent intra-interval event offset $\delta_{\tau_j - t_j}$ made the algorithm unexecutable on actual patient data.

- **Audit of `candidate_r2.json` (Step 5)**:
  Step 5 has been rewritten as follows:
  > *"Step 5: Form the one-step target: on non-event intervals ending in alive visits, the sample update evaluates along the continuation branch $(\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$, corresponding in conditional expectation to the renewal mixture $\mathcal{T} p_j = (1 - \gamma_j) * \mu_{[0, \Delta t_j)} + \gamma_j * (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$, ensuring that under a frozen target network $\theta^-$, the target map is affine in $p$ with Cramer-metric modulus bounded by $\gamma_j < 1$ for positive hazard. On terminal event intervals, the target evaluates at the localized intra-interval projected Dirac $\delta_{\tau_j - t_j}$."*

- **Adversarial Mathematical Scrutiny**:
  1. *Sample Realization vs. Expected Operator*:
     In sample-based Temporal Difference learning (TD), an update rule must be computable from the realized transition tuple $(h_j, \Delta t_j, E_j, h_{j+1})$.
     - When $E_j = 0$ (alive transition to $t_{j+1}$), the empirical realization informs us that the event occurs strictly in $[\Delta t_j, \infty)$. The sample update evaluates directly along the continuation branch $(\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$.
     - When integrated across the underlying conditional survival process, the expected Bellman operator yields the convex renewal mixture:
       $$\mathbb{E}[\mathcal{T}_{\text{sample}} p_j \mid h_j] = (1 - \gamma_j) \cdot \mu_{[0, \Delta t_j)} + \gamma_j \cdot (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$$
     - This preserves the theoretical contraction proof under the frozen target network $\theta^-$ (where the Cramér-metric modulus is bounded by $\gamma_j < 1$ for strictly positive hazard).
  2. *Terminal Event Branching*:
     When an interval ends in an observed death event ($E_j = 1$ at failure time $\tau_j \in [t_j, t_j + \Delta t_j)$), no subsequent visit $t_{j+1}$ exists. The target evaluates at the localized intra-interval projected Dirac $\Pi \delta_{\tau_j - t_j}$.
  3. *Conclusion*: The branch is mathematically complete, operationally well-defined for batch sequence training, and eliminates the undefined operand $\delta_{\tau_j - t_j}$ on surviving transitions.

- **Target 1 Verdict**: **COMPLIANT (PASS)**.

---

### Target 2: Relocate IPCW Censoring Weight to Scalar Squared Cramér Loss in Step 6

- **Prior Flaw & Theoretical Attack**:
  In `candidate_r1.json`, Step 6 asserted:
  > *"Step 6: At a terminal observation $t_M$ where the subject is right-censored alive, complete the target tail with the target network prediction reweighted by inverse probability of censoring from a covariate-conditional censoring model (Cox / Random Survival Forest), with weights truncated at $1 / \hat{G}(t \mid X) \le 10.0$ to eliminate tail division instability."*

  The Theoretician proved that scaling target distribution predictions $p$ directly by $1/\hat{G}(t_M \mid X) \le 10.0$ inflates total distribution probability mass up to $10.0$. This produces an improper defective distribution, violating basic probability axioms ($\sum_k p_k = 1.0$) and breaking the metric-space assumptions of the Cramér distance ($\ell_2^2$ on CDFs).

- **Audit of `candidate_r2.json` (Step 6)**:
  Step 6 has been revised as follows:
  > *"Step 6: At a terminal observation $t_M$ where the subject is right-censored alive, complete the target tail with target network predictions, applying inverse probability of censoring weighting $1 / \hat{G}(t_M \mid X) \le 10.0$ from a covariate-conditional model as a scalar sample weight in the squared Cramer loss rather than scaling distribution probabilities, preserving unit probability mass exactly."*

- **Adversarial Mathematical Scrutiny**:
  1. *Preservation of Probability Measure Space*:
     By completing the target tail with unscaled target network predictions, the target distribution $p^*_{M}$ remains a proper probability distribution satisfying $\sum_k p^*_{M, k} = 1.0$, and its cumulative sum $F^*_{M}$ is a valid monotonic CDF on $[0, 1]$.
  2. *Proper IPCW Loss Formulation*:
     The inverse probability of censoring weighting $w_i = \min\left(\frac{1}{\hat{G}(t_{M, i} \mid X_i)}, 10.0\right)$ enters as an importance weight multiplying the scalar sample loss:
     $$\mathcal{L}_i(F_{M, i}, F^*_{M, i}) = w_i \cdot \ell_2^2(F_{M, i}, F^*_{M, i}) = w_i \sum_{k=1}^K \delta_s \left( F_{M, i}(s_k) - F^*_{M, i}(s_k) \right)^2$$
     This matches standard biostatistical IPCW estimation theory (e.g., Graf et al., Robins & Finkelstein) without creating defective CDFs.
  3. *Interplay with Step 8*:
     Step 8 specifies minimization of squared Cramér distance between predicted CDF $F_j$ and target CDF $F_{G_j}$ as a semi-gradient step through $\theta$. Step 6 provides the sample weight for right-censored endpoints without altering the distribution metric space.

- **Target 2 Verdict**: **COMPLIANT (PASS)**.

---

### Target 3: Nest Interval Survival Discount $\gamma_j$ into Multi-Step Continuation Recursion in Step 7

- **Prior Flaw & Theoretical Attack**:
  In `candidate_r1.json`, Step 7 defined the backward recursion as:
  > *"Step 7: Build the multi-step target by backward recursion $G_j = (1 - \lambda_j) \cdot \mathcal{T} p_j + \lambda_j \cdot \Pi \Phi_{+\Delta t_j} G_{j+1}$ evaluated with duration-geometric mixing weight $\lambda_j = \lambda^{(\Delta t_j / \delta_s)}$..."*

  The Theoretician pointed out that the continuation term $\lambda_j \Pi \Phi_{+\Delta t_j} G_{j+1}$ omitted the interval discount $\gamma_j = S_{\theta^-}(\Delta t_j)$. Consequently, in multi-step bootstrapping as $\lambda_j \to 1$, $G_j \approx \Pi \Phi_{+\Delta t_j} G_{j+1}$, carrying target predictions backward across arbitrary spans of time with zero cumulative mortality discounting.

- **Audit of `candidate_r2.json` (Step 7)**:
  Step 7 has been updated to:
  > *"Step 7: Build the multi-step target by backward recursion $G_j = (1 - \lambda_j) * \mathcal{T} p_j + \lambda_j * \gamma_j * \Pi \Phi_{+\Delta t_j} G_{j+1}$ evaluated with duration-geometric mixing weight $\lambda_j = \lambda^{(\Delta t_j / \delta_s)}$, compounding interval survival discount $\gamma_j$ along multi-step chains and maintaining effective bootstrapping horizon invariance while bounding per-step projection diffusion by $\delta_s^2 / 6$."*

- **Adversarial Mathematical Scrutiny**:
  1. *Compounded Survival Discounting*:
     In continuous-time survival analysis, the probability of remaining event-free across a sequence of visits $t_j, t_{j+1}, \dots, t_{j+m}$ is the product of conditional survival probabilities:
     $$\mathbb{P}(T > t_{j+m} \mid T > t_j) = \prod_{k=j}^{j+m-1} S(t_{k+1} - t_k \mid H_{t_k}) = \prod_{k=j}^{j+m-1} \gamma_k$$
     By inserting $\gamma_j$ into the continuation recursion $\lambda_j \gamma_j \Pi \Phi_{+\Delta t_j} G_{j+1}$, unrolling the recursion over $m$ steps weights the $m$-step return by:
     $$\left( \prod_{k=j}^{j+m-1} \lambda_k \gamma_k \right) \Pi \Phi_{+(t_{j+m} - t_j)} G_{j+m}$$
     This compounds the duration discount $\gamma_k$ along the exact trajectory history, ensuring that the probability of future continuation correctly decays as cumulative mortality risk accumulates.
  2. *Duration-Geometric Horizon Preservation*:
     The compounding of $\lambda_j = \lambda^{(\Delta t_j / \delta_s)}$ ensures that the bootstrapping cutoff timescale remains invariant to the observation density, while the cumulative $\gamma_k$ factor accounts for biological survival probability.
  3. *Defective Mass Semantics on Compounded Returns*:
     Multiplying the continuation distribution by $\gamma_j < 1$ reflects that an individual has a probability $(1 - \gamma_j)$ of failing within interval $j$. The resulting continuation component carries total mass $\gamma_j$. When coupled with Step 4's explicit defective mass accounting ($\ge s_K$ terminal absorbing bin) and Step 5's sample branching, the target recursion is mathematically consistent in the Cramér metric space.

- **Target 3 Verdict**: **COMPLIANT (PASS)**.

---

## 3. Regression, Flaw, and Kill-Switch Audit

### 3.1 Kill-Switch Integrity & Falsification Tampering Check

A common failure mode during revision is relaxing kill-switch thresholds, softening negative controls, or tampering with falsification predictions to artificially pass tests. 

We ran automated structural validation comparing `candidate_r2.json` directly against the `candidate_r1.json` baseline using the skill's official validator:
```bash
python3 .agents/skills/idea-forge/scripts/validate_idea.py \
  --baseline research/continuous-survival-td/phase3/round_1_revision/candidate_r1.json \
  research/continuous-survival-td/phase3/round_2_revision/candidate_r2.json
```
**Validator Output**:
```text
validate_idea: research/continuous-survival-td/phase3/round_2_revision/candidate_r2.json

ok    required_fields: 10 present
ok    pattern_vocabulary: 2 gap(s), 2 distinct pattern(s)
ok    falsification_direction
ok    load_bearing_variable: composed_duration_transition_operator Pi Phi_{+Delta t_j} mo
ok    negative_control
ok    numeric_provenance: 2 bar(s), provenance marker present
ok    named_parameters
ok    differentiation: 6 grounded delta(s)
ok    signature_terms: 5 term(s)
ok    alias_terms: 6 term(s)
ok    kill_switch_integrity: byte-identical to baseline

VERDICT: pass
```

**Audit Finding**:
- `kill_switch_integrity` is **byte-identical to baseline**.
- `falsification_prediction` was NOT modified. The margin $\ge 0.025$ in time-dependent AUC and concordance (derived from $3\times$ the 5-seed bootstrap SE of 0.008 on MIMIC-IV Sepsis-3) remains strictly locked.
- `negative_control` remains strictly locked across all three arms: Arm A1 (discount ablation), Arm A2 (shift ablation), Arm A3 (clamped division comparison), NC-B (within-patient permutation), and NC-C (horizon sweep).
- `compute_budget` remains locked at 120 GPU-hours across 4 baselines, 4 benchmarks, 20 HPO trials, 5 seeds, and 1000 bootstrap iterations.

### 3.2 Prior Art Differentiation & Integrity Check

The six grounded deltas (`Maystre2022`, `DeepTCSR2024`, `Bleistein2024`, `Lee2019`, `Bellemare2017`, `Bradtke1994`) were reviewed. All six remain accurate and consistent with the revised candidate:
- `Bellemare2017`: Properly distinguishes C51's categorical projection of discounted reward from SurvTD's renewal lifetime translation combined with truncated conditional IPCW tail completion.
- `DeepTCSR2024` & `Maystre2022`: Accurately cite the replacement of division by interval survival with the renewal shift and categorical projection.
- `Bradtke1994`: Accurately highlights the extension from scalar value discounting to full distribution discounting via frozen EMA targets.

### 3.3 New Flaw / Anti-Pattern Check

- **Anti-Pattern Check**: The candidate uses two ideation patterns (`assumption_audit_and_pivot` + `architectural_operator_substitution`). This composition is protected by the explicit `composition_note` linking continuous duration audit directly to the renewal shift and categorical projection.
- **End-to-End Algorithmic Closure**: Steps 1 through 8 now form a completely coherent pipeline from continuous-time recurrent encoding $\to$ discrete hazard emission $\to$ frozen target discount evaluation $\to$ renewal shift and projection $\to$ branched sample Bellman target $\to$ scalar IPCW tail completion $\to$ compounded multi-step recursion $\to$ semi-gradient squared Cramér loss optimization.

---

## 4. Final Stage 1 Audit Synthesis

The Round 2 revision of SurvTD in `candidate_r2.json` directly and rigorously remedies the three mathematical objections raised by the Theoretician in Round 2 Evaluation:
1. **Step 5** cleanly resolves the sample-vs-expectation conflation by evaluating along the continuation branch on living transitions while anchoring the expected renewal mixture operator and handling terminal event Diracs.
2. **Step 6** eliminates the improper distribution mass inflation by placing truncated IPCW weights into the scalar squared Cramér loss function, preserving exact unit probability mass.
3. **Step 7** compounds the interval survival discount $\gamma_j$ along multi-step chains, preventing un-discounted survival propagation as $\lambda_j \to 1$.

No regressions, loopholes, or kill-switch modifications were introduced.

---

## 5. Formal Gate Verdict

```text
================================================================================
GATE VERDICT: PASS
================================================================================
```

The candidate `research/continuous-survival-td/phase3/round_2_revision/candidate_r2.json` is **APPROVED** to proceed to Stage 2: Parallel Independent Re-Evaluation Panel (Round 3 Evaluation across 5 seats).
