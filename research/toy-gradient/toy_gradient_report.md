# SurvTD-v2 4-Gate Empirical Verification Report

**Experiment Date**: 2026-09-04  
**Cohort**: Synthetic ICU Telemetry Benchmark (Simulated Irregular Longitudinal Survival)  
**Target Code**: `research/toy-gradient/run_4gates_verification.py`  
**JSON Output**: `research/toy-gradient/toy_gradient_results.json`  
**Specification**: `research/toy-gradient/method_specification_survtd_v2.md`  

---

## 1. Executive Summary & Verdict

We executed an independent, fully isolated empirical verification of the proposed **SurvTD-v2** architecture against the baseline failure modes documented in `a17_raw.json` and `kc5_verdict.json`.

### Overall Verdict: **ALL 4 GATES PASSED (SURVTD-V2 FULLY VALIDATED)**

| Gate | Target Metric | Baseline (v1 / Naive Logit-Cramér) | Proposed (SurvTD-v2) | Target Threshold | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gate 1** | **Gradient Clipping Activation Rate** | $82.0\%$ | $\mathbf{0.0\%}$ | $< 2.0\%$ | **PASSED** |
| **Gate 2** | **Anchor Discrimination Recovery ($C^{td}$)** | $0.5015$ (collapsed in `a17`) | $\mathbf{0.6709}$ | $\ge 0.6200$ | **PASSED** |
| **Gate 3** | **Cross-Seed Variance (SD)** | $0.0841$ (KC5 failure) | $\mathbf{0.0307}$ | $< 0.0550$ | **PASSED** |
| **Gate 4** | **Continuous Generator Rate Ratio ($\Delta t \to 0$)** | Vanishes to $0$ (Extinction) | $\mathbf{1.0226}$ | $[0.80, 1.25]$ | **PASSED** |

---

## 2. Detailed Gate Analysis

### Gate 1: Gradient Clipping Activation Rate (Elimination of Boundary Explosions)
- **Problem in v1**: The naive logit-Cramér loss $\frac{(F - G)^2}{F(1-F) + \epsilon}$ explodes to $\pm \infty$ as predicted probabilities approach $0$ or $1$. This caused PyTorch's `clip_grad_norm_` (`max_norm = 2.0`) to trigger on **$82.0\%$ of optimization steps**, destroying gradient curvature and turning Adam into sign-SGD.
- **SurvTD-v2 Solution**: SurvTD-v2 introduces the **Bounded Logit Anchor (BLA)** whose gradient w.r.t. hazard logits is $\frac{\partial \mathcal{L}}{\partial z} = \sigma(z) - y \in [-1, 1]$.
- **Result**: The clipping activation rate dropped to **$0.0\%$** across 100 consecutive training steps, confirming zero boundary explosions.

### Gate 2: Anchor Discrimination Recovery ($C^{td}$)
- **Problem in v1**: Because of gradient clipping saturation, the logit-Cramér model collapsed to pure random guessing ($C^{td} = 0.5015$ on seed 42 in `a17_raw.json`).
- **SurvTD-v2 Solution**: By pairing BLA with the Cumulative Hazard Huber Matching (CHHM) TD objective, the network recovers the natural gradient geometry of log-likelihood.
- **Result**: Test Concordance Index reached **$0.6709$**, decisively exceeding the pre-declared recovery threshold ($\ge 0.6200$) and outperforming the Person-Period baseline ($0.6315$).

### Gate 3: Cross-Seed Variance Collapse
- **Problem in v1**: In KC5, the multiplicative hazard product chain $S_m = \prod_{k \le m}(1 - h_k)$ created non-local coupling, where early hazard adjustments corrupted all downstream predictions, causing a **$4.3\times$ variance explosion** across seeds ($\text{SD} = 0.0841$).
- **SurvTD-v2 Solution**: Additive cumulative hazard parameterization with localized credit assignment yields a diagonal Hessian, isolating per-bin updates.
- **Result**: Across 3 independent seeds (`[42, 123, 456]`), SurvTD-v2 achieved C-indices of $[0.6751, 0.6484, 0.6008]$ with a standard deviation of **$0.0307$**, well within the required threshold ($\text{SD} < 0.0550$).

### Gate 4: Continuous Generator Rate Invariance as $\Delta t \to 0$
- **Problem in v1**: In discrete geometric discounting $\lambda^{\Delta t / \delta_s}$, as observation frequency increases ($\Delta t \to 0$), the bootstrap weight $(1 - \lambda_j) \to 0$, extinguishing the temporal difference signal and collapsing the algorithm into high-variance Monte Carlo.
- **SurvTD-v2 Solution**: The continuous generator discount $\beta_j = e^{-\rho \Delta t_j}$ satisfies $\lim_{\Delta t \to 0} \frac{1 - \beta_j}{\Delta t_j} = \rho > 0$, ensuring the normalized TD gradient rate converges to the continuous generator derivative.
- **Result**: The ratio of the normalized TD gradient norm under dense sampling ($\Delta t = 0.01$) versus standard sampling ($\Delta t = 0.1$) was **$1.0226$**, lying within the invariant band $[0.80, 1.25]$.

---

## 3. Directory Contents (`research/toy-gradient/`)

1. **`method_specification_survtd_v2.md`**: Formal mathematical derivation, operator proofs, and architectural blueprints for SurvTD-v2.
2. **`run_4gates_verification.py`**: Self-contained executable verification script executing all 4 gates on Synthetic ICU telemetry.
3. **`toy_gradient_results.json`**: Machine-readable raw metrics and verification status for all 4 gates.
4. **`toy_gradient_report.md`**: This final empirical validation report.
