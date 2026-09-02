# Candidate Revision Diff Summary (Round 0 → Round 1)

**Base Candidate**: `candidate_r0.json` (Pre-revision)  
**Revised Candidate**: `candidate_r1.json` (Post-revision)  

---

## Key Diffs by Section

### 1. `core_mechanism`
- **Diff**: Added explicit bound on projection variance diffusion: *"while the categorical projection variance diffusion is bounded by delta_s^2 / 6 per step. The model is fit by squared Cramer distance between predicted and target lifetime CDFs."*

### 2. `core_mechanism_steps`
- **Step 1**: Added explicit conditional unconfoundedness assumption: *"under the explicit assumption that observation times are conditionally unconfounded given h_j."*
- **Step 3**: Added sub-bin linear hazard interpolation rule: *"For sub-bin intervals Delta t < delta_s, linear hazard interpolation S(Delta t) = 1 - h_0 * Delta t / delta_s is applied. Holding theta- frozen treats gamma_j as an exogenous constant during each inner training step to preserve contraction."*
- **Step 4**: Declared terminal absorbing bin semantics: *"accumulating mass carried past horizon s_K into the terminal absorbing bin with >= s_K defective mass semantics."*
- **Step 5**: **CORE FORMULA REWRITE**: Changed from empirical indicator branch to renewal mixture:
  - *Before*: `T p_j = I[event]*delta + (1 - I[event])*(Pi Phi p_theta-)` ($\gamma_j$ missing)
  - *After*: `T p_j = (1 - gamma_j) * mu_{[0, Delta t_j)} + gamma_j * (Pi Phi_{+Delta t_j} p_theta-)_j` ($\gamma_j$ mathematically active as survival weight and contraction modulus).
- **Step 6**: Replaced marginal KM with conditional censoring and weight truncation: *"from a covariate-conditional censoring model (Cox / Random Survival Forest), with weights truncated at 1 / G_hat(t|X) <= 10.0 to eliminate tail division instability."*
- **Step 7**: Added backward recursion formula in $G_{j+1}$ and bounded projection diffusion by $\delta_s^2 / 6$.
- **Step 8**: Specified *squared* Cramér distance ($\ell_2^2$ on CDFs).

### 3. `falsification_prediction`
- Explicitly stated directional drop for Arm A1 (discount) and Arm A2 (shift).
- Declared the irregular Poisson downsampling protocol (50% random cycle drop) on NASA C-MAPSS.
- Anchored 0.025 performance margin with empirical test bootstrap SE provenance (3x 0.008 on MIMIC-IV).
- Declared alarm stability metrics (alarm jitter, false alert episode rate at PPV 0.30) against EMA-smoothed Dynamic-DeepHit.

### 4. `load_bearing_variable`
- Changed to: `"composed_duration_transition_operator Pi Phi_{+Delta t_j} modulated by gamma_j = S_theta-(Delta t_j)"`

### 5. `negative_control`
- Split NC-A into Arm A1 (discount ablation), Arm A2 (shift ablation), and Arm A3 (clamped division comparison).
- Redefined NC-B as within-patient duration permutation preserving total follow-up time.
- Redefined NC-C as count-geometric $\lambda^k$ comparison at matched effective horizon.

### 6. `compute_budget`
- Rebudgeted from 18 to 120 GPU-hours on shared GRU-D/LSTM backbones across 4 explicitly named comparative baselines.
