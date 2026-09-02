# Candidate Revision Diff Summary (Round 1 → Round 2)

**Base Candidate**: `candidate_r1.json` (Evaluated in Round 2)  
**Revised Candidate**: `candidate_r2.json` (Evaluated in Round 3)  

---

## Diffs in `core_mechanism_steps`

### Step 5: Sample Realization Branching
- **Before (`candidate_r1`)**:
  `"Step 5: Form the one-step target in renewal mixture form: T p_j = (1 - gamma_j) * mu_{[0, Delta t_j)} + gamma_j * (Pi Phi_{+Delta t_j} p_theta-)_j, where gamma_j explicitly weights interval survival. The in-interval component mu_{[0, Delta t_j)} is supervised by projecting observed within-interval event offsets delta_{tau_j - t_j} onto the grid..."`
- **After (`candidate_r2`)**:
  `"Step 5: Form the one-step target: on non-event intervals ending in alive visits, the sample update evaluates along the continuation branch (Pi Phi_{+Delta t_j} p_theta-)_j, corresponding in conditional expectation to the renewal mixture T p_j = (1 - gamma_j) * mu_{[0, Delta t_j)} + gamma_j * (Pi Phi_{+Delta t_j} p_theta-)_j, ensuring that under a frozen target network theta-, the target map is affine in p with Cramer-metric modulus bounded by gamma_j < 1 for positive hazard. On terminal event intervals, the target evaluates at the localized intra-interval projected Dirac delta_{tau_j - t_j}."`

### Step 6: Scalar Loss IPCW Weighting
- **Before (`candidate_r1`)**:
  `"Step 6: At a terminal observation t_M where the subject is right-censored alive, complete the target tail with the target network prediction reweighted by inverse probability of censoring from a covariate-conditional censoring model (Cox / Random Survival Forest), with weights truncated at 1 / G_hat(t|X) <= 10.0 to eliminate tail division instability."`
- **After (`candidate_r2`)**:
  `"Step 6: At a terminal observation t_M where the subject is right-censored alive, complete the target tail with target network predictions, applying inverse probability of censoring weighting 1 / G_hat(t_M|X) <= 10.0 from a covariate-conditional model as a scalar sample weight in the squared Cramer loss rather than scaling distribution probabilities, preserving unit probability mass exactly."`

### Step 7: Compounding Continuation Survival Discount
- **Before (`candidate_r1`)**:
  `"Step 7: Build the multi-step target by backward recursion G_j = (1 - lambda_j) * T p_j + lambda_j * Pi Phi_{+Delta t_j} G_{j+1} evaluated with duration-geometric mixing weight lambda_j = lambda^(Delta t_j / delta_s)..."`
- **After (`candidate_r2`)**:
  `"Step 7: Build the multi-step target by backward recursion G_j = (1 - lambda_j) * T p_j + lambda_j * gamma_j * Pi Phi_{+Delta t_j} G_{j+1} evaluated with duration-geometric mixing weight lambda_j = lambda^(Delta t_j / delta_s), compounding interval survival discount gamma_j along multi-step chains and maintaining effective bootstrapping horizon invariance while bounding per-step projection diffusion by delta_s^2 / 6."`
