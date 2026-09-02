# Theoretical Reviewer (`theorist`) — Round 2 Critic Report: SurvTD

**Candidate Evaluated**: `research/continuous-survival-td/phase3/round_1_revision/candidate_r1.json`  
**Report Path**: `/Users/yangjaemo/Desktop/SurvTD/research/continuous-survival-td/phase3/round_2_eval/critic_report_theorist.md`  
**Role**: Theoretical ML Reviewer (`theorist`)  

---

### Scorecard Summary
- **Axis A (Problem Position)**: **4 / 5**
  - Quoted: *"Dynamic survival models that enforce temporal consistency between consecutive observations inherit a unit-step transition (Delta t = 1)..."*
  - Diagnosis: Real, non-obvious gap; correctly diagnoses that division by $S(\Delta t)$ is the root instability.
- **Axis B (Method Quality)**: **3 / 5**
  - **depth: 4 · soundness: 2 · feasibility: 4**
  - Diagnosis: Sound operator theory in expectation, but broken sample estimators and target definitions in Steps 5, 6, and 7.
- **Axis C (Problem-fit)**: **4 / 5**
  - Direct operator substitution $(\Pi \Phi_{+\Delta t}, \gamma_j)$ cleanly resolves both unit-step rigidity and division explosion.
- **Axis D (Falsifiability & Integrity)**: **4 / 5**
  - Sharp 3-arm negative control (NC-A) and derived numeric bars from bootstrap standard errors.

- **Overall Score**: **67 / 100** (Band: `strong`)
- **Gate Status**: Passed
- **Hard Floor Audit**: Passed
- **Final Verdict**: **`REVISE`**

---

### Key Theoretical Findings & Landed Attacks

1. **Step 5 Sample Realization Conflation**:
   - In actual patient trajectories, any interval $[t_j, t_{j+1})$ between two observed visits is by definition a surviving interval without an in-interval event ($E_j = 0$).
   - Step 5 asserts $\mu_{[0, \Delta t_j)}$ is supervised by event offsets $\delta_{\tau_j - t_j}$, which do not exist on surviving transitions. The sample realization must explicitly branch into the continuation branch $(\Pi \Phi p_{\theta^-})$ on living visits and Dirac updates on terminal death visits.
2. **Step 6 IPCW Probability Mass Blowup**:
   - Weighting target distribution predictions by $1/\hat{G}(t_M \mid X) \le 10.0$ inflates total probability mass up to $10.0$, creating an improper defective CDF that violates Cramér metric assumptions. IPCW weighting must be applied as a scalar sample weight in the squared Cramér loss rather than scaling distribution probabilities.
3. **Step 7 Multi-Step Continuation Discount**:
   - The backward recursion $G_j = (1-\lambda_j)T p_j + \lambda_j \Pi \Phi G_{j+1}$ omits the interval discount $\gamma_j$ from the continuation term, implying zero mortality across multi-step chains as $\lambda \to 1$.

---

### Mandatory Revision Targets for Round 2:
1. `[Step 5 sample Bellman realization]`: Branch the sample estimator explicitly into surviving transitions ($\Pi \Phi_{+\Delta t_j} p_{\theta^-}$) and terminal event transitions ($\Pi \delta_{\tau - t_j}$).
2. `[Step 6 IPCW loss weighting]`: Apply IPCW weights $1/\hat{G}(t_M \mid X)$ to the scalar loss function rather than scaling target distribution probabilities.
3. `[Step 7 multi-step continuation discount]`: Nest interval survival discount $\gamma_j$ into the recursive continuation term $\lambda_j \gamma_j \Pi \Phi_{+\Delta t_j} G_{j+1}$.
