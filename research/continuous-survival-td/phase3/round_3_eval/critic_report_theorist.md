role: theorist

## Idea review — SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation

**Candidate Evaluated**: `research/continuous-survival-td/phase3/round_2_revision/candidate_r2.json`  
**Scoop Report**: `research/continuous-survival-td/phase3/round_1_eval/scoop_report.md`  
**Seat**: Theoretician (`theorist`) — Round 3 Evaluation  

---

### Decomposition

- **Problem / gap**:
  1. Dynamic survival models that enforce temporal consistency between consecutive observations (TCSR, DeepTCSR) inherit an artificial unit-step transition ($\Delta t = 1$), treating non-events as a 1-bit signal regardless of elapsed physical time.
  2. The backward update of existing consistency methods renormalises the next-step lifetime distribution by dividing by interval survival probability $S(\Delta t)$, which diverges asymptotically towards infinity as risk rises ($S \to 0$), collapsing predictions towards an uninformative uniform distribution.
- **Method (the move)**:
  Substitute the division operator with an inter-observation renewal transport operator composed of a renewal rightward shift $\Phi_{+\Delta t_j}$ ($R_j = R_{j+1} + \Delta t_j$) and a triangular categorical projection $\Pi$ onto a fixed remaining-lifetime grid, discounted by model-predicted interval survival $\gamma_j = S_{\theta^-}(\Delta t_j)$ evaluated from a frozen EMA target network $\theta^-$. Target evaluation separates into continuation updates on surviving visits and projected Diracs on terminal event visits. Multi-step returns are compounded via a backward recursion using duration-geometric weights $\lambda_j = \lambda^{\Delta t_j / \delta_s}$ and continuation discount $\gamma_j$, trained via squared Cramér loss with scalar truncated conditional IPCW tail completion.
- **Why it should work (formal argument)**:
  - $\Pi \Phi$ is an $\ell_2$-orthogonal projection of a rigid translation, making it strictly non-expansive in the Cramér ($\ell_2$-on-CDF) metric and strictly probability-mass-conserving.
  - Freezing $\theta^-$ treats $\gamma_j$ as an exogenous scalar in $[0, 1)$, ensuring the conditional-expectation renewal mixture operator is affine in $p$ with contraction modulus $\gamma_j < 1$ for any positive hazard $h > 0, \Delta t_j > 0$.
  - IPCW weights applied as scalar sample loss multipliers prevent defective distribution targets while asymptotically unbiasing the right-censored tail.
  - Multi-step compounding $\prod \gamma_j$ reflects continuous survival probability across arbitrarily sampled trajectory sub-chains, while $\lambda^{\Delta t / \delta_s}$ preserves physical bootstrapping horizon invariance.
- **Assumptions audited & verified**:
  - Observation timestamps are conditionally unconfounded given latent recurrent state $h_j$ (stated in Step 1).
  - Target network $\theta^-$ is frozen during inner-loop TD updates to guarantee exogenous $\gamma_j$ contraction (stated in Step 3).
  - Sub-bin intervals $\Delta t < \delta_s$ employ linear hazard interpolation $S(\Delta t) = 1 - h_0 \Delta t / \delta_s$ to preserve well-defined interval discounts (stated in Step 3).
  - Right-censoring is conditionally unconfounded given baseline/dynamic covariates $X$ and handled via conditional IPCW models $\hat{G}(t \mid X)$ (stated in Step 6).
  - Terminal bin carries $\ge s_K$ defective mass semantics, absorbing super-horizon probability mass without loss of unit mass (stated in Step 4).

---

### In-Depth Mathematical Audit of Round 2 Revisions

#### 1. Step 5 Sample Realization Branching & Conditional Renewal Mixture Contraction
- **Sample Realization**: In `candidate_r1`, the target conflated the sample realization with the mixture expectation by asserting that $\mu_{[0, \Delta t_j)}$ was supervised by within-interval event offsets on all transitions. In `candidate_r2`, the sample update is correctly branched:
  - On non-event intervals ending in an alive visit at $t_{j+1}$, $E_j = 0$; the sample target is strictly the continuation branch $(\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$.
  - On terminal event intervals ending in death at $\tau \in [t_j, t_{j+1}]$, $E_j = 1$; the sample target evaluates at the localized intra-interval projected Dirac $\Pi \delta_{\tau - t_j}$.
- **Conditional Expectation**: Taking the conditional expectation of this sample branching with respect to the filtration $\mathcal{H}_{t_j}$:
  $$\mathbb{E}[T_{\text{sample}} p_j \mid \mathcal{H}_{t_j}] = (1 - \gamma_j) \mu_{[0, \Delta t_j)} + \gamma_j (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$$
  where $\gamma_j = \mathbb{P}(T \ge t_{j+1} \mid T \ge t_j, \mathcal{H}_{t_j}) = S_{\theta^-}(\Delta t_j)$, and $\mu_{[0, \Delta t_j)}$ is the conditional event-time distribution over the interval.
- **Contraction Proof**: For two candidate distributions $p, q$ over remaining lifetime with CDFs $F_p, F_q$:
  $$(T p - T q) = \gamma_j \left( \Pi \Phi_{+\Delta t_j} p - \Pi \Phi_{+\Delta t_j} q \right)$$
  In the Cramér metric $\ell_2(F_p, F_q) = \left( \int_0^\infty |F_p(s) - F_q(s)|^2 ds \right)^{1/2}$:
  $$\| F_{Tp} - F_{Tq} \|_{\ell_2} = \gamma_j \| F_{\Pi \Phi p} - F_{\Pi \Phi q} \|_{\ell_2} \le \gamma_j \| F_p - F_q \|_{\ell_2}$$
  Because $\theta^-$ is frozen, $\gamma_j$ is an exogenous scalar. For positive hazard ($h > 0$) and non-zero duration ($\Delta t_j > 0$), $\gamma_j = S_{\theta^-}(\Delta t_j) < 1$. Thus, $T$ is an affine strict contraction mapping with modulus $\gamma_j < 1$.
  *(Note on C51 contrast: Unlike C51 where scalar discounting scales the support yielding modulus $\sqrt{\gamma}$ in $\ell_2$, here $\gamma_j$ is a vertical mixture weight over a shifted support, so it enters linearly as modulus $\gamma_j$.)*

#### 2. Step 6 Truncated Conditional IPCW Loss Weighting & Mass Preservation
- **Mass Preservation**: In `candidate_r1`, scaling the target probability vector by $1 / \hat{G} \le 10.0$ resulted in $\sum_k p_k \le 10.0$, breaking the unit-simplex property and rendering $F_{G}$ an improper, uncalibrated function. In `candidate_r2`, the target tail is completed using target network predictions $p_{\theta^-}$, which strictly satisfy $\sum_k p_{\theta^-, k} = 1$.
- **Loss Weighting Formulation**: The IPCW weight $w_i = \min(1 / \hat{G}(t_M \mid X_i), 10.0)$ is applied as an exogenous scalar weight in the empirical squared Cramér risk:
  $$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^{M_i} w_{i,j} \int_0^{s_K} \left| F_\theta(s \mid \mathcal{H}_{t_j}^{(i)}) - F_{G_j^{(i)}}(s) \right|^2 ds$$
  where $w_{i,j} = 1.0$ for all internal transitions $j < M_i$, and $w_{i, M_i} = \min(1 / \hat{G}(t_{M_i} \mid X_i), 10.0)$ on right-censored terminal visits.
  Both $F_\theta$ and $F_{G_j}$ are mathematically proper CDFs on $[0, s_K]$, conserving unit mass identically. The estimator is standard, unbiased Horvitz–Thompson risk reweighting under conditional unconfounded censoring ($C \perp T \mid X$).

#### 3. Step 7 Multi-Step Continuation Discount $\gamma_j$ Compounding
- **Backward Recursion**: In `candidate_r1`, the recursive continuation branch omitted $\gamma_j$, setting $G_j = (1 - \lambda_j) T p_j + \lambda_j \Pi \Phi G_{j+1}$, which falsely implied zero mortality across chained visits as $\lambda \to 1$.
- In `candidate_r2`, Step 7 correctly compounds the continuation discount:
  $$G_j = (1 - \lambda_j) T p_j + \lambda_j \gamma_j \Pi \Phi_{+\Delta t_j} G_{j+1}$$
  Expanding over an $m$-step sequence of visits $t_j, t_{j+1}, \dots, t_{j+m}$:
  The cumulative continuation multiplier applied to $G_{j+m}$ is:
  $$\prod_{l=0}^{m-1} \left( \lambda_{j+l} \gamma_{j+l} \right) = \left( \lambda^{\sum_{l=0}^{m-1} \Delta t_{j+l} / \delta_s} \right) \left( \prod_{l=0}^{m-1} S_{\theta^-}(\Delta t_{j+l}) \right) = \lambda^{(t_{j+m} - t_j)/\delta_s} \cdot S_{\theta^-}(t_{j+m} - t_j)$$
  Both the bootstrapping eligibility horizon ($\lambda^{\Delta t / \delta_s}$) and cumulative patient survival probability ($S(t_{j+m} - t_j)$) compound consistently in continuous physical time.

#### 4. Non-Expansiveness of $\Pi \Phi$ and Bounded Projection Diffusion
- **Translation Isometry**: $\Phi_{+\Delta t}$ shifts CDFs rigidly: $F_{\Phi_{+\Delta t} p}(s) = F_p(s - \Delta t) \mathbb{1}[s \ge \Delta t]$. By substitution of variables $u = s - \Delta t$, $\| F_{\Phi_{+\Delta t} p} - F_{\Phi_{+\Delta t} q} \|_{\ell_2} = \| F_p - F_q \|_{\ell_2}$. It is an exact isometry.
- **Categorical Projection**: $\Pi$ is an $\ell_2$-orthogonal projection onto the convex set of piecewise-linear CDFs supported on grid $\{s_k\}$. Hence, $\|\Pi F_1 - \Pi F_2\|_{\ell_2} \le \|F_1 - F_2\|_{\ell_2}$.
- **Horizon Clipping**: Mass shifted past $s_K$ is accumulated in the terminal bin ($\ge s_K$). For $s \ge s_K$, $F_1(s) = F_2(s) = 1$, so $|F_1(s) - F_2(s)| = 0$. Clipping strictly contracts or maintains $\ell_2$ distance. Therefore, $\Pi \Phi$ is strictly non-expansive.
- **Projection Variance Diffusion**: For an arbitrary off-grid translation $\Delta t$, let fractional offset $u = (s + \Delta t) \pmod{\delta_s} \in [0, \delta_s)$. Linear interpolation onto adjacent bin centers preserves the first moment ($\mathbb{E}_\Pi[S] = s + \Delta t$) while adding variance $\Delta \sigma^2 = u(\delta_s - u) \le \delta_s^2 / 4$. Integrating over uniform fractional offsets $u \sim \text{Unif}(0, \delta_s)$:
  $$\mathbb{E}[\Delta \sigma^2] = \frac{1}{\delta_s} \int_0^{\delta_s} u(\delta_s - u) du = \frac{\delta_s^2}{6}$$
  This analytically proves the candidate's declared diffusion bound of $\delta_s^2 / 6$ per step.

---

### Four Axes Scorecard

| Axis | Score | Quoted Evidence from `candidate_r2.json` | Theoretical Audit & Reason |
| :--- | :---: | :--- | :--- |
| **A — Problem Position** | **4** | *"Dynamic survival models that enforce temporal consistency between consecutive observations inherit a unit-step transition (Delta t = 1): surviving one step is a one-bit event and the backward update renormalises the next-step lifetime distribution by dividing by the interval survival probability, which diverges as risk rises. SurvTD audits that assumption and replaces the offending operator."* | Real, non-obvious, and technically critical gap. Pinpointing that division $\div S(\Delta t)$ is the singular mathematical failure mode that destabilizes consistency models in high-risk regimes is an authentic and impactful diagnosis. Kept at 4 because temporal consistency is a specific subfield paradigm within dynamic survival rather than a universal bottleneck of standard likelihood models. |
| **B — Method Quality** | **4** | *"The elapsed duration Delta t_j between observations enters as an interval duration discount gamma_j = S_theta-(Delta t_j) computed from an EMA target network, and the next-step lifetime distribution is carried backwards by a rightward shift Phi_{+Delta t_j}... composed with a categorical projection Pi onto the fixed bin grid, instead of by division. The composed operator Pi Phi is non-expansive in the Cramer metric and conserves unit probability mass exactly..."*<br><br>*"Step 5: Form the one-step target: on non-event intervals ending in alive visits, the sample update evaluates along the continuation branch (Pi Phi_{+Delta t_j} p_theta-)_j, corresponding in conditional expectation to the renewal mixture T p_j = (1 - gamma_j) * mu_{[0, Delta t_j)} + gamma_j * (Pi Phi_{+Delta t_j} p_theta-)_j, ensuring that under a frozen target network theta-, the target map is affine in p with Cramer-metric modulus bounded by gamma_j < 1 for positive hazard."*<br><br>*"Step 6: ...applying inverse probability of censoring weighting 1 / G_hat(t_M\|X) <= 10.0 from a covariate-conditional model as a scalar sample weight in the squared Cramer loss rather than scaling distribution probabilities, preserving unit probability mass exactly."*<br><br>*"Step 7: Build the multi-step target by backward recursion G_j = (1 - lambda_j) * T p_j + lambda_j * gamma_j * Pi Phi_{+Delta t_j} G_{j+1} evaluated with duration-geometric mixing weight lambda_j = lambda^(Delta t_j / delta_s), compounding interval survival discount gamma_j along multi-step chains and maintaining effective bootstrapping horizon invariance while bounding per-step projection diffusion by delta_s^2 / 6."* | **depth: 4** — Genuine structural reframing: translating remaining lifetime $R_j = R_{j+1} + \Delta t$ under a renewal shift and duration discount $\gamma_j = S_{\theta^-}(\Delta t_j)$, yielding an affine contraction with modulus $\gamma_j$ in Cramér metric. The projection operator itself is imported from C51 / Rowland et al., keeping depth at 4 per rubric guidance.<br><br>**soundness: 5** — Flawless mathematical execution in `candidate_r2`. All three Round 2 defects (Step 5 sample realization conflation, Step 6 IPCW mass blowup, and Step 7 un-discounted multi-step recursion) have been completely and rigorously resolved. Contraction modulus $\gamma_j < 1$, exact mass conservation, and $\delta_s^2 / 6$ variance diffusion bounds are formally verified.<br><br>**feasibility: 4** — Fully buildable in PyTorch with standard continuous RNN/GRU-D backbones; 120 GPU-hours on RTX 3090/A100 is fully realistic.<br><br>*(Depth dominates B per rubric $\to$ 4).* |
| **C — Problem-Fit** | **5** | *"The division is replaced by the composition of a renewal rightward shift and a categorical projection onto the bin grid in a renewal mixture target. The substitution conserves unit probability mass exactly, is non-expansive in the Cramer metric, and yields an affine target map whose modulus under a frozen target network is bounded by the interval discount gamma_j."* | Direct 1:1 structural correspondence. The identified problem is division explosion $\div S(\Delta t)$ and unit-step rigidity on irregular intervals; the proposed mechanism replaces division with continuous renewal transport $(\Pi \Phi_{+\Delta t}, \gamma_j)$, precisely closing the gap without collateral mathematical anomalies. |
| **D — Falsifiability & Integrity** | **5** | *"load_bearing_variable": "composed_duration_transition_operator Pi Phi_{+Delta t_j} modulated by gamma_j = S_theta-(Delta t_j)"*<br><br>*"negative_control": "NC-A (two-arm operator ablation): Arm A1 replaces interval duration discount gamma_j with unit constant S(delta_s) while keeping continuous shift Phi_{+Delta t_j}; Arm A2 replaces continuous shift Phi_{+Delta t_j} with unit shift Phi_{+delta_s} while keeping duration discount gamma_j = S(Delta t_j); Arm A3 restores DeepTCSR clamped division div S(Delta t_j) on continuous intervals. NC-B (within-patient permutation)... NC-C (bootstrapping horizon)..."*<br><br>*"The prediction is the opposite: discrimination drops toward the unit-step consistency baseline, with performance dropping by at least 0.025 in time-dependent AUC and concordance (derived: 3x the 5-seed bootstrap test standard error measured as 0.008 on MIMIC-IV Sepsis-3). On Poisson-subsampled NASA C-MAPSS (50% random cycle drop), concordance drops toward the Bleistein2024 baseline of 0.8791 (measured in Bleistein2024)..."* | Exemplary falsification design. Load-bearing variable is explicitly isolated. The 3-arm negative control (NC-A) ablates each operator component independently, testing downstream prognostic discrimination (AUC, concordance) and calibration (IBS, alarm jitter). Thresholds have documented provenance (`derived: 3x bootstrap test SE 0.008`, `measured in Bleistein2024`). The claim is strictly calibrated to its argument. |

---

### Score & Gate Arithmetic

$$\text{Overall Score} = \text{round}\left(100 \times \frac{A + B + C - 3}{12}\right) = \text{round}\left(100 \times \frac{4 + 4 + 5 - 3}{12}\right) = \text{round}\left(100 \times \frac{10}{12}\right) = \mathbf{83} / 100$$

- **Score Band**: **`strong`** ($\ge 67$)
- **A / C / D Gate Check**:
  - $A = 4 > 2$
  - $C = 5 > 2$
  - $D = 5 > 2$
  - **Gate Status: PASSED (No cap triggered).**

---

### Structural Checks

1. **Naive-Baseline Audit**:
   - *Independently constructed naive versions*:
     - *Naive Baseline 1 (Encoder-only Delta-t)*: Pass elapsed time $\Delta t$ into the sequential GRU-D encoder, but maintain standard unit-step consistency loss $\div S(1)$ or clamped division $\div \max(S(\Delta t), \epsilon)$.
     - *Naive Baseline 2 (Resampling)*: Interpolate irregular observation trajectories onto a uniform 1-hour grid, applying standard discrete-time survival models.
     - *Naive Baseline 3 (Unconstrained Likelihood)*: Discard inter-visit consistency entirely and train with dynamic cross-entropy / ranking loss (Dynamic-DeepHit).
   - *Classification*: **Branch 1 — The naive version relies on a false premise, and confronting that premise IS the contribution.**
     - The false premise of Naive 1 is that encoder representations can neutralize a mathematically divergent Bellman target; dividing by $S(\Delta t) \to 0$ corrupts the target distribution regardless of feature conditioning.
     - The false premise of Naive 2 is that artificial interpolation respects clinical observation processes without injecting substantial imputation artifacts or computational explosion.
     - The false premise of Naive 3 is that visit-to-visit temporal consistency is unneeded; in deployment, unconstrained models exhibit severe alarm jitter and flickering alerts.
     - SurvTD directly confronts the false premise of division-based consistency by replacing it with mass-conserving renewal transport. The projection machinery is borrowed from distributional RL, but the domain-specific formulation (remaining-lifetime translation, endogenous survival discounting, right-censored tail IPCW) solves a previously open problem.
2. **Novel-but-Empty Detector**:
   - **PASSES**. The idea specifies crisp, quantitative predictions: a drop of $\ge 0.025$ in time-dependent AUC/concordance under operator ablation, degradation toward the 0.8791 baseline on C-MAPSS, and clear failure criteria that would refute the core mechanism.

---

### Summary Judgments

- **Strongest Point**: The formulation of the renewal mixture operator $(\Pi \Phi_{+\Delta t_j}, \gamma_j)$ is now mathematically airtight: the distinction between sample branching and conditional expectation is properly delineated, contraction modulus $\gamma_j < 1$ in Cramér metric is formally established under frozen $\theta^-$, and IPCW loss weighting preserves exact unit probability mass.
- **Most Fixable Weakness**: When implementing the continuous GRU-D encoder across clinical cohorts where multiple diagnostic labs are recorded simultaneously ($\Delta t_j = 0$), explicitly batch or collapse co-timed observations to prevent redundant identity operations ($\Phi_0 = I, \gamma_j = 1$).

---

### Attack Catalog Gauntlet (Leading with §2.3, §2.4, §2.5)

| # | Attack | Severity | Answerability | Where it lands | Audit Verdict & Status |
|---|--------|----------|---------------|----------------|------------------------|
| **2.3** | **Unstated Precondition**: Target contraction requires $\Delta t_j > 0$ and bounded positive hazard; sub-bin interpolation must be defined; censoring unconfoundedness must hold. | minor | now | `core_mechanism_steps` Steps 1, 3, 5, 6 | **Answered & Resolved**. Step 1 explicitly posits conditionally unconfounded observations; Step 3 explicitly specifies linear hazard interpolation for $\Delta t < \delta_s$; Step 5 explicitly bounds contraction modulus by $\gamma_j < 1$ for positive hazard; Step 6 posits covariate-conditional censoring models $\hat{G}(t \mid X)$. All preconditions are formally declared. |
| **2.4** | **Circularity in Target Discount**: If $\gamma_j = S(\Delta t_j)$ is updated online with active network parameters $\theta$, the Bellman operator becomes non-contractive and diverges. | minor | now | `core_mechanism_steps` Steps 3, 8 | **Answered & Resolved**. Step 3 and Step 8 mandate that $\gamma_j = S_{\theta^-}(\Delta t_j)$ is evaluated strictly from a frozen EMA target network $\theta^-$ and trained via semi-gradient updates through $\theta$ only, treating $\gamma_j$ as an exogenous scalar during optimization. |
| **2.5** | **Complexity Mistaken for Depth**: The apparatus (EMA target network, categorical projection, duration-geometric $\lambda$, truncated IPCW) is elaborate; is the core idea real? | minor | now | `core_mechanism` | **Fails / Survives intact**. The one-sentence core contribution—*replacing division by interval survival with a continuous renewal shift and categorical projection discounted by elapsed survival probability*—is genuine and elegant. Each secondary component directly addresses a distinct mathematical constraint (contraction stability, mass conservation, sampling-rate invariance, censoring bias). |
| **1.3** | **Solved at Another Level**: Irregular sampling can be ingested by continuous-time hazard heads without any inter-visit operator. | minor | with evidence | `gap_closure`[0], `compute_budget` | **Answered with evidence**. Unconstrained heads produce severe temporal alarm jitter and alert flickering across sequential patient visits; SurvTD directly benchmarks against Dynamic-DeepHit and discrete hazard baselines to prove alarm stability. |
| **3.3** | **Partial Closure Sold as Full**: Categorical projection adds $\delta_s^2 / 6$ variance per step, so distributional dispersion still depends on observation count over fixed elapsed time. | minor | now | `core_mechanism`, Step 7 | **Answered & Handled**. The candidate explicitly tempers its claim to "effective bootstrapping horizon invariance while bounding per-step projection diffusion by $\delta_s^2 / 6$," accurately documenting the variance tradeoff. |
| **4.3** | **Invented Numbers**: Numeric thresholds in falsification predictions must carry verified provenance. | minor | now | `falsification_prediction` | **Answered & Sourced**. The $\Delta \ge 0.025$ threshold is explicitly derived as $3\times$ the 5-seed bootstrap test standard error (0.008 on MIMIC-IV), and the 0.8791 concordance floor is cited from Bleistein et al. (2024). |

---

### Two-Layer Verdict

#### 1. Hard Floor Audit
- **Scoop check**: Overlap level 2 / 5 in `scoop_report.md` ("adjacent lineage, distinct technical move"). No contemporaneous scoop combining survival analysis with projected distributional Bellman updates. $\to$ **PASS**.
- **Naive-baseline audit**: Classified as Branch 1 (naive version rests on a false premise). Does not return "naive suffices". $\to$ **PASS**.
- **Anti-pattern composition**: The composition (`assumption_audit_and_pivot` $\to$ `architectural_operator_substitution`) has successfully incorporated all required theoretical mitigations. $\to$ **PASS**.
- **Falsification collapse**: Falsification design is rigorous, non-tautological, and fully intact ($D = 5$). $\to$ **PASS**.

**Hard Floor Status**: **PASSED (Floor clear)**.

#### 2. Soft Judgment
In Round 2, this reviewer filed the sole dissenting vote (`REVISE`, score 67) due to three critical mathematical defects:
1. Conflation of sample realizations with conditional expectations in Step 5.
2. Distortion of probability mass via distribution-level IPCW scaling in Step 6.
3. Omission of the continuation survival discount $\gamma_j$ in Step 7 multi-step recursion.

In `candidate_r2.json`, the author has executed a flawless mathematical revision:
- Step 5 cleanly branches into continuation updates on living transitions and projected Diracs on death transitions, proving conditional expectation renewal contraction with modulus $\gamma_j < 1$.
- Step 6 moves IPCW weighting to a scalar loss multiplier, preserving unit probability mass identically.
- Step 7 compounds $\gamma_j$ alongside duration-geometric $\lambda_j$, achieving proper multi-step continuous-time survival discounting.
- Operator non-expansiveness in Cramér metric and $\delta_s^2 / 6$ projection diffusion bounds are analytically solid.

There are no remaining mathematical gaps or unanswerable theoretical attacks. The proposal is formally sound and ready for implementation.

**Final Verdict**: **`ADVANCE`**

---

### Revision Targets for Future Phases (Implementation Advisories)

1. `[Implementation · Step 1 Co-timed Visits]`: In the PyTorch data loader for MIMIC-IV and PBC, ensure simultaneous observation timestamps ($\Delta t_j = 0$) are merged or handled as identity operations to avoid redundant computation.
2. `[Empirical Reporting · Step 7 Diffusion Validation]`: Include an empirical curve measuring target distribution standard deviation across varying synthetic subsampling frequencies to empirically validate the analytical $\sqrt{n} \delta_s / \sqrt{6}$ diffusion bound against the empirical Brier score.
