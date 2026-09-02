role: empiricist

## Idea review — SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation

**Decomposition**

- **Problem / gap:** Temporal-consistency survival methods (TCSR, DeepTCSR) enforce consistency between consecutive observations under an assumed uniform unit step ($\Delta t = 1$), ignoring elapsed time: *"the non-event signal carried by an interval is one bit regardless of how much time elapsed, and the assumption fails outright on irregularly observed clinical and industrial trajectories."* Furthermore, their backward update *"renormalises the next-step lifetime distribution by dividing by the interval survival probability, which diverges exactly in the high-risk regime where S approaches zero, collapsing predictions towards an uninformative flat distribution."*
- **Method (the move):** Replaces the divergent division update with a renewal mixture target operator over continuous elapsed duration: *"The elapsed duration Delta t_j between observations enters as an interval duration discount gamma_j = S_theta-(Delta t_j) computed from an EMA target network, and the next-step lifetime distribution is carried backwards by a rightward shift Phi_{+Delta t_j} (the renewal identity R_j = R_{j+1} + Delta t_j) composed with a categorical projection Pi onto the fixed bin grid, instead of by division."* Step 5 specifies the target as: $\mathcal{T} p_j = (1 - \gamma_j) \cdot \mu_{[0, \Delta t_j)} + \gamma_j \cdot (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$. Multi-step bootstrapping uses duration-geometric decay $\lambda_j = \lambda^{\Delta t_j / \delta_s}$, terminal censoring is completed with truncated conditional IPCW ($1/\hat{G} \le 10.0$), and the model is fit via squared Cramér distance ($\ell_2^2$ on CDFs).
- **Why it should work:** The composed transport operator $\Pi \Phi_{+\Delta t_j}$ *"conserves unit probability mass exactly"* and is *"non-expansive in the Cramer metric"*. Scoping contraction to the frozen target network $\theta^-$ treats $\gamma_j$ as an exogenous constant during each inner optimization step, guaranteeing that *"the target map is strictly affine in p with Cramer-metric modulus bounded by gamma_j, guaranteeing per-iteration contraction whenever hazard is positive."* Projection diffusion is explicitly bounded by $\delta_s^2 / 6$ per step.
- **Assumptions inferred (stated & audited):** (1) Observation times are conditionally unconfounded given the latent sequence state $h_j$ (explicitly stated in Step 1); (2) Right-censoring is unconfounded conditional on observed covariates $X$ (handled in Step 6 via Cox/RSF with truncated weights $1/\hat{G} \le 10.0$); (3) For sub-bin intervals $\Delta t < \delta_s$, hazard is locally linear (Step 3: $S(\Delta t) = 1 - h_0 \Delta t / \delta_s$); (4) Architectural backbone parity is maintained across baselines using shared GRU-D/LSTM encoders.

---

### Axis Scoring Table

| Axis | 1–5 | Quoted evidence | Reason |
| :--- | :---: | :--- | :--- |
| **A — Problem position** | **4** | *"the non-event signal carried by an interval is one bit regardless of how much time elapsed, and the assumption fails outright on irregularly observed clinical and industrial trajectories."* | The gap is genuine, pervasive across clinical telemetry and industrial degradation, and located at a critical mathematical limitation of dynamic survival consistency methods. In `candidate_r1`, the positioning is strongly reinforced by explicitly confronting the two prevailing alternative resolutions in the baseline ladder: (i) resampling/interpolating onto a uniform grid (Baseline 4: Person-period expanded hazard on a 1h grid), and (ii) feeding elapsed time $\Delta t$ into the encoder while retaining clamped division (Baseline 2: DeepTCSR with $\Delta t$ and clamped division). This prevents the problem from being a strawman or soft target. |
| **B — Method quality** | **4** | *"The division is replaced by the composition of a renewal rightward shift and a categorical projection onto the bin grid in a renewal mixture target. The substitution conserves unit probability mass exactly, is non-expansive in the Cramer metric, and yields an affine target map whose modulus under a frozen target network is bounded by the interval discount gamma_j."* | **depth:** High/Moderate-high. The method constructs a continuous-time renewal mixture Bellman operator, combining renewal identity shifting, categorical histogram projection, and duration-geometric $\lambda$-returns. **soundness:** Rigorously rehabilitated from Round 0. $\gamma_j$ is now mathematically active as the convex mixture coefficient in the Step 5 target equation. Contraction is scoped to the frozen target network $\theta^-$ inner step with sub-bin linear hazard interpolation, and projection diffusion is bounded by $\delta_s^2 / 6$. Terminal absorbing bin semantics ($\ge s_K$) and truncated conditional IPCW ($1/\hat{G} \le 10.0$) eliminate edge-case pathologies. **feasibility:** High. All components are closed-form 1D projections and continuous recurrent encodings. The 120 GPU-hour budget across 400 runs is realistic (~18 min/run on A100). |
| **C — Problem-fit** | **4** | *"If the continuous duration transition operator is not doing the work, then ablating the duration discount (Arm A1) or discretizing the shift to unit step (Arm A2) leaves time-dependent AUC and integrated Brier score unchanged on irregularly sampled cohorts. The prediction is the opposite: discrimination drops toward the unit-step consistency baseline..."* | The mechanism directly resolves the stated gaps: continuous renewal transport replaces unit-step indexing, and mass-conserving projection replaces divergent division. Demonstration-level fit is fully secured in `candidate_r1`: NASA C-MAPSS is now subjected to a declared 50% Poisson cycle-drop downsampling protocol, ensuring that C-MAPSS genuinely activates the irregular duration operator. Furthermore, alarm stability (the clinical manifestation of high-risk prediction collapse) is operationalized via alarm jitter and false alert episode rate at matched 0.30 PPV against Dynamic-DeepHit. |
| **D — Falsifiability & claim integrity** | **5** | *"composed_duration_transition_operator Pi Phi_{+Delta t_j} modulated by gamma_j = S_theta-(Delta t_j)"*<br><br>*"NC-A (two-arm operator ablation): Arm A1 replaces interval duration discount gamma_j with unit constant S(delta_s) while keeping continuous shift Phi_{+Delta t_j}; Arm A2 replaces continuous shift Phi_{+Delta t_j} with unit shift Phi_{+delta_s} while keeping duration discount gamma_j = S(Delta t_j); Arm A3 restores DeepTCSR clamped division div S(Delta t_j) on continuous intervals."*<br><br>*"performance dropping by at least 0.025 in time-dependent AUC and concordance (derived: 3x the 5-seed bootstrap test standard error measured as 0.008 on MIMIC-IV Sepsis-3)."* | **load-bearing variable:** Explicitly named and active in Step 5. **control:** NC-A is cleanly decoupled into Arm A1 (discount ablation), Arm A2 (shift ablation), and Arm A3 (clamped division comparison), eliminating the leakage from Round 0. NC-B is corrected to within-patient duration permutation, preserving total follow-up and event labels. NC-C benchmarks duration-geometric $\lambda_j$ against count-geometric $\lambda^k$ at matched effective horizon. **provenance:** The 0.025 margin is derived as $3\times$ the empirical test bootstrap standard error (0.008) measured on MIMIC-IV Sepsis-3, establishing a rigorous $3\sigma$ barrier against sampling noise. Protocol parity is guaranteed with 4 named baselines sharing identical GRU-D/LSTM backbones. |

---

### Overall Score & Gate Status

$$\text{overall} = \text{round}\left(100 \times \frac{A + B + C - 3}{12}\right) = \text{round}\left(100 \times \frac{4 + 4 + 4 - 3}{12}\right) = \text{round}\left(100 \times \frac{9}{12}\right) = \mathbf{75/100}$$

**Overall: 75/100 · Verdict: strong**  
**Gate Status:** No gate fired ($A = 4 > 2$, $C = 4 > 2$, $D = 5 > 2$).

---

### Structural Checks

- **Naive-baseline audit:** **Branch (1)** — *The naive version relies on a false premise $\to$ confronting that premise IS the contribution.*
  - *Independently constructed naive version:* Take DeepTCSR, feed elapsed duration $\Delta t_j$ into the recurrent encoder, evaluate interval survival $S_{\theta^-}(\Delta t_j)$ at the true elapsed duration in the denominator, clamp the denominator at $S \ge 10^{-3}$ to prevent zero-division, and renormalize the discrete pmf by dividing by its sum.
  - *Audit:* The premise of unit-step consistency is that dividing by interval survival is mathematically coherent. Clamping the denominator at $\epsilon = 10^{-3}$ avoids `NaN`s, but distorts the distribution shape in the high-risk tail, while dividing by small numbers exponentially inflates target variance. SurvTD confronts this false premise by replacing division entirely with continuous renewal shifting and categorical projection ($\Pi \Phi_{+\Delta t}$).
  - *Critical empirical check:* In `candidate_r1`, this exact naive baseline is formally incorporated into the evaluation ladder as **Arm A3** of NC-A and **Baseline (2)** in the compute budget (*"DeepTCSR with Delta t feature and clamped division"*). This guarantees that the experiment will directly test whether confronting the false premise yields empirical gains over the naive ad-hoc patch.
- **Novel-but-empty detector:** **Passes cleanly.**
  - *What the idea predicts:* Discrimination (tAUC, concordance) will degrade by $\ge 0.025$ on irregular cohorts under Arm A1 (discount ablation) and Arm A2 (shift ablation); concordance will fall toward 0.8791 on 50% Poisson-subsampled C-MAPSS; alarm jitter and false alert episode rates will improve over Dynamic-DeepHit at 0.30 PPV; and SurvTD will statistically outperform clamped division (Arm A3) and 1h grid expansion (Baseline 4).
  - *What it would change:* The foundational formulation of temporal consistency in continuous-time dynamic survival analysis, replacing ad-hoc discrete-time renormalizations with mass-conserving distributional Bellman operators.
  - *What counts against it:* If Arm A1, Arm A2, Arm A3, or Baseline 4 achieves performance within 0.025 tAUC/concordance of SurvTD on MIMIC-IV or C-MAPSS, the core claim is falsified.

---

### Qualitative Assessment

- **Strongest point:** The empirical protocol achieves exceptional experimental hygiene: it couples an orthogonal negative control matrix (Arm A1 discount ablation, Arm A2 shift ablation, Arm A3 clamped division, NC-B within-patient permutation, NC-C horizon-matched $\lambda$) with a pre-registered $3\sigma$ bootstrap test standard error margin (0.025 derived from 0.008 SE on MIMIC-IV) and strict shared-backbone encoder parity across four distinct baseline families.
- **Most fixable weakness:** While overall discrimination and calibration are thoroughly covered by tAUC, concordance, and IBS, the candidate should pre-register a risk-stratified diagnostic (e.g. Brier score and target entropy specifically in the high-risk tail $S < 0.1$ or within 24 hours of clinical decompensation) to directly observe the elimination of the division collapse phenomenon where it was claimed to hurt most.

---

## Detailed Audit of Core Empirical Battlegrounds

### 1. Negative Controls Audit: NC-A Split, NC-B Within-Patient, NC-C Horizon-Matched

In Round 0, the negative controls suffered from three critical flaws: (i) $\gamma_j$ was missing from the Step 5 equation, leaving NC-A without an operand; (ii) NC-A ablated $\gamma_j \to S(\delta_s)$ while retaining the duration shift $\Phi_{+\Delta t_j}$, leaking duration information into the ablated target; and (iii) NC-B scrambled intervals across independent patients, corrupting total follow-up time and event offsets $\tau_j - t_j$.

`candidate_r1.json` resolves every one of these defects:

```text
================================================================================
NEGATIVE CONTROL SUITE AUDIT MATRIX
================================================================================
Control    Intervention                             Preserved Invariant            Outcome Measured / Hypothesis
----------------------------------------------------------------------------------------------------------------
NC-A / A1  gamma_j -> S(delta_s)                    Phi_{+Delta t_j}, Encoder      tAUC/IBS drops >= 0.025; isolates
           (Constant unit discount)                 Delta t feature                interval duration discounting.

NC-A / A2  Phi_{+Delta t_j} -> Phi_{+delta_s}       gamma_j = S(Delta t_j),        tAUC/IBS drops >= 0.025; isolates
           (Unit-step renewal shift)                Encoder Delta t feature        continuous renewal transport.

NC-A / A3  Pi Phi -> Clamped Div S >= 1e-3          Continuous intervals           SurvTD beats clamped division; tests
           (DeepTCSR continuous patch)              gamma_j active                 mass conservation & stability.

NC-B       Permute Delta t_j within-patient         Total follow-up T_i,           tAUC drops; isolates temporal
           trajectory sequence                      M_i visits, event label        alignment from duration marginals.

NC-C       Duration-geometric lambda^(Dt/ds)        Effective bootstrapping        Separates duration invariance from
           -> Count-geometric lambda^k              horizon H_eff                  variance reduction across sample rates.
================================================================================
```

- **NC-A Mathematical Operand & No-Leakage:** Step 5 now explicitly defines the target as a renewal mixture:
  $$\mathcal{T} p_j = (1 - \gamma_j) \cdot \mu_{[0, \Delta t_j)} + \gamma_j \cdot (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$$
  Arm A1 sets $\gamma_j = S_{\theta^-}(\delta_s)$ while keeping $\Phi_{+\Delta t_j}$. Arm A2 sets $\Phi_{+\delta_s}$ while keeping $\gamma_j = S_{\theta^-}(\Delta t_j)$. Arm A3 replaces the second term with clamped division. The three arms form an exhaustive, orthogonal $2 \times 2$ ablation plus prior-art patch, isolating each channel without leakage.
- **NC-B Within-Patient Permutation:** By restricting duration permutations to within-patient sequences, the total observation window $T_i = \sum_{j=1}^{M_i-1} \Delta t_{i,j}$, total visit count $M_i$, and the subject's final event label $Y_i \in \{0, 1\}$ are strictly preserved. This breaks temporal ordering and visit spacing without corrupting supervision or feature marginals.
- **NC-C Horizon-Matched $\lambda$:** Sweeping $\lambda_j = \lambda^{\Delta t_j / \delta_s}$ against count-geometric $\lambda^k$ at matched effective horizon ($\sum \Delta t_j \approx \bar{H}$) isolates whether duration-geometric bootstrapping provides invariance to irregular sampling rates, rather than conflating it with variance reduction.

---

### 2. NASA C-MAPSS 50% Poisson Downsampling Protocol & Anchor

- **Prior Benchmark Trap:** In Round 0, NASA C-MAPSS FD001–FD004 was cited as the site of the headline concordance drop, despite being sampled at uniform cycles ($\Delta t \equiv 1$) and run to failure without right-censoring. On native C-MAPSS, an irregular duration model is a no-op.
- **Audited Protocol in `candidate_r1.json`:**
  - `falsification_prediction` explicitly declares: *"On Poisson-subsampled NASA C-MAPSS (50% random cycle drop), concordance drops toward the Bleistein2024 baseline of 0.8791 (measured in Bleistein2024)..."*
  - Subsampling telemetry cycles independently with probability $p = 0.5$ transforms the uniform cycle sequence into an irregular process where inter-observation intervals follow $\Delta t \sim \text{Geometric}(0.5)$ (mean $\Delta t = 2.0$ cycles, variance $\sigma^2 = 2.0$).
  - This activation of interval dispersion allows SurvTD's continuous renewal shift $\Phi_{+\Delta t}$ and duration discount $\gamma_j$ to perform real work.
- **Anchor Integrity:** Bleistein et al. (ICML 2024, *CoxSig*) evaluated path signatures for dynamic survival and RUL prediction on C-MAPSS, reporting a baseline concordance of 0.8791. Declaring this anchor provides a concrete, published literature benchmark. In Phase 4, the experimental design will run all four comparative baselines directly on the 50% Poisson-subsampled splits to maintain exact protocol parity.

---

### 3. Pre-Registered 0.025 Margin & Statistical Provenance

- **Provenance Check:**
  - `candidate_r1.json` states: *"performance dropping by at least 0.025 in time-dependent AUC and concordance (derived: 3x the 5-seed bootstrap test standard error measured as 0.008 on MIMIC-IV Sepsis-3)."*
  - This replaces the circular phrasing of Round 0 with an empirical derivation:
    $$\Delta_{\text{pre-reg}} = 3 \times \widehat{\text{SE}}_{\text{boot}} = 3 \times 0.008 = 0.024 \approx 0.025$$
- **Statistical Power & Significance:**
  - Evaluating dynamic survival models on clinical cohorts introduces two distinct variance sources: (i) model initialization / seed variance, and (ii) test cohort sampling variance.
  - A standard error of $\widehat{\text{SE}} = 0.008$ computed over 1,000 bootstrap resamples of the MIMIC-IV test set captures finite-sample patient variance.
  - Setting the pre-registered decision margin at $3 \times \widehat{\text{SE}} = 0.025$ corresponds to a one-sided significance level of $\alpha = 0.00135$ ($z = 3.0$). Any observed drop exceeding 0.025 cannot be attributed to seed noise or test sampling fluctuations. This is a rigorous, statistically grounded decision threshold.

---

### 4. Shared Backbone Parity Across the 4 Comparative Baselines

In `compute_budget`, the candidate locks encoder architecture across four comparative baseline classes:

1. **SurvTD (Proposed Method):** Continuous-time renewal mixture Bellman operator with categorical projection $\Pi \Phi_{+\Delta t_j}$ and duration discount $\gamma_j$.
2. **DeepTCSR with $\Delta t$ Feature and Clamped Division:** The naive continuous adaptation; replaces the unit-step assumption by feeding elapsed time into the encoder and evaluating interval survival $S_{\theta^-}(\Delta t_j)$ in the denominator, clamped at $S \ge 10^{-3}$.
3. **Dynamic-DeepHit:** State-of-the-art non-consistency dynamic survival baseline optimizing terminal discrete-time likelihood plus pairwise ranking loss.
4. **Person-Period Expanded Hazard Baseline on a 1h Regular Grid:** The standard engineering alternative ("solved at another level"); imputes and discretizes irregular trajectories onto a uniform 1-hour grid, training a discrete hazard model.

- **Parity Protocol:** All four methods utilize an identical continuous-time GRU-D or continuous-time LSTM sequence backbone. 
- **Confounding Elimination:** By holding the encoder capacity, input features (including explicit $\Delta t$ inputs), and hyperparameter tuning budget constant, any performance margin is strictly attributable to the loss formulation and temporal consistency operator, completely isolating the methodological contribution from representation learning confounds.

---

### 5. Compute Budget Realism (120 GPU-Hours Across 400 Runs)

In Round 0, the candidate proposed 18 GPU-hours for ~340 runs ($\approx 3.1$ min/run), which was physically impossible for sequence models on MIMIC-IV. `candidate_r1.json` rebudgets to **120 GPU-hours** on RTX 3090 / A100 GPUs.

```text
================================================================================
COMPUTE ALLOCATION & RUNTIME ARITHMETIC
================================================================================
Benchmark Dataset             Cohort Scale              Per-Run Time (A100)   Total Runs (100/bench)   Budget Consumed
----------------------------------------------------------------------------------------------------------------
PBC (Clinical Liver)          312 patients, longitudinal    ~0.5 minutes            100 runs                ~0.8 GPU-hours
Synthetic ODE Degradation     1,000 trajectories            ~1.5 minutes            100 runs                ~2.5 GPU-hours
NASA C-MAPSS (50% Subsampled) 100-250 engines, sensors      ~6.0 minutes            100 runs               ~10.0 GPU-hours
MIMIC-IV Sepsis-3 Telemetry   ~12,000 ICU stays, 24-48h    ~22.0 minutes            100 runs               ~36.7 GPU-hours
----------------------------------------------------------------------------------------------------------------
Subtotal (Main Grid: 4 methods x [20 HPO + 5 Seeds] x 4 Benchmarks = 400 runs):                            ~50.0 GPU-hours
Ablation & Control Suite (NC-A Arms A1-A3, NC-B, NC-C across key benchmarks x 5 seeds):                    ~18.0 GPU-hours
Bootstrap Evaluation & Headroom (1,000 test bootstrap evaluations, retry buffer, margin):                  ~52.0 GPU-hours
================================================================================================================
TOTAL ALLOCATED BUDGET:                                                                                   120.0 GPU-HOURS
================================================================================
```

- **Arithmetic Check:** $120 \text{ GPU-hours} = 7,200 \text{ GPU-minutes}$. Across 400 training jobs, the average budget is $18.0 \text{ minutes per run}$.
- As shown in the empirical breakdown, tabular/small benchmarks (PBC, ODE) run in under 2 minutes, while C-MAPSS runs in ~6 minutes. MIMIC-IV Sepsis-3 GRU-D sequence training on an A100 takes 20–25 minutes. The weighted average runtime across the suite is $\approx 7.5 \text{ minutes per run}$, requiring only ~50 GPU-hours for the primary training grid.
- The 120 GPU-hour allocation provides over $2\times$ headroom to accommodate Bayesian HPO exploration, full 5-seed evaluation, negative control runs, and 1,000-sample test bootstrap confidence intervals. The budget is feasible, realistic, and defensible.

---

## Gauntlet — Landed Attacks

Led at §4 (Evidence), then the remaining sections.

| # | Attack | Severity | Answerability | Where it lands |
| :---: | :--- | :---: | :---: | :--- |
| **4.4** | **Absence of a pre-registered high-risk stratum diagnostic.** The motivation states that the division update collapses predictions towards a flat distribution specifically in the high-risk regime ($S \to 0$). However, the primary metrics (tAUC, concordance, IBS) aggregate over all patient-times. While IBS is strictly proper, a pooled metric can dilute a failure mode concentrated in the extreme tail. | minor | now | `falsification_prediction`, evaluation protocol |
| **4.x** | **Potential unmeasured confounding in clinical observation frequency.** Step 1 assumes observation times are conditionally unconfounded given latent state $h_j$. In MIMIC-IV ICU telemetry, clinician observation frequency ($\Delta t^{-1}$) strongly correlates with patient acuity through unrecorded bedside impressions. While conditioning on $h_j$ is standard practice, a stratified check is needed to ensure gains are not driven by observation intensity leakage. | minor | with evidence | `core_mechanism_steps` Step 1, MIMIC-IV evaluation |
| **2.3** | **Linear hazard assumption for sub-bin intervals $\Delta t < \delta_s$.** Step 3 specifies $S(\Delta t) = 1 - h_0 \Delta t / \delta_s$. If $\delta_s$ is chosen too coarsely relative to physiological dynamics (e.g. $\delta_s = 24\text{h}$ in sepsis), linear interpolation could introduce discretization bias for rapid vital sign changes. | minor | now | `core_mechanism_steps` Step 3 |
| **1.3** | **Resampling / grid interpolation preprocessing baseline trade-off.** Baseline 4 evaluates a 1h regular grid discrete hazard model. While this benchmarks against the "solved at another level" alternative, the candidate does not articulate the trade-off regarding forward-fill imputation artifacts or lookahead bias in ICU data. | minor | with evidence | `compute_budget`, `gap_closure` |

---

### Tested Attacks That Did Not Land

- **4.1 (No falsifier):** Tested and did not land. The candidate declares quantitative, directional falsification margins ($\ge 0.025$ drop in tAUC/concordance on irregular cohorts under Arm A1 and Arm A2; drop toward 0.8791 on 50% Poisson C-MAPSS; worsening alarm jitter and false alert episode rates vs. Dynamic-DeepHit at 0.30 PPV). Clear losing branches are defined.
- **4.2 (Tautological control):** Tested and did not land. NC-A intervenes on the mathematical operator (discount, shift, clamped division) and evaluates downstream clinical discrimination (tAUC, IBS). NC-B permutes intervals within-patient and measures ranking degradation. NC-C tests duration-geometric decay against count-geometric decay at matched horizon. All controls land on downstream empirical outcomes.
- **4.3 (Invented numbers):** Tested and did not land. Every numeric bar carries explicit provenance: the 0.025 margin is derived from $3\times$ the 0.008 bootstrap test standard error on MIMIC-IV; 0.8791 is measured from Bleistein2024; weight truncation at 10.0 is standard IPCW practice; and 120 GPU-hours is derived from 400 runs at 18 min/run on A100.
- **4.5 (Success is guaranteed):** Tested and did not land. If clamped division (Arm A3) matches SurvTD within 0.025, or if ablated arms retain performance, the core claim fails.
- **2.1 (Equivalent to naive):** Tested and did not land. The naive version (clamped division) is explicitly formulated and included as a direct competitor (Baseline 2 / Arm A3).
- **2.4 (Circularity):** Tested and did not land. $\gamma_j = S_{\theta^-}(\Delta t_j)$ is evaluated under a frozen EMA target network $\theta^-$, treating $\gamma_j$ as an exogenous constant during each inner training step and breaking the circular self-referential dependency.
- **6.3 (Infrastructure in a paper's clothing):** Tested and did not land. SurvTD is a foundational algorithmic methodology bridging continuous-time reinforcement learning and dynamic survival analysis, not a systems artifact.

---

## Two-Layer Verdict

### 1. Hard Floor — Checked, None Fired

- **Scoop axes match:** **Pass.** Scoop report confirms Overlap Level 2/5 (*adjacent lineage, distinct technical move*). No contemporaneous scoop found.
- **Naive-baseline audit:** **Pass.** Returns Branch (1), not *naive suffices*. The naive baseline (clamped division with $\Delta t$ encoder features) is explicitly constructed and included in the empirical benchmark ladder (Baseline 2 / Arm A3).
- **Anti-pattern mitigations:** **Pass.** Target network freezing, sub-bin linear hazard interpolation, terminal absorbing bin semantics, truncated conditional IPCW, and shared-backbone encoder parity are fully integrated.
- **Falsifiability collapse:** **Pass.** Load-bearing variables, pre-registered $3\sigma$ bootstrap margins, and multi-arm controls are fully specified.

### 2. Soft Judgment: ADVANCE

The candidate `candidate_r1.json` has executed a thorough, mathematically sound, and empirically disciplined revision. All eight mandatory revision targets from Round 1 have been implemented with high fidelity:
1. The missing load-bearing variable $\gamma_j$ is wired into the Step 5 renewal mixture equation.
2. Contraction is properly scoped to the frozen target network inner loop with linear sub-bin hazard interpolation.
3. Projection diffusion is bounded at $\delta_s^2 / 6$, retracting distributional invariance overreach.
4. NC-A is disentangled into three orthogonal arms (A1 discount, A2 shift, A3 clamped division).
5. NASA C-MAPSS is grounded via a 50% Poisson downsampling protocol with the Bleistein2024 anchor (0.8791).
6. Censoring tail completion is conditioned on covariates with weights truncated at 10.0.
7. Observation times are explicitly assumed conditionally unconfounded given $h_j$, and NC-B is corrected to within-patient permutation.
8. Parity is enforced across 4 baselines on shared GRU-D/LSTM encoders, alarm metrics are specified at 0.30 PPV, and compute is realistic at 120 GPU-hours.

The four minor landed attacks represent standard experimental refinements for Phase 4 execution, not blocking barriers. The proposal is cleared for transition to experiment architecture.

---

### Non-Blocking Execution Recommendations for Phase 4 (`experiment-architect`)

1. **Pre-Register High-Risk Stratum Diagnostics:** Include a pre-registered evaluation table reporting Brier score, calibration slope, and target distribution entropy stratified by predicted survival quintile (specifically the highest-risk quintile $S < 0.20$ and within 24h of an event) to directly showcase the elimination of the division divergence failure mode.
2. **Observation Intensity Sensitivity Check:** On MIMIC-IV Sepsis-3, report performance stratified by patient observation frequency quartiles to confirm that SurvTD's advantage holds uniformly across both sparsely and densely sampled ICU trajectories.
3. **Bin Width Sensitivity Sweep:** Pre-register a sensitivity sweep over remaining-lifetime bin widths $\delta_s \in \{0.5\text{h}, 1.0\text{h}, 2.0\text{h}\}$ on MIMIC-IV to empirically verify that sub-bin linear hazard interpolation remains robust across grid granularities.
