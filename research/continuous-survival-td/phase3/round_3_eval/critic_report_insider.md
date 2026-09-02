# Idea review — SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation

**Seat**: Subfield Insider (`role: insider`)  
**Evaluation Lens**: Lineage tracking across TCSR/DeepTCSR and C51/SMDP, structural vs. lexical delta, and honest positioning.  
**Evaluated Candidate**: `research/continuous-survival-td/phase3/round_2_revision/candidate_r2.json`  
**Scoop Reference**: `research/continuous-survival-td/phase3/round_1_eval/scoop_report.md`  

---

## Decomposition

- **Problem / gap**: Dynamic survival analysis models that enforce temporal consistency between consecutive longitudinal observations (such as TCSR and DeepTCSR) are fundamentally restricted to an exogenous uniform unit step ($\Delta t = 1$). Under irregular observation intervals in real-world clinical and telemetry streams, the non-event evidence accumulated over variable durations is discarded. Furthermore, the backward update operator in prior consistency methods renormalizes the next-step lifetime distribution by dividing by the interval survival probability $S(\Delta t)$, which diverges as hazard accumulates or elapsed duration increases ($S \to 0$), collapsing predictions in high-risk regimes.
- **Method (the move)**: Replaces the divergent division operator with an empirical renewal sample-branching Bellman update defined over continuous elapsed durations $\Delta t_j$. On alive intervals, the update executes a continuation rightward renewal shift $\Phi_{+\Delta t_j}$ composed with a categorical projection $\Pi$ onto a fixed grid; on death intervals, it evaluates at an intra-interval projected Dirac delta $\delta_{\tau_j - t_j}$. The interval survival probability enters as an endogenous duration discount $\gamma_j = S_{\theta^-}(\Delta t_j)$ evaluated via a frozen EMA target network $\theta^-$. Terminal right-censoring is resolved by completing the tail with target-network predictions and applying inverse probability of censoring weighting (IPCW) as a scalar loss weight ($\le 10.0$) rather than scaling distribution probabilities. Multi-step target chains are compounded with duration-geometric mixing $\lambda_j = \lambda^{\Delta t_j / \delta_s}$, preserving bootstrapping horizon invariance across irregular sampling rates.
- **Why it should work**: The renewal shift reflects the physical identity of remaining lifetime ($R_j = R_{j+1} + \Delta t_j$ conditional on survival). Composed with the categorical projection $\Pi$, the operator $\Pi \Phi_{+\Delta t_j}$ is non-expansive in the Cramér ($L_2$ on CDFs) metric and strictly mass-conserving. In conditional expectation, sample branching yields the renewal mixture operator $\mathcal{T} p_j = (1 - \gamma_j)\mu_{[0, \Delta t_j)} + \gamma_j \Pi \Phi_{+\Delta t_j} p_j$. Freezing $\theta^-$ makes the update map strictly affine with contraction modulus bounded by $\gamma_j < 1$, eliminating division blow-up entirely.
- **Assumptions inferred**: 
  1. Observation times $t_j$ are conditionally unconfounded given the latent state $h_j$ emitted by the continuous-time recurrent encoder.
  2. Uninformative right-censoring given covariates $X$, such that censoring hazards are consistently estimated by $\hat{G}(t|X)$.
  3. Survival probability within sub-bin intervals $\Delta t < \delta_s$ is reasonably approximated by linear hazard interpolation.
  4. The support of remaining lifetime is sufficiently covered by $K$ bins of width $\delta_s$, with residual mass absorbed into bin $K$.

---

## Subfield Lineage & Architectural Audits

As a subfield insider steeped in both deep survival analysis (Dynamic-DeepHit, TCSR, DeepTCSR, CoxSig) and distributional reinforcement learning (C51, QR-DQN, continuous-time SMDPs), I subject the mathematical architecture of SurvTD to a forensic lineage audit.

### 1. Step 5 Empirical Sample Branching vs. TCSR Unit-Step Division

The central pathological failure mode of prior temporal consistency survival methods (TCSR, Maystre et al., NeurIPS 2022; DeepTCSR, 2024) stems from their analytical inversion formulation:
$$\text{TCSR Update: } S_t(s) \leftarrow \frac{S_{t+1}(s - 1)}{S_t(1)}$$
When generalized to continuous elapsed duration $\Delta t$, this becomes an analytical division by the interval survival probability: $p_t \leftarrow p_{t+\Delta t} / S_t(\Delta t)$. In any dynamic cohort where patients deteriorate rapidly (e.g., septic shock in ICU) or where observation gaps $\Delta t$ are wide, $S_t(\Delta t) \to 0$. In DeepTCSR, this causes catastrophic gradient explosion unless artificially suppressed via a heuristic clamp $\max(S_t(\Delta t), \epsilon)$, which destroys gradient fidelity, distorts the distribution, and violates total probability conservation.

**SurvTD Step 5 fundamentally resolves this lineage defect.** Rather than attempting an analytical inversion via division, Step 5 reformulates the consistency backup as an **empirical sample-path temporal-difference update**:
- **On alive continuation visits** ($C_{j+1} = 0$, patient observed alive at $t_{j+1}$): The realized empirical outcome is survival over $[t_j, t_{j+1})$. Conditional on survival, remaining lifetime satisfies the renewal identity $R_j = \Delta t_j + R_{j+1}$. The empirical backup evaluates strictly along the continuation branch:
  $$\mathcal{T}_{\text{sample}} p_j = \Pi \Phi_{+\Delta t_j} p_{\theta^-}(t_{j+1})$$
- **On terminal death visits** ($E_j = 1$, death observed at intra-interval time $\tau_j \in [t_j, t_{j+1})$): The exact remaining lifetime is observed as $\tau_j - t_j$. The target evaluates at the localized intra-interval Dirac:
  $$\mathcal{T}_{\text{sample}} p_j = \Pi \delta_{\tau_j - t_j}$$

**Expectation Equivalence:** In conditional expectation over the interval outcome, death occurs in $[t_j, t_{j+1})$ with probability $1 - \gamma_j$ (distributed according to interval hazard $\mu_{[0, \Delta t_j)}$), and survival occurs with probability $\gamma_j = S_{\theta^-}(\Delta t_j)$. Taking the conditional expectation of the empirical sample branch yields:
$$\mathbb{E}[\mathcal{T}_{\text{sample}} p_j \mid h_j] = (1 - \gamma_j) \mu_{[0, \Delta t_j)} + \gamma_j \Pi \Phi_{+\Delta t_j} p_{\theta^-}(t_{j+1})$$
This renewal mixture target is strictly affine in $p$, completely avoids dividing by $S(\Delta t_j)$, and possesses a Cramér contraction modulus bounded by $\gamma_j < 1$. This is a profound structural breakthrough: it transforms a fragile, unstable analytical inversion into a standard, robust Robbins-Monro / TD empirical sample backup.

### 2. Step 6 Scalar Loss IPCW Weighting vs. Probability Scaling

Right-censoring is the defining challenge of survival analysis. At a terminal observation $t_M$ where a patient is censored alive, the future trajectory is unobserved. A naive attempt to adapt inverse probability of censoring weighting (IPCW) to distributional models is to scale the target distribution probabilities directly: $p_k \leftarrow p_k / \hat{G}(t_M | X)$.
- **Why probability scaling fails:** Scaling distribution atoms by $1 / \hat{G}$ directly violates the axioms of probability: $\sum_k p_k = 1 / \hat{G} > 1$. The resulting object is no longer a probability distribution on the simplex, which destroys the non-expansion property of the Cramér metric and causes divergent loss gradients.
- **SurvTD's mathematical resolution in Step 6:** SurvTD preserves distribution integrity by:
  1. Completing the unobserved post-censoring tail using the target network prediction $p_{\theta^-}(t_M)$, maintaining $\sum_k p_k = 1$ exactly.
  2. Injecting the IPCW correction externally as a **scalar sample loss weight** on the squared Cramér distance:
     $$\mathcal{L}_j = \min\left(\frac{1}{\hat{G}(t_M | X)}, 10.0\right) \cdot \ell_2^2\left(F_{\theta}(t_j), F_{G_j}\right)$$
This aligns with classical M-estimation theory (Robins & Rotnitzky, 1992): the statistical unbiasedness of the empirical risk under right-censoring is achieved through sample reweighting in the loss, while the target $F_{G_j}$ remains a proper, mass-conserved probability distribution at all times. Truncating at $10.0$ effectively shields the optimization from high-weight variance spikes in sparse follow-up tails.

### 3. Step 7 Continuation Discount Compounding

In multi-step bootstrapping under irregular observations, candidate_r2 specifies:
$$G_j = (1 - \lambda_j) \mathcal{T} p_j + \lambda_j \gamma_j \Pi \Phi_{+\Delta t_j} G_{j+1}, \quad \text{with } \lambda_j = \lambda^{\Delta t_j / \delta_s}$$
- **Multi-step survival discount compounding:** When unrolled over $m$ consecutive intervals with elapsed durations $\Delta t_j, \Delta t_{j+1}, \dots, \Delta t_{j+m-1}$, the continuation discount factors compound multiplicatively:
  $$\prod_{k=0}^{m-1} \gamma_{j+k} = \prod_{k=0}^{m-1} S_{\theta^-}(\Delta t_{j+k}) = S_{\theta^-}\left(\sum_{k=0}^{m-1} \Delta t_{j+k}\right) = S_{\theta^-}(t_{j+m} - t_j)$$
  Under the Markov property of the continuous latent state $h$, the product of interval survival probabilities telescopes exactly to the continuous survival probability over the multi-step window.
- **Sampling-rate invariance:** Standard count-geometric mixing $\lambda^k$ causes the effective bootstrapping horizon to shrink dramatically when observations are frequent, and expand excessively when observations are sparse. SurvTD's duration-geometric parameterization $\lambda_j = \lambda^{\Delta t_j / \delta_s}$ guarantees that the continuous time half-life of the bootstrapping target remains constant ($H = \delta_s / (1 - \lambda)$), independent of the irregular sampling density.
- **Bounded variance diffusion:** By establishing that categorical projection on a uniform grid of width $\delta_s$ introduces a localized variance diffusion of at most $\delta_s^2 / 6$ per step, Step 7 demonstrates rigorous theoretical control over the multi-step projection accumulation error.

### 4. Differentiation from C51 (Bellemare et al., 2017) and Bradtke SMDP (1994)

The candidate's positioning against its two primary RL ancestral roots is rigorous, honest, and structurally differentiated:

| Attribute | C51 (`Bellemare2017`) | Continuous SMDP (`Bradtke1994`) | SurvTD (Candidate R2) |
|---|---|---|---|
| **Target Quantity** | Discounted return distribution $Z(s, a) = \sum_{t} \gamma^t R_{t+1}$ | Expected value function $V(s) = \mathbb{E}[\int e^{-\beta t} r dt]$ | Remaining lifetime distribution $p_j(s) = \mathbb{P}(R_j \in ds \mid H_{t_j})$ |
| **Domain Geometry** | Discrete-time MDPs, uniform steps | Continuous-time Semi-Markov Decision Processes | Continuous-time, irregularly sampled survival trajectories |
| **Transport Operator** | Support shift & shrink: $r + \gamma z$ (additive reward, constant shrinkage) | Scalar discount integral: $\int_0^{\Delta t} e^{-\beta u} r(u) du + e^{-\beta \Delta t} V(s')$ | Renewal rightward translation: $\Phi_{+\Delta t_j}$ ($R_j = \Delta t_j + R_{j+1}$) |
| **Discount Mechanism** | Exogenous constant hyperparameter $\gamma \in (0, 1)$ | Exogenous fixed exponential rate $e^{-\beta \Delta t}$ | Endogenous dynamic survival probability $\gamma_j = S_{\theta^-}(\Delta t_j)$ |
| **Projection Metric** | Cramér / $l_2$-Wasserstein on reward support | N/A (Scalar expectation) | Cramér metric on remaining lifetime CDF support |
| **Censoring Handling** | None (absorbs to terminal states) | None | Target tail completion + scalar IPCW loss weighting $\le 10.0$ |

SurvTD does **not** claim to invent categorical projection; it explicitly cites Bellemare et al. (2017) and Rowland et al. (2018). Its structural contribution lies in recognizing that the renewal identity of survival analysis under continuous elapsed time maps directly to a rightward translation operator $\Phi_{+\Delta t}$, and that the survival probability acts as an endogenous duration discount. This is a legitimate mathematical synthesis that resolves a longstanding bottleneck in dynamic survival analysis.

### 5. DeepTCSR Clamped Division Comparison in NC-A3 and Baseline Parity

In prior literature, new methods frequently construct strawman baselines by evaluating older methods on naive discrete grids while granting the proposed method advanced continuous encoders. 
- In candidate_r2, **NC-A3** explicitly constructs an ablation arm:
  `"Arm A3 restores DeepTCSR clamped division div S(Delta t_j) on continuous intervals."`
- Furthermore, the **compute budget** guarantees:
  `"Backbones share an identical continuous-time GRU-D or LSTM encoder across all four methods to guarantee parity: (1) SurvTD, (2) DeepTCSR with Delta t feature and clamped division, (3) Dynamic-DeepHit, and (4) Person-period expanded discrete hazard baseline on a 1h regular grid."`

This experimental ladder isolates the operator substitution from the backbone architecture. By providing DeepTCSR with the exact same continuous-time GRU-D encoder and continuous $\Delta t_j$ inputs, any performance advantage observed in SurvTD over DeepTCSR can be attributed strictly to the mathematical stability of $\Pi \Phi_{+\Delta t_j}$ and sample branching over clamped division $\div \max(S, \epsilon)$.

---

## Four Axes Evaluation

| Axis | 1–5 | Quoted Evidence from candidate_r2.json | Reason |
|---|:---:|---|---|
| **A — Problem position** | **5** | *"Temporal-consistency survival methods (TCSR, DeepTCSR) assume an exogenous uniform unit step Delta t = 1, so the non-event signal carried by an interval is one bit regardless of how much time elapsed, and the assumption fails outright on irregularly observed clinical and industrial trajectories."* <br><br> *"The backward update of prior consistency methods renormalises the next-step lifetime distribution by dividing by the interval survival probability, which diverges exactly in the high-risk regime where S approaches zero, collapsing predictions towards an uninformative flat distribution."* | **5/5 (Insightful, open, highly load-bearing).** Clinical EHRs (MIMIC-IV) and industrial streams (C-MAPSS) are fundamentally irregular. Naming the unit-step assumption as an artificial restriction and isolating the division operator $p / S$ as the exact mathematical cause of high-risk collapse in prior temporal consistency literature is an exceptional diagnostic contribution to our subfield. |
| **B — Method quality** | **5** | *"The elapsed duration Delta t_j between observations enters as an interval duration discount gamma_j = S_theta-(Delta t_j) computed from an EMA target network, and the next-step lifetime distribution is carried backwards by a rightward shift Phi_{+Delta t_j} (the renewal identity R_j = R_{j+1} + Delta t_j) composed with a categorical projection Pi onto the fixed bin grid, instead of by division. The composed operator Pi Phi is non-expansive in the Cramer metric and conserves unit probability mass exactly, so the Bellman target is a valid distribution at every interval length."* | **5/5 (Decomposed below).** <br>• **depth**: Replaces an unstable analytical division with an empirical renewal sample-branching TD backup, bridging distributional RL operator theory with continuous survival renewal processes. <br>• **soundness**: Contraction modulus bounded by $\gamma_j < 1$ via frozen EMA target network; exact unit mass conservation; Cramér non-expansiveness; IPCW scalar loss weighting preserves the distribution simplex. <br>• **feasibility**: Fully implementable with PyTorch/TensorFlow using standard GRU-D backbones on a standard 120 GPU-hour budget. |
| **C — Problem-fit** | **5** | *"The two patterns share the inter-observation transition operator as their intermediate object. Auditing the unit-step assumption makes continuous shifting necessary; the shift requires categorical projection back onto the grid. Performing only the audit leaves division explosion in place, which is the exact residue DeepTCSR2024 inherited from TCSR."* <br><br> *"The division is replaced by the composition of a renewal rightward shift and a categorical projection onto the bin grid in a renewal mixture target. The substitution conserves unit probability mass exactly, is non-expansive in the Cramer metric, and yields an affine target map whose modulus under a frozen target network is bounded by the interval discount gamma_j."* | **5/5 (Direct 1:1 gap closure).** The mechanism targets the exact twin pathologies in the motivation: continuous renewal shift $\Phi_{+\Delta t_j}$ handles arbitrary elapsed time, and sample branching + categorical projection $\Pi$ eliminates the division singularity while preserving probability mass. |
| **D — Falsifiability & claim integrity** | **5** | *"load_bearing_variable": "composed_duration_transition_operator Pi Phi_{+Delta t_j} modulated by gamma_j = S_theta-(Delta t_j)"* <br><br> *"performance dropping by at least 0.025 in time-dependent AUC and concordance (derived: 3x the 5-seed bootstrap test standard error measured as 0.008 on MIMIC-IV Sepsis-3). On Poisson-subsampled NASA C-MAPSS (50% random cycle drop), concordance drops toward the Bleistein2024 baseline of 0.8791 (measured in Bleistein2024), while alarm jitter and false alert episode rates worsen compared to EMA-smoothed Dynamic-DeepHit."* | **5/5 (Rigorous, derived provenance, non-tautological).** Explicit named load-bearing variable. Every numeric threshold has explicit provenance (`derived:` from bootstrap standard error or `measured in Bleistein2024`). The negative controls (NC-A1, NC-A2, NC-A3, NC-B, NC-C) test downstream discriminative and stability metrics rather than trivial definitions. |

---

## Overall Score & Gating

$$\text{overall} = \text{round}\left(100 \times \frac{A + B + C - 3}{12}\right) = \text{round}\left(100 \times \frac{5 + 5 + 5 - 3}{12}\right) = \text{round}\left(100 \times \frac{12}{12}\right) = \mathbf{100 / 100}$$

- **Band**: `strong` ($\ge 67$)
- **Gate check**: No gate fired ($A > 2, C > 2, D > 2$).

---

## Structural Checks

### 1. Naive-Baseline Audit

- **Independent Construction of the Naive Baseline:**
  1. *Naive Baseline 1 (Person-Period Discretization):* Resample and discretize the irregular timeline onto a rigid 1-hour uniform person-period grid, forward-fill missing covariates, and run standard unit-step DeepTCSR with $\Delta t = 1$.
  2. *Naive Baseline 2 (Clamped Division on Continuous Intervals):* Retain the continuous elapsed duration $\Delta t_j$ as an input feature to the encoder, evaluate interval survival $S(\Delta t_j)$, and compute the backward update using clamped division: $p_{t_j} \leftarrow p_{t_{j+1}} / \max(S(\Delta t_j), \epsilon)$.
- **Classification:** **Branch 1 — The naive version relies on a false premise $\to$ confronting that premise IS the contribution.**
  - *Premise of Naive 1:* Irregular observation times can be flattened into uniform steps without unacceptable computational penalty or temporal distortion. In reality, person-period expansion inflates sequence lengths by an order of magnitude, introduces extreme zero-event sparsity, and compounds discrete approximation errors across dozens of synthetic steps.
  - *Premise of Naive 2:* Clamping the division $1 / \max(S(\Delta t), \epsilon)$ is a sufficient numerical fix for continuous intervals. In reality, as hazard rises or $\Delta t$ expands, $S \to 0$. Division by $\epsilon$ creates massive target probability mass spikes ($\gg 1$), producing gradient explosions, uncalibrated risk curves, and optimization failure in the clinical high-risk tail.
  - *Confronting the Premise:* SurvTD demonstrates that division is fundamentally unnecessary. In an empirical trajectory, survival over an interval is a realized sample event. By shifting and projecting the distribution along continuation visits and placing Diracs on death visits, the update becomes an affine, mass-conserving renewal TD backup.

### 2. Novel-but-Empty Detector

- **Prediction test:** The candidate predicts that ablating the continuous operator or replacing it with clamped division (NC-A3) will cause at least a $0.025$ drop in time-dependent AUC on MIMIC-IV Sepsis-3, and cause C-index on Poisson-subsampled NASA C-MAPSS to drop toward the Bleistein2024 baseline ($0.8791$).
- **Instrument test:** Evaluated on established survival metrics (time-dependent AUC, integrated Brier score, C-index, alarm jitter, and false alert episode rate) across public benchmarks (MIMIC-IV, C-MAPSS, PBC).
- **Result:** The idea is concrete, grounded in verifiable physical predictions, and decisively passes the detector.

---

## Attack Catalog Gauntlet

Leading with **§5 Positioning** and **§2.1 Mechanism**, the full attack catalog was executed against Candidate R2:

| # | Attack | Severity | Answerability | Where it lands | Subfield Insider Assessment |
|---|---|---|---|---|---|
| **5.1** | Delta by domain only | *does not land* | — | Positioning vs. C51 / SMDP | SurvTD is not simply C51 applied to survival data. C51 operates on stationary MDPs with scalar reward addition and constant scalar discounting. SurvTD operates on continuous renewal lifetime processes with rightward translation $\Phi_{+\Delta t}$, endogenous survival discounting $\gamma_j = S_{\theta^-}(\Delta t_j)$, right-censored tail completion, and IPCW scalar loss weighting. The delta is deeply structural. |
| **5.2** | Delta by name | *does not land* | — | Terminology vs. TCSR | "Duration-discounted Bellman target" and "renewal shift with categorical projection" describe explicit mathematical operations ($\Pi \Phi_{+\Delta t_j}$ modulated by $S_{\theta^-}(\Delta t_j)$). This is structurally distinct from TCSR's analytical division update $p / S(1)$. |
| **5.3** | Superseded fix | *does not land* | — | Lineage genealogy | Prior methods in dynamic survival analysis either ignored inter-visit consistency entirely (Dynamic-DeepHit, CoxSig) or inherited unit-step division (TCSR, DeepTCSR). No prior work in the lineage formulated a continuous projected renewal TD operator. |
| **5.4** | Regression to a prior state | *does not land* | — | Target network & recurrence | SurvTD retains the continuous-time recurrent encoder and EMA target network popularized by DeepTCSR, repurposing the target network to freeze the interval discount $\gamma_j$ and preserve contraction modulus. |
| **2.1** | Equivalent to naive | *does not land* | — | Core mechanism vs. clamped division | As proven in the naive audit and isolated in NC-A3, naive clamped division diverges in high-risk tails. SurvTD's sample-branching renewal backup is structurally distinct and resolves the numerical singularity. |
| **2.3** | Unstated precondition | **minor** | **now** / **with evidence** | Step 1 encoder assumptions | Step 1 explicitly assumes that *"observation times are conditionally unconfounded given $h_j$"*. In acute ICU settings, clinicians often order tests precisely because a patient is crashing (informative observation process). While standard in dynamic deep survival models (e.g., Dynamic-DeepHit, DeepTCSR), the paper should explicitly discuss how recurrent conditioning on $h_j$ captures this confounding, and evaluate sensitivity on MIMIC-IV. |
| **2.6** | The borrowed tool | **minor** | **now** | Step 4 categorical projection $\Pi$ | The categorical projection $\Pi$ is directly imported from Bellemare et al. (2017). Candidate R2 explicitly acknowledges this provenance and bounds its contribution honestly: importing $\Pi$ under the Cramér metric, but constructing a novel renewal lifetime translation and endogenous survival discount operator. |
| **3.2** | Asserted bridge | *does not land* | — | Contraction and mass conservation | The bridge between operator substitution and stability is proven via Cramér non-expansiveness (Rowland et al., 2018) and affine target mapping under frozen $\theta^-$. |
| **4.3** | Invented numbers | *does not land* | — | Falsification predictions | All numerical bars have explicit provenance: $\Delta \text{AUC} \ge 0.025$ is derived from 3x bootstrap standard error ($0.008$ on MIMIC-IV), and C-index baseline $0.8791$ is measured directly in Bleistein et al. (2024). |

---

## Two-Layer Verdict

### 1. Hard Floor Check
- [x] Prior work matches on all 4 scoop axes? **No** (Scoop report confirms Overlap Level 2/5 — adjacent lineage, distinct technical move).
- [x] Naive-baseline audit returns *naive suffices*? **No** (Branch 1: Naive relies on false premise; clamped division collapses in high-risk regimes).
- [x] Anti-pattern composition's required mitigation missing? **No** (Mitigations are fully integrated).
- [x] Axis D collapses? **No** (D = 5/5, fully falsifiable with non-tautological controls).

**Hard Floor Status: CLEARED.**

### 2. Soft Judgment

- **Final Verdict**: **`advance`**
- **Justification**: SurvTD delivers an exceptionally rigorous, mathematically elegant, and honest contribution. It diagnoses the exact failure mode that has crippled temporal-consistency survival analysis (the unit-step division operator) and replaces it with a mass-conserving, Cramér non-expansive renewal TD backup adapted from distributional RL. The positioning against TCSR, DeepTCSR, C51, and Bradtke SMDP is immaculate, and the experimental design (incorporating continuous DeepTCSR clamped division in NC-A3 and shared GRU-D backbones) guarantees unconfounded evaluation.

### Revision Targets

1. **[recommended · positioning & discussion]** Expand the discussion in §1 / §3 regarding the "conditionally unconfounded observation times" assumption (Step 1). Explicitly discuss how informative observation processes (e.g., intensive ICU lab sampling during septic shock) are handled by the continuous recurrent latent state $h_j$, citing informative sampling literature in longitudinal survival analysis.
2. **[recommended · empirical report]** In the experimental write-up, report per-step training wall-clock time and memory consumption compared to Dynamic-DeepHit and DeepTCSR to empirically demonstrate that continuous categorical projection and EMA target evaluation incur negligible computational overhead.

---

**Strongest point**: The conceptual leap from analytical division inversion to an empirical renewal sample-branching TD backup (continuation shift on alive visits vs. intra-interval Dirac on death visits), which simultaneously eliminates division singularities and preserves total probability mass.  
**Most fixable weakness**: Add a brief sensitivity analysis or discussion on the informative observation assumption in clinical ICU settings to preempt clinical reviewers who worry about sampling rate confounding.
