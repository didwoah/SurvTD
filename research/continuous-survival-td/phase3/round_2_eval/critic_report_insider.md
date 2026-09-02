# Critic Report — Subfield Insider (Deep Survival Analysis & Distributional RL)

**Role**: `insider`  
**Candidate**: `SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation` (`candidate_r1.json`)  
**Evaluator Lens**: Lineage tracking across TCSR/DeepTCSR and C51/SMDP, structural vs. lexical delta, honest positioning, and clinical survival baseline realism.

---

## Idea review — SurvTD

### Decomposition
- **Problem / gap**: Dynamic survival models that enforce inter-visit temporal consistency (TCSR, DeepTCSR) inherit two critical flaws from their discrete-time foundation: (1) an assumed uniform unit-step transition ($\Delta t = 1$) that treats inter-visit duration as a 1-bit non-event indicator regardless of elapsed time; and (2) a backward Bellman-style update that renormalises the lifetime distribution by dividing by interval survival probability $S(\Delta t)$, which diverges to infinity as patient mortality risk rises or as intervals widen, collapsing predictions toward an uninformative flat distribution.
- **Method (the move)**: Replaces the unit-step division operator with a continuous-time renewal mixture operator. The elapsed duration $\Delta t_j$ enters as an interval duration discount $\gamma_j = S_{\theta^-}(\Delta t_j)$ computed from an EMA target network $\theta^-$. The next-step distribution is transported backwards via a rightward renewal shift $\Phi_{+\Delta t_j}$ ($R_j = R_{j+1} + \Delta t_j$) composed with a categorical projection $\Pi$ onto a fixed remaining-lifetime grid in Cramér metric (conserving unit mass exactly). The one-step target forms a renewal mixture $\mathcal{T} p_j = (1 - \gamma_j) \mu_{[0, \Delta t_j)} + \gamma_j (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$, extended to multi-step bootstrapping via a duration-geometric $\lambda$-return $\lambda_j = \lambda^{\Delta t_j / \delta_s}$ and completed at right-censored endpoints via truncated IPCW tail completion.
- **Why it should work**: In survival theory, remaining lifetime satisfies the renewal identity $R(t) = \Delta t + R(t + \Delta t)$ on the survival branch. Projecting shifted distributions with a triangular kernel is non-expansive in the Cramér ($L_2$ on CDF) metric. Evaluating the endogenous discount $\gamma_j$ under a frozen target network $\theta^-$ renders the target mapping strictly affine with contractive modulus bounded by $\gamma_j = S_{\theta^-}(\Delta t_j) < 1$ whenever hazard is positive. This completely eliminates division divergence while strictly preserving probability mass conservation.
- **Assumptions inferred**:
  1. Observation times $t_j$ are conditionally unconfounded given the latent trajectory state $h_j$.
  2. The hazard is strictly positive on $[0, \Delta t_j)$ such that $\gamma_j < 1$, guaranteeing contractive modulus during inner updates.
  3. Truncated IPCW weights ($1/\hat{G} \le 10$) sufficiently control tail variance without introducing asymptotic bias under heavy right-censoring.
  4. At terminal censoring times, survival dynamics follow the learned population hazard encoded in $\theta^-$.

---

### Quantitative Scorecard

| Axis | Score (1–5) | Quoted Evidence from `candidate_r1.json` | Reason |
| :--- | :---: | :--- | :--- |
| **A — Problem position** | **4** | *"Dynamic survival models that enforce temporal consistency between consecutive observations inherit a unit-step transition (Delta t = 1): surviving one step is a one-bit event and the backward update renormalises the next-step lifetime distribution by dividing by the interval survival probability, which diverges as risk rises. SurvTD audits that assumption and replaces the offending operator."* | **Strong**. The gap is genuine, technically precise, and directly addresses the core roadblock that caused deep survival analysis to abandon consistency learning in favor of point-in-time terminal losses (Dynamic-DeepHit). Exposing the joint vulnerability of unit-step transitions and division divergence is a sharp diagnostic. |
| **B — Method quality** | **4** | *"The elapsed duration Delta t_j between observations enters as an interval duration discount gamma_j = S_theta-(Delta t_j) computed from an EMA target network, and the next-step lifetime distribution is carried backwards by a rightward shift Phi_{+Delta t_j} (the renewal identity R_j = R_{j+1} + Delta t_j) composed with a categorical projection Pi onto the fixed bin grid, instead of by division. The composed operator Pi Phi is non-expansive in the Cramer metric and conserves unit probability mass exactly..."* | **depth**: High. Reframing the backward update from division-based renormalization to renewal mixture transport and categorical projection is a substantive mathematical upgrade.<br>**soundness**: High with minor clarification needed. Contraction under frozen $\theta^-$, mass conservation, and Cramér non-expansiveness are rigorous. Terminal absorbing bin semantics ($\ge s_K$) cleanly handle boundary mass. A minor drafting ambiguity remains in Step 5 regarding sample-level vs. expected realization of $\mu_{[0, \Delta t_j)}$ on non-terminal surviving steps.<br>**feasibility**: High. Uses standard 1D triangular projection kernels, continuous-time RNN backbones, and Cramér distance via cumulative sums in PyTorch. |
| **C — Problem-fit** | **5** | *"The division is replaced by the composition of a renewal rightward shift and a categorical projection onto the bin grid in a renewal mixture target. The substitution conserves unit probability mass exactly, is non-expansive in the Cramer metric, and yields an affine target map whose modulus under a frozen target network is bounded by the interval discount gamma_j."* | **Flawless**. The method targets the exact dual failure modes specified in the problem statement: the continuous shift and duration discount resolve the irregular interval problem, while the renewal mixture and categorical projection eliminate the division divergence without sacrificing mass conservation. |
| **D — Falsifiability & claim integrity** | **5** | *"composed_duration_transition_operator Pi Phi_{+Delta t_j} modulated by gamma_j = S_theta-(Delta t_j)"*<br>and<br>*"performance dropping by at least 0.025 in time-dependent AUC and concordance (derived: 3x the 5-seed bootstrap test standard error measured as 0.008 on MIMIC-IV Sepsis-3). On Poisson-subsampled NASA C-MAPSS (50% random cycle drop), concordance drops toward the Bleistein2024 baseline of 0.8791 (measured in Bleistein2024)..."* | **Exemplary**. Load-bearing variable is unambiguous. Negative controls (3-arm operator ablation NC-A including continuous clamped division NC-A3, within-patient permutation NC-B, and duration-geometric horizon sweep NC-C) are non-tautological. Numerical margins have explicit provenance (`derived:` from bootstrap standard errors, `measured in Bleistein2024`). |

**Overall Score**: **83 / 100**  
$$\text{overall} = \text{round}\left(100 \times \frac{4 + 4 + 5 - 3}{12}\right) = \text{round}\left(\frac{1000}{12}\right) = 83$$  
**Band**: **`strong`** ($\ge 67$)  
**Gate Evaluation**: No gate fired ($A = 4 > 2$, $C = 5 > 2$, $D = 5 > 2$).

---

## Subfield Insider Deep-Dive: Lineage, Mechanics & Baselines

### 1. Step 5 Renewal Mixture Formulation vs. TCSR Unit-Step Division
In unit-step consistency methods (TCSR, Maystre et al. 2022; DeepTCSR 2024), Bayes' theorem is invoked to enforce consistency across consecutive observations:
$$S_j(s) = S_j(1) \cdot S_{j+1}(s - 1) \implies S_{j+1}(s - 1) = \frac{S_j(s)}{S_j(1)}$$
When translated into a backward target for lifetime probability distributions or densities, this inverted formulation demands division by interval survival: $p_{target}(s) \propto p_{j+1}(s - 1) / S_j(1)$.
In high-risk clinical cohorts (such as septic ICU patients where hazard spikes) or across wide observation gaps ($\Delta t \gg 1$), $S(\Delta t) \to 0$. Dividing by this probability causes gradient explosions, catastrophic numerical instability, and pushes tail predictions toward an uninformative uniform distribution.

SurvTD's Step 5 bypasses this by returning to the primal **law of total probability / renewal mixture**:
$$\mathcal{T} p_j = (1 - \gamma_j) \cdot \mu_{[0, \Delta t_j)} + \gamma_j \cdot (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$$
Here, $\gamma_j = S_{\theta^-}(\Delta t_j) \in [0, 1]$ appears as a **multiplicative mixture coefficient**, not a divisor. 
- Surviving the interval carries remaining lifetime forward by $\Delta t_j$ with probability $\gamma_j$, requiring a rightward translation $\Phi_{+\Delta t_j}$.
- Experiencing the event within the interval has probability $1 - \gamma_j$, governed by within-interval failure density $\mu_{[0, \Delta t_j)}$.
Because $\gamma_j$ is bounded in $[0, 1]$, the operator is numerically stable at all risk levels, conserves total probability mass $\sum_k (\mathcal{T} p_j)_k = 1$ identically, and exhibits no division singularities. This is a bona fide **structural delta**, not a cosmetic reparameterization.

### 2. Differentiation from C51 (Bellemare 2017) and Bradtke SMDP (1994)
SurvTD sits at the confluence of continuous-time reinforcement learning and categorical distributional RL. Its positioning against both lineages in `candidate_r1.json` is remarkably honest and technically accurate:
- **Vs. C51 (Bellemare et al. 2017)**:
  - In C51, the Bellman target operates on cumulative discounted reward return $Z(x, a) \stackrel{D}{=} R(x, a) + \gamma Z(x', a')$. The discount $\gamma \in (0, 1)$ is an **exogenous scalar constant**, and the update applies a support **contraction/scaling** $z_k \mapsto r + \gamma z_k$.
  - In SurvTD, the target operates on remaining survival lifetime $R_j$. The discount $\gamma_j = S_{\theta^-}(\Delta t_j)$ is **endogenous**, state-dependent, and duration-dependent. Furthermore, the temporal update is an **additive renewal shift** $R_j = R_{j+1} + \Delta t_j$ (shifting the support rightward), rather than a support contraction. Additionally, SurvTD must handle right-censoring via truncated IPCW tail completion (Step 6), a challenge absent in standard MDPs.
  - SurvTD explicitly credits C51 for importing the projection operator $\Pi$ under the Cramér metric, making its novelty claim transparent and defensible.
- **Vs. Continuous-Time SMDP (Bradtke & Duff 1994)**:
  - Bradtke & Duff formulate semi-Markov TD learning for **scalar value functions** $V(s)$, discounting expected future return by $e^{-\beta \Delta t}$ across variable decision epochs.
  - SurvTD operates on the **full probability distribution** of remaining lifetime, applying continuous discounting to the mixture weight while translating the underlying random variable along the real line.

### 3. DeepTCSR Clamped Division Comparison in NC-A3 and Baseline Set
A common defense of unit-step division is that one can simply "clamp" the denominator: $\max(S(\Delta t), \epsilon)$ with $\epsilon = 10^{-3}$, and feed $\Delta t$ into the encoder.
SurvTD anticipates and dismantles this objection:
1. It includes **Arm A3 in Negative Control NC-A**: restoring DeepTCSR continuous clamped division $\div \max(S(\Delta t_j), \epsilon)$.
2. It includes **DeepTCSR with $\Delta t$ feature and clamped division** as Baseline (2) in the 120 GPU-hour compute budget ladder.
From a theoretical perspective, clamped division violates probability axioms: when $S(\Delta t) < \epsilon$, dividing by $\epsilon$ scales probability mass by $1/\epsilon = 1000$, destroying mass conservation and distorting the survival curve. SurvTD's negative control NC-A3 will empirically isolate whether the mass-conserving renewal mixture out-discriminates heuristic clamping.

### 4. Person-Period Expanded Discrete Hazard Comparison
In clinical epidemiology, biostatisticians routinely handle irregular longitudinal observations without RL: they discretize time into fixed epochs (e.g., 1-hour bins), carry forward clinical vitals, and fit a pooled person-period discrete hazard model (e.g., binary cross-entropy on whether an event occurred in that 1h block).
Machine learning researchers frequently ignore this baseline, comparing complex neural TD models only against weak or static Cox models.
SurvTD explicitly includes:
- `compute_budget`: **"(4) Person-period expanded discrete hazard baseline on a 1h regular grid."**
This is a critical inclusion. If SurvTD cannot outperform a properly tuned 1h person-period discrete hazard network on dynamic AUC and calibration, the TD machinery would be unnecessary overhead. Testing against it demonstrates genuine methodological confidence.

### 5. Terminal Absorbing Bin Semantics ($\ge s_K$)
In Step 4, SurvTD specifies:
*"accumulating mass carried past horizon s_K into the terminal absorbing bin with >= s_K defective mass semantics. Pi Phi is non-expansive in Cramer (L2-on-CDF) metric and conserves total mass exactly."*
In clinical survival analysis, observation trajectories are truncated at a study follow-up horizon $s_K$. The true remaining lifetime distribution is often defective on $[0, s_K)$, with residual mass $S(s_K) = P(R \ge s_K)$ surviving beyond the study window.
By establishing an absorbing bin at $s_K$ that accumulates all probability mass translated past $s_K$:
- Total probability mass remains exactly 1 ($\sum_{k=1}^K p_k = 1$).
- The cumulative distribution function satisfies $F(s_K) = 1$.
- The squared Cramér distance $\sum_{k=1}^K (F_j(s_k) - F_{G_j}(s_k))^2$ remains bounded, well-conditioned, and respects the truncated survival boundary without boundary reflection artifacts.

---

## Two Structural Checks

### 1. Naive-Baseline Audit: **Branch 1** (The naive version relies on a false premise)
- **Independent construction of the naive baseline**: Take DeepTCSR, concatenate elapsed duration $\Delta t_j$ to the recurrent encoder state, and apply clamped division:
  $$\mathcal{T}_{naive} S(s) = \frac{S_j(s)}{\max(S_j(\Delta t_j), \epsilon)}$$
  Alternatively, interpolate/forward-fill irregular observations onto a regular 1h grid and run standard unit-step TCSR.
- **The false premises exposed**:
  1. *Clamped division is a benign numerical stabilizer*: False. Clamping the denominator destroys probability normalization whenever $S(\Delta t) < \epsilon$, introducing massive artificial density into the tails and distorting calibration.
  2. *Forward-filling irregular clinical observations onto regular grids preserves event dynamics*: False. Forward-filling introduces artificial observation synchrony and distorts the true hazard rate across variable rest periods.
- **Audit Verdict**: SurvTD's contribution lies precisely in confronting and removing these false premises. While the projection tool is imported from C51, adapting it to continuous renewal translation, endogenous survival discounting, and right-censoring constitutes a legitimate, domain-grounded methodological contribution.

### 2. Novel-but-Empty Detector: **PASS**
SurvTD makes specific, non-vacuous empirical predictions that would conclusively falsify the mechanism if violated:
- Dropping duration discounting (Arm A1) or discretizing shift to unit-step (Arm A2) predicts a drop of $\ge 0.025$ in time-dependent AUC and concordance on MIMIC-IV Sepsis-3 ($3\times$ bootstrap SE).
- Performance on Poisson-subsampled C-MAPSS degrades toward the Bleistein2024 baseline of 0.8791.
- Alarm jitter and false alert episodes per patient-day at matched 0.30 PPV worsen relative to Dynamic-DeepHit.
The idea is concrete, falsifiable, and structurally complete.

---

## Attack Catalog Gauntlet (Leading §5 Positioning & §2.1)

| # | Attack | Severity | Answerability | Where it lands | Insider Assessment |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **5.1** | Delta by domain only | *Dismissed* | now | `core_mechanism` | **Does not land**. SurvTD does not simply apply C51 to medical data. It replaces C51's reward scaling with renewal rightward translation, replaces fixed discount $\gamma$ with endogenous survival discount $S(\Delta t)$, and handles right-censoring via IPCW tail completion. |
| **5.2** | Delta by name | *Dismissed* | now | `core_mechanism`, Step 5 | **Does not land**. The renewal mixture is mathematically distinct from TCSR's division. One is an affine convex combination; the other is a rational function with a singular denominator. The delta is operational and algebraic. |
| **5.3** | Superseded fix | *Dismissed* | now | `gap_closure`, Step 5 | **Does not land**. DeepTCSR added target networks but preserved unit-step division. SurvTD is the first to repair the update operator itself. |
| **5.4** | Regression to prior state | *Dismissed* | now | `core_mechanism_steps`, Step 3 | **Does not land**. SurvTD retains the target network $\theta^-$ to freeze $\gamma_j$, explicitly preserving contraction modulus. |
| **2.1** | Equivalent to naive | *Dismissed* | now | `negative_control`, NC-A3 | **Does not land**. The naive version (clamped division $\div \max(S, \epsilon)$) is explicitly benchmarked in NC-A3 and shown theoretically to violate probability conservation. |
| **2.3** | Unstated precondition / Semantic ambiguity | **minor** | **now** | `core_mechanism_steps`, Step 5 | **Lands**. In Step 5: *"The in-interval component mu_{[0, Delta t_j)} is supervised by projecting observed within-interval event offsets delta_{tau_j - t_j} onto the grid."*<br>On non-terminal transitions $(t_j \to t_{j+1})$, the patient is observed alive at $t_{j+1}$, meaning no event occurred in $[t_j, t_{j+1})$. Thus, no empirical event offset $\tau_j - t_j$ exists for non-terminal steps! The text must explicitly distinguish the empirical sample-path target (where non-terminal transitions evaluate only the survival branch $\Pi \Phi p_{\theta^-}$ with weight 1) from the expected Bellman operator. Fully answerable now in drafting. |
| **2.6** | The borrowed tool | **minor** | **now** | `differentiation_from_lit` | **Lands cleanly**. Categorical projection is borrowed from Bellemare (2017). However, candidate_r1.json completely discharges this attack by explicitly acknowledging the borrowing and detailing the domain-specific adaptations (renewal translation, endogenous discount, censoring). |
| **1.1 / 1.3** | Competition against non-TD pooled hazard models | **minor** | **with evidence** | `compute_budget`, Baseline (4) | **Lands**. Reviewers will ask whether an ordinary continuous-time neural hazard model or 1h person-period pooled logistic model matches SurvTD without any TD complexity. SurvTD answers this by including the 1h person-period baseline and predicting lower inter-visit alarm jitter. |

---

## Two-Layer Verdict

### 1. Hard Floor (Abandon Triggers)
- **4-Axis Scoop Collision**: NO. Scoop report confirms overlap level 2/5 (adjacent lineage, distinct technical move).
- **Naive Baseline Suffices**: NO. Naive clamped division and simple grid discretization suffer from severe theoretical and empirical defects.
- **Anti-Pattern Composition Unmitigated**: NO. The composition between the unit-step audit and operator substitution is explicitly harmonized in `composition_note`.
- **Falsification Collapsed**: NO. Load-bearing variable and non-tautological controls are solidly specified.
**Hard Floor Verdict**: **PASS**.

### 2. Soft Judgment & Recommendation
SurvTD demonstrates superior subfield maturity. It cleans up the deep survival consistency literature by diagnosing why TCSR and DeepTCSR suffered from instability, replacing their broken division operator with a mathematically rigorous renewal mixture and categorical projection. It properly credits C51 and Bradtke SMDP, specifies non-tautological negative controls (including continuous clamped division), and incorporates the critical person-period discrete hazard baseline.

The remaining issue is a minor semantic ambiguity in Step 5 regarding the realization of $\mu_{[0, \Delta t_j)}$ on non-terminal surviving visits, which is easily clarified in drafting.

**Final Verdict**: **`advance`**  
*(Ready to proceed to Phase 4 Experiment Design)*

---

### Revision / Drafting Recommendations for Phase 4 & Paper Architecture
1. **[Recommended · Step 5 Transition Clarification]**: Explicitly state that on non-terminal transitions $(t_j \to t_{j+1})$ where the patient is observed alive, the sample transition target evaluates exclusively along the survival branch: $\mathcal{T}_{sample} p_j = \Pi \Phi_{+\Delta t_j} p_{\theta^-}$, while the within-interval failure component $\mu_{[0, \Delta t_j)} = \delta_{\tau - t_j}$ applies strictly to the terminal event transition $(t_M \to \tau)$. Alternatively, clarify if $\mu_{[0, \Delta t_j)}$ is filled by the model's own in-interval hazard prediction $h_{\theta^-}(s)$ when computing an expected Bellman target.
2. **[Recommended · Non-TD Stability Metric]**: In the experimental results table against the 1h person-period baseline, lead with inter-visit prediction jitter / consecutive prediction variance $\mathbb{E}[\|p_{j+1} - \Phi_{-\Delta t_j} p_j\|_1]$ alongside C-index and time-dependent AUC to decisively prove the value of Bellman consistency over point-in-time maximum likelihood.

---

**Strongest Point**: The replacement of TCSR's singular division-by-survival operator with a mass-conserving renewal mixture rightward shift and categorical projection under Cramér distance is a genuine mathematical breakthrough that makes continuous-time temporal consistency viable.

**Most Fixable Weakness**: Disentangle the sample-path realization from the expected renewal mixture formula in Step 5 so reviewers do not confuse non-terminal survival steps with observed event intervals.
