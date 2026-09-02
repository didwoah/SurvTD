# SurvTD Critic Report: Reinforcement Learning & Policy Evaluation Expert Review

**Reviewer Role**: Top-tier Conference Reviewer (NeurIPS / ICML / RLDM Expert in Reinforcement Learning, Temporal Difference Theory, and Policy Evaluation)  
**Target Document**: [`candidate.json`](file:///Users/yangjaemo/Desktop/SurvTD/research/continuous-survival-td/phase2/candidate.json)  
**Supporting Documents**: [`bottleneck.md`](file:///Users/yangjaemo/Desktop/SurvTD/research/continuous-survival-td/phase1/bottleneck.md), [`lit_table.md`](file:///Users/yangjaemo/Desktop/SurvTD/research/continuous-survival-td/phase0/lit_table.md)  
**Evaluation Date**: 2026-09-02  
**Final Decision**: **advance** (with technical precision requirements for Phase 4)

---

## 1. Executive Summary

The proposed candidate, **SurvTD** (*Continuous-Time Semi-Markov Temporal Difference Learning for Irregular Longitudinal Dynamic Survival Analysis*), presents a mathematically elegant and principled paradigm shift: recasting continuous-time dynamic survival prediction over irregular longitudinal streams as policy evaluation in an absorbing Semi-Markov Decision Process (SMDP).

By substituting the numerically catastrophic downstream division $\div S(\Delta t)$ of prior discrete temporal consistency methods (`Maystre2022`, `DeepTCSR2024`) with a contractive prospective Bellman operator scaled by endogenous interval survival discount $S(\Delta t)$, and mapping continuous temporal shifts via non-expansive Cramér projections (`Bellemare2017`, `Rowland2018`), SurvTD establishes an analytically sound foundation.

The RL theoretical mechanisms are remarkably well-conceived. However, rigorous scrutiny from the perspective of Sutton & Barto's TD theory, distributional RL contraction, and the "Deadly Triad" necessitates tempering overclaimed convergence guarantees in non-linear function approximation and formalizing the exact distributional backward $\lambda$-return recursion under right-censored absorption boundaries.

---

## 2. Four-Dimensional Evaluation Matrix

| Evaluation Dimension | Score (1-5) | Assessment |
| :--- | :---: | :--- |
| **A. Problem Position** | **5 / 5** | Flawlessly positions dynamic survival analysis as an off-policy policy evaluation problem on irregular trajectories. Accurately identifies the foundational flaw of discrete uniform TD. |
| **B. Method Quality** | **4 / 5** | Mathematically rigorous operator design using Cramér non-expansive projection and target network decoupling. Deducted 1 point due to overclaiming "fixed-point convergence guarantee" under deep neural nets and leaving the continuous-time $\lambda$-return backward recursion under right-censoring informal. |
| **C. Problem-Fit** | **5 / 5** | Perfect topological and structural alignment between absorbing SMDP dynamics and longitudinal clinical time-to-event trajectories with irregular sampling intervals. |
| **D. Falsifiability** | **5 / 5** | Clear load-bearing parameter ($\lambda$), well-defined negative controls ($\lambda=1.0$, $\Delta t=1.0$), and quantitative performance baselines on published benchmarks. |
| **Total Score** | **19 / 20** | **Outstanding (Decision: `advance`)** |

---

## 3. Detailed Dimension Scoring with Direct Textual Citations

### 3.1 Dimension A: Problem Position (Score: 5 / 5)
- **Direct Citation**:
  > *"Gap: Discrete unit-step restriction (Delta=1) and numerical division explosion (div S) in prior survival consistency models (TCSR, DeepTCSR) on irregular continuous-time observations."* (lines 14–15)  
  > *"Gap: Catastrophic gradient variance and temporal inconsistency of terminal Monte Carlo NLL loss across long-horizon longitudinal trajectories."* (lines 19–20)
- **Reviewer Commentary**:
  The positioning addresses the fundamental dichotomy in continuous-time longitudinal survival modeling. Classical methods like Dynamic-DeepHit (`Lee2019`) and CoxSig (`Bleistein2024`) discard intermediate temporal consistency, relying solely on terminal episodic Monte Carlo returns, which suffer from $\mathcal{O}(T)$ variance accumulation. Conversely, discrete consistency frameworks (`Maystre2022`) impose uniform step discretizations ($\Delta t = 1$) that artificially bin continuous trajectories and divide by small survival probabilities $S(\Delta t) \to 0$, inducing numerical gradient explosions. Framing this as a failure of temporal difference learning over irregular continuous-time SMDPs is conceptually rigorous and spot-on.

### 3.2 Dimension B: Method Quality (Score: 4 / 5)
- **Direct Citation**:
  > *"Step 3: Evaluate interval duration \\Delta t_j = t_{j+1} - t_j and compute interval discount factor S_{\\theta^-}(\\Delta t_j) using a frozen EMA Target Network \\theta^- to maintain quasi-linear contraction stability."* (line 7)  
  > *"Step 4: Shift future target probability distribution leftward by continuous duration \\Delta t_j and project onto discrete support via categorical projection \\Pi: (\\Pi \\Phi p)_k = \\sum_m p_m * \\max(0, 1 - |(s_m - \\Delta t_j) - s_k| / delta_s), which is non-expansive under the Cramér metric."* (line 8)  
  > *"Step 5: Formulate Survival Bellman target T_{\\theta^-} p_j(s) = I[Event in \\Delta t_j] + S_{\\theta^-}(\\Delta t_j) * (\\Pi \\Phi p_{\\theta^-})_j(s)."* (line 9)  
  > *"Step 7: Minimize Cramér distance loss between predicted cumulative distribution F_pred and target cumulative distribution F_{G^lambda} to guarantee fixed-point convergence."* (line 11)
- **Reviewer Commentary**:
  - **Strengths**: 
    1. Replacing division $\div S$ with forward prospective Bellman multiplication $\times S_{\theta^-}(\Delta t)$ resolves the numerical singularity.
    2. Utilizing the categorical projection $\Pi$ under the Cramér ($l_2$-Wasserstein) distance is theoretically sound. As proven by Rowland et al. (AISTATS 2018), while C51's projection is expansive under KL-divergence, it is strictly non-expansive ($\|\Pi P - \Pi Q\|_{l_2} \le \|P - Q\|_{l_2}$) under the Cramér distance. When scaled by $S(\Delta t) \le 1 - \epsilon < 1$ for $\Delta t > 0$, the composite operator $\mathcal{T} = \Pi T$ is a valid contraction mapping on the space of distributions.
    3. Decoupling the endogenous discount $S_\theta(\Delta t)$ via an exponential moving average (EMA) target network $\theta^-$ transforms an otherwise non-linear coupled fixed-point problem into a sequence of quasi-linear contraction steps.
  - **Weaknesses / Criticisms**:
    1. *Overclaimed Convergence Guarantee*: Step 7 claims to *"guarantee fixed-point convergence"*. In reinforcement learning, combining function approximation (deep recurrent encoders), bootstrapping (Bellman targets), and off-policy/offline trajectory distributions triggers the **Deadly Triad** (Sutton & Barto, 2018). While the target network and Cramér non-expansiveness empirically mitigate divergence and bound bootstrap drift, they do *not* mathematically guarantee global fixed-point convergence with general non-linear neural networks (Baird, 1995; Tsitsiklis & Van Roy, 1997). This phrasing must be toned down to "ensures quasi-contractive stability and bounds distribution drift".
    2. *Informal $\lambda$-Return Formulation*: Step 6 mentions constructing a geometric multi-step $\lambda$-return via backward recursion, but does not provide the explicit distributional operator equation that compounds continuous spatial-temporal shifts $\Phi_{\Delta t_j}$ across irregular intervals.

### 3.3 Dimension C: Problem-Fit (Score: 5 / 5)
- **Direct Citation**:
  > *"Reformulate dynamic survival analysis over continuous irregularly sampled longitudinal trajectories as policy evaluation in an absorbing Semi-Markov Decision Process (SMDP)."* (line 3)  
  > *"Composing assumption_audit_and_pivot with reframe_as_solvable_object is necessary because relaxing the discrete unit-step assumption introduces variable continuous intervals that require contractive SMDP policy evaluation machinery to resolve."* (line 25)
- **Reviewer Commentary**:
  The structural isomorphism between longitudinal survival analysis and absorbing SMDPs is mathematically genuine:
  - An event (e.g., mortality, engine failure) is an irreversible absorption state ($S_\infty$).
  - Clinical/sensor observations occur at irregular epochs $t_0, t_1, \dots, t_J$, exactly matching sojourn times in continuous-time Semi-Markov processes (Bradtke & Duff, 1994).
  - The survival probability $S(t_{j+1} - t_j \mid H_{t_j})$ functions as the transition probability of non-absorption over interval $\Delta t_j$.
  Mapping time-to-event dynamic prediction directly onto distributional policy evaluation is not an awkward analogy; it is an exact functional embedding.

### 3.4 Dimension D: Falsifiability (Score: 5 / 5)
- **Direct Citation**:
  > *"falsification_prediction": "When multi-step bootstrapping is ablated by setting lambda to 1.0 (collapsing to terminal Monte Carlo NLL), ranking concordance drops significantly below the intermediate bootstrap baseline (measured in Bleistein2024 as 0.8791 on NASA turbofan degradation) due to unmitigated trajectory variance."* (line 26)  
  > *"load_bearing_variable": "lambda_return_mixture_parameter"* (line 27)  
  > *"negative_control": "Ablate lambda_return_mixture_parameter by sweeping lambda to 1.0 (pure Monte Carlo NLL) and fixing Delta t to 1.0 (discrete uniform discretization), which disables continuous temporal difference bootstrapping and causes time-dependent AUC and C-index to degrade to baseline performance."* (line 28)
- **Reviewer Commentary**:
  The empirical falsification protocol is airtight. By designating $\lambda$ as the load-bearing parameter, the hypothesis isolates the exact source of value added: intermediate temporal difference bootstrapping. The ablation cleanly demonstrates whether continuous-time TD outperforms both pure episodic Monte Carlo ($\lambda = 1.0$) and uniform-discretized TD ($\Delta t = 1.0$). Furthermore, citing the published C-index benchmark ($0.8791$ from Bleistein et al., ICML 2024) establishes a concrete, quantifiable bar for rejection.

---

## 4. In-Depth RL & TD Technical Critique

As a specialist reviewer in policy evaluation and temporal difference methods, I offer the following technical verifications and critiques:

### 4.1 Endogenous Discounting and Contraction Validity
In standard MDPs, the discount factor $\gamma \in [0, 1)$ is an exogenous scalar constant. In SurvTD, the discount factor is state- and duration-dependent:
$$\gamma(h_j, \Delta t_j) = S_{\theta^-}(\Delta t_j \mid h_j)$$
Because survival functions are monotonically non-increasing and satisfy $S(0) = 1$ and $\lim_{s \to \infty} S(s) = 0$, for any non-zero transition interval $\Delta t_j > 0$ with non-zero hazard, we have:
$$\sup_{j} S_{\theta^-}(\Delta t_j \mid h_j) \le \gamma_{\max} < 1$$
Under fixed target parameters $\theta^-$, the Survival Bellman Operator:
$$(\mathcal{T}_{\theta^-} p)(s) = I[\text{Event in } \Delta t_j](s) + S_{\theta^-}(\Delta t_j) \cdot (\Pi \Phi_{\Delta t_j} p_{\theta^-})(s)$$
is a contraction mapping under the Cramér metric $\ell_2(F_P, F_Q) = \left( \int_0^\infty |F_P(u) - F_Q(u)|^2 du \right)^{1/2}$ with contraction modulus $\gamma_{\max}$.  
Crucially, using the target network $\theta^-$ is mandatory. If $S_\theta$ were updated simultaneously with the predictions, the operator would become non-linear in $\theta$, destroying contraction guarantees and creating runaway positive feedback loops (risk overestimation driving discount underestimation). The candidate's architectural decision to freeze $S_{\theta^-}$ via an EMA target network is strictly required.

### 4.2 Deadly Triad and Off-Policy Dynamics
Longitudinal survival data is collected offline from logged historical cohorts. In offline RL, policy evaluation encounters the Deadly Triad:
1. **Bootstrapping**: Step 5 bootstraps downstream predictions $p_{\theta^-}(s - \Delta t_j)$.
2. **Function Approximation**: Step 1 uses a recurrent sequential neural network.
3. **Off-Policy Data Distribution**: The empirical observation trajectory distribution $P(H_{t_j}, \Delta t_j)$ does not match the forward unrolled state visitation frequency.

To prevent divergence:
- SurvTD leverages **regularized $\lambda$-returns** ($\lambda < 1$), which exponentially dampens bootstrap error accumulation.
- SurvTD restricts updates to prospective distributions bounded on compact support $[0, s_{\max}]$.
- **Recommendation for Phase 4**: The proposal must formally state that the semi-gradient TD update is employed (i.e., gradients do not propagate through the target $G_j^\lambda$), preventing catastrophic off-policy gradient drift.

### 4.3 Formulation of the Distributional $\lambda$-Return Backward Recursion
To ensure exact implementation, Step 6 must be formalized. In scalar TD($\lambda$), the backward recursive target is:
$$G_j^\lambda = R_{j+1} + \gamma_j \left( (1 - \lambda) V(S_{j+1}) + \lambda G_{j+1}^\lambda \right)$$
In SurvTD, the return is a probability density/mass function over residual lifetime $s \in [0, s_{\max}]$. Let:
- $\Phi_{\Delta t_j}$: Shift operator on residual time, $\Phi_{\Delta t_j} p(s) = p(s + \Delta t_j)$ for $s \ge 0$.
- $\Pi$: Categorical Cramér projection onto the fixed bins $\{s_1, \dots, s_K\}$.
- $E_j(s)$: Immediate event mass distribution if failure occurs in $[t_j, t_{j+1}]$, with total mass $1 - S(\Delta t_j)$.

Then the exact backward recursion from terminal observation $J$ to step $j$ is:
$$G_J(s) = p_{\text{terminal}}(s)$$
$$G_j^\lambda(s) = E_j(s) + S_{\theta^-}(\Delta t_j \mid h_j) \cdot \Pi \left[ (1 - \lambda) \Phi_{\Delta t_j} p_{\theta^-}(s \mid h_{j+1}) + \lambda \Phi_{\Delta t_j} G_{j+1}^\lambda(s) \right]$$
This recursive relationship correctly compounds the temporal shift $\Delta t_j$ across multi-step targets while maintaining non-expansiveness at each step via projection $\Pi$.

### 4.4 Right-Censoring Absorption Boundary
A vital question in survival analysis is how right-censoring at final observation $t_J$ (where $I[\text{Event}] = 0$) is handled.
- If an individual is censored at $t_J$, no event occurred up to $t_J$. The model has no terminal empirical event delta $\delta(s - 0)$.
- In conventional RL terms, this is an episode termination without reward.
- In SurvTD, at a censoring boundary $t_J$, the bootstrap target must truncate: the residual lifetime distribution beyond $t_J$ must rely on the model's self-consistent tail prediction $p_{\theta^-}(s \mid h_J)$, conditioned on survival up to $t_J$. This boundary condition must be explicitly integrated into the backward recursion in Phase 4.

---

## 5. Actionable Requirements for Phase 4 (Planning & Implementation)

1. **Tone Adjustment on Theoretical Claims**:
   - Replace *"guarantee fixed-point convergence"* (Step 7) with *"guarantees contraction of the Bellman target under the Cramér metric for frozen target network parameters $\theta^-$, stabilizing semi-gradient neural TD learning"*.
2. **Explicit Distributional $\lambda$-Return Specification**:
   - Implement the explicit recursive distributional formula detailed in Section 4.3, ensuring that the continuous interval shift $\Phi_{\Delta t_j}$ and projection $\Pi$ are applied at every recursion step.
3. **Censoring Masking & Loss Weighting**:
   - Ensure the backward recursion gracefully handles right-censored trajectories by masking the terminal immediate event distribution $E_J(s)$ and relying on target network survival predictions for unobserved future horizons.
4. **EMA Target Decay Hyperparameter Sensitivity**:
   - In offline SMDPs, the EMA decay rate $\tau$ (e.g., $0.005$ or $\theta^- \leftarrow 0.995 \theta^- + 0.005 \theta$) directly governs the contraction stability. Include an ablation over $\tau \in \{0.001, 0.01, 0.1\}$ in the experimental design.

---

## 6. Final Verdict

- **Decision**: **`advance`**
- **Justification**: 
  The core mechanism of SurvTD is mathematically sound, highly novel, and directly solves the structural failure modes of previous temporal consistency survival models (eliminating $\div S$ explosion and bridging discrete $\Delta t = 1$ to irregular continuous time). The application of Cramér non-expansive projection from Distributional RL (`Rowland2018`) to continuous Semi-Markov survival policy evaluation is a first-class contribution for top-tier venues (NeurIPS / ICML / RLDM). The identified theoretical clarifications can be easily integrated into Phase 4 and Phase 5 without requiring fundamental structural changes.
