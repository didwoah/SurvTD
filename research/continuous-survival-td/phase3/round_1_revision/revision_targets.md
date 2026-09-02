# Phase 3 Round 1: Deduplicated Mandatory Revision Targets

These revision targets were synthesized from the 5 independent Opus review seats (`chair`, `theorist`, `empiricist`, `domain`, `insider`) in Round 1 Evaluation (`research/continuous-survival-td/phase3/round_1_eval/`). Each item below was raised by ≥ 3 seats and constitutes a mandatory kill-criteria repair.

---

### Target 1: Wire the Load-Bearing Variable into the Step 5 Bellman Target Equation
- **Raised by**: Chair, Theorist, Empiricist, Domain, Insider (5/5 seats - Unanimous).
- **Flaw**: $\gamma_j = S_{\theta^-}(\Delta t_j)$ was declared as the primary load-bearing variable and target of NC-A, but was completely absent from the Step 5 target formula ($T p_j = \mathbb{I} \delta + (1-\mathbb{I})\Pi\Phi p_{\theta^-}$). NC-A had no operand.
- **Mandatory Fix**: Rewrite Step 5 in explicit renewal mixture form: $\mathcal{T} p_j = (1 - \gamma_j) \cdot \mu_{[0, \Delta t_j)} + \gamma_j \cdot (\Pi \Phi_{+\Delta t_j} p_{\theta^-})_j$.

### Target 2: Scope the Contraction Claim to the Frozen-Target Inner Loop
- **Raised by**: Theorist, Domain, Insider (3/5 seats).
- **Flaw**: When $\gamma$ is an endogenous model prediction $1 - F_\eta(\Delta t)$, the operator has modulus up to 5.04 in Cramér and is NOT contractive.
- **Mandatory Fix**: Scrupulously state that contraction modulus $\gamma_j < 1$ holds strictly for the inner loop where $\gamma_j$ is held exogenous by the frozen target network $\theta^-$. Joint outer-loop convergence is an empirical two-timescale scheme. Add sub-bin hazard interpolation ($S(\Delta t) = 1 - h_0 \Delta t / \delta_s$) to prevent vanishing contraction as $\Delta t \to 0$.

### Target 3: Acknowledge and Bound Projection Diffusion
- **Raised by**: Theorist, Empiricist, Domain (3/5 seats).
- **Flaw**: The claim that bootstrapping is "invariant to sampling rate" is false for distribution shape: categorical projection $\Pi$ adds $\approx \delta_s^2 / 6$ variance per step, diffusing target dispersion as $O(\sqrt{n})$ over identical physical time $T$.
- **Mandatory Fix**: Retract the claim of distributional invariance. Restrict invariance strictly to the *effective bootstrapping horizon*, and bound per-step projection diffusion by $\delta_s^2 / 6$.

### Target 4: Disentangle Negative Control A (`NC-A`)
- **Raised by**: Chair, Empiricist, Domain, Insider (4/5 seats).
- **Flaw**: Replacing $\gamma_j \to S(\delta_s)$ while keeping $\Phi_{+\Delta t_j}$ duration-dependent leaked duration information, creating a hybrid arm rather than the unit-step baseline.
- **Mandatory Fix**: Split NC-A into:
  - Arm A1: Discount ablation ($\gamma_j \equiv S(\delta_s)$, shift kept continuous).
  - Arm A2: Shift ablation ($\Phi_{+\delta_s}$ unit shift, discount kept duration-dependent).
  - Arm A3: DeepTCSR clamped division comparison ($\div S(\Delta t_j)$ with $S \ge 10^{-3}$).

### Target 5: Realize the NASA C-MAPSS Benchmark Protocol
- **Raised by**: Empiricist, Domain (2 seats, fatal benchmark trap).
- **Flaw**: NASA C-MAPSS is cycle-based ($\Delta t \equiv 1$) and uncensored (ground-truth RUL provided), making it incapable of testing irregular duration discounting natively.
- **Mandatory Fix**: Formally declare an irregular Poisson downsampling protocol (50% random cycle drop) to induce controlled continuous-time irregularity.

### Target 6: Truncate and Condition the IPCW Tail Completion (Step 6)
- **Raised by**: Theorist, Domain (2 seats, mathematical paradox).
- **Flaw**: Using marginal Kaplan-Meier $1/\hat{G}(t)$ violated covariate independence (discharge alive in MIMIC-IV tracks health) and reintroduced divergent division in the tail.
- **Mandatory Fix**: Use a covariate-conditional censoring model (Cox / Random Survival Forest) with weights truncated at $1/\hat{G}(t \mid X) \le 10.0$ to eliminate tail division instability.

### Target 7: Formally State the Observation Process Assumption
- **Raised by**: Chair, Empiricist, Domain, Insider (4/5 seats).
- **Flaw**: ICU observation intervals are informative ($\lambda_{\text{obs}}(t) \propto \text{Risk}(t)$). Reading $\Delta t$ as pure survival evidence conflates clinical protocols with disease risk.
- **Mandatory Fix**: Formally declare the conditional unconfoundedness assumption ($T \perp \Delta t \mid h_j$) and add within-patient duration permutation (NC-B) to isolate temporal alignment.

### Target 8: Declare Comparative Baselines and Rebudget Compute Realistically
- **Raised by**: Chair, Empiricist, Domain, Insider (4/5 seats).
- **Flaw**: 18 GPU-hours was fantasy for 340 runs; baselines lacked tuning parity and shared encoders.
- **Mandatory Fix**: Allocate 120 GPU-hours on shared GRU-D/LSTM backbones, define explicit comparative baselines (Dynamic-DeepHit, DeepTCSR with clamped division, CoxSig, Person-period expanded discrete hazard), and instrument alarm stability metrics (PPV 0.30, alert jitter).
