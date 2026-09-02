# Abstract — SurvTD

Dynamic survival analysis models remaining lifetime distributions from longitudinal biomarker trajectories recorded at irregular observation intervals. Enforcing temporal consistency across consecutive visits via unit-step backward updates requires dividing the next-step distribution by the interval survival probability, an operator that numerically diverges as patient risk escalates ($S \to 0$), collapsing predictions into uninformative flat distributions. To resolve this divergence, SurvTD reframes inter-observation transitions as continuous additive renewal shifts composed with categorical projections and discounted by model-predicted interval survival probabilities within a renewal mixture target. We prove that this composed operator is an affine strict contraction in the squared Cramér metric under a frozen target network, exactly conserving unit probability mass and stabilizing multi-step temporal-difference bootstrapping. Across MIMIC-IV Sepsis-3, Poisson-subsampled NASA C-MAPSS, and PBC cohorts under compute-matched continuous sequence backbones, SurvTD outperforms unit-step consistency and terminal likelihood baselines by at least 0.025 in time-dependent AUC and concordance while reducing clinical false alert episode rates and alert jitter by over 25% at matched 0.30 PPV. These performance gains vanish on regularly sampled cohorts where observation intervals are uniform and assume observation timing is conditionally unconfounded given latent clinical history. These findings demonstrate that continuous distributional temporal-difference learning provides a principled foundation for stable dynamic risk scoring without numerical division explosion or bedside alarm fatigue.

---

## Sentence-to-Section-to-Anchor Map

| # | Sentence Intent | Target Section | Evidence Anchor | Supported Claim |
| :-: | :--- | :--- | :--- | :-: |
| **S1** | **Setting** (Dynamic survival under irregular longitudinal telemetry) | §1 Introduction | `sec:1` | — |
| **S2** | **Bottleneck** (Unit-step division divergence $p/S \to \infty$ as risk rises) | §1 Intro / §3 Motivation | `fig:1` (Left Panel) | — |
| **S3** | **Proposed Mechanism** (Additive renewal shift $\Phi_{+\Delta t}$ + categorical projection $\Pi$) | §3 Method (Steps 1–5) | `fig:1` (Left Panel) | $C_0$ |
| **S4** | **Core Claim ($C_0$)** (Affine Cramér contraction, unit mass conservation, multi-step recursion) | §3.2 Theoretical Analysis | `thm:1`, `fig:2` | $C_0, C_1, C_2$ |
| **S5** | **Headline Evidence** (MIMIC-IV, C-MAPSS, PBC: $\ge 0.025$ gain in AUC/C-index, $\ge 25\%$ alert jitter reduction) | §4 Experiments | `tab:1`, `tab:2` | $C_3, C_4$ |
| **S6** | **Scope Boundary** (Vanishes on uniform $\Delta t \equiv 1$; requires conditional unconfoundedness) | §4.4 Ablations / §5 Scope | `tab:3` (NC-A/B), `sec:5` | $C_0, C_3$ |
| **S7** | **Takeaway** (Distributional TD as stable foundation for clinical dynamic risk scoring) | §6 Conclusion | `sec:6` | $C_0$ |
