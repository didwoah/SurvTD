# Abstract — SurvTD

Dynamic survival analysis models remaining lifetime distributions from longitudinal biomarker trajectories recorded at irregular observation intervals. While temporal consistency frameworks have demonstrated the promise of temporal-difference (TD) learning in survival analysis, existing methods remain restricted to uniform discrete steps ($\Delta t = 1$) and lack theoretical convergence guarantees under non-linear neural representations. When applied to real-world clinical telemetry, unregularized landmark models exhibit severe trajectory variance and bedside alarm thrashing, while naive continuous extensions of discrete consistency encounter boundary leaks and numerical instability in high-risk regimes. To establish a sound continuous-time formulation, SurvTD reframes inter-observation transitions as continuous additive renewal shifts composed with categorical projections and discounted by model-predicted interval survival probabilities within a renewal mixture target. We prove that this composed operator is an affine strict contraction in the squared Cramér metric under a frozen target network, exactly conserving unit probability mass and providing the first formal contraction guarantee for survival temporal-difference learning. To maintain temporal fidelity across heterogeneous observation rates, SurvTD compounds interval survival discounting through duration-geometric $\lambda$-returns, preserving an invariant effective prediction horizon on the physical time axis. Across MIMIC-IV Sepsis-3, Poisson-downsampled NASA C-MAPSS, and PBC cohorts under compute-matched continuous sequence backbones, SurvTD improves time-dependent concordance by over 0.025 while suppressing clinical false alert episode rates and alert jitter by over 25% at matched 0.30 PPV. These findings demonstrate that continuous distributional temporal-difference learning provides a theoretically grounded, sampling-invariant foundation for dynamic survival monitoring that directly mitigates bedside alarm fatigue.

---

## Sentence-to-Section-to-Anchor Map

| # | Sentence Intent | Target Section | Evidence Anchor | Supported Claim |
| :-: | :--- | :--- | :--- | :-: |
| **S1** | **Setting** (Dynamic survival under irregular longitudinal telemetry) | §1 Introduction | `sec:1` | — |
| **S2** | **Motivation** (Discrete unit-step constraint, lack of convergence theory, and clinical alarm thrashing) | §1 Intro / §3 Motivation | `fig:1` (Left/Right) | — |
| **S3** | **Proposed Mechanism** (Additive renewal shift $\Phi_{+\Delta t}$ + categorical projection $\Pi$) | §3 Method (§3.1) | `fig:1` (Left Panel) | $C_0$ |
| **S4** | **Theoretical Guarantee** (Theorem 1: Affine strict Cramér contraction and unit mass conservation) | §3.2 Theoretical Analysis | `thm:1` | $C_1$ |
| **S5** | **Horizon Invariance** (Duration-geometric $\lambda$-returns maintaining physical half-life) | §3.3 Multi-Step Target | `fig:2` | $C_2$ |
| **S6** | **Headline Evidence** (MIMIC-IV, C-MAPSS, PBC: $\ge 0.025$ C-index gain, $\ge 25\%$ alert jitter reduction) | §4 Experiments | `tab:1`, `tab:2` | $C_3, C_4$ |
| **S7** | **Takeaway** (Continuous TD as a sound, alarm-stable foundation for dynamic survival monitoring) | §6 Conclusion | `sec:6` | $C_0$ |
