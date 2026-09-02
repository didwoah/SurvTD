# Scoop-Radar report — SurvTD (attempt 2 candidate)

**Date**: 2026-09-03 · **Channels searched**: signature (arxiv, openreview; openalex/s2 returned
HTTP 400/429 and contributed nothing — recall on published venues is therefore thinner than the
raw counts suggest) and alias (arxiv, openreview; dblp/crossref likewise unavailable).
Pools: `scoop/signature.json` (14 on-topic of 54 raw), `scoop/alias.json` (47 on-topic of 140 raw).

## 1. Four-axis decomposition of the claim

| Axis | This candidate |
|---|---|
| **Problem framing** | Dynamic (updated-at-each-visit) time-to-event prediction from irregularly sampled longitudinal covariates under right-censoring; evaluated on discrimination, calibration, and prediction stability across consecutive visits. |
| **Core mechanism** | Replace the `÷ S(Δt)` backward update of unit-step temporal-consistency survival methods with a renewal rightward shift composed with a categorical projection onto a fixed lifetime grid, discounted by a model-predicted interval survival `γ_j = S_{θ⁻}(Δt_j)` from an EMA target network; multi-step targets mixed by a λ that decays in elapsed duration rather than in visit count. |
| **Key insight** | Elapsed time between observations is not a nuisance to be rounded to 1 — it is the amount of accumulated non-event evidence, and it enters the operator as a duration discount. The division that made this impossible is removable, and what replaces it preserves mass conservation and non-expansiveness in the Cramér metric. |
| **Application domain** | ICU / clinical telemetry (Sepsis-3, PBC), industrial degradation (C-MAPSS), simulated ODE/SDE cohorts. |

## 2. Nearest prior art

| Rank | Work | Axes matched | Why it is not this |
|---|---|---|---|
| 1 | **TCSR** (`Maystre2022`, NeurIPS 2022) | framing, insight (partial) | Same objective family — consistency between consecutive predictions — but the transition is a uniform unit step and the update renormalises by dividing by interval survival. The operator this candidate substitutes *is* TCSR's update. Nearest work; the delta is the operator, not the goal. |
| 2 | **DeepTCSR** (`DeepTCSR2024`) | framing, mechanism (partial) | Adds target networks and end-to-end training; inherits the unit-step transition unchanged. Shares the target network but uses it for stability of a different update. |
| 3 | **CoxSig** (`Bleistein2024`, ICML 2024) | framing, domain | Addresses irregular sampling in the *encoder* (path signatures) with a linear PH head and terminal MC likelihood. Orthogonal: could be used as SurvTD's encoder. |
| 4 | **Dynamic-DeepHit** (`Lee2019`) | framing, domain | Terminal likelihood + pairwise ranking; imposes no inter-visit relation at all. |
| 5 | **C51 / categorical projection** (`Bellemare2017`) and follow-ups retrieved this round — *Foundations of Multivariate Distributional RL*, *A Finite-Iteration Theory for Asynchronous Categorical Distributional TD*, *Parameterized Projected Bellman Operator* | mechanism (the projection machinery) | The projection operator and its non-expansiveness are imported, not invented. None of the retrieved distributional-RL work treats remaining lifetime, an endogenous model-predicted discount, or censoring. This is the honest boundary of the mechanism's novelty. |
| 6 | *Decoupling risk and masking in mammographic density under irregular follow-up using a latent Markov progression* (alias channel, 2024–25) | framing, domain | Irregular follow-up in survival, but a latent-state statistical model with no consistency objective or bootstrapped target. |

**No contemporaneous scoop found.** No retrieved record combines a survival/time-to-event target
with a projected distributional Bellman update over variable elapsed durations. The alias channel
returned continuous-time RL (SMDP, PhiBE, diffusion control) and distributional-RL operator theory
as two separate literatures; the candidate sits at their intersection with survival analysis, and
that intersection is unoccupied in this pool.

## 3. Overlap level and delta

**Overlap level: 2 / 5 — adjacent lineage, distinct technical move.**
(1 = unoccupied · 2 = adjacent lineage · 3 = same mechanism, different domain · 4 = near-duplicate ·
5 = published.)

> **Delta statement.** Temporal-consistency survival methods obtain their target by renormalising the
> next-visit lifetime distribution by the interval survival probability, which is defined only for a
> uniform unit step and diverges as that probability approaches zero. SurvTD substitutes that update
> with a renewal shift and categorical projection discounted by a model-predicted interval survival,
> which is defined at any real interval length, conserves probability mass exactly, and is
> non-expansive in the Cramér metric — importing the projection machinery from categorical
> distributional RL, where the projected quantity is a reward distribution under an exogenous
> discount, and adapting it to remaining lifetime under an endogenous, censored one.

**Caveat for the panel.** OpenAlex, Semantic Scholar, DBLP and Crossref were all unreachable this
run, so published-venue recall rests on arXiv and OpenReview alone. Treat the "no scoop" finding as
provisional on the two channels that answered.
