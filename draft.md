# Mean Rescaling Is Not Enough: Semi-Markov Temporal-Difference Learning for Real-Time Survival Distributions

> **Status:** ICLR-style 8-page draft. Linear-setting results are final; deep-learning results
> (§5.4) are **placeholders to be filled**. Marked `[TBD]` throughout.

---

## Abstract

Clinical and industrial monitoring produce observations at *irregular* intervals, yet the
dominant temporal-difference (TD) approach to survival analysis treats every consecutive
observation as one discrete step. The elapsed time between observations is then either discarded
or recovered post-hoc by rescaling visit counts with an average inter-observation interval.
We show this is not merely lossy but **structurally insufficient**: we construct states whose
visit-count law, mean holding time, and mean residual lifetime are all identical, yet whose
real-time event distributions differ substantially — a 0% versus 20% risk of death within
half an hour. No rescaling by any mean, *including the population-optimal one*, can recover the
difference. We introduce **SM-TCSR**, a distributional TD method that treats the observation
process as a Markov renewal process and shifts the bootstrapped target by the *actual* elapsed
time. The method is a single masked Cramér objective with a type-dispatched inner penalty, uses
the exact censoring likelihood for partially observed intervals, and supports both a finite
horizon and an integrable-weight infinite horizon — the latter being necessary because the
unweighted Cramér integral diverges under competing risks. On a controlled benchmark of 1,385
runs where all methods share an identical linear survival head, SM-TCSR improves real-time
probabilistic prediction on AIDS ($\sqrt{\mathrm{IBS}}$ 0.362 vs 0.380) and PBC2 (0.329 vs 0.334),
while — by design — winning nothing in the unit-time control condition. On a synthetic process
with state-dependent holding times, SM-TCSR is the only method that does not degrade as
irregularity increases, winning on all 5/5 seeds. We report the conditions where it loses.

---

## 1 Introduction

Intensive-care records, longitudinal clinical trials, and machine telemetry all share a
structure: the state of the subject is revealed at *irregular*, often informative, times. A
patient may be re-measured after 30 minutes or after 5 hours. The prediction we want is
continuous — *what is the probability of death within the next 6, 24, or 72 hours?* — and it must
be updated whenever a new observation arrives.

Temporally-consistent survival analysis (TCSR; Maystre & Russo, 2022) and its deep extension
(DeepTCSR; Vargas Vieyra & Frossard, 2024) address the *learning* side of this problem
elegantly. They observe that the predicted event-time distribution at one visit should agree
with the prediction at the next visit, shifted forward in time, and turn this consistency into a
temporal-difference target. This uses censored trajectories efficiently and improves
data efficiency in the small-sample regime.

But both operate on a **visit-index** clock: the shift is always exactly one step. A transition
spanning 30 minutes and one spanning 9 hours are treated identically.

### 1.1 The obvious fix does not work

The natural repair is to convert visit counts back to wall-clock time by multiplying by a mean
inter-observation interval — globally, or per state. We call this a *mean clock*. Figure 1 shows
why this fails.

**[FIGURE 1: (a) state graph with holding-time densities; (b) State B real-time CDF]**

Two states $A$ and $B$ have an *identical* transition graph: from either, 40% of subjects die
immediately and 60% pass through an intermediate state $C$ before dying. Their visit-count law is
identical ($P(N{=}1) = 0.40$), their mean holding time is identical (1 hour), and their mean
residual lifetime is identical (1.6 hours). The only difference is the *shape* of the holding-time
density: $A$ is unimodal, $B$ is a 50/50 mixture of a fast and a slow path.

The consequence is not subtle. The true probability of death within 0.5 hours is **0% for $A$ and
20% for $B$**. Every mean clock — global, per-state, and even an *oracle* clock given the exact
population means — predicts 0% for both. One subject in five dies within half an hour and the
model reports no risk at all. Integrated over the horizon, the squared CDF error is
$5.9\times10^{-3}$ for all three mean clocks versus $1.5\times10^{-5}$ for a method that uses the
real elapsed time: a factor of ~390, and it does **not shrink with sample size**, because it is a
limit of expressiveness rather than of estimation.

### 1.2 Contributions

1. **A structural impossibility result, empirically instantiated.** We construct a semi-Markov
   process where visit-count law, mean holding time, and mean residual lifetime are matched
   exactly across states, yet real-time risk differs by 20 percentage points at a clinically
   relevant horizon. No mean-based rescaling can close the gap (§5.1).

2. **SM-TCSR**, a distributional TD method on the real-time axis. The target shifts the next
   state's cumulative incidence by the *actual* elapsed time $D_n$ using exact evaluation
   (continuous head) or CDF interpolation (grid head), never integer rounding. The objective is a
   **single** masked Cramér expectation with a per-transition-type inner penalty, so no balancing
   hyperparameter between a TD term and a supervision term is introduced (§3).

3. **A principled treatment of the horizon.** We show the unweighted Cramér integral
   $\int_0^\infty$ **diverges** under competing risks, because cause-specific CIFs converge to
   $\pi_c \ne 1$; a difference of 0.1 in the limit makes the loss infinite. We give two
   convergent alternatives — finite horizon and integrable weighting — that differ only in a
   sampler (§3.5).

4. **A controlled benchmark that separates two confounded effects.** Across 1,385 runs with zero
   failures, all five arms share an identical linear survival head. A unit-time control condition
   isolates the *learning rule*; a real-time condition isolates the *time representation*.
   SM-TCSR wins nothing in the control — which is what licenses attributing its real-time gains
   to the time representation (§5.2–5.3).

We report negative results in the main text. SM-TCSR does not improve concordance on the random
walk, and coefficient recovery is best for the simplest baseline for a reason we explain.

---

## 2 Problem Setup

### 2.1 Observations and transitions

A subject produces observations at times $t_0 < t_1 < \dots$, with covariates $x_n$. Write
$\mathcal H_{t_n}$ for the history up to $t_n$ and $s_n = \mathrm{Encoder}(\mathcal H_{t_n})$ for
the state representation. A **transition** ends at whichever comes first: the next observation,
or a terminal event. Its duration is $D_n \in \mathbb{R}^{+}$.

Terminal events are *competing*: for ICU mortality the causes are death and alive-discharge.
Treating discharge as censoring biases mortality upward, since a discharged subject does not
experience the event at all. We therefore predict cause-specific **cumulative incidence
functions** (CIFs):

$$F_c(\tau \mid s) = P\bigl(Z \le \tau,\ J = c \mid s,\ \text{no terminal event yet}\bigr),
\qquad S(\tau\mid s) = 1 - \sum_c F_c(\tau\mid s),$$

where $Z$ is the time to the first terminal event and $J$ its cause. Note $\tau$ (a *query* time
asked of the model) and $D_n$ (an *observed* interval) are distinct real-valued quantities; they
meet only in the target construction as $\tau - D_n$.

### 2.2 The semi-Markov assumption

We assume the observation-time pairs form a **Markov renewal process**:

$$P\bigl(s_{n+1}\in B,\ D_n \le u \mid \mathcal H_{t_n}\bigr) = Q(B, u \mid s_n).$$

The current state suffices to predict the *joint* law of the next state and the time to reach it.
Two properties matter. First, the next state and the elapsed time need not be independent — sicker
states may transition sooner, and the model may exploit this. Second, the holding time is not
restricted to be exponential; restricting it would collapse the process to a Markov chain and make
elapsed time uninformative. This is what makes the setting *semi-*Markov.

We state clearly what is **not** claimed. The model uses time between *observations*, not between
latent clinical states; it does not assume the subject is unchanged between observations, nor
does it estimate the true onset time of a latent state. Irregular sampling is not by itself
evidence of semi-Markov structure — the above is an explicit modelling assumption. Finally, using
a recurrent or attention encoder does not automatically make $s_n$ sufficient.

### 2.3 What must the head provide

The CIF must be evaluable at arbitrary real $\tau$, with $F_c(0\mid s)=0$, monotone in $\tau$, and
$\sum_c F_c \le 1$. Our default is a cause-specific Weibull mixture, which satisfies all three by
construction and admits exact shifting by substituting $\tau - D_n$. A discrete-hazard grid head
is a supported alternative (§3.4).

---

## 3 Method: SM-TCSR

### 3.1 The temporal-consistency relation

If no event occurs between $t$ and $t+\Delta t$, the residual lifetime shrinks by exactly the
elapsed amount: $R_t = \Delta t + R_{t+\Delta t}$. In distribution, the two predictions are
related by a pure time shift. Using this as a learning signal is the TD idea; using the *actual*
$D_n$ as the shift is what distinguishes SM-TCSR from visit-index TD.

### 3.2 TD targets

**[FIGURE 2: the three transition types]**

Each transition is one of three types, producing a target $\widetilde F$ and an observation mask
$m$.

**(a) `survive` — the next observation is reached alive.**

$$\widetilde F_{n,c}(\tau) =
\begin{cases}
0, & 0 \le \tau < D_n,\\
F_{\bar\theta,c}(\tau - D_n \mid s_{n+1}), & \tau \ge D_n,
\end{cases}\qquad m \equiv 1.$$

Since $F_c(0)=0$ the target is continuous at $\tau = D_n$. Here $\bar\theta$ are target-network
parameters.

**(b) `event` — a terminal event of cause $c^\ast$ is observed.**

$$\widetilde F_{n,c}(\tau) = \mathbf 1\{c = c^\ast\}\,\mathbf 1\{\tau \ge D_n\},\qquad m \equiv 1.$$

No bootstrapping. These step targets anchor the whole fixed point to observed outcomes.

**(c) `censor` — follow-up ends without a next observation.** All that is known is that no event
occurred before $r^c$. The information is used through the exact censoring likelihood (§3.3).

We never convert censoring into an event, never fabricate a next state, and never drop censored
subjects. Crucially, *knowing the post-censoring outcome* and *letting the model predict from the
last observed state* are different things; the latter is legitimate, and the last state's
prediction serves as the bootstrap target of the preceding transition.

Under the assumption that the state is sufficient and all transition types are included, the true
model satisfies $F^\ast_c(\tau\mid s) = \mathbb E[\widetilde F^{\ast}_{n,c}(\tau)\mid s_n = s]$.
A single next state does not represent all futures; the conditional mean over many transitions
does.

### 3.3 A single objective

Let $\kappa(n)\in\{\texttt{survive},\texttt{event},\texttt{censor}\}$. The loss is one expectation
over transitions:

$$\boxed{\ \mathcal L(\theta) = \mathbb E_n\Bigl[\ \omega_{\kappa(n)}\cdot
\ell_{\kappa(n)}\bigl(F_\theta(\cdot\mid s_n),\ \mathrm{sg}[\widetilde F_n]\bigr)\Bigr]\ }$$

with a type-dispatched inner penalty and a per-type weight $\omega_\kappa$ (default 1). Every
transition contributes to exactly one penalty; there is no additive second term and hence no
balancing coefficient to tune.

For `survive` and `event`:

$$\ell = \sum_c \int_0^\infty w(\tau)\bigl(F_{\theta,c}(\tau\mid s_n) - \widetilde F_{n,c}(\tau)\bigr)^2 d\tau .$$

For `censor` we use the **survival-conditional continuation target (SCCT)** — the exact likelihood
contribution of a censored observation:

$$\ell_{\texttt{censor}} = -\log S_\theta(r^c \mid s_n)
= -\log\Bigl(1 - \sum_c F_{\theta,c}(r^c\mid s_n)\Bigr).$$

SCCT requires no IPCW weights, so it avoids IPCW variance inflation and clamping bias, and it
recovers information from the post-last-observation interval rather than discarding it. Its
gradient grows without bound as $F_\theta(r^c)\to1$; we clip gradients. An alternative
squared-path penalty $\int_0^{r^c} w\,F_{\theta,c}^2\,d\tau$ has bounded gradients and shares the
Cramér scale. The two encode the same information with different geometry, so we treat the choice
as an ablation (A1, §5.5).

**Why squared CDF error.** The objective is a (weighted, masked) squared Cramér distance. Unlike
the Wasserstein distance, it admits unbiased minibatch gradient estimates — decisive when
bootstrapping with stochastic optimisation. It is scale-sensitive of order 1, so error is
penalised in proportion to *how far off in time* the prediction is, which the visit-index setting
does not require but the real-time setting does. For `event` transitions the target is a step
function and the penalty is exactly the CRPS, a proper scoring rule in the same family as the
Brier/IBS metrics we evaluate with.

### 3.4 Shifting without rounding

Visit-index methods shift by exactly one index, so an array roll is exact. Real $D_n$ is not a
grid multiple; rounding to the nearest bin injects up to half a bin of error *per transition*, and
TD recursion accumulates it. With a continuous head, substituting $\tau - D_n$ is exact. With a
grid head we interpolate the **CDF**, not the hazard: discrete hazards $h_k$ condition on
different events per bin, so an interpolate of $h_2$ and $h_3$ corresponds to no distribution,
whereas $F$ is an ordinary monotone function of $\tau$ and remains a valid CDF under
interpolation. Empirically, on a Weibull reference shifted by a non-grid amount, CDF
interpolation attains max error 0.0028 and recovers the tail mass to 0.0338 (truth 0.0337), while
hazard interpolation gives 0.0549 and 0.0278 respectively.

### 3.5 Horizon: two convergent choices

Under competing risks, $\lim_{\tau\to\infty}F_c(\tau) = \pi_c \neq 1$. If the model and the target
disagree about $\pi_c$ by even a small amount, the integrand converges to a positive constant and

$$\int_0^\infty \bigl(F_{\theta,c}(\tau)-\widetilde F_c(\tau)\bigr)^2 d\tau = \infty .$$

Numerically, with $F = 0.2(1-e^{-0.5\tau})$ and $G = 0.3(1-e^{-0.5\tau})$, the truncated integral
grows exactly linearly in the upper limit (1.97 at 200, 19.97 at 2000). Two convergent choices:

| | weight | sampler | use |
|---|---|---|---|
| **finite horizon** | $w(\tau)=\frac1H\mathbf 1\{\tau\le H\}$ | $\tau_m \sim \mathrm{Unif}(0,H)$ | fixed clinical horizons |
| **weighted infinite** | $w(\tau)>0$, $\int w = 1$, e.g. $\rho e^{-\rho\tau}$ | $\tau_m \sim w$ | no fixed cutoff |

The first is the special case $w = \frac1H\mathbf 1\{\tau\le H\}$ of the second; in code only the
sampler changes. Importantly, integrating to $H$ does **not** assume the event occurs by $H$: we
never impose $F_\theta(H)=1$, and $1-\sum_c F_{\theta,c}(H)$ is retained as tail mass. Note also
that $\rho$ selects *which prediction times matter*, and is not a reward discount — the TD target
still shifts by the actual $D_n$.

### 3.6 Algorithm

Targets are computed under `stop_gradient` from an EMA target network,
$\bar\theta\leftarrow(1-\eta)\bar\theta+\eta\theta$. Integration points $\tau_m$ are drawn from
$w$ and are *independent of the model's output grid*. Splits are at the subject level.
Normalisation statistics are fit on the training split only. Pseudocode is in Appendix A.

---

## 4 Related Work

**Temporal consistency in survival analysis.** TCSR (Maystre & Russo, 2022) introduced TD learning
over discrete visit indices with a proportional-odds discrete-hazard head, and DeepTCSR (Vargas
Vieyra & Frossard, 2024) added an end-to-end encoder and an EMA target network. Both apply a
one-step shift and weight the cross-entropy by predicted survival; neither divides by a survival
probability, and neither models elapsed time. We inherit their principle for using censored
trajectories and change the shift from an index to a duration.

**Dynamic survival with irregular sampling.** Dynamic-DeepHit (Lee et al., 2020) and SurvLatent
ODE (Moon et al., 2022) model longitudinal covariates with competing risks; they are trained by
direct likelihood/ranking rather than temporal-difference bootstrapping, and serve as our external
baselines in §5.4.

**Semi-Markov decision processes and distributional RL.** Variable-duration transitions in
value-based RL date to Bradtke & Duff (1994); Fatemi et al. (2022) apply semi-Markov offline RL to
healthcare, predicting scalar value rather than an event-time distribution. Frost et al. (2024)
use TD on irregular ICU data for a scalar mortality risk. Our objective is the squared Cramér
distance, standard in distributional RL (Bellemare et al., 2017; Rowland et al., 2018) for its
unbiased sample gradients; we apply it to cause-specific CIFs on the real-time axis rather than to
return distributions.

**Positioning.** Real-time shifting, target networks, and Cramér losses each exist. The
contribution is the combination as a semi-Markov survival TD target, the single-objective
formulation with SCCT censoring, and the horizon analysis that the competing-risks setting forces.

---

## 5 Experiments

We separate two questions that are conflated in prior work: *does the learning rule help?* and
*does the time representation help?* Every arm in §5.2–5.3 shares an **identical linear survival
head** $h(k\mid x)=\sigma(\alpha_k+\beta^\top x)$, present in both reference implementations, so
architecture is held fixed.

**Arms.** `Init. state` (MLE on first observations), `Landmarking` (MLE on all landmark
sub-sequences), `TCSR` ($\lambda=0$ TD on the visit axis), `DeepTCSR` (EMA target network, shipped
linear backbone), and `SM-TCSR` (ours).

**Protocol.** Subject-level 60/20/20 splits, 5 seeds (a repeated random split, *not* 5-fold CV),
nested training subsets at fractions $\{0.2,\dots,1.0\}$ so all arms see exactly the same subjects.
Preprocessing and the mean clock are fit on the training subset only; the IPCW censoring
distribution is estimated on training data. Hyperparameters are selected by validation IPCW-IBS
under an identical rule for all arms. Total 1,385 unique runs, **0 failures**.

| dataset | subjects | transitions | censored | real $D$ range | train $n$ (20%→100%) |
|---|---|---|---|---|---|
| AIDS | 467 | 938 | 59.7% | 2–6 months (grid) | 56 → 280 |
| PBC2 | 312 | 1,633 | 55.1% | **48–2,107 days (44×)** | 37 → 187 |
| Random Walk | 5,000 | ~18,500 | 39.0% | controlled by $\gamma$ | 600 → 3,000 |

*Table 1: Datasets. AIDS and PBC2 raw sources contain real visit timestamps that the standard
preprocessing pipeline discards; we retain them.*

### 5.1 Mean rescaling is structurally insufficient

The construction of Figure 1 is a semi-Markov process with $Q(s',du\mid s)=p(s'\mid s)q_s(du)$,
three transient states, no censoring, and at most two transitions, evaluated against the
*analytic* population CDF so that no finite test set noise enters. We compare mean clocks of
increasing strength: global, per-state, and an **oracle** given the exact population means (the
transition probabilities are still estimated).

At $N=5{,}000$ subjects over 30 replicates, the integrated squared CDF error per unit horizon is
$5.85\times10^{-3}$ (global), $5.85\times10^{-3}$ (per-state), $5.85\times10^{-3}$ (oracle), and
$1.49\times10^{-5}$ for real-duration TD — a factor of ~390. The three mean clocks are
indistinguishable from each other and their error **does not decrease with sample size**: it is an
expressiveness limit. At $\tau = 0.5$ h the true risk for state $B$ is 0.200; all mean clocks
report 0.000, real-duration TD reports 0.206.

**A diagnostic that does not favour us.** Against a *direct empirical survival* baseline that
regresses total event times without any TD or shifting, real-duration TD attains
$1.487\times10^{-5}$ versus $1.500\times10^{-5}$; the paired difference is
$-1.3\times10^{-7}\pm3.9\times10^{-7}$ (MC SE) — **indistinguishable**. We therefore claim that
the time distribution is recovered, *not* that TD is superior here. This toy is not a TD-specific
problem.

### 5.2 Linear benchmark: separating learning rule from time representation

**[FIGURE 3: Setting A (control) and Setting B (real time), three datasets]**

**Setting A (control, $D \equiv 1$).** All arms predict on the visit axis; SM-TCSR receives no
temporal advantage. It should not win — and it does not, on any dataset. `Landmarking` attains the
best visit-axis concordance on all three (AIDS 0.7063, PBC2 0.8297, RW 0.8485). This null result
is the licence for interpreting Setting B.

**Setting B (real time).** SM-TCSR uses actual $D_n$; the other arms convert visit counts with the
training-subset mean $\bar D$ and are labelled *mean-clock*. All arms are scored against the same
event times and horizon.

| | AIDS C-index ↑ | AIDS $\sqrt{\mathrm{IBS}}$ ↓ | PBC2 C-index ↑ | PBC2 $\sqrt{\mathrm{IBS}}$ ↓ |
|---|---|---|---|---|
| Init. state (mean-clock) | 0.6930 ± 0.0267 | 0.3808 ± 0.0193 | 0.8017 ± 0.0182 | 0.3547 ± 0.0166 |
| Landmarking (mean-clock) | 0.6961 ± 0.0260 | 0.3976 ± 0.0177 | **0.8242 ± 0.0279** | 0.3343 ± 0.0163 |
| TCSR (mean-clock) | 0.6887 ± 0.0297 | 0.3801 ± 0.0215 | 0.8149 ± 0.0247 | 0.3457 ± 0.0141 |
| DeepTCSR (mean-clock) | 0.5974 ± 0.0409 | 0.4200 ± 0.0042 | 0.8013 ± 0.0303 | 0.3759 ± 0.0357 |
| **SM-TCSR (ours)** | **0.7046 ± 0.0240** | **0.3619 ± 0.0202** | **0.8258 ± 0.0364** | **0.3293 ± 0.0273** |

*Table 2: Real-time evaluation at 100% of the training pool, mean ± s.d. over 5 seeds.*

SM-TCSR gives the best probabilistic prediction on both clinical datasets, with the larger margin
on AIDS (0.0182 in $\sqrt{\mathrm{IBS}}$). The PBC2 concordance margin over `Landmarking` (0.0015)
is far below the seed s.d. and we report it as a tie.

**Two caveats we do not hide.** (i) On the random walk at $\gamma=0$, SM-TCSR does not win; all
inter-arm margins there (0.0002–0.003) are one to two orders of magnitude smaller than the seed
s.d. (0.02–0.03) and should be read as ties. (ii) DeepTCSR degrades as AIDS data grows
(0.652→0.602). AIDS is 59.7% censored with at most five visits, so zero-padding dominates, and the
as-published censoring mask admits padded rows into the loss. We retained the published behaviour;
this weakness should therefore not be attributed to the algorithm alone.

### 5.3 Performance as a function of irregularity

**[FIGURE 4: sweep over state-dependent duration strength $\gamma$]**

To vary irregularity in a controlled way we hold the embedded chain *bit-identical* — same initial
states, transitions, events, and 39.0% censoring — and attach only the durations, so that
$\mathbb E[D]$ is preserved and only the *shape* and *state-dependence* change. The knob $\gamma$
scales how strongly the **mean** holding time depends on the state,
$m(s)\propto\exp(-\gamma z(s))$ with $z$ the standardised risk score: higher-risk states are
observed sooner, which is the defining property of a semi-Markov process and the pattern seen in
clinical follow-up. At $\gamma=1$ the spread of $D$ is ~35×, still milder than PBC2's measured 44×.

At $\gamma=0$ (the control, $\mathbb E[D\mid s]$ constant) the mean clock is exact in expectation
and all methods coincide. As $\gamma$ grows, every mean-clock arm degrades while **SM-TCSR stays
flat**:

| $\gamma$ | Landmarking | TCSR | DeepTCSR | SM-TCSR |
|---|---|---|---|---|
| 0.0 | **0.3488** | 0.3536 | 0.3502 | 0.3488 |
| 0.5 | 0.3449 | 0.3442 | 0.3476 | **0.3440** |
| 1.0 | 0.3757 | 0.3697 | 0.3764 | **0.3442 ± 0.0084** |

*Table 3: $\sqrt{\mathrm{IBS}}$ at 100% training data, 5 seeds. At $\gamma=1$ SM-TCSR wins on
**5/5 seeds**, paired difference vs TCSR $-0.0256$.*

**Concordance tells a different story, and we report it.** Across all $\gamma$, `Landmarking`
retains a small concordance edge (0.8657 vs 0.8495 at $\gamma=1$). The mechanism is clear: in this
construction $m(s)$ is a monotone function of the risk score, so higher-risk subjects die earlier
in *both* visit count and real time. The visit-count ranking already induces the correct real-time
ranking, and a scalar clock — being monotone — does not disturb it. Mean rescaling gets the
*ordering* right and the *absolute timing* wrong; concordance is blind to the latter. Indeed
concordance for mean-clock arms *improves* with $\gamma$ (0.815→0.866) because the clock amplifies
risk separation. This is precisely why rank metrics and probabilistic error must be read together.

### 5.4 Deep-learning setting `[TBD — placeholder values]`

> **These numbers are placeholders.** They exist to fix the table structure and will be replaced
> by measured results. Do not cite.

| Method | NASA C-MAPSS C-idx ↑ | IBS ↓ | MIMIC-IV C-idx ↑ | IBS ↓ | eICU C-idx ↑ | IBS ↓ |
|---|---|---|---|---|---|---|
| DeepHit | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` |
| Dynamic-DeepHit | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` |
| SurvLatent ODE | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` |
| TCSR | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` |
| DeepTCSR | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` |
| **SM-TCSR (ours)** | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` | `.xxx` |

*Table 4: Deep-learning benchmarks with a shared GRU encoder. **Placeholder — to be filled.**
MIMIC-IV is the development cohort and eICU the external validation cohort; cohort definitions,
feature sets, prediction times and horizons are matched across methods.*

### 5.5 Pre-registered ablations `[TBD]`

A1 SCCT versus squared-path censoring penalty · A2 $\omega_{\texttt{event}}\in\{1,2,5\}$ (event
transitions carry only 7–13% of the loss signal: AIDS 13.4%, PBC2 7.2%, RW 9.6%) · A3 actual $D_n$
versus a fixed substitute (an intentional control; *not* a faithful TCSR reproduction) · A4 TD
versus direct real-time likelihood on the same head · A5 target network on/off · A6 finite versus
weighted-infinite horizon · A7 Weibull-mixture versus grid head.

---

## 6 Limitations

1. **No convergence guarantee.** The target accumulates elapsed time with no discount, so standard
   distributional-RL contraction results do not transfer, and the target network is a heuristic
   stabiliser rather than a proof.
2. **Selection bias among completed transitions.** Collecting only transitions completed before
   administrative end can under-sample long transitions; the empirical law of completed transitions
   need not match the generative one.
3. **Informative observation times.** If sicker subjects are measured more often, the transition
   sample is biased. Subject-level sampling does not correct this.
4. **State sufficiency.** Whether a finite history window or a learned representation satisfies the
   Markov renewal assumption is a separate hypothesis.
5. **Scope of evidence.** The motivating study has three states, two transitions, and no censoring.
   The linear benchmark holds the head fixed by design and is not an architecture comparison. Five
   seeds do not establish statistical significance, and nested subsets correlate results across
   sample sizes; we therefore report paired differences and mark sub-s.d. gaps as ties.
6. **Baseline fidelity.** DeepTCSR is run as published, including a censoring mask that admits
   padded visits into the loss (§5.2).

---

## 7 Conclusion

Visit-index temporal-difference survival analysis discards the elapsed time between observations,
and rescaling by any mean — even the population-optimal one — cannot recover the resulting loss of
distributional information. SM-TCSR shifts the bootstrapped cumulative incidence by the actual
duration under a Markov renewal assumption, using a single masked Cramér objective with an exact
censoring likelihood and a horizon treatment that remains finite under competing risks. It
improves real-time probabilistic prediction on two clinical datasets and is the only method that
does not degrade as state-dependent irregularity grows, while — by construction — offering no
advantage when durations are uninformative. Whether the gains persist under learned encoders,
informative observation processes, and real ICU cohorts is the subject of §5.4, which remains to
be completed.

---

## References

*(to be formatted with `iclr2026_conference.bst`)*

- Bellemare, Dabney & Munos. A Distributional Perspective on Reinforcement Learning. ICML 2017.
- Bellemare et al. The Cramér Distance as a Solution to Biased Wasserstein Gradients. 2017.
- Bradtke & Duff. Reinforcement Learning Methods for Continuous-Time MDPs. NeurIPS 1994.
- Fatemi, Killian, Subramanian & Ghassemi. Semi-Markov Offline RL for Healthcare. CHIL 2022.
- Frost, Li & Harris. Robust Real-Time Mortality Prediction in the ICU using TD Learning. ML4H 2024.
- Graf, Schmoor, Sauerbrei & Schumacher. Assessment and comparison of prognostic classification
  schemes for survival data. Statistics in Medicine, 1999.
- Lee, Yoon & van der Schaar. Dynamic-DeepHit. IEEE TBME, 2020.
- Maystre & Russo. Temporally-Consistent Survival Analysis. NeurIPS 2022.
- Moon, Groha & Gusev. SurvLatent ODE. MLHC 2022.
- Rowland, Bellemare, Dabney, Munos & Teh. An Analysis of Categorical Distributional RL. AISTATS 2018.
- Vargas Vieyra & Frossard. Deep End-to-End Survival Analysis with Temporal Consistency. 2024.

---

## Appendix (outline)

- **A** Algorithm and pseudocode; implementation checklist.
- **B** Full result tables: Setting A/B all fractions, coefficient error, per-seed values.
- **C** Coefficient-recovery metric. The reference $\beta_{\rm ref}$ is a full-data
  *initial-state* MLE, so `Init. state` is structurally favoured as $n\to N$ (AIDS 0.3196, RW
  0.0199). We report both the paper-original unnormalised $L_2$ and $\lVert\cdot\rVert_2/\sqrt p$.
- **D** Motivating study: visit-count panel, State A panel, resolution sensitivity (grid 0.05 →
  0.01 → 0.005 gives $1.87\!\times\!10^{-5}\!\to\!3.5\!\times\!10^{-6}\!\to\!3.1\!\times\!10^{-6}$),
  and the control scenarios (`Regular`: all four methods identical at $8.4\times10^{-6}$;
  `Different means`: per-state clock improves 59× over global, which is why it is *not* the main
  condition).
- **E** Random-walk generator. We fix a recursion defect in the reference implementation
  (`seqs[i, j+1]` read a zeroed slot, making the process i.i.d. rather than a random walk) and
  remove the constant feature that is collinear with the baseline intercepts; consequently the
  original RW numbers are **not** reproduced. RW CDF-RMSE uses 3 seeds and 3 fractions; MC
  evaluation error (0.0065) is an order of magnitude below between-model gaps.
- **F** Reproducibility: TCSR at commit `fe2aa73`; DeepTCSR obtained as an anonymous archive with
  no commit hash and therefore not verifiable as the official implementation. Data provenance,
  seeds, and split rules.
- **G** Note: on the random walk, Setting A visit-axis concordance and Setting B real-time
  concordance at $\gamma=0$ coincide exactly for the four mean-clock arms, because with
  $D\equiv1$ the conversion is a monotone rescaling and concordance is rank-based.
