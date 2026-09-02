role: domain (clinical AI / biostatistics)

## Idea review — SurvTD: Duration-Discounted Temporal-Difference Consistency for Dynamic Survival Analysis under Irregular Observation

Seat lens: clinical-AI reviewer and biostatistician who works with ICU telemetry and deploys
early-warning scores. Instantiated on the three real cohorts the candidate names — MIMIC-IV
Sepsis-3, PBC, and NASA C-MAPSS FD001–FD004.

**Decomposition**

- **Problem / gap.** Temporal-consistency survival methods (`Maystre2022` TCSR, `DeepTCSR2024`)
  "assume an exogenous uniform unit step Delta t = 1, so the non-event signal carried by an
  interval is one bit regardless of how much time elapsed," and their backward update
  "renormalises the next-step lifetime distribution by dividing by the interval survival
  probability, which diverges exactly in the high-risk regime where S approaches zero."
- **Method (the move).** Replace the division with a composed operator: a renewal rightward shift
  `Phi_{+Delta t_j}` (from the pathwise identity `R_j = R_{j+1} + Delta t_j`) followed by a
  triangular categorical projection `Pi` back onto a fixed lifetime bin grid, with an interval
  discount `gamma_j = S_theta-(Delta t_j)` from an EMA target network, an IPCW tail completion at
  censored terminal observations, a lambda-return whose decay is "geometric in elapsed duration
  rather than in observation count," and a Cramér-distance fit.
- **Why it should work.** "The composed operator Pi Phi is non-expansive in the Cramer metric and
  conserves unit probability mass exactly, so the Bellman target is a valid distribution at every
  interval length," and the frozen-target map's "Cramer-metric modulus is bounded by gamma_j."
- **Assumptions inferred (none of these is stated in the candidate).**
  1. Observation times are conditionally non-informative given the encoded history — required for
     `gamma_j` to be readable as "the accumulated probability of surviving the whole interval …
     evidence that scales with how much time actually elapsed."
  2. Censoring is independent of the event process marginally, so that "a Kaplan-Meier estimate of
     the censoring distribution" yields valid IPCW weights.
  3. Censoring is censoring, not a competing risk — i.e. discharge alive / transplant are treated
     as uninformative right-censoring rather than as competing events.
  4. The benchmark cohorts actually exhibit the irregularity and the censoring the mechanism needs
     in order to be exercised.
  5. `gamma_j` is an intervenable term in the target, since NC-A ablates it — but Step 5's target
     formula contains no `gamma_j`.

---

| Axis | 1–5 | Quoted evidence | Reason |
|------|-----|-----------------|--------|
| A — Problem position | **3** | "the assumption fails outright on irregularly observed clinical and industrial trajectories"; "diverges exactly in the high-risk regime where S approaches zero" | The `÷ S(Δt)` divergence is a real, correctly-diagnosed numerical defect and the TD-consistency-under-irregular-Δt slot is genuinely open. But the party who pays a cost from the bottleneck *as stated* is an author writing the next paper after TCSR — neither TCSR (NeurIPS 2022) nor DeepTCSR (2024 preprint) runs in any ICU. The urgency is borrowed from alarm fatigue, which the candidate never instruments. And "handle non-unit Δt" is the most obvious criticism any longitudinal method receives; the field's standard answer (fine-grained person-period expansion onto a regular grid) is never named or beaten. Real and open, but narrow and one level away from the cost. |
| B — Method quality | **3** | "the renewal identity R_j = R_{j+1} + Delta t_j"; "conserves unit probability mass exactly"; "the claim made is per-iteration contraction under a frozen target, not convergence of the joint two-timescale scheme" | **depth:** genuine — this is an operator substitution, not another loss term, and it survives one-sentence statement without machinery ("shift the lifetime distribution by the elapsed time and re-project, instead of dividing by interval survival"). Discounted by the scoop report's own honest boundary: the projection machinery is "imported, not invented" from C51. **soundness:** the renewal identity is pathwise exact and mass conservation is a real, checkable property that the `÷S` update genuinely lacks — credit. Against it: `gamma_j` is declared load-bearing but appears nowhere in the Step-5 target; "a Kaplan-Meier estimate of the censoring distribution" assumes marginal independent censoring, which is false in the headline cohort, and `÷ Ĝ(t)` reintroduces at the tail precisely the divergent division the method is sold as removing, with no truncation or stabilisation mentioned; the "accumulated non-event evidence" reading silently requires non-informative sampling. **feasibility:** strong — public data, credentialed-access-only for MIMIC, "roughly eighteen GPU-hours" on a 3090 is optimistic for a 50-trial search per method but not fantastical. |
| C — Problem-fit | **3** | "The division is replaced by the composition of a renewal rightward shift and a categorical projection onto the bin grid"; "the assumption fails outright on irregularly observed clinical … trajectories" | Against the *narrow* gap the `gap_closure` block states — the update is undefined and divergent at non-unit Δt — the fit is near-perfect and would score 4: the operator is exactly the broken object, and removing it removes the defect. Against the *sold* clinical gap it drops to 2, and §3.3 lands. In ICU telemetry the interval is short *because* the patient is deteriorating; the mechanism's central interpretive claim needs Δt to be ancillary to the event process, and in the one benchmark with real irregularity it is not. The regime where the operator's story is clean (exogenous irregular sampling) is covered by the simulated ODE/OU cohorts; the regime where the gap hurts is the one where the interpretation is unjustified. Net 3. |
| D — Falsifiability | **3** | load-bearing variable: "interval_duration_discount_factor gamma_j = S(Delta t_j)" · control: "NC-B (orthogonal): scramble the observation durations across independent trajectories while preserving the feature values, which must destroy the gain if it comes from temporal alignment rather than from regularisation" | A real falsifier exists and lands downstream (tAUC/IBS), so this clears the D≤2 gate. NC-B and NC-C are genuinely good, non-tautological orthogonal controls, and NC-C ("sweeping lambda to 1.0") correctly refuses to confound variance reduction with duration discounting. The theory claim is honestly scoped — explicit credit. Against it: the *primary* control NC-A intervenes on a term the target formula does not contain, so it is either a no-op or is silently an ablation of the shift magnitude, in which case degradation is arithmetically forced (breaking an exact identity), not evidence for the discount reading; NC-A is a literal no-op on C-MAPSS, where Δt is constant. "reports the alarm-stability consequence directly" is a headline selling point with no metric, no threshold, and no appearance in the falsification prediction. The 0.03 bar carries a `derived:` tag but is derived from seed noise rather than from any clinical quantity, and sits at PBC's split-to-split noise floor. |

**Overall: 50/100 · Verdict: borderline**
No gate fired — A=3, C=3, D=3 are all above the ≤2 cap thresholds; the band stands on the
arithmetic alone.

**Naive-baseline audit: branch 1, with a caveat that must be closed.**
Constructing the naive version independently, not from the candidate's strawman: a biostatistician
handed irregular longitudinal survival data does *person-period expansion* — discretise onto a fine
regular grid (e.g. 1 h for MIMIC, 672 rows for a 28-day stay), forward-fill covariates with a
missingness indicator, carry Δt as a feature, and fit a discrete-time hazard model. Every irregular
interval then becomes an integer number of unit steps, and TCSR's unit-step update applies verbatim
by being applied `Δt/δ_s` times. This is the field-standard preprocessing for every MIMIC benchmark
and it costs nothing new.

That naive version does rest on a false premise — compounding `÷ S(δ_s)` across `n = Δt/δ_s` steps
multiplies the divergence rather than removing it, which is exactly the defect the candidate names —
so confronting it *is* the contribution, and branch 1 is the right classification. Two caveats:
(i) the confrontation is a textbook tool from a neighbouring field (C51's categorical projection),
and the candidate does discharge the SKILL's requirement here — it names real domain-specific
structure ("the projected quantity is remaining lifetime, the shift is a renewal identity …, the
discount is an endogenous survival probability that the model itself predicts, and censoring
requires an IPCW tail completion that has no analogue in reward distributions"). That is a
sufficient answer, and I credit it. (ii) The naive version is never named, never run, and its
failure is asserted rather than shown. Until person-period expansion appears as a baseline, branch 1
is a claim, not a finding.

**Novel-but-empty detector:** does not fire on the operator claim — I can state what it predicts
(tAUC/IBS degrade to the unit-step baseline under NC-A/NC-B) and what would count against it. It
*does* fire on the alarm-stability claim, which predicts nothing measurable as written.

**Strongest point:** the renewal identity `R_j = R_{j+1} + Δt_j` is pathwise exact and holds for
every subject regardless of *why* the interval had the length it did — so the shift operator itself,
unlike the interpretation layered on top of it, is fully robust to informative observation, and
mass-conservation is a genuine, checkable defect of the `÷S` update that a clinician would recognise
as the cause of collapsed predictions in exactly the sickest patients.

**Most fixable weakness:** replacing "a Kaplan-Meier estimate of the censoring distribution" with a
covariate-conditional censoring model, splitting administrative censoring from discharge-alive,
saying whether discharge-alive is censoring or a competing risk, and specifying weight truncation —
this is a one-paragraph repair that removes the seat's sharpest soundness objection and stops Step 6
from reintroducing the divergent division that Step 4 was built to eliminate.

---

## Gauntlet — attack table

| # | Attack | Severity | Answerability | Where it lands |
|---|--------|----------|---------------|----------------|
| 1.2 | **Manufactured urgency (lead).** The stated bottleneck is "TCSR/DeepTCSR assume Δt = 1." Nobody deployed pays a cost from that — no production ICU early-warning system uses a TD-consistency survival loss, and DeepTCSR is a 2024 arXiv preprint. A real cost *does* exist one level away (alarm fatigue from low-PPV sepsis alerts is documented and clinicians pay it daily), but the candidate borrows that urgency without ever connecting to it: alarm stability appears exactly once, in a `differentiation_from_lit` line, and never in the falsification prediction. As written, the only party paying the stated cost is the author of the next paper after TCSR. | major | now | motivation / `gap_closure[0]`, `differentiation_from_lit[Lee2019]` |
| 3.3 | **Partial closure sold as full (lead).** The mechanism is clean where inter-observation intervals are exogenous — which is true of the "tumour-growth ODE and Ornstein-Uhlenbeck cohorts [that] are simulated," and of C-MAPSS. It is *not* true of MIMIC-IV Sepsis-3, the only rung with real, large irregularity, and that is where the gap is claimed to hurt. So the benchmark that demonstrates the mechanism cleanly is the one where the clinical difficulty has been removed by construction, and the benchmark where the difficulty is real is the one where the mechanism's central interpretation is unjustified. | major | with evidence | `gap_closure[0]`, `compute_budget` benchmark suite |
| 2.3 | **Unstated precondition — informative observation.** Step 3 reads `gamma_j` as "the accumulated probability of surviving the whole interval … evidence that scales with how much time actually elapsed." That reading requires the observation process to be conditionally non-informative. In an ICU it is the opposite: labs go q1h because the patient is crashing and q24h because they are stable, so Δt is a *severity marker*, and `gamma_j` conflates non-event evidence with the clinician's own risk assessment. The mechanism can absorb the sampling artifact and report it as risk. Two aggravations: (a) the encoder already "receives the elapsed durations as inputs," so the discount is computed from a network that itself conditions on Δt — the discount and the feature are entangled; (b) NC-B scrambles durations, which destroys the physiological and the clinician-behaviour channel *simultaneously*, so a positive NC-B is equally consistent with "the model learned deterioration" and "the model learned the ordering behaviour of the ICU team." Notably, the project's own Phase-1 bottleneck statement builds its case on exactly this literature (`Lin2001`, `Alaa2017`, "λ_obs(t) ∝ Risk(t)") and neither paper survives into `differentiation_from_lit`. | major | now | `core_mechanism_steps[3]`, `load_bearing_variable`, `negative_control` NC-B |
| 2.3b | **Unstated precondition — censoring, and the division comes back.** Step 6 uses "a Kaplan-Meier estimate of the censoring distribution." Marginal KM requires censoring independent of the event process. In MIMIC-IV Sepsis-3, right-censoring is dominated by *discharge alive*, the most state-dependent transition in the unit — a clinician discharges because the patient improved — so Ĝ systematically misprices exactly the low-risk patients who leave early, and the "unbiased target" claim fails. In PBC the same applies to transplant. Two further edges: discharge-alive and transplant are arguably *competing events*, not censoring, which changes the estimand (crude vs. net risk) — and the candidate cites Dynamic-DeepHit, a competing-risks method, as a baseline while ignoring competing risks in its own construction; and `1/Ĝ(t)` diverges in the tail where few remain at risk, so a method whose entire rhetorical premise is "the division by a small survival probability is the defect" reintroduces a divergent division at Step 6, with no truncation or stabilisation stated. | major | now | `core_mechanism_steps[5]` |
| 1.3 | **Solved at another level.** For MIMIC-IV the standard pipeline resamples onto a regular hourly grid with forward-fill and missingness indicators; under it Δt is constant and the entire motivation evaporates. Independently, person-period expansion applies the unit-step update `Δt/δ_s` times and needs no new operator. There is a good rebuttal available — gridding forward-fills away the informative-sampling signal and manufactures spurious constant-covariate stretches, and compounding `÷S(δ_s)` compounds the divergence — but the candidate makes neither argument and lists no such baseline. | major | with evidence | motivation, `compute_budget` (baseline set) |
| 4.4 | **Unmeasurable claim — "alarm stability."** Invoked as a differentiator ("reports the alarm-stability consequence directly") with no definition, no instrument, no threshold, and no appearance in `falsification_prediction`, which names only "time-dependent AUC and integrated Brier score." As a deployed-systems reviewer I need it as: alerts per patient-day and *distinct alert episodes* at a fixed threshold; re-alarm (threshold-crossing) rate per patient per 24 h, which is what actually drives fatigue; PPV at a fixed alert budget; and lead time to first true alarm. It must also be reported *jointly* with discrimination and against a trivially-smoothed baseline (EMA-smoothed Dynamic-DeepHit), because a constant predictor has perfect stability and any smoothing regulariser improves any jitter metric for free. | major | now | `differentiation_from_lit[Lee2019]`, `falsification_prediction` |
| 4.5 | **NC-A cannot lose, or cannot act.** NC-A "replace[s] the duration-dependent interval discount factor gamma_j = S(Delta t_j) in the Bellman target with the unit-step constant S(delta_s)" — but the Step-5 target `T p_j = I[·] δ + (1 − I[·]) (Pi Phi_{+Delta t_j} p_theta-)` contains no `gamma_j` term; `gamma_j` enters only as the analytic contraction modulus. So NC-A is either a no-op on the target, or it is tacitly `Phi_{+Delta t_j} → Phi_{+delta_s}`, which mis-shifts the target by up to `Δt − δ_s` and breaks a pathwise-exact identity. Degradation is then arithmetically forced and confirms nothing about the *discount* interpretation. On C-MAPSS, where Δt is constant at one cycle, NC-A is a no-op by construction. | major | now | `negative_control` NC-A, `core_mechanism_steps[4]`, `load_bearing_variable` |
| 4.3 | **Numbers with provenance tags that the packet cannot support.** "the concordance level measured in Bleistein2024 as 0.8791" — four significant figures, and the retrieved `phase0/lit_table.md` entry for `Bleistein2024` contains no numbers at all, so it is unverifiable from the input packet; it is also a C-index quoted on a benchmark that is effectively uncensored and where the field reports RMSE and the asymmetric NASA score, not concordance. The 0.03 bar is tagged `derived:` from "the seed-to-seed standard deviation … roughly one third of that gap," i.e. 3σ of a single-run SD — but with five seeds the standard error is ≈σ/√5, making 0.03 ≈ 6.7 SE, and more importantly a *clinically* meaningful ΔC-index is derived from net benefit or alert burden, not from one's own noise. On PBC (n = 312, ~140 deaths) 0.03 sits at or below the split-to-split noise floor, so the pre-declared bar is not reachable with five seeds without repeated CV or bootstrap intervals. Recorded as minor rather than fabrication because both figures do carry provenance tags — the defect is that the derivations are the wrong *kind*, not that they are absent. | minor | now | `falsification_prediction` |
| 3.3b | **The C-MAPSS rung does not exercise either novel component.** C-MAPSS emits one record per operating cycle — regular sampling, Δt ≡ 1 — so `gamma_j` and `Phi_{+Delta t}` are constant and SurvTD degenerates to a unit-step categorical TD method there. Training units are run to failure and test units come with a supplied ground-truth RUL, so there is no right-censoring and the Step-6 IPCW completion is a no-op. Either the rung is used as-is, in which case it cannot test the load-bearing variable, or it is artificially subsampled, in which case the induced irregularity is exogenous *by construction* and is therefore the easy regime — and the candidate must say which. | major | now | `compute_budget`, `falsification_prediction` |
| 2.6 | **Borrowed tool.** "the triangular categorical kernel" and the non-expansiveness argument are C51's, and the scoop report concedes it: "The projection operator and its non-expansiveness are imported, not invented." Recorded as minor and substantially answered — the candidate discharges the catalog's requirement by naming the domain-specific structure (remaining lifetime as the projected quantity, the renewal identity as the shift, an endogenous model-predicted discount, censoring with no reward-distribution analogue), which is the right response. It should be stated with that honesty in the paper rather than left to the scoop report. | minor | now | `differentiation_from_lit[Bellemare2017]` |

**Attacks I checked that do not land, and I record as not landing:**
2.1 (equivalent to naive) — the shift-and-project operator is not what a competent practitioner would
reach for by default; the naive move is person-period expansion, which is structurally different.
2.4 (circularity) — `gamma_j` from a frozen EMA target is the standard target-network device, not a
quantity requiring the problem to be solved first. 6.2 (two half-papers) — the core claim fits well
inside 25 words. 5.4 (regression to a prior state) — the candidate removes the unit-step assumption
while explicitly *retaining* DeepTCSR's target network and stating the changed purpose.

---

## Two-layer verdict

**Hard floor: not triggered.** Checked each of the four floor conditions explicitly.
- No prior work matches on all four scoop axes — the scoop report records "Overlap level: 2 / 5 —
  adjacent lineage, distinct technical move," and the nearest work (TCSR) matches on framing and
  partially on insight, not on mechanism.
- The naive-baseline audit returns **branch 1** (naive rests on a false premise), not "naive
  suffices."
- Every required mitigation is insertable without dismantling the idea: a conditional censoring
  model with truncated weights, an alarm-burden instrument, a person-period-expansion baseline, a
  disambiguation of what NC-A intervenes on, and an explicit statement of the observation-process
  assumption.
- D does not collapse — a falsification is constructible and NC-B/NC-C already land downstream.

**Soft judgment: `revise`.** Not `advance`: this is not a trivial borderline. Two of the three real
cohorts cannot exercise the load-bearing variable as specified (C-MAPSS by regularity, PBC by scale
relative to the declared effect size), the third violates the mechanism's unstated interpretive
precondition, the primary negative control acts on a term absent from the target formula, and the
one clinical quantity a deployment reviewer cares about is named but never measured. None of this is
fatal; all of it is repairable at the write-up level, and the operator at the centre is genuinely
sound.

**revision_targets[]**

1. **State and test the observation-process assumption.** Say explicitly that reading `gamma_j` as
   "accumulated non-event evidence" requires Δt to be conditionally ancillary given `h_j`, and that
   this is false in MIMIC-IV. Report the empirical association between Δt and short-horizon hazard
   in the cohort, and add an arm that separates the two channels — inverse-intensity weighting of
   the observation process, or a *within-patient* Δt permutation (as distinct from NC-B's
   across-trajectory scramble, which destroys both channels at once).
2. **Repair the censoring model (Step 6).** Replace the marginal Kaplan-Meier Ĝ with a
   covariate-conditional censoring model; separate administrative end-of-window censoring
   (independent, KM fine) from discharge-alive and transplant (state-dependent); declare whether
   discharge-alive is treated as censoring or as a competing risk, and justify the choice against
   the cited competing-risks baseline; specify weight truncation, and acknowledge that `1/Ĝ`
   reintroduces the divergent division the method is built to remove.
3. **Instrument alarm stability or drop the claim.** Define it as alerts per patient-day, distinct
   alert episodes, and re-alarm (threshold-crossing) rate at a fixed PPV or fixed alert budget,
   reported jointly with discrimination and against an EMA-smoothed Dynamic-DeepHit, so that trivial
   smoothing cannot win the metric.
4. **Fix or drop the C-MAPSS rung.** It is regularly sampled and effectively uncensored, so both
   novel components are inert on it; if it is artificially subsampled, say so and state plainly that
   the induced irregularity is exogenous by construction and therefore the friendly regime.
5. **Disambiguate what NC-A intervenes on.** Write `gamma_j` into the Step-5 target explicitly, or
   restate NC-A as an ablation of the shift magnitude `Phi_{+Delta t} → Phi_{+delta_s}` and concede
   that degrading a pathwise-exact identity is a near-certain outcome rather than evidence for the
   discount reading.
6. **Re-derive the 0.03 bar from a clinical quantity** (net benefit, or ΔPPV at a fixed alert
   budget) rather than from seed-to-seed SD, and add repeated CV or bootstrap intervals on PBC,
   where n = 312 with ~140 events puts 0.03 at the split-to-split noise floor for five seeds.
7. **Add person-period expansion as a baseline.** Fine-grained discrete-time hazard on a regular
   grid with Δt as a feature is what a biostatistician does first; the paper must show that
   compounding the unit-step update `Δt/δ_s` times actually fails, rather than asserting it.
8. **Connect the motivation to a payer, or re-scope it honestly.** Either build the bridge from
   "TCSR assumes Δt = 1" to a cost a clinician bears — via the alarm-burden instrument of target 3 —
   or state that this is a methods contribution to the TD-survival lineage and let it stand on the
   operator, which is strong enough to carry a paper on its own terms.
