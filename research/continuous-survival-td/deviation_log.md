# Deviation Log

Every divergence between the preregistration / design document
(`phase3/round_2_revision/candidate_r2.json`) and the executed code, plus every
amendment made during the 2026-09 repair effort.

For each entry: what was declared, what the code did, the fix, and **whether the
fix was decided before or after seeing post-fix results**. Entries marked
`decided-before-results` were fixed on the basis of code inspection and unit
tests alone, with no downstream metric consulted.

Status legend: `[ ]` open · `[x]` landed · `[~]` in progress

---

## 0. Standing rules for this effort

1. `alpha` (anchor weight), `lambda`, the ablation arm definitions, and every
   kill threshold are selected or declared **once**, on the validation split or
   a priori, and frozen before the 5-seed run. Any later adjustment is itself a
   deviation and gets an entry here.
2. Operator correctness is validated by unit tests and training-free
   target-level diagnostics only — **never** by downstream C-index. A
   well-supervised model can score well despite a broken operator.
3. A hypothesis that does not survive the repair is the correct output of the
   repair. The kill criteria are honoured, not re-tuned around.

---

## 1. Withdrawn results

**[x] W-01. The previously circulated Table 1, Table 2, Table 3 and
`falsification_report.json` are withdrawn, not superseded.**

- Moved to `experiments/results/invalidated_2026-09-03/` with a README stating
  every reason. Retained for audit trail only.
- Table 1 was produced with `--dry_run`: 2 epochs, 1 seed, `max_units=25`. The
  C-MAPSS column came from 5 test engines / 10 permissible pairs; the reported
  `0.100 / 0.800 / 0.900` are exactly 1/10, 8/10, 9/10.
- **The original "ALL ADVERSARIAL STRESS TESTS PASSED" verdict carried no
  information in either direction.** A pipeline in which gamma was equivalent to
  a lambda reparameterization (D-gamma), the sole ground-truth target sat in the
  wrong bin (D9), the durations were shifted by one index (D8), the
  preregistered IPCW weighting did not exist (D11), and the reference baseline
  scored below chance could neither falsify nor confirm anything.
- `decided-before-results`

---

## 2. Implementation defects: code did not implement the design

These are cases where the design document is correct and the code diverged from
it. Fixing them is not a change of claim.

**[ ] D-gamma. The design's renewal mixture was never implemented, making the
duration discount equivalent to a reparameterization of lambda.**

- *Declared* (`candidate_r2.json` Step 5): the one-step target is a renewal
  mixture, `T p_j = (1 - gamma_j) * mu_[0,dt_j) + gamma_j * (Pi Phi_{+dt_j} p_theta-)_j`.
- *Code*: `survtd_operator.py:221` implements only the second term. No identifier
  relating to `mu` exists anywhere in the codebase. gamma was instead placed in
  the lambda recursion at `:227` and then cancelled by the renormalization at
  `:230`.
- *Consequence*: because both branches are normalized to mass 1, the denominator
  is exactly `(1-lambda) + lambda*gamma`, so the post-normalization weight on
  the bootstrap branch is identically `lambda*gamma / ((1-lambda) + lambda*gamma)`.
  Verified to float32 precision by
  `test_operator_defects.TestGammaIsNotInert`: observed `0.5876288414` vs
  predicted `0.5876288660` at gamma=0.95.
- *Note*: gamma is not literally a no-op on the target — it does move mass
  between branches. The defect is that its effect is *exactly* a lambda
  reparameterization, so it carries no information the lambda knob does not
  already carry. **This is why EXP-03 / NC-A1 retained 93.9% of the gain: it was
  a near-no-op ablation, not a falsification of C_1.**
- *History*: the Phase 3 panel caught a "gamma_j ghost variable" defect in Round
  1 and the design was amended to fix it. The implementation reverted that fix.
- *Fix*: implement Step 5's mixture; delete the renormalization; assert mass
  conservation instead. See A-02 for the accompanying correction to Step 7.
- `decided-before-results`

**[ ] D8. Inter-visit durations misaligned by one index — the "full" model was
already running the NC-B negative control.**

- The loaders use a backward difference, `dts[j] = times[j] - times[j-1]`
  (`sepsis_loader.py:54-55`, `cmapss_loader.py:78`, `pbc_loader.py:51`). The
  renewal identity for transition `j -> j+1` is
  `R_j = R_{j+1} + (times[j+1] - times[j]) = R_{j+1} + dts[j+1]`.
- `survtd_operator.py:202` reads `dt_j = dts[j]` and pairs it with
  `target_pmfs[j+1]`. Every transition is affected; the shift amount, `gamma_j`
  and `lambda_j` are all computed from the wrong interval. Same at
  `ablations.py:56` and `deeptcsr_clamped.py:96`.
- Verified: for `times = [1, 4, 8, 20]` (`dts = [1, 3, 4, 12]`) transition 0->1
  needs 3.0 and the code uses 1.0; 1->2 needs 4.0 and uses 3.0; 2->3 needs 12.0
  and uses 4.0.
- *Consequence*: because the durations are i.i.d. within a trajectory the
  marginal distribution is preserved, but the **per-transition correspondence
  between duration and transition is destroyed — which is precisely what the
  NC-B duration permutation does.** EXP-04's verdict was therefore invalid for a
  reason internal to the operator, independent of the data generator. Fixing D8
  is a prerequisite for EXP-04 meaning anything, and may on its own flip the
  verdict.
- `decided-before-results`

**[ ] D9. The terminal ground-truth Dirac was placed at the length of the
previous interval instead of the residual time to event.**

- `survtd_operator.py:191-193`:
  `delta_tau = max(0, min(dt_last, tau_event - (tte - dt_last)))`. When
  `tau_event = tte` (every event trajectory in these cohorts) the second
  argument equals `dt_last`, so the `min` always selects `dt_last`.
- Verified: `times=[1,4,8]`, `tte=8.5`, `delta_s=2.5` puts the Dirac in bin 1
  (s=4.0); the truth is bin 0 (residual 0.5).
- Same at `ablations.py:47-50` and `deeptcsr_clamped.py:88-90`. Structurally the
  identical "elapsed interval instead of residual time" error as the
  Person-Period defect.
- *Consequence*: combined with lambda-attenuation, the ~0.1% of ground-truth
  signal that survived back-propagation was also pointing at the wrong bin. On
  the sepsis cohort events occur strictly after `times[-1]`, so the in-interval
  event branch (`:211-216`) never executes at all: one misplaced Dirac per event
  trajectory, zero per censored trajectory.
- *Fix*: derive interval-event flags from residual times; drop the fragile
  `events` / `tau_event` pair from the operator signature.
- `decided-before-results`

**[ ] D10. gamma = S(dt) interpolated half a bin too early.**

- `survival[k] = prod_{m<=k}(1-h_m) = P(R > (k+1)*delta_s)`, so `S(dt)` must be
  read at index `dt/delta_s - 1`. `compute_interval_discount:107` uses
  `dt/delta_s - 0.5`. Systematic over-discounting.
- Currently harmless because gamma is inert; becomes load-bearing the moment
  D-gamma is fixed. The sub-bin branch (`:96-104`) is correct and is continuous
  with the `-1.0` convention at `dt = delta_s`, which confirms `-1.0` is intended.
- `decided-before-results`

**[ ] D11. The preregistered IPCW Cramer weighting was never implemented.**

- Preregistration section 3 Rung 4 lists "scalar IPCW Cramer loss weighting" as
  part of the *definition* of the full method.
- `compute_loss_trajectory(..., ipcw_weight=1.0)` is never passed a value by
  `trainer.py`, so `step_weights` is always all-ones. It has never run.
- `decided-before-results`

**[ ] D12. Bin-index convention diverges across the baseline ladder.**

- SurvTD uses centre-based projection (`u = s/delta_s - 0.5`), Dynamic-DeepHit
  uses `floor(rem/delta_s)` (`dynamic_deephit.py:80`), DeepTCSR uses
  `round(dt/delta_s)` (`ablations.py:101`). Up to a half-bin systematic offset
  between contenders in a comparison whose headline delta is 0.025.
- Preregistration section 3 promises identical backbones; a shared bin
  convention and a shared hazard head are a free strengthening of that parity
  claim.
- `decided-before-results`

**[ ] D-PP. The Person-Period baseline supervised one bin per visit and scored
below chance.**

- `person_period.py:74-84` supervises only `k_step = round(dt_j/delta_s)` — the
  bin of the *elapsed interval*, not the residual time. On sepsis `k_step ~ 1`,
  so bins 2..29 are never trained, yet evaluation reads bin `K//2 = 15`.
- Reported C-index 0.263 (sepsis) and 0.100 (C-MAPSS), both far below 0.5. **A
  below-chance Rung-1 baseline voids the baseline-ladder parity claim and
  distorts every gain-retention denominator in Table 3.**
- The docstring and preregistration section 3 Rung 1 also promise a "1-hour
  forward-filled regular grid" expansion that the code never performs.
- `decided-before-results`

**[ ] D-CENS. Right-censored trajectories received a self-copy as their target.**

- `survtd_operator.py:197` sets the terminal target to `target_pmfs[-1]` — the
  target network's own output, carrying zero information. The sepsis cohort is
  ~72% censored, so the majority of trajectories had no ground-truth anchor
  anywhere.
- `decided-before-results`

**[ ] D14. `squared_cramer_distance_loss` is not the Cramer distance.**

- Uses `mean` over K rather than `delta_s * sum`, so its magnitude depends on K
  and it is not in time units. `delta_s` ranges 0.5 to 100 across cohorts, so
  losses are not comparable between cohorts.
- `decided-before-results`

**[ ] D-EVAL. Risk was read at each subject's last visit, with no landmarking.**

- `run_track_a.py:68` and the duplicated `run_track_b.py:53` take the risk score
  as `cdf[-1, K//2]`. The last-visit timestamp differs per subject and
  correlates with the outcome — in C-MAPSS the final observation *is* the
  failure cycle. This is simultaneously leakage and a clock mismatch, and it
  means the "dynamic" claim was never measured.
- `compute_concordance_td` (`metrics.py:14`) is plain Harrell's C on a static
  scalar, not Antolini's time-dependent C^td as the name and preregistration
  claim.
- IBS (`metrics.py:109`) indexes an absolute evaluation time into a
  last-visit-relative survival curve, and normalizes IPCW by a count rather than
  by the sum of weights — hence the reported 0.78-0.86 against a ~0.25 ceiling.
- `decided-before-results`

**[ ] D-STAT. No statistics were reported.**

- `compute_bootstrap_ci` and `paired_wilcoxon_test` are imported at
  `run_track_a.py:37` and never called. No table carried an SD, CI or p-value,
  against preregistration section 5.
- `decided-before-results`

**[ ] D-HPO. The preregistered HPO never ran.**

- `run_tuning_search` is called from nowhere. Even when called,
  `hpo.py:104` sets `val_loss = float(trial)` as a "surrogate" for every
  non-SurvTD model, so trial 0 always wins. The code's search space
  (`{64,128}` hidden, `{0.1,0.2,0.3}` dropout, no batch size) also contradicts
  the declared space.
- `decided-before-results`

**[ ] D-DEAD. Declared mechanisms that never executed.**

- Every loader sets `mask: None`, so GRU-D's decayed-imputation branch
  (`backbones.py:57-58`) never runs and `x_hat = x` identically.
- `set_empirical_mean` (`backbones.py:96`) is never called, so the `x_mean`
  buffer stays all zeros. Note the ordering: because `mask` is None, `x_mean` is
  never read, so fixing the mean alone changes nothing — **mask first, then mean**.
- `ContinuousLSTM.forward` accepts a `times` argument no caller supplies.
- *Honest consequence for the paper*: until the mask is real, GRU-D is
  functionally a GRU with `dt` concatenated, and the "continuous-time decay
  backbone" claim is unsupported.
- `decided-before-results`

**[ ] D15. A second, divergent implementation of the core operator exists.**

- `run_lambda_ablation.py` has hardcoded macOS paths (`:23`, `:328`), an
  `os.chdir`, and its own `compute_lambda_returns` with a different recursion
  (multiplies by `S_dt` on both branches, no projection, no mass conservation)
  plus the same self-distillation defect (`:69`). It cannot execute on the
  current machine.
- Two implementations of the operator in one repository is a reproducibility
  hazard: a reviewer who runs it gets different mathematics.
- *Fix*: move to `experiments/legacy/` with a superseded header; implement the
  preregistered lambda sweep as a mode of the refactored `run_track_b` built on
  the real `SurvTDModel` / `train_model` path.
- `decided-before-results`

---

## 3. Data defects

**[ ] X-01. The cohort labelled "MIMIC-IV Sepsis-3" is a synthetic generator.**

- No MIMIC-IV data exists in this repository and none was used. The label
  appears in Table 1/2 headers, the README, and preregistration sections
  1/3/5/6 including Kill Criterion 3.
- *Fix*: renamed to **"Synthetic ICU Telemetry (simulated)"** throughout.
- `decided-before-results`

**[x] X-10. Landmark grids were selected from label distributions alone, and
several declared candidates had to be rejected as infeasible.**

Every (cohort, landmark, horizon) must have enough at-risk subjects, cases and
controls for the metrics to exist. Checked with no model involved, before any
training. Rejected candidates, recorded so the selection is auditable:

| candidate | why rejected |
|---|---|
| synthetic_icu L=48, Delta=24 | 0 controls in every split: L+Delta = 72 is the maximum stay |
| cmapss FD001 | a 20-unit test split yields 1 case at L=100; switched to FD002 (260 units) |
| cmapss L=50, Delta=50 | 0 cases: the minimum time-to-failure is 128 cycles |
| cmapss L=200, Delta=50 | 5 controls in test |
| pbc L=365, Delta=365 | 4 controls in test |
| tumor L=2.5, Delta=2.0 | 9 controls in test |
| tumor lethal_threshold=2.0 | event rate 0.85 exhausts the control set by the second landmark; raised to 2.6 (rate 0.63) |

Also corrected: **administrative censoring must not cap at the evaluation
horizon.** Capping residual times at exactly Delta empties the AUC control set
(residual > Delta) by construction, which made all three C-MAPSS landmarks
degenerate. LandmarkSpec.admin_censor_at is now explicit, must exceed
max(horizons), and is set to 2*Delta for C-MAPSS.

The declared grids live in src/data/cohorts.py, which the gate and the
experiments both read, so they cannot drift apart. Switching C-MAPSS to FD002
also changes the setting: it spans six operating conditions rather than one, so
it is harder as well as larger.

**[ ] X-02. The generator tied event time to the observation schedule.**

- `sepsis_loader.py:82` sets `tte = times[-1] + U(0.1, 2.0)` for events, giving
  `corr(n_obs, tte | event=1) = 0.778`, and `:85` assigns **every** censored
  subject `tte = 72.0` exactly (1 distinct value).
- *Consequence 1*: the empty control set that made the whole AUC column read a
  hardcoded 0.500.
- *Consequence 2*: `permute_patient_durations` preserves `sum(dt)` and therefore
  preserves the tte ordering, so EXP-04 could not fail.
- **Correction to an earlier diagnosis**: fixing the generator is *not* what
  makes EXP-04 valid. Measured `corr(n_obs, tte | event=1) = 0.843` in the
  **real PBC2 data** — the schedule/outcome correlation is intrinsic to
  longitudinal survival data (sicker subjects die sooner and are observed fewer
  times) and cannot be engineered away. The load-bearing fix is landmarking. The
  generator fix is still required, because `tte = times[-1] + noise` and a
  single censored time are degenerate rather than merely natural. **Generator
  surgery must not be used to make a negative control pass.**
- `decided-before-results`

**[ ] X-03. Per-patient z-scoring erased the label-generating signal.**

- `sepsis_loader.py:78` standardizes each subject against their own trajectory,
  forcing every subject's feature mean to 0 (measured max |mean| = 3e-5). The
  signal that determines the label is precisely the between-subject level offset
  (HR +25, MAP -20, lactate +3, SOFA +5).
- `decided-before-results`

**[ ] X-04. PBC competing risks silently mishandled — train and eval disagree on
9% of the cohort.**

- `label` has three levels: `{0.0: 143, 1.0: 140, 2.0: 29}`, where 2 is
  transplant. `pbc_loader.py:46` stores `event = 2.0` verbatim, then `:58`'s
  `if event > 0.5` treats transplant as a **death** in the training loss, while
  `metrics.py:27`'s `event_indicators[i] == 1` treats the same subject as
  **censored** in evaluation.
- *Fix*: declare transplant as censoring at the transplant time.
- `decided-before-results`

**[ ] X-05. PBC `delta_s = 100.0` is mis-scaled by roughly 3x.**

- Measured: tte median 328 / max 744, but **residual time (tte - last visit)
  median 36**. With `delta_s=100, K=30` the grid spans 3000 (4x the max tte, so
  ~75% of bins are structurally empty) while the bin width is ~3x the median
  quantity being predicted.
- *Fix*: `delta_s = 30, K = 30`.
- `decided-before-results`

**[ ] X-06. Tumor generator couples the latent ODE path to the visit schedule.**

- `tumor_loader.py:40-51` integrates using the **visit gaps as the Euler step**
  and accumulates noise per visit, so a subject with 11 visits has a *different
  latent trajectory* than one with 4 — not a differently-sampled version of the
  same path. This is upstream of the event-time quantization.
- Test-split event rate is 1.00: there is no censoring, so it is not a survival
  problem.
- `decided-before-results`

**[ ] X-07. C-MAPSS has no censoring and leaks the failure cycle.**

- Every unit has `event = 1.0` and the trajectory runs to the failure cycle.
- `cmapss_loader.py:108` also returns the wrong quantity as `max_horizon`:
  `df['cycle'].quantile(0.95)` is a percentile of pooled observation cycle
  indices (including test rows), not of time-to-event. Same class of error at
  `pbc_loader.py:80`, computed pre-split.
- *Fix*: administrative censoring at `L + Delta` under the landmark protocol,
  which creates genuine censoring from real data with no synthetic tinkering.
- `decided-before-results`

**[ ] X-08. Standardization and imputation statistics leak across the split.**

- `pbc_loader.py:32-33` imputes with whole-dataframe medians and `:36-38`
  standardizes with whole-dataframe statistics, both before the split.
- No validation split exists in any loader (train/test only).
- `decided-before-results`

**[ ] X-09. PBC docstring misstates the censoring rate.**

- Claims ">55% right-censoring"; the measured event rate is 0.62, i.e. 38%
  censored.
- `decided-before-results`

---

## 4. Amendments to the preregistration and the design document

These change what is claimed or how it is tested, and are the entries a reviewer
will scrutinise most closely. All are committed **before** the 5-seed run.

**[ ] A-01. Seeds corrected to the preregistered list.**

- Declared: `[42, 123, 456, 789, 101112]`. `run_track_a.py:108` defaulted to
  `[0, 1, 2, 3, 4]`.
- `--dry_run` output is redirected to `results/dry_run/` so it can no longer
  overwrite reported tables, and any table built from `dry_run` or fewer than 5
  seeds is stamped `WARNING: DRY RUN OUTPUT - NOT FOR PUBLICATION`.
- `decided-before-results`

**[ ] A-02. Correction to design Step 7: the lambda recursion was not
mass-conserving.**

- Step 7 as written, `G_j = (1-lambda_j) * T p_j + lambda_j * gamma_j * Pi Phi G_{j+1}`,
  gives `sum(G_j) = (1-lambda_j) + lambda_j*gamma_j < 1` — a defective
  distribution. The code masked this with a renormalization, which is what made
  gamma inert (D-gamma).
- *Amended form*: place gamma only in the one-step bootstrap term and let the
  renewal mixture carry the near/far split:
  - `T-hat p_j = restrict_[0,dt_j](p_theta-,j)  (+)  gamma_j * Pi Phi_{+dt_j} p_theta-,j+1`
    where the first term's mass is exactly `1 - gamma_j` by the definition
    `gamma_j = S_theta-(dt_j)`, so total mass is exactly 1.
  - `G_j = (1 - lambda_j) * T-hat p_j + lambda_j * Pi Phi_{+dt_j} G_{j+1}`
- *Verified numerically before implementation*: mass `= 1.000000` for all
  `gamma in {1.0, 0.7, 0.3} x lambda in {0.0, 0.6, 1.0}`; `lambda=0` reduces
  exactly to the one-step target and `lambda=1` exactly to Monte-Carlo
  propagation (`allclose` True); and gamma moves early-bin mass
  `0.005 / 0.304 / 0.702` at `gamma = 1.0 / 0.7 / 0.3`, i.e. no longer a lambda
  reparameterization.
- *Also corrects an error in the repair plan's own prose*, which had proposed
  routing gamma's mass to a "survived beyond horizon" bin. That is the wrong
  direction: `1 - gamma_j` is the probability of dying *inside* the interval, so
  the mass belongs in the **early** bins. Routing it beyond the horizon would
  invert the risk ordering.
- *C_1 becomes testable for the first time*: the linear part of `T-hat` in the
  far argument is `gamma_j * Pi Phi`, with Cramer-metric modulus `<= gamma_j < 1`,
  which is what `thm:1` asserts. Under the shipped code the linear part was
  `lambda_eff * Pi Phi`, so the theorem did not describe the implementation.
- `decided-before-results`

**[ ] A-03. C_2's diffusion bound restated as an expectation bound; EXP-08 loses
its verdict-bearing role.**

- Declared (C_2): "bounding per-step projection diffusion by `delta_s^2 / 6`".
  **This is false as a per-step bound.** `Pi Phi` is a two-tap linear filter with
  taps `(1-f, f)`, so it adds exactly `f(1-f) * delta_s^2` to the variance of
  *any* input distribution. Measured: `f = 0.25 -> 0.1875`, `f = 0.5 -> 0.2500`,
  against `delta_s^2/6 = 0.1667`. The worst case is `0.25 * delta_s^2`.
- `delta_s^2 / 6` is exactly `E_{f~U(0,1)}[f(1-f)]`, so the bound holds **in
  expectation over the offset distribution only**.
- The existing unit test (`test_operators.py:87`) silently weakened the
  assertion to `delta_s^2 / 4` and its own comment admits the discrepancy, so
  the test suite and the preregistration disagreed.
- Because the quantity is an algebraic identity of the operator, **EXP-08 cannot
  fail** — measuring it on trained models does not change this. It is demoted to
  a unit-test identity check and carries no verdict in Table 3. Its
  verdict-bearing role is replaced by two tests that *can* fail: target
  sharpness under compounded projection (E8b) and the empirical contraction
  modulus (E8c, which the docstring already promised as "& Contraction Test" but
  never implemented).
- `decided-before-results`

**[ ] A-04. Anchor term introduced; the contribution is reframed.**

- The shipped objective's only ground-truth signal was the terminal Dirac,
  attenuated by `lambda_j = 0.6^(dt/delta_s) ~ 0.489` on sepsis, i.e. `~0.001`
  after ten visits — and misplaced (D9). Everything else was self-distillation
  against the EMA target.
- *Amendment*: `loss = (1 - alpha) * L_TD + alpha * L_anchor` with a
  **censored-CRPS** anchor at every visit, expressed in the Cramer metric so
  that alpha is a genuine convex weight in `[0,1]` with no auxiliary scale
  hyperparameter. This is also where D11's preregistered IPCW weighting finally
  exists.
- **The risk, stated plainly: at `alpha = 1` the objective is a per-visit proper
  scoring rule on residual time, i.e. Dynamic-DeepHit's L1 in Cramer form. There
  exists a value of alpha at which SurvTD *is* the baseline it claims to beat.**
  This reframes the contribution: SurvTD is not "a TD objective replacing the
  likelihood" but **a duration-aware TD consistency regularizer added to a
  likelihood**, and C_3 must become "the TD term adds >= 0.025 over the same
  backbone with the same supervision".
- *Controls this obliges us to add*: an `alpha = 1, no TD term` arm as a
  first-class condition, and a new **Kill Criterion 5** — if `alpha=1` matches
  full SurvTD within seed noise, C_0 and C_3 are falsified. This is the most
  likely way the paper dies after the repair and is declared here in advance.
- alpha is selected **once** (validation split, `lambda=0.6`, seed 42,
  `alpha in {0, 0.25, 0.5, 0.75, 1}`) then frozen across every lambda, arm,
  cohort and seed, and applied identically to DeepTCSR-Clamped so that EXP-03 /
  EXP-05 measure the operator rather than who received supervision.
- `decided-before-results`

**[ ] A-05. HPO budget reduced.**

- Declared (section 3): 20-trial TPE under subject-level 5-fold CV = 1600 fits
  across 4 methods x 4 cohorts. Infeasible on the available hardware (one RTX
  3050 OEM; measured 49.2 s/epoch for SurvTD on 240 subjects pre-optimization).
- *Amended*: 12-trial search under subject-level 3-fold CV, 6 epochs per trial,
  run once per (cohort, method) at seed 42 and reused across the 5 evaluation
  seeds. Search **bounds** unchanged from section 3
  (`lr in [1e-4, 5e-3]`, hidden `{64,128,256}`, dropout `[0.1,0.5]`,
  batch `{32,64,128}`; `num_layers = 2` fixed for parity). TPE is used when
  optuna is available, otherwise random search.
- Budget parity across methods — the property section 3 was protecting — is
  preserved exactly.
- `decided-before-results`

**[ ] A-06. Section 5's standard-error claim withdrawn.**

- Declared: "test bootstrap standard error was measured as 0.008 on MIMIC-IV
  Sepsis-3", from which the 0.025 meaningful delta is derived as `3 x SE`.
- No MIMIC-IV data exists in the repository and no bootstrap was ever run. **The
  claimed measurement does not exist.**
- *Amended*: 0.025 is retained as a fixed, pre-declared threshold, but its
  `3 x SE` derivation is withdrawn. The actual bootstrap SE will be reported
  alongside it.
- `decided-before-results`

**[ ] A-07. Primary endpoint restated as landmarked.**

- Replaces the unlandmarked last-visit score (D-EVAL) with landmarked Antolini
  C^td and Uno cumulative-dynamic AUC at a `(L, Delta)` grid declared here,
  before any result is read:
  - Synthetic ICU: `delta_s=2.0, K=36`, `L in {12, 24, 48} h`, `Delta = 24 h`
  - C-MAPSS: `delta_s=5.0, K=30`, `L in {50, 100, 150} cyc`, `Delta = 50 cyc`,
    administrative censoring at `L + Delta`
  - PBC: `delta_s=30, K=30`, `L in {0, 180, 365}`, `Delta = 365`
  - Tumor: `delta_s=0.5, K=25`, `L in {2, 4, 6}`, `Delta = 2`
- `decided-before-results`

**[ ] A-08. Adjudicating statistic changed, because the declared test cannot
reach significance.**

- Section 5 mandates a paired two-tailed Wilcoxon signed-rank test over 5 seeds.
  **The minimum attainable exact two-tailed p for n=5 is `2/2^5 = 0.0625`**, so
  the declared significance requirement is unsatisfiable as written.
- *Amended*: the Wilcoxon result is reported as declared, but the statistic that
  adjudicates the 0.025 threshold is the 1000-sample **subject-level paired
  bootstrap CI of the per-subject metric difference** — also mandated by section
  5, also never implemented.
- `decided-before-results`

**[ ] A-09. EXP-05's disjunctive kill criterion split into three.**

- `run_track_b.py:177` used `c_survtd > c_clamped + 0.01 **or** grad_norm > 1.5`.
  The preregistered hypothesis is that clamped division "will exhibit severe
  gradient variance spikes"; the observed gradient norm was **0.03**, which
  *contradicts* it — yet the run was recorded as PASSED on the C-index limb
  alone.
- Also, the probe itself measured nothing usable: one backward pass on
  `train_set[0]`, one seed (`:168-175`).
- *Amended* into three independently reported verdicts: E5a discrimination
  (paired, 5 seeds, CI); E5b gradient stability (per-step grad norms over a full
  epoch, pre-declared spike index / CV / clip-rate statistics, **stratified by
  the smallest-gamma decile** as the hypothesis actually requires); E5c
  numerical degeneracy (clamp activation rate, uniform-fallback rate from
  `clamped_division_target:112-114`, total-variation error against the exact
  renewal target).
- *Pre-declared interpretation*: if the clamp never binds on these cohorts, E5b
  is **expected** to fail and the correct conclusion is that the
  numerical-degeneracy argument is a theoretical contribution not evidenced
  here. That will be stated in Table 3 rather than worked around.
- `decided-before-results`

**[ ] A-10. EXP-06's missing subsampling sweep added; criterion tests the
interaction.**

- Section 4 NC-C requires a sweep at 100% / 50% / 25% observation retention.
  `run_track_b.py:190-193` ran a single condition and the verdict rested on
  `c_survtd >= c_count_geom`, observed 0.809 vs 0.806 — a 0.003 delta from one
  seed with no interval.
- *Amended*: full `{duration, count} x {1.0, 0.5, 0.25} x 5 seeds` grid. Because
  the claim is *invariance* rather than level, the criterion is the retention
  slope per mixing rule: `mean |slope_duration| <= 0.5 * mean |slope_count|`,
  paired Wilcoxon over seeds, full grid reported with CIs.
- *Plus a training-free analytic test (E6a) run first*: the lambda-return weight
  on the n-step term is `w_n = (1-lambda) * prod_{i<n} lambda_i`. For
  duration-geometric mixing `prod lambda_i = lambda^(sum dt_i / delta_s)`, a
  function of elapsed time only, so the effective horizon in time units is
  invariant to visit density; for count-geometric it is `lambda^n` and scales as
  `1/retention`. Pre-declared: duration-geometric `H_eff` varies < 5% across
  retention, count-geometric > 50%. **If this fails, C_2 is falsified
  analytically at zero GPU cost.**
- `decided-before-results`

**[ ] A-11. Gain-retention statistic replaced by the paired absolute delta.**

- `run_track_b.py:93,107,134,140` computed
  `max(0, c_arm - c_base) / max(1e-4, total_gain)` — a ratio of two noisy
  quantities, clipped at both ends, undefined when `total_gain ~ 0`, and
  normalized against a baseline that scored below chance (D-PP).
- Section 4 states the hypothesis in **absolute** units ("Arms A1 and A2 will
  lose at least 0.025 in AUC/C-index") while the code's kill criterion was
  **relative** (">50% gain retained"). These are different criteria.
- *Amended*: primary statistic is the paired absolute delta `c_full - c_arm` per
  seed with a paired Wilcoxon and bootstrap CI against 0.025 — what was actually
  preregistered. Retention % becomes secondary, computed per seed only when that
  seed's `total_gain > 0.01`, reported with a CI and explicitly `undefined`
  otherwise. Both AUC and C-index are reported; the code computed only C-index.
- **Verdict flips are expected** once the baseline is repaired.
- `decided-before-results`

**[ ] A-12. EXP-04 gains a positive control.**

- A negative control with no floor is uninterpretable. Adding
  `shuffle_durations_across_patients` (which destroys all duration information)
  establishes the floor, so retention normalizes against a real `[floor, full]`
  interval instead of a broken baseline.
- `decided-before-results`

**[ ] A-13. Threshold calibration moved off the test set.**

- `run_track_a.py:174` passes test-set predictions into
  `evaluate_alarm_fatigue`, which calls `calibrate_threshold_for_ppv` on those
  same predictions. Selecting the 0.30-PPV operating point on the evaluation
  data directly inflates the C_4 / EXP-07 claim.
- `compute_decision_curve_analysis:118` has the same last-visit leak, using
  `max_risks` over the whole trajectory.
- *Fix*: freeze the threshold on validation, apply to test; move DCA onto the
  landmark-conditional risk at `(L, Delta)`.
- `decided-before-results`

**[ ] A-14. "Alert Jitter Count" was mislabelled.**

- The `break` at `alarm_fatigue.py:96` caps the counter at 1 per subject, so the
  reported quantity is a **subject fraction**, not a count.
- The relative-reduction column divided by `max(1e-4, 0)`, producing
  `-1999900.0%`.
- The window loop (`:86-89`) also recomputes `in_window` over the whole `times`
  array per index (O(L^2)) and anchors overlapping windows at every observation,
  double-counting.
- *Fix*: report "unstable 6h-window count per subject-day" as the headline rate
  and "fraction of subjects with >= 1 unstable 6h window" as a correctly
  labelled secondary column; render `n/a` when the reference is 0 rather than
  dividing by an epsilon.
- `decided-before-results`

**[ ] A-15. `gamma_placement` exposes a genuine tension in C_2.**

- C_2's prose says the recursion "compounds interval survival discounting",
  which corresponds to multiplying the recursive branch by gamma as well
  (`gamma_placement='compounded'`). But under that choice **`lambda = 1` no
  longer reduces to Monte Carlo**, contradicting Kill Criterion 2's phrasing and
  `run_lambda_ablation.py`'s own "lambda=1.0 = Pure Monte Carlo / NLL".
- Both placements are implemented, `'bootstrap'` is the default, and **both are
  reported**. The tension is recorded here rather than resolved by quietly
  picking whichever performs better.
- `decided-before-results`

---

**[ ] A-16. Hazard-head initialization defect, and the exploratory re-run it licenses.**

- **This entry is `decided-AFTER-results`.** Kill Criterion 3 fired on Synthetic
  ICU (5 seeds, alpha = 0.0, default initialization) and this amendment was
  written in response to that outcome. It does **not** carry the standing of
  A-01 .. A-15 and must never be presented as if it did.
- **The measured defect.** Under PyTorch's default `nn.Linear` initialization the
  hazard-head logits sit at ~0, so at step 0:

  | quantity | measured | cohort truth (synthetic_icu, seed 42 train split) |
  |---|---|---|
  | mean per-bin hazard | 0.4999 | 0.010 - 0.039 |
  | `S(K * delta_s)` | 1.41e-11 | 0.59 |
  | `gamma = S(dt)` at median gap (2.11h) | 0.481 | 0.979 |
  | `gamma = S(dt)` at p90 gap (6.92h) | 0.091 | 0.933 |

  `gamma_j` is the contraction modulus of the renewal operator, so the effective
  credit-assignment horizon `1/(1 - gamma)` collapses from ~48 steps to ~1.9.
  At alpha = 0 the terminal Dirac is the only ground truth in the objective, and
  over a ~10-visit trajectory it reaches the first visit attenuated by
  `0.48^10 ~ 1e-3`. The failure is self-reinforcing: small gamma routes mass to
  the near branch, which keeps S small, which keeps gamma small.
- **Three corroborating observations in the primary run**, none of which the
  original diagnosis note anticipated:
  1. `corr(SurvTD, DeepHit)` across seeds is **-0.211** while
     `corr(Person-Period, DeepHit)` is **+0.988**. Every arm except SurvTD tracks
     per-seed cohort learnability; SurvTD's output is dominated by optimization
     noise rather than data signal.
  2. SurvTD IBS on seed 456 is **0.468** (0.084 - 0.136 elsewhere), the
     signature of a survival curve collapsed toward 0. DeepTCSR shows the same
     signature in 4 of 5 seeds (IBS 0.56 - 0.60).
  3. Seed 456's SurvTD run took **335.9s** against 675 - 952s on the other seeds,
     i.e. `patience = 5` terminated it around epoch 7. During cold start the
     validation C^td sits at chance and does not improve, so the stopping rule
     and the initialization defect compound.
- **What changes.** (a) `DiscreteHazardHead.init_prior_bias` /
  `init_constant_bias` and `apply_hazard_prior_init`, the latter re-syncing
  `target_head` -- without that resync the target network, which is what supplies
  gamma_j, would keep the collapsed bias and the fix would be inert. (b) The KM
  prior is the marginal KM of **residual times** `R_j = tte - t_j` pooled over
  training-split (subject, visit) pairs, **not** the time-from-enrolment KM the
  research note specified; the head's support is residual time, and on cohorts
  whose visit density varies over time the two curves differ materially.
  `eps = 1e-4` is fixed here and defines the tail prior for bins past the last
  observed event; it is declared, not tuned. (c) `train_model` gains
  `es_warmup = 5`: patience does not accrue before epoch 5. (d) `run_track_a`
  flushes results after every (cohort, seed, method).
- **Parity is mandatory.** The initialization is applied identically to all four
  neural arms (SurvTD, DeepTCSR-Clamped, Dynamic-DeepHit, Person-Period). The
  honest expectation is that Dynamic-DeepHit improves too -- its per-visit
  likelihood already pulls the hazards down within one epoch, which is why it
  scored 0.646 from the same broken initialization. If the TD term only wins
  when the baselines are crippled, there is no contribution.
- **The alpha = 0.0 selection is void.** It was chosen on a single seed-42
  validation run whose full sweep spread (0.078) is the same order as the
  initialization noise HANDOVER measured (0.042), and seed 42 is the **only** one
  of five seeds on which SurvTD beat Dynamic-DeepHit (+0.026; the other four:
  -0.204, -0.096, -0.155, -0.035). It was also selected under the defective
  initialization, which changes the TD term's scale. alpha must be re-selected
  after the initialization fix, over **3 seeds** rather than 1.
- **Gate, declared before running.** Synthetic ICU, 5 seeds, `km_prior`. Proceed
  to a full re-run only if **both**: mean landmarked C^td for SurvTD >= 0.60,
  **and** `corr(SurvTD, DeepHit)` across seeds turns positive. The second
  condition is the load-bearing one -- a mean can rise by luck, but tracking
  cohort difficulty is direct evidence the model is learning the data. If the
  gate fails, the initialization diagnosis is rejected and the preregistered
  result stands as final.
- **Reporting.** The alpha = 0.0 / default-initialization Cohort 1 result is the
  **preregistered primary outcome and C_3 is falsified on it**; the raw log is
  preserved at `experiments/results/preregistered_primary_2026-09-04/`. Anything
  produced under this amendment is reported in a separate, explicitly
  **secondary / exploratory** block and never replaces it.
- `decided-after-results`

---

**[ ] A-16b. The first A-16 gate failed; one final initialization variant, declared before it runs.**

- **A-16's gate is failing.** Under `km_prior` (KM bias + `W ~ N(0, 1e-4)`),
  alpha = 0, 20 epochs, SurvTD scored **0.5365** on seed 42 (primary: 0.5831) and
  **0.4797** on seed 123 (primary: 0.5076) -- both *below* the defective-init run.
  G1 requires a 5-seed mean >= 0.60; the remaining three seeds would have to
  average 0.661 against primary values of 0.519 / 0.517 / 0.641. The run is being
  carried to all five seeds so the record is complete, but the verdict is STOP.
- **The substantive result: gamma was not the binding constraint.** Raising
  gamma from 0.481 to 0.980 did not improve discrimination; it slightly reduced
  it. That is evidence *against* the credit-assignment reading of A-16 and *for*
  the more basic one -- at alpha = 0 the TD objective lacks the supervision to
  learn discrimination at all. This is exactly what **Kill Criterion 5** tests,
  and what HANDOVER named as the likeliest failure (`C_0`).
- **A confound in A-16's own design, found by measurement, not by argument.**
  The declared intervention changed two things at once. Between-subject SD of the
  predicted risk `F(K/2)` at initialization, over 6 inits on synthetic_icu seed 42:

  | init | risk SD | gamma(bin0) |
  |---|---|---|
  | default | 0.00000 | 0.55 |
  | `km_prior` (`W ~ N(0,1e-4)`) — what the gate ran | 0.00001 | 0.98 |
  | KM bias, default `W` | **0.00894** | 0.97 |

  Both of the first two are 0, for *different* reasons: default init saturates
  (hazard ~ 0.5 drives `F(K/2)` to 1.0 for everyone), while `km_prior` zeroes the
  covariate pathway. The research note's `W ~ N(0, 1e-4)` was adopted for
  "zero-step calibration", but under alpha = 0 it starts the model at a
  covariate-free population curve that is close to a fixed point of the
  bootstrapped target, with no strong gradient to leave it. Dynamic-DeepHit is
  unaffected because its per-visit likelihood rebuilds the weights within an epoch.
  **This is a defect in the amendment's experimental design, not a result.**
- **A-16b, declared before running.** `km_bias_only`: the KM prior bias with the
  **default weight matrix retained**, so gamma ~ 0.97 *and* the covariate pathway
  stays alive. Same cohort, same 5 seeds, same alpha = 0.0, same gate thresholds
  (mean C^td >= 0.60 **and** corr(SurvTD, difficulty) > 0).
- **This is the LAST initialization variant.** Running variants until one passes
  is the failure mode this log exists to prevent. If A-16b does not clear the gate,
  the initialization diagnosis is rejected outright, the preregistered primary
  outcome is final, and no further initialization work is done. The `km_prior`
  failure is reported alongside A-16b's outcome either way -- it is not superseded.
- **Order of work.** Kill Criterion 5 runs **first**, at the preregistered
  configuration (default init, alpha = 0.0, 5 seeds), because it decides `C_0`
  independently of any initialization question and has never once executed. Its
  one deviation is `es_warmup = 5`, applied identically to both arms; it can only
  help the cold-starting alpha = 0 arm, so a falsification under it is
  conservative.
- `decided-after-results`

---

**[x] A-16/A-16b OUTCOMES and KILL CRITERION 5 (recorded as observed, 2026-09-04).**

**Kill Criterion 5 — FALSIFIED.** First execution ever. Synthetic ICU, 5 seeds,
preregistered configuration (default init, alpha = 0.0 for the full arm), full
SurvTD against the anchor-only alpha = 1 control on an identical backbone:

| seed | full (TD) | anchor-only | paired delta |
|---|---|---|---|
| 42 | 0.6089 | 0.5619 | +0.0470 |
| 123 | 0.4350 | 0.5185 | -0.0835 |
| 456 | 0.5120 | 0.5416 | -0.0296 |
| 789 | 0.4861 | 0.5689 | -0.0828 |
| 101112 | 0.6346 | 0.4679 | +0.1667 |
| **mean** | **0.5353 ± 0.0841** | **0.5318 ± 0.0407** | **+0.0036**, 95% CI **[-0.0724, +0.0929]** |

Requirement was delta >= 0.015 AND CI lower bound > 0. Neither holds; only 2 of 5
seeds are positive. **Per preregistration §6.5, `C_0` and `C_3` are falsified.**

The sharpest way to state it: the TD term moved the mean by **+0.0036** and
multiplied the across-seed variance by **4.3x** (SD 0.0407 -> 0.0841). It is not
that the TD term is slightly worse than the anchor -- it adds noise and no signal.
Note also that the largest positive delta (+0.1667, seed 101112) comes from the
anchor arm scoring 0.4679, i.e. below chance, not from the TD arm excelling.

`es_warmup = 5` was in effect for both arms. It can only help the cold-starting
alpha = 0 arm, so this falsification is conservative.

**A-16 gate — STOP.** `km_prior`, 5 seeds: mean C^td **0.5256** (primary 0.5536),
corr with cohort difficulty **-0.084**. Both gate conditions failed, and **all five
seeds moved down**. What the initialization did fix was calibration: mean IBS
0.186 -> 0.131, and the collapsed seed 456 went 0.468 -> 0.156. Discrimination did
not move. The initialization defect caused the survival-curve collapse; it did not
cause the discrimination failure. Two separate failures; A-16 diagnosed one.

**A-16b gate — STOP.** `km_bias_only` (KM bias, default weight matrix retained, so
gamma ~0.97 with a live covariate pathway) scored **0.4880 ± 0.0645**, *worse* than
both `km_prior` (0.5256) and the primary run (0.5536), with corr **-0.658**. The
covariate-pathway confound identified in A-16b was real as a measurement and wrong
as an explanation.

**As declared in A-16b, this was the last initialization variant.** The
initialization diagnosis is rejected outright. No further initialization work.
The preregistered primary outcome stands: `C_3` falsified on Cohort 1.

**What remains unexplained, and is now the live question.** SurvTD's anchor-only
arm (0.5318) and Person-Period (0.6367) share a backbone, a head, a bin convention
and a prediction target, and neither uses TD -- yet differ by ~0.10. The two
remaining differences are the loss geometry (squared Cramer/CRPS versus masked BCE
on per-bin hazards) and the supervision density (Person-Period trains on the
1-hour grid expansion, ~3x the rows). Squared Cramer is L2 on the CDF, with a
gradient linear in the residual and dominated by the population-level curve shape;
BCE's gradient diverges on confident errors and separates subjects harder. That
predicts the observed pattern -- SurvTD arms have acceptable IBS and poor C^td.
This is a **fourth** candidate cause and, per the lesson of A-16, it must be
isolated by measurement (Person-Period without grid expansion) before any loss
redesign is attempted.

- `decided-after-results`

---

## 5. Environment

**[x] E-01. `scikit-survival` cannot be installed normally on Python 3.14.**

- `pip install scikit-survival==0.28.0` fails because its transitive dependency
  `ecos` has no cp314 wheel and its source build errors out. `ecos` is required
  only by `FastSurvivalSVM`, which this project does not use.
- Resolved with `pip install --no-deps scikit-survival==0.28.0`. Verified:
  `concordance_index_ipcw`, `cumulative_dynamic_auc` and
  `integrated_brier_score` all import; scikit-learn stays at 1.9.0 (no
  downgrade). Recorded in `requirements.txt`.

**[x] E-02. `lifelines` is deliberately not a dependency.**

- `metrics.py:11` imported it, and it is not installed — **the shipped pipeline
  could not even be imported in this environment**, so the published results
  were produced elsewhere and are not reproducible here.
- Rather than install it, the Kaplan-Meier censoring estimator moves to
  `src/evaluation/censoring.py` so that `G(t-)` (left limit) and
  train-fit / test-apply behaviour are under explicit control.

**[x] E-03. `experiments/unit_tests/` was not discoverable.**

- Missing `__init__.py` made `unittest discover` fail outright, so the suite had
  never run as a suite. Added; 11 pre-existing tests pass.

---

## 5b. Measured outcomes of the repair (recorded as observed)

**[x] M-01. D8 did NOT explain the EXP-04 collapse. Prediction not supported.**

The Stage 1 rationale predicted that repairing the duration misalignment would
separate the full model from the NC-B duration-permutation control, because
shifting the duration sequence by one index is itself a form of permutation.
Measured after the fix (sepsis cohort, n=250, seed 42, 8 epochs, single seed):

| condition | C |
|---|---|
| Person-Period reference | 0.7468 |
| SurvTD full | 0.7861 |
| SurvTD permuted (NC-B) | **0.7962** |
| gap (full - permuted) | **-0.0101** |
| gap before the fix (archived) | +0.0030 |

The gap did not open; the permuted control scored marginally *higher*. **The
hypothesis that D8 was the reason EXP-04 collapsed is not supported.**

Moreover the checkpoint was underpowered by construction, and that is a flaw in
the diagnostic rather than merely an inconclusive result. It was run before the
evaluation was repaired, so both conditions were scored by the leaky last-visit
risk `cdf[-1, K//2]`. Because the generator sets `tte ~ times[-1] + U(0.1,2)` for
events and `tte = 72.0` for everyone censored, that evaluation largely rewards
encoding trajectory *length* -- and `permute_patient_durations` preserves
`sum(dt)` exactly. The two conditions therefore score on the same leak and cannot
be separated by this measurement, whatever the operator does.

Consequences:
- D8 remains a real defect and its fix stands on the unit tests
  (`test_total_shift_uses_forward_gaps`), not on this outcome.
- EXP-04 must be re-measured only after **both** the landmark evaluation and the
  generator fix (X-02) are in place, plus the positive control from A-12.
- The Person-Period figure moving from the archived 0.263 to 0.7468 is **not**
  evidence of a repair: the archived number came from a 2-epoch, 12-subject dry
  run, and the Person-Period loss defect (D-PP) had not yet been touched at the
  time of this measurement. The two numbers are not comparable.
- Do not cite any number in this table. Single seed, no CI, leaky evaluation.

**[x] M-05. The anchor repaired the supervision deficit. My parity smoke test
was mis-specified, and single-seed comparisons at this precision are worthless.**

Stage 4 checkpoint, on the OLD generator with the leaky last-visit evaluation and
a single seed. Diagnostic only; no number here may be reported.

| model | C |
|---|---|
| Person-Period | 0.8013 |
| Dynamic-DeepHit (alpha_rank=0.5) | 0.8899 |
| DeepTCSR-Clamped (alpha=0.5) | 0.7937 |
| SurvTD alpha=0.0 (pure TD) | 0.8203 |
| SurvTD alpha=0.5 | 0.8038 |
| SurvTD alpha=1.0 (pure anchor) | 0.8367 |

**Person-Period is 0.8013, not the archived 0.263.** The below-chance defect is
gone even under the leaky evaluation.

**My smoke test premise was wrong.** It asserted that at alpha=1 SurvTD and
Dynamic-DeepHit have near-identical supervision, so their C-index must meet.
The gate therefore reported "alpha=1 does NOT meet Dynamic-DeepHit; something
beyond the supervision deficit is still wrong". That conclusion was unfounded: it
overlooked DDH's pairwise ranking term (L2, alpha_rank=0.5), which directly
optimizes the reported metric while CRPS is a proper score for the whole
distribution. Decisive check, same seed and configuration:

| condition | C |
|---|---|
| DDH with ranking loss | 0.8481 |
| DDH with ranking loss REMOVED (alpha_rank=0) | **0.6633** |
| SurvTD alpha=1.0 | 0.8367 |

The ranking term is worth +0.185 on its own. Against DDH's pure likelihood,
SurvTD's censored-CRPS anchor is far ahead (0.837 vs 0.663), which is the
expected direction: CRPS scores the entire CDF, while DDH's L1 scores a single
bin per visit. No hidden defect is implied.

**A larger caveat, which matters for the whole exercise.** DDH scored 0.8899 in
the checkpoint and 0.8481 in the follow-up under the same seed and config: the
checkpoint seeded once and trained six models in sequence, so the RNG state at
model construction differed. **Initialization alone moves the result by 0.042**
-- larger than the 0.053 "parity gap" I was investigating, and larger than the
0.025 delta the paper claims. No conclusion may be drawn from a single-seed
difference at this precision. This is exactly why preregistration section 5
mandates 5 seeds plus a subject-level paired bootstrap, and it is a reason to
distrust any of the withdrawn single-seed Track B verdicts independently of the
operator defects.

**[x] M-06. EXP-05 / E5c: the clamp essentially never binds, so NC-A3's
numerical degeneracy is absent from this cohort.**

Instrumented over 18,528 clamped-division targets during training: the clamp
`max(S(dt), 1e-3)` bound on **0.01%** of them and the uniform fallback fired on
**0.00%**. This explains the observed gradient norm of 0.03 that contradicted the
preregistered hypothesis of "severe gradient variance spikes": there is nothing
to blow up, because `S(dt)` essentially never approaches zero at these hazards
and inter-visit durations.

Pre-declared consequence (A-09): a null gradient-stability result on this cohort
is EXPECTED and uninformative rather than falsifying. The honest conclusion to
report in Table 3 is that the numerical-degeneracy argument behind C_1 is a
theoretical contribution not evidenced empirically here. Demonstrating it would
require a pre-declared high-hazard regime in which the clamp actually binds; that
regime must be declared before it is run and labelled exploratory.

**[x] M-04. The landmark harness is verified not to leak. Gate passed.**

The check that would have caught the last-visit leak, run before any training:
does an uninformative predictor score at chance? Two predictors are used, and
both must.

*Covariate-free Kaplan-Meier reference* (constant prediction, fit on train,
applied to test) -- **C^td = 0.500 exactly at all 10 (cohort, landmark)
combinations**, with IBS 0.048-0.187, below the 0.25 reference level everywhere.

*Untrained networks, averaged over 12 random initializations*:

| cohort | landmark | mean null AUC | s.e. | band | verdict |
|---|---|---|---|---|---|
| synthetic_icu | 12 / 24 / 36 | 0.518 / 0.516 / 0.490 | .030/.043/.026 | .059/.087/.052 | PASS |
| cmapss FD002 | 125 / 150 / 175 | 0.475 / 0.499 / 0.495 | .027/.028/.028 | .054/.056/.055 | PASS |
| pbc | 0 / 180 | 0.507 / 0.482 | .043/.048 | .086/.095 | PASS |
| tumor | 1.5 / 2.0 | 0.549 / 0.522 | .026/.061 | .051/.122 | PASS |

Three corrections to the GATE ITSELF were needed, all found by running it. Each
was a flaw in the guard, not in the pipeline, and they are recorded because a
guard that was never wrong was probably never tested:

1. **An untrained network is not a null model.** Random weights are a random but
   non-degenerate linear projection of the features, so on the tumour cohort --
   where tumour size nearly determines the outcome -- a single initialization
   reached AUC 0.882. The expectation over initializations is 0.5; one draw is
   not. The gate now averages over initializations.
2. **An absolute Brier ceiling is the wrong guard.** The first version asserted
   IBS <= 0.30 inside integrated_brier and immediately tripped on untrained
   models. An untrained network has per-bin hazard ~0.5, so its survival decays
   like 0.5^k and it legitimately scores ~1.0 on survivors: the assertion
   conflated "the computation is broken" with "the model is bad". Replaced by a
   range check on [0, 1] plus the KM-marginal row as the yardstick.
3. **The decision rule ignored Monte-Carlo error.** Comparing the mean to a fixed
   0.08 band flagged pbc L=0 (0.581) and tumor L=2.0 (0.583) as leaks when both
   sat within 1.2-1.7 standard errors of chance at 6 initializations. The rule is
   now "within max(0.05, 2 s.e.) of 0.5". **The band was not widened until things
   passed** -- that is the behaviour this effort exists to prevent. Instead the
   error was reduced by doubling the initializations, and the estimates moved
   TOWARD the null (0.581 -> 0.507, 0.583 -> 0.522), which is the evidence that
   distinguishes noise from leakage.

**[x] M-03. The anchor is scale-matched to within ~12x, not exactly.**

Measured at initialization on a 5-visit synthetic trajectory (delta_s = 1, K = 20):
TD term 0.2139, anchor term 2.6103, ratio **12.2x**. An IPCW-NLL anchor would sit
around 100x above the Cramer term, so expressing the anchor in the same metric
buys roughly an order of magnitude -- but it does not make the two terms equal.

Stated plainly: **alpha is a genuine convex weight, but it is not a balance
parameter.** alpha = 0.5 puts equal weight on two terms of unequal magnitude, so
at that setting the anchor dominates the gradient by roughly 12:1 at
initialization (the gap should narrow as the model approaches the data). No
rescaling is applied, because a normalizing constant would reintroduce exactly
the free scale hyperparameter the Cramer-metric anchor was chosen to avoid. The
consequence is handled by reporting the whole alpha curve rather than defending a
single value, and by the alpha = 0 and alpha = 1 endpoint arms.

**[x] M-02. The gamma reformulation is verified at the target level.**

After Stage 3, with gamma derived from the mass of the restricted near branch:
- near-branch mass equals `(1-lambda)(1-gamma)` and the far branch
  `(1-lambda)*gamma + lambda`, to 5 decimal places, for gamma in {0.95, 0.7, 0.4};
- the defect identity `lambda*gamma/((1-lambda)+lambda*gamma)` is now false;
- `lambda = 1` reproduces an independently constructed projected Monte-Carlo
  target to 1e-6 at every visit, and `lambda = 0` the one-step renewal target;
- all four arms in `ARMS` produce measurably different targets;
- `gamma_placement='compounded'` demonstrably breaks the `lambda = 1` Monte-Carlo
  endpoint, which is the tension recorded in A-15.

These are training-free, target-level checks, per standing rule 2.

---

## 6. Red-test guardrails

**[x] R-01. `experiments/unit_tests/test_operator_defects.py` added.**

Characterization tests that fail against the pre-repair operator and must pass
after each fix lands. Current status (pre-repair):

| Test | Defect | Status |
|---|---|---|
| `test_total_shift_uses_forward_gaps` | D8 | FAIL (expected) |
| `test_terminal_dirac_at_residual_time` | D9 | FAIL (expected) |
| `test_gamma_at_one_bin_equals_first_entry` | D10 | FAIL (expected) |
| `test_gamma_is_continuous_at_the_sub_bin_boundary` | D10 | FAIL (expected) |
| `test_future_branch_weight_is_not_the_lambda_reparameterization` | D-gamma | FAIL (expected) |
| `test_branches_are_separated` | test self-guard | ok |
| `test_gamma_changes_the_target_shape` | characterization | ok |
| `test_returns_sum_to_one_across_lambda_and_gamma` | invariant | ok |

The D-gamma test is the decisive one: the bootstrap-branch mass matches
`lambda*gamma/((1-lambda)+lambda*gamma)` to float32 precision
(`0.5876288414` vs `0.5876288660` at `gamma=0.95`).

Note on test design: at `j = L-2` the one-step and bootstrap branches are
identical by construction on a censored trajectory, because `G_{L-1}` *is*
`target_pmfs[-1]`. gamma is unobservable there. The test therefore uses `L=4`
and reads the mixture at `j=1`, with a `test_branches_are_separated` self-guard
so a future degenerate construction cannot silently make the red test vacuous.
