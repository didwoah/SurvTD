# Handover — experiment repair effort (2026-09-03)

Branch: **`feature/experiment-repair`** (7 commits ahead of `main`, working tree clean)
Remote: `origin` → `https://github.com/didwoah/SurvTD.git`

---

## 0. Pushing

Nothing has been pushed. Everything is committed locally on the branch above.

```bash
git checkout feature/experiment-repair
git push -u origin feature/experiment-repair
```

If that fails on auth, either configure a credential helper / PAT, or export the
branch as a bundle and move it:

```bash
git bundle create survtd-repair.bundle main..feature/experiment-repair
# then, wherever the repo is reachable:
git fetch survtd-repair.bundle feature/experiment-repair:feature/experiment-repair
```

Do **not** fast-forward `main` onto this branch yet: the experiment runners do not
execute (see §3).

---

## 1. What the repair was for

`experiments/results/` contained Table 1–3 and a falsification report that could
not serve as evidence in either direction. The reasons were not "the model is
weak" — they were defects in the evaluation harness, the data generators, and the
operator. The four originals are now in
`experiments/results/invalidated_2026-09-03/` with a README recording every reason;
they are **withdrawn, not superseded**.

Full audit trail: `research/continuous-survival-td/deviation_log.md` (~850 lines,
every defect and every preregistration amendment, each tagged
`decided-before-results`).

The three findings that mattered most, all verified by measurement:

| ID | Defect | Evidence |
|---|---|---|
| **D-gamma** | The design document's Step 5 renewal mixture was never implemented, so the duration discount γ was **exactly** a reparameterization of λ | bootstrap-branch mass matched `λγ/((1−λ)+λγ)` to float32 precision (`0.5876288414` vs `0.5876288660`). This — not a falsification of C₁ — is why EXP-03/NC-A1 retained 93.9% of the gain |
| **D8** | Transition `j→j+1` needs `dts[j+1]` but the operator used `dts[j]`; every transition was shifted one index | `times=[1,4,8,20]`: needed 3/4/12, code used 1/3/4 |
| **D9** | The only ground-truth target sat at the length of the *previous interval* instead of the residual time | `tte=8.5, t_last=8.0` → code bin 1, truth bin 0 |

Two further facts a reviewer will ask about:

- The cohort labelled **"MIMIC-IV Sepsis-3"** in Tables 1–2, the README and
  preregistration §1/§3/§5/§6 was a **synthetic generator**. No MIMIC data exists
  in this repository and none was used.
- Preregistration §5's "test bootstrap standard error was measured as 0.008 on
  MIMIC-IV Sepsis-3" — from which the whole 0.025 threshold is derived as `3×SE` —
  **describes a measurement that does not exist**.

---

## 2. What is done (7 commits)

| Commit | Stage | Content |
|---|---|---|
| `8a670ab` | 0 | Environment pin, 5 red tests, results withdrawn, deviation log started |
| `922a6ea` | 1–2 | D8/D9/D10 fixed; explicit `⊥` overflow coordinate; Cramér loss made an integral (D14) |
| `9dce7fa` | 3 | Renewal mixture implemented, γ made load-bearing, arms unified into `ARMS` |
| `80ecbe0` | 4 | Censored-CRPS anchor, censoring-aware targets, real IPCW (D11) |
| `60c0ed1` | 5 | Person-Period rebuilt; the promised regular-grid expansion implemented |
| `4f72702` | 6 | Data + evaluation layers rebuilt; **landmark harness passes the leak gate** |
| `8babb45` | — | Stage 4 checkpoint measurements recorded, with two corrections to my own reasoning |

**Tests: 96, all passing.** At the start of this effort the suite could not even be
discovered (`experiments/unit_tests/__init__.py` was missing, so
`unittest discover` failed outright — it had never run as a suite).

```bash
python -m unittest discover -s experiments/unit_tests -t .
```

### Verified by measurement

- **γ is a real mechanism.** Near branch carries `(1−λ)(1−γ)`, far branch
  `(1−λ)γ+λ`, to 5 decimals. The defect identity is now false. `λ=1` reproduces an
  independently constructed projected Monte-Carlo target to 1e-6 at every visit;
  `λ=0` gives the one-step renewal target exactly. **C₁'s contraction claim becomes
  true for the first time**: the linear part of `T̂` in its far argument is
  `γ_j·ΠΦ` with modulus `γ_j < 1`, which is what `thm:1` asserts — under the
  shipped code it was `λ_eff·ΠΦ`, so the theorem did not describe the code.
- **Person-Period: 0.263 → 0.8013.** The below-chance defect is gone (C-index
  1.000 on a trivially separable unit test). Every gain-retention denominator in
  Table 3 had been normalized against the broken version.
- **The anchor repaired the supervision deficit.** Censored trajectories now carry
  a proper score at every visit; previously ~72% of the sepsis cohort contributed
  nothing.
- **The leak gate passes** on 4 cohorts × 10 (landmark, horizon): the KM-marginal
  reference scores `C^td = 0.500` exactly everywhere with IBS 0.048–0.187, and
  untrained networks averaged over 12 initializations score AUC 0.475–0.549.
- **Synthetic ICU cohort repaired**: AUC case/control at the median horizon went
  from 12/**1** (which is why an entire AUC column read a hardcoded `0.500`) to
  108/**250**; distinct censored times 1 → 344; max per-subject feature mean
  3e-5 → 2.8–3.8 (the between-subject level signal is no longer standardized away);
  mask density 0.659, the first genuine missingness in the project.
- **EXP-05/E5c**: over 18,528 clamped-division targets the clamp bound on **0.01%**
  and the uniform fallback never fired. This explains the observed GradNorm of 0.03
  that contradicted the preregistered "severe gradient spikes" hypothesis — there
  is nothing to blow up. Pre-declared consequence: a null result here is expected
  and uninformative, and C₁'s numerical-degeneracy argument is a theoretical
  contribution **not evidenced empirically on these cohorts**.

### Where I was wrong, and corrected

Recorded in the deviation log rather than quietly fixed:

1. **D8 did not explain the EXP-04 collapse.** I predicted the full-vs-permuted gap
   would open; measured **−0.0101** (permuted scored slightly *higher*). The
   checkpoint was also mis-designed: it ran before the evaluation was repaired, so
   both conditions scored on the same last-visit leak. D8 stands on its unit test,
   not on that outcome. (M-01)
2. **My λ=1 prototype check was circular.** It defined the expected Monte-Carlo
   target to include the near-death term and compared against that. The near branch
   must vanish at λ=1; the test now builds the target independently. (Stage 3 commit)
3. **The γ-mass direction in my plan was backwards.** I had written "route the
   discounted mass to a survived-beyond-horizon bin". `1−γ` is the probability of
   dying *inside* the interval, so it belongs in the **early** bins; the other
   direction would invert the risk ordering. (A-02)
4. **My IBS sanity ceiling was a mis-specified assert.** It conflated "the
   computation is broken" with "the model is bad" and tripped on untrained models.
   Replaced by a `[0,1]` range check plus the KM-marginal reference as the yardstick.
   (M-04)
5. **An untrained network is not a null model.** One initialization reached AUC
   0.882 on the tumour cohort, because random weights are a random projection of a
   near-deterministic feature. The gate now averages over initializations. (M-04)
6. **My leak decision rule ignored Monte-Carlo error.** A fixed 0.08 band flagged
   two landmarks that sat within 1.2–1.7 standard errors of chance. The band was
   **not** widened until things passed — the initializations were doubled, and the
   estimates moved *toward* the null (0.581→0.507, 0.583→0.522). (M-04)
7. **My supervision-parity smoke test was mis-specified.** It assumed SurvTD at
   α=1 and Dynamic-DeepHit have near-identical supervision, so the gate declared
   "something is still wrong" when they did not meet. It overlooked DDH's pairwise
   ranking term. Removing it drops DDH from 0.8481 to **0.6633**, far below SurvTD's
   0.8367 — the anchor is ahead of DDH's pure likelihood, in the expected
   direction, and no hidden defect is implied. (M-05)

### The single most important caveat

**Initialization variance exceeds the effect the paper claims.** Dynamic-DeepHit
scored **0.8899** and **0.8481** under the same seed and configuration, differing
only in RNG state at model construction. That 0.042 spread is larger than the
0.053 gap I was investigating and larger than the preregistered 0.025 delta.

No single-seed comparison at this precision means anything. This is an independent
reason to distrust the withdrawn Track B verdicts, separate from the operator
defects.

**There are still no reportable benchmark numbers.** Everything in
`experiments/results/diagnostics/` is labelled non-publishable in its own JSON.

---

## 3. Code you must update yourself — the repo does not run end to end

This is the important operational section. **Three runners currently fail at
import**, by design (loud failure beats a silent plausible number), but they need
work before anything can be executed.

### 3a. Broken: the experiment runners

```
experiments/run_track_a.py     ImportError: cannot import name 'compute_concordance_td'
experiments/run_track_b.py     ImportError: cannot import name 'compute_concordance_td'
experiments/run_all.py         (imports the two above)
```

They were left untouched deliberately: rewriting them is Stage 7+ work and depends
on decisions not yet made (frozen α, HPO budget). What they need:

- `compute_concordance_td` / `compute_time_dependent_auc` /
  `compute_integrated_brier_score` no longer exist. Use
  `src.evaluation.landmark.evaluate_landmarked(model, train, test, spec, delta_s, device)`,
  which returns one metric dict per `(landmark, horizon)`.
- Loaders now return **6 items** `(train, val, test, input_dim, max_horizon, x_mean)`,
  not 4. Prefer `src.data.cohorts.COHORTS[name].load(seed)` → `CohortData`.
- Delete the `cdf[-1, K//2]` risk extraction at `run_track_a.py:68` and the
  duplicate at `run_track_b.py:53` — that is the last-visit leak.
- Seeds must default to the preregistered `[42, 123, 456, 789, 101112]`, and
  `--dry_run` must write to a separate directory so it can never overwrite a
  reported table.
- `run_track_b.py` should read arm definitions from
  `src.operators.survtd_operator.ARMS` rather than restating them.

### 3b. Stale exports and a live footgun

- **`src/data/__init__.py`** still exports the old `generate_sepsis_icu_cohort` and
  does not export `load_synthetic_icu`, `cohorts`, or `preprocessing`. Two-line fix.
- **`src/data/sepsis_loader.py` still exists with the degenerate generator**
  (`tte = times[-1] + U(0.1,2.0)`, every censored subject at exactly 72.0). It is
  still importable and the two diagnostic scripts still use it on purpose, for
  comparability with the archived numbers. It should become a shim that raises with
  a pointer to `synthetic_icu_loader`, or be deleted once the diagnostics are
  retired. **Until then it is possible to accidentally run an experiment on the
  broken cohort.**

### 3c. Untouched and still carrying known defects

| File | Outstanding |
|---|---|
| `src/training/trainer.py` | Selects "best" weights by **training** loss (`:119-121`); `val_dataset` is accepted and never used despite the docstring promising early stopping. A validation split now exists, so this is wiring. Should select on validation **C-index**, not loss — the TD loss is not comparable across arms. Also imports `collate_patient_batch` and never calls it (D13) |
| `src/training/hpo.py` | Never called from anywhere; `:104` sets `val_loss = float(trial)` for every non-SurvTD model so trial 0 always wins; search space contradicts preregistration §3 |
| `src/evaluation/alarm_fatigue.py` | `break` at `:96` caps the "jitter count" at 1 per subject, so it is a subject *fraction*, not a count; the 0.30-PPV threshold is calibrated on the **test** set (`run_track_a.py:174`); `compute_decision_curve_analysis:118` has the same last-visit leak; the window loop is O(L²) with overlapping windows |
| `experiments/run_lambda_ablation.py` | A **second, divergent implementation** of the core operator: hardcoded macOS paths (`:23`, `:328`), an `os.chdir`, its own recursion multiplying by `S_dt` on both branches with no projection and no mass conservation, plus the same self-distillation defect (`:69`). Cannot execute on this machine. Move to `experiments/legacy/` — two operator implementations in one repo is a reproducibility hazard |
| `experiments/battle_*.py`, `demo_survtd.py`, `visualize_survtd.py` | Unchecked; they call loaders with the old 4-tuple signature. `CohortData.as_legacy_tuple()` exists for them |
| `src/models/backbones.py` | `ContinuousLSTM.forward` accepts a `times` argument no caller supplies — wire it or delete it |
| `README.md` | Links to a `docs/` directory that does not exist; still describes the cohort as MIMIC-IV |
| `research/continuous-survival-td/preregistration.md` | **Not yet amended.** All 15 amendments are drafted in `deviation_log.md` §4 but the preregistration itself is unchanged. It must be amended and committed **before** the 5-seed run, or the amendments stop being pre-registered |

### 3d. Environment

`pip install scikit-survival` **fails on Python 3.14** — its transitive dependency
`ecos` has no cp314 wheel and its source build errors out. `ecos` is only needed by
`FastSurvivalSVM`, which this project does not use:

```bash
pip install --no-deps scikit-survival==0.28.0
```

`lifelines` is deliberately **not** a dependency. `metrics.py` imported it while it
was not installed, which means the shipped pipeline could not be imported in this
environment at all — the published results were produced elsewhere and are not
reproducible here. The Kaplan-Meier censoring estimator now lives in
`src/evaluation/censoring.py`, which controls the `G(t⁻)` left limit and
train-fit/test-apply explicitly. See `requirements.txt`.

---

## 4. What remains, in order

Stages 0–6 of the plan are done. `~/.claude/plans/elegant-booping-castle.md` holds
the full plan with two appendices.

### Blocking, before any number can be reported

1. **Amend `preregistration.md`** and commit it *first*. All 15 entries are drafted
   in `deviation_log.md` §4. The load-bearing ones:
   - **Kill Criterion 5 (new)**: if the `α=1, no TD term` arm matches full SurvTD
     within seed noise, C₀ and C₃ are falsified. This is the most likely way the
     paper dies and must be declared in advance.
   - **C₂'s diffusion bound is false as stated.** Per-step diffusion is exactly
     `f(1−f)δs²`, worst case `0.25δs²` > `δs²/6 ≈ 0.167δs²`. `δs²/6` is
     `E_{f~U(0,1)}[f(1−f)]`, so it holds **in expectation only**. It is also an
     algebraic identity of a two-tap filter, so **EXP-08 cannot fail** — measuring
     it on trained models does not change that. Demote it to a unit test and give
     its verdict role to E8b (target sharpness) and E8c (empirical contraction).
   - HPO budget reduced to 12 trials × 3-fold; §5's fabricated SE withdrawn; the
     adjudicating statistic changed to the subject-level paired bootstrap because
     the minimum attainable exact two-tailed Wilcoxon p at n=5 is **0.0625**, so
     the declared significance requirement is unsatisfiable as written.
2. **Freeze α.** One selection on validation (λ=0.6, seed 42, α ∈ {0, .25, .5, .75, 1}),
   then frozen across every λ, arm, cohort and seed, and applied identically to
   DeepTCSR-Clamped. Note α is a convex weight but **not** a balance parameter: the
   measured anchor:TD scale ratio is 12.2× at initialization.
3. **Wire the trainer** to validation-based selection and early stopping (§3c).
4. **Rewrite the runners** (§3a).

### Then

5. **Vectorize.** A full 5-seed run is ~15 h as-is. Targets: batch the trajectory
   loss (`collate_patient_batch` already exists and is unused), remove the `.item()`
   sync points, cache the projection matrix per distinct `Δt/δs`. Estimated 15–40×.
   Two hazards: the current reduction is `mean over subjects(mean over visits)` — a
   flat mean would silently re-weight long trajectories; and **do not compose shifts
   across steps**, since `Π∘Φ_a∘Φ_b ≠ Φ_{a+b}∘Π` and the per-step reprojection *is*
   the diffusion C₂ studies.
6. **Redesign Track B**: split EXP-05 into three pre-declared tests; add EXP-06's
   missing 100/50/25% sweep (run the training-free analytic horizon check first — it
   could falsify C₂ at zero GPU cost); replace the clipped gain-retention ratio with
   the paired absolute delta; add EXP-04's positive control.
7. **Re-measure EXP-04** — only now is it meaningful, with landmarking *and* the
   repaired generator in place.
8. **5-seed run**, then regenerate Tables 1–3 with mean±SD, bootstrap 95% CI and
   paired Wilcoxon, plus the KM-marginal reference row.

### Expected outcomes — plan for these

- **C₃ (≥0.025 over baselines) is most at risk.** Dynamic-DeepHit was correctly
  supervised all along and Person-Period was broken; the ladder is
  supervision-matched for the first time. A small or zero gain is the honest
  expectation.
- **C₀ is the likeliest failure.** At α=1 the objective *is* Dynamic-DeepHit's L1 in
  Cramér form, so the contribution must be reframed as "a duration-aware TD
  consistency regularizer added to a likelihood", and C₃ as "the TD term adds
  ≥0.025 over the same backbone with the same supervision".
- **EXP-03/NC-A1 may still retain >50% of the gain, and legitimately so.** γ's
  effect size is bounded by the spread of `γ = S(Δt)`. Run the training-free power
  check first: dump the empirical `γ` distribution and report `P(γ < 0.9)`. If under
  ~10% of transitions have `γ < 0.9`, the arm is **underpowered by construction** on
  that cohort and a null result is uninformative rather than falsifying — which must
  be stated, not presented as either outcome.

If the claims do not survive, that is the correct output of this work. A pipeline in
which γ was a λ reparameterization, the target Dirac was misplaced, the durations
were shifted, the preregistered IPCW never ran, and the reference baseline scored
below chance could not have falsified anything — which is why the original
"ALL ADVERSARIAL STRESS TESTS PASSED" verdict carried no information in either
direction.

---

### 9. Follow-up Research Directions (Weight Initialization)
- Detailed proposal documented in [`research/continuous-survival-td/notes/weight_initialization_strategies.md`](research/continuous-survival-td/notes/weight_initialization_strategies.md):
  1. **Marginal Kaplan-Meier Prior Bias Initialization**: Initialize final hazard head bias $b_k = \text{logit}(\hat{h}_{\text{KM}}(k))$ so the network starts exactly at the population survival curve ($C^{td} = 0.500$), eliminating early death explosion and preventing division collapse in DeepTCSR.
  2. **Anchor-Supervised Warm-Start**: Curriculum annealing $\alpha: 1.0 \to 0.0$ over early epochs to prevent bootstrap instability.
  3. **Optimistic Survival Bias**: Constant negative bias $b_k = -3.5$ ensuring high initial survival discount $\gamma_j \in [0.90, 0.99]$.

