---
name: experiment-architect
description: >-
  Design the experiment set that could prove a research claim wrong, and lock it before any
  runs happen — claim-to-experiment binding, a compute-matched baseline ladder with declared
  tuning parity, the ablation that isolates the claimed mechanism, negative controls, a
  statistics plan, and pre-registered kill criteria. TRIGGER when the user asks what
  experiments to run, how to design an evaluation or ablation study, which baselines are
  needed, how many seeds, or wants an experiment plan before starting runs. DO NOT TRIGGER for
  interpreting results that already exist (that is evidence-auditor), for debugging or
  launching training jobs, or for writing an experiments section.
---

# Experiment Architect

Write the plan that could kill the claim, before the first run. The output is
`evidence-plan.json` plus `preregistration.md`, and both are meant to be uncomfortable: a plan
that cannot lose is not a test.

**Why this exists.** In an analysis of 19k ICLR submissions and 74k reviews, the dominant
weaknesses in the experiments section were *insufficient or weak baselines, too few datasets,
missing ablations, and reproducibility*. None of those are discovered late — they are decided
here, before anything runs, and become unanswerable afterwards. "You undertuned the baseline"
cannot be rebutted once the runs are done.

## Input

`claim-tree.json` if it exists; otherwise an idea card, a candidate, or a described claim —
build a minimal claim tree first (`C0` plus any sub-claims) and say that you did.

---

## 1. Bind claims to experiments, both directions

Every claim needs at least one experiment that could falsify it. Every experiment needs at
least one claim that depends on it.

- **A claim with no experiment is an overclaim** — it just has not been noticed yet.
- **An experiment with no claim is page count.** It will still cost compute, still take space,
  and a reviewer will ask what it is for.

Write the bipartite map explicitly before anything else. This is the check that makes an
overclaim mechanical rather than a matter of taste, and it is the reason the claim-id
namespace exists.

## 2. The baseline ladder

Four rungs. Missing rungs are where papers lose.

| Rung | What | Why a reviewer asks |
|---|---|---|
| **Naive** | The dumbest thing that could work — random, majority class, nearest neighbour, no-op | If it matches you, the problem was not what you said it was |
| **Strongest published, compute-matched** | The current best method, given *your* compute budget, not its paper's | An uncontrolled compute comparison is not a comparison |
| **Yours minus the novelty** | Your system with exactly the claimed mechanism removed | This is what attributes the gain to the mechanism rather than the engineering around it |
| **Yours** | The full method | |

**Declare tuning parity in writing, now.** State the search budget, the range searched, the
data, and the wall-clock each baseline receives — and make it the same as yours. This single
sentence is what stands between the paper and its most common fatal review comment, and it can
only be written honestly *before* results exist.

If a baseline cannot be tuned equally (no public code, prohibitive cost), say so in the plan
and in the paper. A stated limitation survives review; an unstated one does not.

## 3. The mechanism-isolating ablation

Not "ablate each component." The one ablation that removes **exactly the causal element the
core claim names**, and nothing else.

Write it as a prediction: *"Removing X should cost Y, because the claim is that X produces Y."*

If removing the claimed mechanism predicts no specific loss, the claim is not mechanistic —
it is a claim that the system as a whole is better, and it should be reworded to say that.
Discovering this now costs a sentence; discovering it in review costs the paper.

## 4. Negative controls

The intervention that **should not work**. Permute the labels the method depends on, randomize
the structure it exploits, hold the load-bearing quantity numerically fixed while destroying
its meaning.

If the negative control *also* produces the gain, the result is confounded — the gain comes
from something other than the stated mechanism (extra compute, variance reduction, a longer
schedule). This is the most informative experiment in most plans and the most frequently
omitted.

**A control that intervenes on X and measures X is not a control**; it verifies a definition.
Land it on a downstream outcome.

## 5. The statistics plan — declared before results

| Field | Rule |
|---|---|
| **seeds** | ≥ 3 to speak about spread at all; **5 is the defensible default** |
| **variance_reported** | Say what the bars are: std over seeds, 95% CI, IQR. A table of bare means cannot support a comparative claim |
| **meaningful_delta** | The effect size that counts as a real difference. **Decide it now.** Deciding after seeing results is how noise becomes a contribution |
| **declared_before_results** | `true`. `evidence-auditor` treats anything added later as exploratory, not confirmatory |

## 6. Kill criteria

For the core claim, and ideally each sub-claim: **the result that makes you abandon it.**

Write it as a concrete outcome, not a mood. "If the negative control retains more than half the
gain, the mechanism is not what produces the effect, and the claim is withdrawn."

This is the field authors skip, and skipping it is what converts a disappointing result into
six weeks of reframing.

## 7. Cost model and cut order

Per experiment: GPU-hours (or API dollars), and a **`cut_order`** — the drop sequence if the
budget halves. Decide the order now, while it is still an honest judgment about evidential
value rather than a rationalization of what happened to finish.

The rung you must never cut is the mechanism-isolating ablation. Without it there is no paper,
only a system report.

---

## Output

The output is structured across two distinct human-readable tracks and one unified machine contract:

1. **Track A — Paper Experiments (`paper_experiments.md`)**:
   - Focus: The constructive narrative for §4 of the manuscript.
   - Content: Multi-cohort benchmark ladder, compute-matched baselines, primary clinical/empirical metrics.
2. **Track B — Adversarial Stress Tests (`adversarial_stress_tests.md`)**:
   - Focus: The hypothesis-destroying Red-Team track for §5 and Appendix/Rebuttal armor.
   - Content: Lethal negative controls, extreme boundary collapse stress tests, and confound-hunting ablations designed specifically to break the proposal.
3. **Unified Machine Contract (`evidence-plan.json`)**:
   - Merges all experiments from both tracks with explicit `track: "paper"` or `track: "adversarial"` tags under the pre-declared budget.
   - Validate with:
     ```bash
     python3 "$SKILL_DIR/scripts/validate_plan.py" evidence-plan.json --tree claim-tree.json
     ```
4. **Pre-registration (`preregistration.md`)**:
   - The human-readable version, dated, stating in prose: claims, experiments, tuning parity, statistics plan, and kill criteria. **Commit it before the first run.**

## Scope

This designs the evidence. It does not run anything, tune anything, or debug anything. When
results arrive, hand them to `evidence-auditor` together with this plan — the comparison
between what you promised to check and what you actually checked is the audit's most useful
input.
