---
name: falsification-pivot
description: >-
  Structurally enforce the recovery and re-ideation loop when experimental evidence contradicts
  a hypothesis or a pre-registered kill criterion fires — perform a causal autopsy, archive the
  failed run to attempt_N/, extract hard negative constraints, and seed idea-forge Phase 2
  directly without losing literature context. TRIGGER when evidence-auditor reports a claim as
  contradicted, when a negative control fails, or when a pre-registered kill criterion fires.
  DO NOT TRIGGER for minor effect-size downgrades (that is evidence-auditor scope refinement).
---

# Falsification Pivot

When an experiment kills a claim, do not paper over it, do not p-hack, and do not start from
blank-slate scratch. The failure is the most expensive and valuable negative knowledge the project
owns.

**The artifact is `attempt_N/negative_constraints.json` plus the seeded workspace for `idea-forge Phase 2`.**

## Why this exists

A common pathology in ML research is the *sunk-cost shuffle*: an experiment proves the core
mechanism is not what produces the gain, and the authors spend six weeks massaging hyperparameters,
changing metrics, or rewriting the story to pretend they predicted the null result all along.

`falsification-pivot` enforces the honest alternative:
1. **Face the autopsy**: The kill criterion fired; the hypothesis is dead.
2. **Preserve the evidence**: Archive the entire failed run to `$RUN/attempt_N/` with all logs and code intact.
3. **Extract the negative constraint**: State the exact causal mechanism that broke, and forbid the next iteration from using it.
4. **Fast-path to re-ideation**: Skip Phase 0 (Problem discovery) and Phase 1 (Literature lineage) — the problem is still real. Re-enter `idea-forge` directly at **Phase 2 (Candidate Generation)** under the hard negative constraint.

---

## Input

- `evidence-audit.json` / `falsification_verdict.md` from `evidence-auditor`
- `claim-tree.json` and `evidence-plan.json`
- `results/` containing the raw experimental run outputs

---

## 1. Causal Autopsy

Read the driving finding from `evidence-auditor`. Formulate the autopsy along three axes:

| Axis | Question | Example Finding |
|---|---|---|
| **What failed** | Which exact experiment, metric, or control broke? | NC-B within-patient permutation retained 68% of the concordance gain. |
| **Why it broke** | What unstated assumption or confound caused the failure? | The gain stemmed from sequence visit count regularisation, not continuous temporal alignment. |
| **What is forbidden** | What structural property must the next candidate avoid? | No operator that aggregates visits without strict duration-scaled decay. |

---

## 2. Automated Archival & Workspace Setup

Execute the deterministic pivot script:

```bash
python3 "$SKILL_DIR/scripts/pivot_workspace.py" \
  --audit "$RUN/evidence-audit.json" \
  --workspace "$RUN"
```

The script performs:
1. Scans for existing `attempt_*` directories and selects the next sequential folder `$RUN/attempt_{N+1}/`.
2. Moves active tracking artifacts (`claim-tree.json`, `evidence-plan.json`, `preregistration.md`, `results/`, `phase2/`, `phase3/`) into `$RUN/attempt_{N+1}/`.
3. Copies forward `$RUN/phase0/` and `$RUN/phase1/` so the background and literature context are preserved.
4. Writes `$RUN/attempt_{N+1}/negative_constraints.json`.

---

## 3. The Negative Constraint Contract

`$RUN/attempt_{N+1}/negative_constraints.json` must declare:

```json
{
  "failed_attempt": "attempt_2",
  "falsified_claim": "C0",
  "fatal_evidence": "NC-B within-patient permutation retained 68% of gain",
  "root_cause_diagnosis": "Model learned passive visit count regularisation rather than continuous duration alignment",
  "hard_negative_constraints": [
    "The next candidate must not rely on unweighted visit count pooling",
    "The transition operator must vanish under uniform permutation"
  ]
}
```

---

## 4. Re-Entry to `idea-forge Phase 2`

Invoke `idea-forge` targeting `Phase 2b (Empirical Pivot)`:
- Phase 0 and Phase 1 are skipped.
- The generator receives `negative_constraints.json`.
- The new candidate (`candidate.json`) must explicitly declare how it bypasses the forbidden patterns before proceeding to Phase 3.

---

## Rules

- **Never delete a failed attempt.** `attempt_1`, `attempt_2`, etc. form the honest scientific lab notebook.
- **A negative constraint is permanent.** A subsequent attempt cannot quietly reintroduce a mechanism that was empirically falsified in an earlier attempt.
- **Three candidate cycles spent without survival $\to$ write `gauntlet_failed.md` and stop.**
