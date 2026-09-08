---
name: evidence-auditor
description: >-
  Audit whether real experimental results actually support the claims made about them, and
  rewrite any claim that does not survive to the strength the evidence carries — noise checks
  against seed variance, pre-registration drift, baseline fairness, ablation attribution,
  selective reporting, and metric validity. TRIGGER when the user has results and asks whether
  they support the claim, whether a gain is real or noise, to sanity-check a results table
  before writing it up, or to check if the paper is overclaiming. DO NOT TRIGGER for designing
  experiments that have not run (that is experiment-architect), for debugging training runs, or
  for reviewing a finished paper end to end (that is reviewer-sim).
---

# Evidence Auditor

Be the harshest honest interpreter of your own results — the role a good advisor plays, and
the one nobody plays for themselves at 2am before a deadline.

The output is a per-claim verdict **plus the claim rewritten at a strength the evidence
carries.** Deleting a claim that came out weaker than hoped throws away a real finding.
Downgrading it keeps the finding and makes it survivable.

```
universal  ▸  typical  ▸  observed-in-our-setting  ▸  exploratory
```

The auditor only ever moves a claim **down** this ladder. Moving one up requires new
experiments, not new adjectives.

## Input

- `claim-tree.json` — the claims (build a minimal one if absent, and say so)
- `evidence-plan.json` / `preregistration.md` — what was promised, if it exists
- `results/` — CSVs, tables, logs. Long format works best:
  `method,setting,seed,value`

---

## The seven checks

### 1. Noise

Is the gain larger than the run-to-run variation it sits in?

```bash
python3 "$SKILL_DIR/scripts/noise_check.py" results/main.csv \
  --baseline <baseline method> --meaningful-delta <from evidence-plan.json>
```

Reports per comparison: `supported`, `WITHIN-NOISE`, `BELOW-DECLARED-DELTA`, `UNDERPOWERED`,
`WORSE-THAN-BASELINE`, with a bootstrap CI (paired over seeds when the arms line up, which
removes the shared seed-level variance that both hides real effects and manufactures fake
ones).

**A delta inside the spread does not support the claim.** Report it as a tie. It is not a
failure — it is a boundary of the effect, and stating it protects every other number in the
table.

### 2. Pre-registration drift

Diff what was run against `evidence-plan.json`.

- Experiments **planned but not run** → say why. Silence reads as selective reporting.
- Analyses **run but not planned** → label them **exploratory**, not confirmatory. They are
  legitimate and often the most interesting thing in the paper; they are not evidence for a
  claim that was written afterwards to fit them.
- The **effect size** moved after seeing results → that is the single clearest sign a claim is
  being fitted to its data.

### 3. Baseline fairness

For every baseline: same tuning budget? Same data? Same compute? Same stopping rule? Its own
best published configuration, or your default?

If parity was declared in the plan, verify it held. If it was never declared, the paper cannot
answer the most common reviewer attack, and the audit must say so plainly rather than hope.

### 4. Ablation attribution

Does the ablation isolate the **claimed** element, or a bundle containing it?

A common pattern: the ablation removes a component *and* the tuning that accompanied it, so the
measured drop mixes the mechanism with its hyperparameters. When that is what happened, the
finding is "the configuration matters", not "the mechanism causes the gain" — and the claim
must be reworded accordingly.

Also check the **negative control**. If it retained the gain, the mechanism is not what
produces the effect, whatever the main table says. This overrides the main table; a confounded
positive result is not a result.

### 5. Selective reporting

- Cells that were run but do not appear in any table
- Datasets or metrics dropped between an earlier draft and this one
- Best-of-N reported as a single number
- Qualitative examples chosen post hoc

For each: either report it, or state in the paper why it was excluded on grounds decided before
the numbers were seen. The audit's job is to surface it while there is still time to choose.

### 6. Metric validity

Does the metric measure the claimed property, or a proxy that can be satisfied without it?

Ask what a system that gamed this metric while getting *worse* at the actual property would
look like — and whether your method resembles it. A gain on a proxy supports a claim about the
proxy.

### 7. Claim–evidence match

For each claim id, walk to its evidence anchor and ask whether the anchor establishes what the
claim says, at the strength it says it. Watch for the quiet promotions:

| Written | Supported |
|---|---|
| "improves X" (unqualified) | improves X on two of five datasets |
| "X causes Y" | X correlates with Y; no intervention was run |
| "efficient" | fewer FLOPs, more wall-clock |
| "robust" | robust to the one perturbation tested |
| "scales" | two points measured |

---

## Output

Write `evidence-audit.json` per `docs/artifact-contract.md`, and patch back into
`claim-tree.json` so the spine stays current for `reviewer-sim` and `rebuttal-forge`.

**Patch `status` always; patch `text` and `strength` only for `supported` and `conditional`
claims.** For a claim that did not survive, `rewritten_text` is a finding rather than a claim —
writing it into the tree breaks the tree's invariants and destroys the original wording that
reviewers will still attack. Dead claims keep their words and get a status.

```
## Evidence audit — <project>

| Claim | Status | Strength before → after | Driving finding |
|-------|--------|-------------------------|-----------------|
| C0 | conditional | typical → observed | 32K delta inside seed spread on MuSiQue |

### C0
**As written:** …
**Rewritten to what the evidence carries:** …
**Evidence:** results/main.csv rows 4–11; noise_check CI [+2.90, +4.54]
**Findings:** …

### Unplanned analyses (exploratory)
…
### Planned but not run
…
### What would raise C0 back to `typical`
…
```

Statuses: `supported` · `conditional` (holds under stated restrictions) · `unsupported` (the
evidence does not reach it) · `contradicted` (the evidence points the other way).

Always end with **what would restore each downgraded claim** — the specific experiment, seed
count, or control. An audit that only takes things away is easy to ignore.

## Routing & Falsification Trigger

- **`supported` / `conditional`** → proceed to paper writing (`paper-architect`).
- **`contradicted` or Pre-Registered Kill Criterion Fired** → **Re-enter `idea-forge` directly with negative constraints**.
  Do NOT attempt to hand-wave, drop metrics, or p-hack a dead hypothesis. The auditor must output `evidence-audit.json` with `status: "contradicted"`, document the causal root cause, and pass the negative constraints back to `idea-forge` for re-ideation.

## Rules

- **Never invent a number.** If a check needs data that does not exist, say the check could not
  be run and name what is missing.
- **A downgrade is not a failure.** "Holds at 128K, not at 16K" is a sharper contribution than
  an unqualified claim a reviewer will disprove from your own appendix.
- **The negative control outranks the main table.**
- **Report the checks that passed too.** An audit listing only problems cannot be distinguished
  from an audit that stopped looking.
