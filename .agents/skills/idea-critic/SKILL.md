---
name: idea-critic
description: >-
  Judge how strong a research idea is at the idea stage, before any experiments exist — score
  it on problem position, method quality, problem-fit, and falsifiability with quoted evidence
  for every score, or rank several ideas blind. TRIGGER when the user asks how good an idea is,
  to rate or grade a research idea or proposal, whether a contribution is strong enough to
  pursue, which of several ideas is strongest, or hands over an idea and wants a judgment. Runs its gauntlet as a panel of five
  independent critics that each file their own report.
  DO NOT TRIGGER for prior-art or "has someone done this" checks (that is scoop-radar), for
  reviewing a finished paper that already has results (that is reviewer-sim), or for generating
  an idea.
---

# Idea Critic

Judge whether an idea is **good** — before experiments exist, and deliberately **without
consulting the literature**. Whether a near-duplicate exists is a different question with a
different skill (`scoop-radar`). Merging them produces the review everyone has received and
nobody has used: *"seems reasonable, I couldn't find anything identical."*

A strong score does not predict acceptance, which also turns on execution this skill cannot
see.

## Scope

- **No experiments exist.** Every axis is a reasoning-level judgment — what a reviewer forms
  from an abstract. **Never invent results to score with.** If a claim's truth needs an
  experiment, judge the plausibility of its *argument*.
- **Not a novelty check.** Judge the contribution's intrinsic ambition, not its distance from a
  literature search.

## Input

An idea as `candidate.json`, an idea card, or prose with a title, a motivation, and a method.
If a part is missing or thin, infer the most reasonable reading and note the assumption. Do not
stall asking for clarification.

---

## The four axes

Score each **1–5**, integer. There are deliberately **no fixed level descriptors** — instead
**every score must quote the specific phrase from the idea that justifies it.** A score with no
quoted evidence is not allowed. That requirement, not a rubric table, is what keeps scores
honest and the comparison auditable.

Judge **substance, not length or fluency.** A short crisp idea can score 5; a long polished one
can score 2. B-soundness is where fluent prose most easily hides an unsound step.

### A — Problem position

Is the gap the motivation identifies real, important, non-obvious, and genuinely open?

- **Strong** — a gap that matters and the field has not closed; naming it is itself insightful.
- **Weak** — obvious, already handled, or a soft target chosen so it can be "solved" cheaply.
- *Why it exists:* when authors pick their own bottleneck, the cheapest route to looking
  successful is to pick an easy one. This axis is what stops a soft bottleneck plus a trivial
  fix from scoring well.

### B — Method quality

Is the method good in itself? One number, but the reason **must** decompose into three named
sub-judgments, because a single number hides which one is weak:

- **depth** — a genuinely new mechanism, construction, or reframing, versus an incremental
  tweak (one more loss term, a new schedule, a hyperparameter).
- **soundness** — does the "why it should work" argument hold internally? Are the assumptions
  justified, or is there hand-waving or an unstated condition? *Judge the logic on its own
  terms, whatever problem it aims at — fit is axis C.*
- **feasibility** — buildable in principle, or does it need oracles, data, or resources that do
  not plausibly exist? This means **buildable**, not "will get good numbers." It is the softest
  sub-judgment: let depth dominate B and treat feasibility as a tie-breaker.

Michael Black's warning applies directly here: reviewers routinely mistake complexity,
difficulty, and technicality for novelty. A method that is elaborate is not thereby deep. Score
depth on what is *new*, not on what is *hard to implement*.

### C — Problem-fit

Does the method target and plausibly resolve **the specific gap in the motivation**, rather
than an adjacent, easier, or different problem?

- **Strong** — the mechanism clearly bears on the identified gap; if it works, that gap closes.
- **Weak** — aimed at a different problem than the one claimed, or the connection is asserted
  rather than shown.
- *Boundary vs B-soundness:* B asks "is the logic coherent?", C asks "does that logic connect to
  THIS problem?" A method can be internally sound and solve the wrong problem, or be on target
  and hand-wavy. Keep them separate.

### D — Falsifiability and claim integrity

Could this idea be shown to be wrong, and is it stated at a strength its argument can carry?

- Is there a named **load-bearing variable** — the quantity whose behavior carries the claim?
- Is the **negative control non-tautological**? "Intervene on X so X becomes zero" tests a
  definition. A real control lands on a *downstream* outcome.
- Does every numeric bar carry provenance (`derived:` / `measured in <paper>`), or is it
  invented to sound precise?
- Is the claim's wording at the strength the argument supports, or is it a guarantee-shaped
  sentence resting on an empirical hunch?

*Why it exists:* an idea too vague to be wrong is also too vague to collide with prior work,
which is how a bare model outscores a grounded one on novelty while producing nothing usable.
D is the axis that catches **novel-but-empty**.

---

## Two structural checks

**Naive-baseline audit.** Independently construct the naive version of this idea — do not use
the author's stated one, which is usually a strawman. Then classify:

1. *The naive version relies on a false premise* → confronting that premise **is** the
   contribution. If the confrontation is a textbook tool from another field, name the
   domain-specific structure that makes this instance unsolved, or call the contribution
   application-grade honestly.
2. *The naive version suffices* → the idea is incremental. Say so plainly.
3. *The naive version works but the field disbelieves it* → minimalism, and legitimate, if the
   evidence for the disbelief is named.

**Novel-but-empty detector.** If you cannot state what the idea would predict, what it would
change, or what would count against it — the idea is not novel, it is unfinished. Report that
rather than a high score.

---

## Modes

### `absolute` (always run)

```
overall = round(100 * (A + B + C - 3) / 12)      # D reported separately, see the gate
```
Bands: `strong` ≥ 67 · `borderline` 34–66 · `weak` < 34.

**The A/C/D gate overrides the band.**

- **A ≤ 2 or C ≤ 2 → cannot be `strong`** (cap at borderline). A great method on a trivial
  problem, or a great problem with a method that does not address it, is not a strong idea, and
  an equal-weight mean would say otherwise.
- **D ≤ 2 → cannot be `strong`.** An unfalsifiable idea has no upper bound on how good it looks
  and no lower bound on how wrong it is.
- **A low B alone does not cap.** A simple method that genuinely addresses an important problem
  is counter-intuitive minimalism, not weakness.

State the cap and why, in one line.

### `pairwise` / `listwise` (when comparing)

Per axis, say which idea is stronger and why in one line; for B, name which arm wins on depth,
soundness, and feasibility so the diagnostic survives. Then pick an overall winner
**holistically** — but a decisive **C** loss usually decides it, even for an idea that wins A
and B, because a method that does not fit its problem is not the better idea.

**Judge blind to source.** Which model, person, or system produced an idea is not evidence
about its quality. Treat relative judgments as the trustworthy signal: absolute scores bunch
around 3/5, relative rankings do not.

### `gauntlet` (called by idea-forge Phase 3)

Run `absolute`, then attack. Read `references/attack-catalog.md` and work the catalog against
this idea. For each landed attack record: the attack, its **severity**, whether it is
**answerable now / answerable with evidence / unanswerable**, and the claim id it hits if a
`claim-tree.json` exists.

Then a **two-layer verdict**:

- **Hard floor — judgment cannot override these.** `abandon` when: a prior work matches on all
  four scoop axes (from the `scoop-radar` report, if supplied); the naive-baseline audit returns
  *naive suffices*; an anti-pattern composition's required mitigation cannot be inserted; or D
  collapses (no falsification is constructible for this idea at all).
- **Soft judgment above the floor** — `advance` for trivial borderlines only, otherwise
  `revise` with concrete `revision_targets[]`.

**The gauntlet judges; it never edits the candidate.** Splitting attack-finding from
attack-fixing is what keeps a critique from quietly becoming a rewrite that agrees with itself.

**The gauntlet is run by a panel, not by one reviewer** — see below. A single-critic gauntlet is
a fallback for a cheap run, and must be labelled `panel: solo` in its output so a later reader
knows the verdict rests on one lens.

---

## Panel mode — five independent critics

One critic produces one lens, and a lens has blind spots that correlate with its own strengths:
the theorist waves through an unmeasurable claim, the empiricist waves through an unsound proof.
Running five critics **in parallel and in mutual isolation** turns those blind spots into
observable disagreement, and the disagreement is the most useful output the gauntlet produces.

### The five seats

Each seat runs the *same* rubric — the four axes, the quoted-evidence requirement, the two
structural checks, the two-layer verdict. What differs is the lens each one is told to lead
with and the section of `references/attack-catalog.md` it works first. Every seat still works
the whole catalog; the assignment sets priority, not scope.

| Seat | `role` id | Lens | Leads the catalog at |
|---|---|---|---|
| Area chair | `chair` | Is this a paper? Ambition, venue fit, whether the contribution survives being stated in one sentence | §6 ambition, §1 position |
| Theoretician | `theorist` | Is the "why it should work" argument *formally* sound — assumptions, operators, estimators, unstated preconditions | §2.3, §2.4, §2.5 |
| Empirical auditor | `empiricist` | Could this be measured, and would the measurement be believed — instruments, baselines, tuning parity, control quality | §4 evidence |
| Domain practitioner | `domain` | Does the mechanism match how the data and the deployed system actually behave, and does anyone pay the cost today | §1.2, §3.3 |
| Subfield insider | `insider` | Knows the lineage cold — is the delta structural or lexical, and is the naive version already this | §5 positioning, §2.1 |

`domain` and `insider` are **instantiated per run**: name the actual field and the actual
nearest subfield in the seat's brief (e.g. "ICU clinician / clinical-AI reviewer",
"deep-survival-analysis method author"). A generic "domain expert" produces a generic report.

### Dispatch rules

- **One agent per seat, all five launched together**, each in its own context.
- **Identical input packet** to all five: `candidate.json`, the `scoop-radar` report if one
  exists, and `claim-tree.json` if one exists. Nothing else.
- **No critic sees another critic's report**, no critic sees the context that generated the
  candidate, and no critic is told what any other seat scored. Correlated reports are worth
  less than one honest report.
- **No prior-round reports** are supplied on a re-run. A critic shown last round's verdict
  anchors to it; that is how a panel launders a score upward across attempts.
- Each seat writes `<RUN>/phase3/critic_report_<role>.md` in the standard output format below,
  and returns its four scores and its verdict.

### Synthesis

The dispatching context — never a sixth critic, and never one of the five — writes
`<RUN>/phase3/panel_verdict.md`. Synthesis is arithmetic plus recording; it is not a place to
re-litigate a seat's score.

1. **Scorecard.** One row per seat: A, B, C, D, overall, verdict.
2. **Panel score per axis = the median of the five**, not the mean. One outlier seat must not
   move the panel, and a mean lets it.
3. **Dispersion is a finding.** For any axis whose range across the five is ≥ 2, mark it
   `contested` and state, in one line, what the high seat and the low seat each saw. A contested
   axis goes into `revision_targets[]` regardless of the median: a panel that cannot agree on
   whether the method is sound has located a real ambiguity in the write-up.
4. **Verdict rule.** A hard floor fired by *any single seat* is absolute — verify the trigger
   against the floor list, and if it holds, the panel verdict is `abandon`. Otherwise the panel
   verdict is **the most severe verdict held by at least two seats** (`abandon` > `revise` >
   `advance`). With five seats and three possible verdicts this rule always resolves.
5. **`revision_targets[]`** is the union across seats, deduplicated and ordered by how many
   seats raised each. A target raised by ≥ 3 seats is `mandatory`; the rest are `recommended`.
6. **Preserve the dissent.** A lone `abandon`, or any `fatal` attack no other seat addressed, is
   quoted verbatim in the synthesis even when the panel advances. Averaging a dissent away is
   the failure this whole mode exists to prevent.

Cost note: five seats is the default for a candidate heading into an experiment budget. For a
quick sanity read, run `solo` and say so.

## Output

```
## Idea review — <title>

**Decomposition**
- Problem / gap: …
- Method (the move): …
- Why it should work: …
- Assumptions inferred: …

| Axis | 1–5 | Quoted evidence | Reason |
|------|-----|-----------------|--------|
| A — Problem position | | "…" | |
| B — Method quality   | | "…" | depth: … · soundness: … · feasibility: … |
| C — Problem-fit      | | "…" | |
| D — Falsifiability   | | "…" | load-bearing variable: … · control: … |

**Overall: NN/100 · Verdict: strong | borderline | weak**
<the gate line, if it fired>

**Naive-baseline audit:** branch (1|2|3) — …
**Strongest point:** <one sentence>
**Most fixable weakness:** <one sentence, phrased as what would raise the score>
```

For `gauntlet`, append the attack table and the two-layer verdict with `revision_targets[]`.

For **panel mode**, each seat emits the block above (headed with its `role`), and the
dispatching context emits the synthesis:

```
## Panel verdict — <title>

| Seat | A | B | C | D | Overall | Verdict |
|------|---|---|---|---|---------|---------|
| chair | | | | | | |
| theorist | | | | | | |
| empiricist | | | | | | |
| domain | | | | | | |
| insider | | | | | | |
| **median** | | | | | **—** | |

**Contested axes:** <axis — high seat saw … · low seat saw …>, or "none (all ranges ≤ 1)"
**Panel verdict: advance | revise | abandon** — <the rule that produced it, one line>
**Preserved dissent:** <verbatim, or "none">

**revision_targets[]**
1. [mandatory · raised by N/5] …
2. [recommended · raised by N/5] …
```

## Anti-bias rules

- **Quote evidence for every score.** A bare number is a vibe, and vibes do not separate strong
  ideas from weak ones.
- **Stay at the idea stage.** Do not imagine results. Score the argument's plausibility.
- **Judge substance, not presentation.** A confident write-up dresses up a thin idea.
- **In comparisons, judge blind to source.**
- **Do not reward complexity.** Ask what is new, not what is hard.
