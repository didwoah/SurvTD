---
name: idea-forge
description: >-
  Turn an under-specified research direction into ONE literature-grounded, reviewer-defensible
  research idea with a concrete mechanism and a falsification plan. TRIGGER when the user asks
  for a research idea, asks what the bottleneck in an area is, wants a vague direction sharpened
  into a paper-shaped proposal, or wants an idea they already have grounded against recent
  literature. DO NOT TRIGGER for literature reviews with no idea to produce, for summarizing a
  single paper, for benchmark or dataset construction, for engineering tasks, or for
  free-association brainstorming with no research context.
---

# Idea Forge

Convert an under-specified direction into one research idea that can survive review. The
skill's stance is that **a publishable idea is constructed, not invented**: you read a
structural gap out of recent literature, choose the kind of move that closes that shape of
gap, then instantiate the move on the specific gap. The flash of insight is real, but it is
the last step of that chain, and the earlier steps can be done deliberately.

Three outcomes per run, and all three are legitimate:

1. `idea-card.md` + `candidate.json` — the idea, with its kill-switch attached
2. `do_not_generate.md` — the direction cannot be grounded, with what to change
3. `gauntlet_failed.md` — every candidate died in audit, with each verdict

## Two entry points

**Cold start.** The user has a direction, no idea. Run all five phases.

**Idea injection.** The user already has an idea — it came out of a conversation, a hunch, a
talk. This is the more common case and it is served here explicitly. Run **Phase 0 and
Phase 1 only**, to answer: is the bottleneck this idea attacks actually open in the
literature, or only open in the user's memory? Then write the idea into `candidate.json`
format, attach a kill-switch, and send it to `scoop-radar` and `idea-critic`. Skip Phase 2.
Say which entry point you took.

## Setup

No dependencies, no API keys. `S2_API_KEY` is used if exported, never required.

```bash
SKILL_DIR=<this folder>
RUN="research/<topic-slug>"      # one run per directory; never reuse one that has phase0/
```

---

## Phase 0 — Literature grounding

**Write four queries before searching.** Read `references/query-design.md` and apply both of
its tests to every query. One of the four must be the **escape-mechanism query**, phrased in
solution vocabulary rather than problem vocabulary — a paper that already fixed this
bottleneck titles itself by its fix, so problem-keyed queries miss exactly the papers that
matter most.

Two windows, because no single source is good at both ends:

```bash
# recent: preprints and in-review work, where the live gaps are
python3 "$SKILL_DIR/scripts/lit_search.py" \
  --query "q1" --query "q2" --query "q3" --query "q4" \
  --sources arxiv,s2,openreview --from <today-6mo> \
  --limit 25 --out "$RUN/phase0/recent.json" --markdown "$RUN/phase0/recent.md"

# established: published work, where the lineage is
python3 "$SKILL_DIR/scripts/lit_search.py" \
  --query "q1" --query "q2" --query "q3" --query "q4" \
  --sources openalex,s2,crossref --from <today-24mo> --to <today-6mo> \
  --published-only --limit 25 --out "$RUN/phase0/published.json"
```

The script drops records that share less than a third of a query's content terms and lists
them under `dropped_off_topic` — check that list once. If something important was dropped,
your query was too generic, not the gate too strict.

**Partition by relevance.** Over every record's title and abstract, label
`core | adjacent | off_topic`. Be conservative: when torn between core and adjacent, pick
core. A wrong `off_topic` is an unrecoverable recall loss; a wrong `core` costs one row.
Only `core` papers earn a full-text read.

**Read the method sections.** For the top 10–20 `core` papers, fetch the full text and read
the method and limitations sections. **This is a hard gate: do not proceed to Phase 1 from
abstracts.** Abstracts state what a paper achieved; they omit the assumptions and scope
conditions that determine whether a gap is actually open. A bottleneck written from
abstracts is a bottleneck written from marketing copy.

Write `$RUN/phase0/lit_table.md` with one row per paper: `paper_id | date | venue | title |
bottleneck this paper targets | unresolved residue | relevance`.

## Phase 1 — Bottleneck identification

One statement, and it must be a **structural gap, not a topic**:

> ✗ topic — "improving retrieval accuracy for RAG systems"
> ✓ bottleneck — "GRPO broadcasts one group-normalized score uniformly over a 128K rollout, so
> the useful signal per decisive token shrinks like 1/length — the method collapses exactly in
> the long-context regime it was built for" *(cites arxiv:2510.xxxxx, arxiv:2511.xxxxx)*

Cite at least two `paper_id`s inline. A bottleneck that cites nothing is model memory.

**Build the method-lineage tree.** Arrange the retrieved methods as a tree of what refined or
replaced what. Roots are foundational, leaves are current. This exposes three gap types a flat
list hides:

| Gap type | Where it sits | Why it matters |
|---|---|---|
| **additive** | at a leaf | an unmet need no current method achieves |
| **subtractive** | at a shared ancestor | a load-bearing assumption every descendant inherits — strong papers often *remove* rather than add |
| **regression** | already in an ancestor | the "fix" is superseded, not novel. The tree blocks it before you spend months on it |

Ancestors you know but did not retrieve may be added as **awareness-only** nodes, marked
non-citable. They make no novelty claim; they exist to catch regressions.

**Routing.** `proceed`, or `do_not_generate` when the direction is too broad, has no
literature anchor, retrieved fewer than five genuinely relevant papers, or is a
benchmark/system-construction task this skill does not serve. Write `do_not_generate.md` with
concrete remedial steps. **Never ask the user mid-flow** — infer missing intake and record
what you inferred.

## Phase 2 — Pattern-guided generation

Separate *which move* from *what the move becomes*.

**2.1 Select.** Pick the anchor gap and at most two siblings that could plausibly form one
paper. Read `references/ideation-patterns.md` and choose the pattern whose operational
signature structurally closes each gap. **Prefer two patterns** — composition is the empirical
norm — but a single pattern is legitimate if you write a `composition_note` defending why one
move suffices. Never choose a pattern because it is common; frequency is audit context, not a
generation prior.

**2.2 Generate.** Write `$RUN/phase2/candidate.json` (schema: `docs/artifact-contract.md`):

```json
{
  "title": "...", "core_mechanism": "...", "core_mechanism_steps": ["..."],
  "gap_closure": [{"gap": "...", "pattern": "<one of the 15 ids>", "how_closed": "..."}],
  "falsification_prediction": "...", "load_bearing_variable": "...",
  "negative_control": "...", "compute_budget": "...",
  "differentiation_from_lit": [{"paper_id": "...", "delta": "..."}],
  "signature_terms": ["..."], "alias_terms": ["..."]
}
```

Four rules bind the writing:

- **Every artifact must already exist.** Datasets, model access levels, annotations, tools —
  name a real one, or count the cost of building it in `compute_budget`.
- **Every number needs provenance.** Method parameters are named symbols with a default and a
  selection rule; a bare `top 5` or `≥ 95%` cannot be swept or graded. Outcome bars in the
  falsification carry `derived:` or `measured in <paper_id>` — an invented bar is fabrication.
- **The kill-switch is a mechanism test, not a definition test.** The negative control must
  intervene on the load-bearing variable and predict a *downstream* metric returning to
  baseline. "Set X to zero, therefore X is zero" tests a definition and is the most common
  self-confirming failure.
- **Two collision channels.** `signature_terms` is your own vocabulary; `alias_terms` is what
  *other communities* call this same mechanism. The renamed-ancestor blind spot is lexical,
  not temporal — no date window catches a mechanism under a different name.

Run the gate before spending any audit budget:

```bash
python3 "$SKILL_DIR/scripts/validate_idea.py" "$RUN/phase2/candidate.json"
```

Fix what it names. **Never edit a kill-switch field to make a check pass** — that is the exact
failure the gate exists to catch.

### Phase 2b — Empirical Pivot Entry (When a Hypothesis is Falsified)

When an empirical hypothesis is falsified during experiments, re-enter `idea-forge` directly at Phase 2:
- **Skip Phase 0 and Phase 1**: The problem context and literature lineage are inherited from the
  existing run without repeating searches.
- **Mandatory Input**: The negative constraint statement and causal failure root cause from `evidence-auditor`.
- **Constraint Compliance**: The new candidate must explicitly declare how its new operator
  bypasses the fatal failure mode identified in `negative_constraints.json`.
- **Gate**: Run `validate_idea.py "$RUN/phase2/candidate.json"`. The validated candidate then
  proceeds to Phase 3 for full 3-Round Two-Stage Gauntlet evaluation.

## Phase 3 — Quality gauntlet

Run both, in this order, in separate contexts. The context that wrote a candidate rubber-stamps it.

1. **`scoop-radar`** on the candidate → a 1–5 overlap level and a delta statement.
2. **`idea-critic`** in `gauntlet` mode, with the scoop report → `advance | revise | abandon`.

Also check `references/anti-patterns.md`: if the set of patterns used matches a documented
reject-favored composition, the required mitigation must be **substantively present in the
mechanism**, not merely mentioned.

**Routing and Iterative Revision Loop.**

- `advance` → Phase 4. **Strict Requirement: Advancing to Phase 4 requires an unambiguous UNANIMOUS ADVANCE (5 out of 5 seats).** If even a single critic seat issues a `revise` verdict, `idea-forge` rejects advancing to Phase 4 and enters the revision loop until that seat's specific objections are fully resolved.
- `revise` → enter the **Two-Stage Re-Evaluation Protocol**:
  1. **Archive and Structure Rounds**: Chronologically track iterations in `$RUN/phase3/` as `$RUN/phase3/round_N_eval/` (critic reports and panel synthesis) and `$RUN/phase3/round_N_revision/` (revision targets, candidate snapshots `candidate_rN.json`, diff, and validation logs).
  2. **Stage 1 (Revision Audit Gate)**: A single independent compliance auditor subagent checks `candidate_rN.json` against previous `revision_targets[]`. The auditor must issue a `PASS` verdict before proceeding to Stage 2. If `FAIL`, return to revision.
  3. **Stage 2 (Parallel Independent Re-Evaluation Panel)**: Upon `PASS`, dispatch independent evaluator subagents concurrently in parallel (5 seats: Chair, Theorist, Empiricist, Domain, Insider) on `candidate_rN.json`. Write reports to `$RUN/phase3/round_{N+1}_eval/` and synthesize into `panel_verdict.md`.
  4. **Unanimous Advance Gate**: If all 5 seats vote `advance` (5/5 unanimous), proceed to Phase 4. If any seat votes `revise`, increment `N` and loop to resolve the remaining targets. **Single-seat hotfixing within the same round is strictly prohibited**: whenever a candidate is edited, it constitutes a new revision (`candidate_rN.json`) that must undergo Stage 1 Audit Gate followed by a fresh, full 5-seat parallel panel evaluation (`round_{N+1}_eval/`) to audit for cross-component regressions and side effects.
- `abandon` → retry **only if this attempt produced lessons the previous generation did not
  have.** Archive to `$RUN/attempt_N/` and regenerate with the archived audit as a negative
  constraint. If the same subsumption lesson repeats, the problem is the framing: go back to
  Phase 1 once. No new lessons, or three candidate cycles spent → write `gauntlet_failed.md`
  citing every verdict, and stop. A run that honestly fails is worth more than a card that
  survived by lowering the bar.

## Phase 4 — Idea card

Render `$RUN/idea-card.md`: title · motivation (the bottleneck, with at least two "why prior
work stopped here" points) · method as numbered steps · falsification · compute · the closest
prior work and the delta · reviewer concerns surfaced by the gauntlet.

Then register the claim on the spine so downstream skills can address it: write
`claim-tree.json` with the idea's central assertion as `C0` and copy the four kill-switch
fields verbatim. `paper-architect` will expand it; `experiment-architect` will bind
experiments to it.

Re-run `validate_idea.py --baseline "$RUN/phase2/candidate.json"` as the last act.

## Scope

Produces an idea plus an honest feasibility judgment. It does not produce experiment matrices,
ablation plans, baseline tables, or a calendar — those are `experiment-architect`'s. It
produces **one** idea, not twenty; a ranked list of twenty is a way of avoiding the judgment
that makes one of them defensible.
