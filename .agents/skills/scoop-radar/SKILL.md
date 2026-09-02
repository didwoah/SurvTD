---
name: scoop-radar
description: >-
  Check whether a proposed research novelty has already been published — decompose the claim
  into four axes, search prior art on two channels, deep-read the closest candidates, and
  return a 1–5 overlap level plus a defensible delta statement. TRIGGER when the user asks
  whether an idea is novel, whether they have been scooped, how their contribution differs
  from prior work, or wants a prior-art check on a specific claim. DO NOT TRIGGER for general
  literature reviews with no claim to check, for judging whether an idea is *good* (that is
  idea-critic), or for writing a related-work section for an already-validated idea (that is
  related-work-positioner).
---

# Scoop Radar

Decide whether a specific novelty claim is already occupied, and if not, produce the sentence
that says why. Two outputs matter: an overlap **level**, and a **delta statement** a reviewer
would accept.

**This skill judges occupancy, not quality.** An idea can be perfectly novel and worthless.
Novelty read alone is actively misleading — a vague idea collides with nothing and scores as
maximally novel, which is why a bare model beats a grounded one on novelty while losing badly
on quality. Always read a scoop level next to `idea-critic`'s verdict, never alone.

## Input

Two things, and if either is missing, infer the most reasonable reading and proceed. Never
stop to ask.

1. **Research problem** — the difficulty being addressed.
2. **Novelty** — the specific claimed contribution.

If the "novelty" only restates the problem, say so at the top of the report: a claim that
cannot be distinguished from its problem cannot be checked for occupancy, and that is itself
the finding.

---

## Step 1 — Decompose the claim into four axes

| Axis | What it captures |
|---|---|
| **Problem framing** | Task definition, inputs, outputs, evaluation regime |
| **Core mechanism** | The technical move — architecture, algorithm, proof technique, data construction |
| **Key insight** | Why it works; what prior state of the art lacked |
| **Application domain** | Where it applies, and how broadly |

Write all four explicitly. Every later step compares against this list, so vagueness here
propagates into a falsely comfortable verdict.

## Step 2 — Search two channels

**Signature channel** — your own vocabulary, recent window. Catches contemporaneous scoops.

```bash
python3 "$SKILL_DIR/scripts/lit_search.py" \
  --query "<restate the original research problem>" \
  --query "<broad domain, 3-5 words>" \
  --query "<method signature, 5-8 words>" \
  --sources arxiv,openalex,s2,openreview --from <today-10mo> \
  --limit 25 --out scoop/signature.json
```

**Alias channel** — what *other communities* call this same mechanism, wide window. Catches
the renamed ancestor.

```bash
python3 "$SKILL_DIR/scripts/lit_search.py" \
  --query "<alias term 1>" --query "<alias term 2>" \
  --sources arxiv,openalex,s2,dblp,crossref --from <today-48mo> \
  --limit 25 --out scoop/alias.json
```

The alias channel exists because **the renamed-ancestor blind spot is lexical, not temporal**.
A goal-conditioned success detector and a goal-image conditioned scorer can be the same
mechanism; no date window finds one from the other's name. Generate alias terms from what you
know of neighbouring literatures — control theory, statistics, information retrieval,
econometrics, cognitive science — not from the paper's own words.

**Then augment from memory.** Keyword search misses landmark work that is phrased differently,
and missing the canonical reference is the most common way a novelty check fails. Add papers
you are confident exist and are directly relevant — typically 0–5, tagged
`source: model-recall`. Leave date and venue blank rather than guessing; Step 4 resolves them
from the PDF. **Fabricating a citation is far worse than a smaller pool.**

## Step 3 — Abstract triage

For every deduplicated paper, record: title · date · the four axes as that paper instantiates
them · **overlap score 0–4** (how many axes plausibly match) · source.

The score is a triage signal, not a verdict. Abstracts hide the assumptions and scope
conditions that actually decide overlap.

## Step 4 — Deep-read the candidates

Promote a paper to deep read if **any** hold:

- overlap ≥ 2 on abstracts
- it matches on **core mechanism** specifically, even with a different domain or framing —
  mechanism overlap is the dangerous kind
- it is from the same narrow subfield within 24 months, even at a lower score
- its abstract is ambiguous in a way that could confirm *or* rule out overlap

Cap at **3–7 papers** after filtering. Fewer than 3 qualify → lower the bar and take the next
most similar, so the deep read still has coverage.

```bash
bash "$SKILL_DIR/scripts/fetch_paper.sh" "<pdf url>" "<slug>"
```

Read the extracted text — introduction for claimed contributions, method for what the
mechanism actually is, experiments for scope and assumptions, limitations for what the authors
themselves concede. Skim their related-work for prior art you missed.

**Budget: skim, do not read end to end.** Once you can quote three concrete passages pinning
down the setup, the mechanism, and the scope, stop. If extraction fails, fall back to the
abstract and **record the limitation explicitly** — never report having read a paper you could
not read.

Update the four axes per paper from the body: `match` / `partial` / `differ`. Downgrade papers
the body clears; upgrade papers the body incriminates. Correcting triage noise is the entire
point of this step.

## Step 5 — Level and verdict

Per prior work: `level = 5 − (axes matching)`.

| Axes matching | Level | Label |
|---|---|---|
| 0 | **5** | No overlap — most novel |
| 1 | **4** | Low overlap |
| 2 | **3** | Medium overlap |
| 3 | **2** | High overlap |
| 4 | **1** | Full overlap — scooped |

**The verdict is the minimum level across all prior works — the worst case, never the mean.**
One sufficiently close paper scoops an idea; averaging lets unrelated retrievals dilute a
decisive collision into comfortable-looking novelty. No prior works at all → Level 5.

## Step 6 — The delta

> Unlike **[closest prior work]**, which **[does X under assumption Y]**, the proposed work
> **[does X′ / drops Y / extends to Z]**, yielding **[concrete measurable benefit]**.

Fill all three slots concretely; "extends prior work to a new setting" tells a reviewer
nothing. Draw the assumption and mechanism wording from the deep-read record, not the abstract.

**If no crisp delta can be written, say so and name the paper blocking it.** Forcing a sentence
that does not hold is worse than reporting the block — the reviewer will find it anyway, and
later.

## Report

Render inline, in this order, complete and untruncated:

1. **Verdict** — level number and label
2. **Delta** — the sentence, or the block and the paper causing it
3. **Decomposed claim** — the four axes
4. **All retrieved papers** — every deduplicated record with its fields and overlap score; if
   zero were found, say so and list the queries that returned nothing
5. **Comparison** — proposed work first, then each prior work, with per-paper axes, level, and
   label

**A thorough search that finds nothing is evidence, not a failure.** Document the queries so
the negative result is inspectable.

## Interpreting the levels

- **5 / 4** — the delta stands on its own. Cite the neighbours and move on.
- **3** — related work exists; the delta is defensible but must be stated explicitly in the
  paper. This is where most healthy ideas land.
- **2** — one axis separates you from a competitor. Fragile. Sharpen the axis or reframe; a
  reviewer who reads that paper will ask, and "we differ in domain" rarely survives.
- **1** — reframe the contribution. Write down what residue is left over after the prior work
  is subtracted, and whether that residue is a paper.
