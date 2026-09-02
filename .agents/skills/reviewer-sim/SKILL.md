---
name: reviewer-sim
description: >-
  Simulate the review a paper will actually receive at ICLR, NeurIPS, ICML or a similar venue —
  three independent reviewers with separated personas plus an area-chair meta-review, scored on
  the venue's own scale, with every weakness tagged fixable-before-deadline or rebuttal-only.
  TRIGGER when the user asks how their paper will be reviewed, to review a draft before
  submission, what reviewers will attack, or what score to expect. DO NOT TRIGGER for judging
  an idea that has no results (that is idea-critic), for auditing results against claims (that
  is evidence-auditor), or for responding to reviews that already exist (that is
  rebuttal-forge).
---

# Reviewer Sim

Produce the review before the reviewers do, while there is still time to act on it.

**Three separate reviewers, not one averaged one.** An averaged review hides the reviewer who
will actually sink the paper. Real review outcomes are driven by the *distribution*: a paper
with 8/6/3 is in far more trouble than one with 6/6/5, and the 3 is a specific person with a
specific objection who has to be identified by name of concern.

## Input

The draft (or its claim tree, abstract, method, and results), plus `evidence-audit.json` and
`related-work.md` if they exist. Read `references/venue-rubrics.md` for the target venue's
scale and norms.

---

## The three reviewers

Run each **independently**. Do not let one reviewer's conclusion inform another's — the point
is three genuinely different readings, and a shared context collapses them into one voice with
three headings.

### R1 — The methods skeptic
Reads the method section first and hard. Asks: is the mechanism what the paper says it is?
Would a simpler thing do this? Is there an unstated precondition? Is the novelty in the
mechanism or in the vocabulary? Typically the most technically detailed review and the source
of the deepest objections.

### R2 — The empirical rigor reviewer
Reads the tables first. Asks: are the baselines strong, current, and **tuned as well as the
proposed method**? How many seeds? Are the error bars real and defined? Does the ablation
isolate the claimed mechanism? Are there datasets or metrics conspicuously absent? Is the
comparison compute-matched?

**This reviewer decides most empirical papers.** In large-scale ICLR review analysis the
dominant experiment-section weaknesses are weak baselines, too few datasets, missing ablations,
and reproducibility — R2's entire remit.

### R3 — The scope and positioning reviewer
Reads the abstract, intro, and related work. Asks: is this the right venue? Is the delta from
the two closest works real and stated? Does the abstract promise more than the results deliver?
Would a practitioner in this area change anything after reading it? Often the shortest review
and the one that produces "not enough for this venue."

**Weakness volume is calibrated, not polite.** Rejected papers at ICLR carry ~5.9 weaknesses
per review against ~5.2 for accepted ones. A simulated review with two mild concerns is not a
review — it is encouragement. Find what is actually there.

## Every weakness gets four fields

| Field | Values |
|---|---|
| **claim id** | which `C*` it attacks, or `presentation` / `positioning` |
| **severity** | `fatal` · `major` · `minor` |
| **timing** | `fixable-before-deadline` · `rebuttal-only` · `unfixable` |
| **evidence** | a quote or a specific table/section reference |

The **timing** field is what makes this skill actionable rather than demoralizing. A major
weakness that is fixable in three days before submission is a task; the same weakness
discovered after submission is a rebuttal problem with a much lower success rate. Sort by
severity within `fixable-before-deadline` and work that list first.

## Scores and the meta-review

Each reviewer gives a rating and a **confidence** on the venue's scale
(`references/venue-rubrics.md`), with the confidence honestly reflecting how much of the paper
that persona actually engaged with. A confident low score does far more damage than an
uncertain one, and knowing which you are facing changes the rebuttal strategy entirely.

Then an **area chair meta-review**: where the reviewers disagree, which objection carries most
weight, and the likely outcome. Give a distribution — "accept if R2's baseline concern is
answered, reject otherwise" — not a point estimate. The AC's job is to weigh, and a simulated
AC that just averages has skipped the only interesting part.

## Output

```
## Simulated review — <title> · <venue> <year>

### R1 — methods skeptic · Rating: N/10 · Confidence: N/5
**Summary:** <2 sentences the reviewer would write>
**Strengths:** …
**Weaknesses:**
| # | Weakness | Claim | Severity | Timing | Evidence |
**Questions to authors:** …

### R2 — empirical rigor · …
### R3 — scope and positioning · …

### Area chair meta-review
Score distribution: N / N / N
Points of disagreement: …
The objection that decides this paper: …
Likely outcome: … (conditional on what)

### Action list before the deadline
| Priority | Fix | Addresses | Est. effort |

### Rebuttal-only items
<what cannot be fixed now, and what evidence would answer it later>
```

## Rules

- **Ground every weakness in the text.** A review of a paper you did not read closely is worth
  as little from a model as from a human. Quote or reference.
- **Do not soften.** The purpose is to surface what a real reviewer will say while it can still
  be acted on. A gentle simulated review is a disservice with a deadline attached.
- **Do not manufacture weaknesses either.** Padding to hit a count teaches the author to
  discount the list, and the real objection gets discounted with it.
- **Report strengths honestly.** Reviewers do, and an AC weighs them. A review with no strengths
  is not calibrated to any real venue.
- **Stay inside what the paper claims.** Reviewing the paper you wish had been written is the
  most common failure of simulated review.
