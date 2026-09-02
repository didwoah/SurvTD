---
name: rebuttal-forge
description: >-
  Turn a set of reviews into a rebuttal that moves scores — classify every reviewer point as
  concede, clarify, refute with existing evidence, or refute with a new experiment; check which
  new experiments fit the rebuttal window; prioritize the reviewer closest to flipping; and
  draft the responses. TRIGGER when the user has received reviews and asks how to respond, what
  to run during the rebuttal period, how to prioritize reviewer concerns, or wants a rebuttal
  drafted. DO NOT TRIGGER for predicting reviews before submission (that is reviewer-sim) or
  for revising the paper itself.
---

# Rebuttal Forge

The rebuttal is where borderline papers are decided. In ICLR 2024–2025, papers whose scores
moved after rebuttal were accepted **55.7–57.6%** of the time; papers whose scores did not move,
**7.8–12.4%**. The modal transitions are 5→6, 6→8, 3→5.

Two things follow, and they shape everything below.

1. **The goal is score movement, not being right.** A response that wins the argument and
   leaves the score unchanged has failed.
2. **Target the reviewer closest to flipping, not the most hostile one.** A confident 3 with a
   structural objection may be immovable in two weeks. A 5 with two answerable concerns is where
   the acceptance is. Hostility feels urgent; borderline is where the return is.

## Input

The reviews, the paper, `evidence-audit.json` if it exists, and the **hard constraints**:
days remaining, compute available, word limit per response.

---

## 1. Extract every point

Reviews mix real objections with narration. Extract each **distinct actionable point**, with:

- who raised it (a point raised by two reviewers is worth more than either alone)
- the claim id it attacks, from `claim-tree.json`
- whether it is a **stated objection** or an **implied doubt** (a lukewarm summary line is a
  concern that was not written down; address it anyway)

A point that appears in two reviews and is answered well moves two scores.

## 2. Classify every point — four buckets

| Bucket | When | How it reads |
|---|---|---|
| **concede** | The reviewer is right | Say so in the first sentence, state the change, move on. **Do not argue.** |
| **clarify** | The paper contains the answer and the reviewer missed it | The paper was unclear. Say *where* it will be made clearer, quote the passage, never imply they failed to read |
| **refute with existing evidence** | You have data that settles it | Lead with the number or table. One sentence of framing, then the evidence |
| **refute with new experiment** | Only a new run settles it | Go to step 3 |

**Concede early and cleanly.** A rebuttal that contests every point reads as defensive and
reviewers discount all of it, including the parts that were right. Conceding one real weakness
buys credibility for the four refutations that follow — this is the most reliably underused
move in rebuttal writing.

**Never contest a reviewer's premise without evidence.** "We disagree that baselines are weak"
loses. "Table R1 adds the baseline at matched tuning budget; the gap holds" wins.

## 3. Triage the new experiments

For each candidate experiment, estimate honestly: wall-clock, compute, and whether it can be
*written up* in the window, not just run.

Then rank by **expected score movement per hour**, which is not the same as scientific
interest:

- Which reviewer does it address, and **how close are they to flipping?**
- Is it a point raised by multiple reviewers?
- Does it address the objection the AC will weigh most (usually R2-type: baselines, ablations,
  breadth)?
- Would a *partial* result help? An added baseline at 3 of 5 datasets, clearly labeled as
  partial, usually beats promising the full grid later.

**Then cut the list to what actually fits, with slack for writing.** An overcommitted rebuttal
that arrives incomplete is worse than a smaller one delivered whole. Name explicitly what you
are not running and why — reviewers accept a stated scope limit far better than a missing
answer.

## 4. Draft

**Per reviewer.** Open with the single strongest thing you have for *that* reviewer — most
reviewers read the opening and skim the rest. Order: their most severe concern first, not the
easiest one.

**A common response** when two or more reviewers raise the same point: answer it once,
thoroughly, and reference it from each individual response. Repeating an answer three times
wastes the word budget that the specific concerns need.

**Structure per point:**

```
> [quoted reviewer concern, trimmed]

<Verdict in the first sentence: agree / here is the evidence / here is the new result.>
<Then the evidence: a number, a table reference, or the specific change.>
<Then, if relevant, what remains open — stated honestly.>
```

**Tone contract:**

- No defensiveness, no "as we clearly stated in Section 3."
- No new claims that the results do not carry. **A rebuttal that overclaims gets audited by an
  irritated reviewer**, and they will find it.
- Word limits are hard. Cut framing, never evidence.
- Thank reviewers once, at the top, briefly. Gratitude paragraphs consume the budget answers
  need.

## 5. Track the paper edits

Every concede and clarify implies a change to the paper. Keep `rebuttal/changelog.md` mapping
each promise to the edit that fulfills it, so the camera-ready actually contains what the
rebuttal promised. Promising a change and not making it is noticed, and at open-review venues
it is noticed publicly.

---

## Output

```
rebuttal/
├── point-map.md        every extracted point → reviewer, claim id, bucket, priority
├── experiment-plan.md  what runs in the window, what does not, and why
├── response-R1.md      per-reviewer drafts, inside the word limit
├── response-R2.md
├── response-common.md  shared answer to shared points
└── changelog.md        promise → paper edit
```

Lead the point map with the **flip analysis**: for each reviewer, current score, what would move
it, and how plausible that is in the time available. That table is what decides where the
remaining days go.

## Rules

- **Answer every point.** An unanswered concern is read as conceded, and silently.
- **Lead with evidence, not framing.**
- **Concede what is true, immediately.**
- **Never promise an experiment you cannot deliver in the window.**
- **Prioritize by expected score movement**, not by how annoying the reviewer was.
