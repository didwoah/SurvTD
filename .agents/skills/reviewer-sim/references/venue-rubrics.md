# Venue rubrics

Scales and norms for the venues this skill simulates. Confirm against the current year's
reviewer guide — venues revise these, sometimes substantially, between editions.

## ICLR

**Rating 1–10.** In practice the mass sits at 3, 5, 6, 8.

| | |
|---|---|
| 1–3 | Reject. Strong reject at 1–2 |
| 5 | Marginally below threshold — the most common rejected score |
| 6 | Marginally above threshold — the most common accepted score |
| 8 | Accept, good paper |
| 10 | Award quality |

**Confidence 1–5.** 4 = confident, checked the math and related work carefully. 5 = absolutely
certain, deeply familiar with the area. Reviewers use 3–4 most often.

Open review with a public discussion period. Rebuttal is a threaded conversation, not a single
response, and reviewers are expected to reply.

**Score movement matters more here than anywhere.** In ICLR 2024–2025, papers whose scores moved
after rebuttal were accepted 55.7–57.6% of the time; papers whose scores did not move, 7.8–12.4%.
The modal transitions are 5→6, 6→8, 3→5. Borderline is where rebuttals pay.

## NeurIPS

Recent editions score on separate axes rather than one number:

- **Soundness** 1–4 (poor / fair / good / excellent)
- **Presentation** 1–4
- **Contribution** 1–4
- **Overall** 1–6, where 4 = borderline accept, 5 = accept, 6 = strong accept
- **Confidence** 1–5

Reviews carry slightly fewer weaknesses than ICLR (~5.1 vs ~6.1 per review). A **checklist** is
mandatory and reviewers are asked to flag violations; a mis-answered checklist item is an easy,
avoidable weakness.

## ICML

**Overall 1–5**, with confidence. Emphasis on soundness and clarity of contribution. Shorter
rebuttal window than ICLR and no extended discussion — the response has to land in one shot,
which changes the strategy: lead with the strongest evidence rather than building to it.

---

## What actually moves scores

Across venues, the objections that decide papers:

1. **Baseline strength and fairness** — "you compared against a weak or undertuned baseline."
   The most common fatal empirical objection, and unanswerable after the runs are done unless
   parity was declared in advance.
2. **Missing ablation** — the reader cannot attribute the gain to the claimed mechanism.
3. **Insufficient evaluation breadth** — one dataset, one model scale, one seed.
4. **Overclaiming** — the abstract's scope exceeds the results' scope. Cheap to fix before
   submission, expensive after.
5. **Unclear contribution** — the reviewer cannot state in one sentence what is new. Usually a
   claim-tree problem wearing a writing costume.
6. **Positioning** — the delta from the two closest works is not stated, so the reviewer
   supplies their own guess.

## Norms worth encoding in a simulated review

- Reviewers read **title → abstract → Figure 1 → first table**, then decide how carefully to
  read the rest. Simulate that path when judging presentation.
- **Confidence scales the damage.** A confident 3 outweighs an uncertain 3 in the AC's
  weighing, and needs a different rebuttal.
- **A reviewer who asks a question is engaged**, and engagement predicts score movement more
  than initial tone does. Questions are the most valuable part of a simulated review.
- **The AC weighs, it does not average.** A well-argued minority objection can decide a paper
  against two positive reviews.
