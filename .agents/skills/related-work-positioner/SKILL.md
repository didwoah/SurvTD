---
name: related-work-positioner
description: >-
  Write the related-work section as a positioning argument rather than a bibliography — group
  prior work into method lineages, build an explicit delta table against the closest works,
  state why each stopped where it did, and verify every citation resolves to a real paper.
  TRIGGER when the user asks to write or improve a related-work section, to position their
  contribution against prior work, to build a comparison table of prior methods, or to check
  a .bib file for fabricated references. DO NOT TRIGGER for deciding whether an idea is scooped
  (that is scoop-radar) or for a general literature survey with no contribution to position.
---

# Related-Work Positioner

`scoop-radar` answers *am I scooped*. This answers *how do I place this* — and they need
different artifacts. A scoop check is a worst-case search; a positioning section is an argument
that a specific gap exists and that this work fills it.

**Related work is a scientific claim, not a courtesy.** A list of papers with one sentence each
tells a reviewer nothing about why your work needed to exist. A lineage plus an explicit delta
tells them exactly that.

## Input

The contribution (`claim-tree.json` / idea card), `scoop-report.md` if one exists, and the
current `.bib` if there is one.

---

## 1. Retrieve, broadly this time

The scoop check searched for collisions. This search is for **coverage** — the works a reviewer
in this area expects to see.

```bash
python3 "$SKILL_DIR/scripts/lit_search.py" \
  --query "<lineage 1>" --query "<lineage 2>" --query "<lineage 3>" \
  --sources arxiv,openalex,s2,dblp --from <today-36mo> --limit 25 \
  --out rw/pool.json --markdown rw/pool.md
```

Then add, from your own knowledge, the foundational work the recent window misses. A related
work section that starts in 2024 signals that the authors do not know the area. Mark these
`source: model-recall` — **`verify_citations.py` will check every one of them, and anything
that does not resolve is treated as fabricated.**

## 2. Group into lineages, not buckets

A bucket is "Methods for X" — a list wearing a heading. A **lineage** is a chain: this method
refined that one, this one replaced that assumption, this one is a branch from here.

For each lineage write:

- The **shared premise** every member holds
- How the line **progressed** — what each step fixed
- Where it **stopped**, and *why*: was the next step technically infeasible, was the problem
  believed unimportant, or did nobody look?

That third question is the heart of the section. "Prior work stopped here because X was
believed necessary" sets up a contribution far better than "prior work does not address Y."

Two to four lineages. More than four means the framing is too broad and the reader loses the
plot.

## 3. The delta table

For the three to six closest works, an explicit comparison on the four axes. This is the table
reviewers actually use, and its absence is why "how is this different from [paper]?" is the
most common review question.

| Work | Problem framing | Core mechanism | Key insight | Domain | Δ from ours |
|---|---|---|---|---|---|
| LoongRL (2025) | long-context RL | uniform group-normalized advantage | curriculum over hop depth | text QA | leaves the advantage untouched; we redistribute it onto pivotal phases |

The Δ column is one sentence, concrete, and **not** "different approach". If a Δ cell is hard
to write, that work is closer than the framing admits — go back to `scoop-radar` before
writing prose around it.

## 4. Write the section

Lineage-first prose, not paper-first. Each paragraph carries one lineage: shared premise,
progression, stopping point, and one sentence relating it to the contribution.

Three failure modes to avoid:

- **The wall of citations.** Fifteen papers in one parenthetical does not show breadth, it
  shows they were not read.
- **The strawman.** Understating prior work is the fastest way to draw a hostile reviewer, and
  the author of that paper is often in the pool.
- **The undifferentiated close.** Every closest work gets its Δ stated in the text, not only in
  a table a reviewer may skip.

Position **honestly relative to the audit.** If `evidence-audit.json` downgraded a claim, the
delta is against the downgraded version. Claiming a delta the results no longer support is how
a related-work section becomes the paragraph a reviewer quotes.

## 5. Verify every citation

```bash
python3 "$SKILL_DIR/scripts/verify_citations.py" refs.bib --out citations.verified.json
```

Resolves each entry by DOI, then arXiv id, then title against Crossref, OpenAlex, and Semantic
Scholar, with year-aware matching so a landmark paper does not resolve to a re-indexed
duplicate. Three outcomes:

- `verified` — resolved, title matches
- `mismatch` — resolved to a paper with a different title. A **chimeric citation**: real
  authors, real venue, wrong or invented title. Reads perfectly and is wrong.
- `unresolved` — no index has it. **Treat as fabricated until proven otherwise.**

This is not optional hygiene. A fabricated citation found by a reviewer costs the paper more
than any weak result, and it is invisible to proofreading — that is precisely why it needs a
mechanical check.

---

## Output

- `related-work.md` — the section, plus the delta table
- `citations.verified.json` — the resolution record
- A list of the works your positioning *depends* on, so `reviewer-sim` knows who the hostile
  reviewer is likely to be

## Rules

- **Never cite a paper you have not verified exists.** Not "probably fine" — verified.
- **Never cite a paper for a claim you have not checked it makes.** Citing a real paper for
  something it does not say is the same failure with better camouflage.
- **State the delta against the strongest reading of prior work**, not the most convenient one.
