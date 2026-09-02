---
name: paper-architect
description: >-
  Build the claim tree that a paper is organized around — one core claim, sub-claims each bound
  to exactly one figure or table, an abstract written before the sections, a Figure 1 that
  renders the claim rather than the architecture, and a section budget against the venue page
  limit. TRIGGER when the user asks how to structure a paper, what the story or narrative
  should be, to write or fix an abstract or contribution bullets, how to organize sections, or
  what Figure 1 should show. DO NOT TRIGGER for writing full section prose, for related-work
  positioning (that is related-work-positioner), for figure rendering (that is figure-smith),
  or for reviewing a finished draft (that is reviewer-sim).
---

# Paper Architect

Decide what the paper claims before deciding what it says. The artifact is
`claim-tree.json` — the spine every other skill in this pack addresses — plus `abstract.md`
and `outline.md`.

The method is old and reliable: **write the claim, then the abstract, then the figure, then
the paper.** Peyton Jones and Widom both teach it, and the reason it works is that a paper
written outward from a claim cannot accumulate sections that serve no claim, whereas a paper
written forward from an introduction always does.

## Input

`evidence-audit.json` if results are audited, `claim-tree.json` if `idea-forge` seeded one,
otherwise the idea and whatever results exist. Say what you started from.

---

## 1. The core claim: one sentence, ≤ 25 words

> Phase-stratified credit recovers the long-context RL signal that uniform advantage broadcast
> dilutes.

**If it does not fit in 25 words, the paper has more than one core claim.** That is the single
most useful diagnostic in this skill. Split it, promote one half to a sub-claim, or accept that
you have two papers and pick one. Two half-developed contributions in one submission is a
recognized reject pattern — reviewers attack the weaker half and the stronger one never gets
its due.

The core claim is `C0`. It is **falsifiable** ("X does Y under Z"), not an activity ("we
propose X").

## 2. Sub-claims, each bound to exactly one anchor

Two to four. Each names the claim it `supports`, and each is bound to **exactly one**
`fig:N` / `tab:N` / `thm:N` / `app:N`.

- **Bound to nothing** → cut the claim, or plan the experiment. There is no third option, and
  every unbound sub-claim is an overclaim that has not been noticed.
- **Bound to two anchors** → it is two claims. A claim whose evidence is spread across two
  exhibits is one neither exhibit establishes.
- **Stronger than the claim it supports** → the support cannot outrank the conclusion.

```bash
python3 "$SKILL_DIR/scripts/check_claim_tree.py" claim-tree.json
```

## 3. The abstract, written now

Before any section. Every sentence maps to a section **and** to an evidence anchor. The
mapping is the point: a sentence with no anchor is a promise the paper does not keep, and it is
the sentence a reviewer quotes back.

A serviceable shape, five to seven sentences:

1. The setting, in one clause
2. The bottleneck — specific enough that someone could disagree with it
3. What is proposed, in mechanism terms, not marketing terms
4. The core claim, at the strength `evidence-audit.json` supports
5. The headline evidence, with the condition it holds under
6. The scope boundary — where it does *not* hold. Reviewers trust papers that state this
7. What it means, in one clause

Write it at the strength the audit permits. An abstract stronger than the results is the most
expensive sentence in a paper.

## 4. Contribution bullets that can be wrong

| ✗ | ✓ |
|---|---|
| We propose PS-GRPO, a novel credit assignment method | Redistributing GRPO's advantage onto pivotal phases widens the 128K accuracy gap; uniform broadcast does not |
| We conduct extensive experiments | The gain vanishes under label permutation at fixed C, isolating the mechanism from variance reduction |
| We show our method is effective | The effect holds at 64K and 128K, and is inside seed noise at 16K |

"We propose X" describes an activity. Nothing in the world contradicts it, which is exactly why
it carries no information. Every bullet states something that could be false, and names the
condition it holds under.

## 5. Figure 1 renders the claim, not the architecture

The most common wasted page in an ML paper is a system diagram as Figure 1. It shows what you
built; the reader wants to know what you found.

**Figure 1 should be the picture of `C0` being true.** For a comparative claim, that is usually
the curve or the gap. For a mechanism claim, it is often the negative control sitting next to
the main result. The architecture diagram is valuable — put it in the method section, where
someone who has decided to care is ready for it.

Write the spec, hand it to `figure-smith`:

```json
{"id": "fig:1", "supports_claim": "C0", "claim_kind": "comparative",
 "caption": "<the takeaway sentence, not the subject>", ...}
```

## 6. Section budget

Allocate pages against the venue limit before writing. A section over budget must name what it
cuts — deciding at the end, under deadline, cuts whatever is least finished rather than least
important.

Typical 9-page allocation: intro 1.25 · related work 0.75 · method 2.5 · experiments 3 ·
analysis 1 · conclusion 0.5. Adjust for the paper's kind — a theory paper moves pages from
experiments to method, a benchmark paper the reverse — but *decide* it.

## 7. The reader-path check

Reviewers read title → abstract → Figure 1 → the first table, then decide how carefully to read
the rest. Simulate exactly that path and ask: **does the claim and its main evidence arrive
inside those four elements?**

If it takes until section 4.3 to learn what was found, the paper will be reviewed as though
that finding does not exist.

---

## Output

- `claim-tree.json` — validated by `check_claim_tree.py`
- `abstract.md` — with the sentence → section → anchor map beneath it
- `outline.md` — sections, page budget, and which claims each section carries

## Rules

- **Never introduce a claim the evidence does not carry.** If `evidence-audit.json` downgraded
  a claim, the abstract uses the downgraded wording. This is the exact point where audited
  results get quietly re-promoted on their way into prose.
- **Ids are append-only.** A cut claim becomes `"status": "withdrawn"`, never deleted — reviewer
  comments and audit findings that referenced it still need to resolve.
- **The tree outranks the outline.** If a section serves no claim, the section goes, not the
  invariant.
