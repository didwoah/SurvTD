# Query design for Phase 0

Retrieval quality caps everything downstream. A bottleneck can only be as real as the papers
it was read from, so four queries are written deliberately before any search runs.

## The four slots

| # | Slot | What it recalls |
|---|---|---|
| 1 | **Problem** | Papers that name the same difficulty you are naming |
| 2 | **Mechanism** | Papers whose method operates on the same object, whatever they call the problem |
| 3 | **Escape-mechanism** | Papers that *already fixed* this bottleneck — see below |
| 4 | **Adjacent-field** | The same structural problem under another community's vocabulary |

A fifth query is allowed when a slot genuinely splits, never to round the number up.

## Slot 3 is the one people skip

A paper that solved your bottleneck does not title itself by your problem. It titles itself
by its solution. So a query set written entirely in problem vocabulary has a structural blind
spot precisely where the most dangerous papers live — the ones that make your idea a
regression rather than a contribution.

Write slot 3 by finishing this sentence in *solution* words:

> "If someone had already fixed this, their method would be called …"

```
problem framing   →  "advantage dilution in long-context RL"          (slots 1-2)
escape framing    →  "per-step credit assignment policy gradient"     (slot 3)
adjacent framing  →  "temporal credit assignment reward shaping"      (slot 4)
```

## Two tests every query must pass

**Vocabulary ownership.** Does your community actually use these words for this thing? Terms
you coined retrieve nothing. Terms borrowed from an adjacent field retrieve that field.
Sanity-check: could you point at three papers likely to use this phrasing in their abstract?

**Concrete object.** Does the query name a thing, not a quality? `"efficiency"`,
`"robustness"`, `"challenges"`, `"landscape"`, `"overview"` all retrieve surveys.
`"KV cache eviction"`, `"group-normalized advantage"`, `"phase-level credit"` retrieve
methods. Mechanism-first phrasing is the difference between a corpus of gaps and a corpus of
restatements.

## Two windows, on purpose

| Window | Sources | Why |
|---|---|---|
| 0–6 months | arxiv, s2, openreview | Live gaps and scoop risk live here. Preprints and in-review submissions. |
| 6–24 months | openalex, s2, crossref | Lineage and baselines live here. Published, with citation counts and clean identifiers. |

No single source covers both ends: arXiv has recency and a weak lexical API, OpenAlex has
breadth and no in-review signal, OpenReview has the forward-looking signal and nothing else,
Crossref has clean DOIs and no preprints. Query each where it is strongest. Overlap is fine —
deduplication merges the records and each source enriches fields the others lack.

## Reading the yield report

`lit_search.py` writes `dropped_off_topic` — the records it discarded for sharing too little
with the query that retrieved them. Read that list once per run.

- **Many drops from one query** → that query is spending retrieval budget on noise. Rewrite it
  with a more concrete object, or drop it.
- **A drop you recognize as important** → your query is too generic, not the gate too strict.
  Lower `--min-overlap` for that run *and* fix the query.
- **Zero results** → widen the window before widening the query. A narrow query over 24 months
  beats a vague query over 6.
