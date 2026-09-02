---
name: theory-rigor
description: >-
  Audit the theoretical content of a paper — build an assumption ledger, track every proof
  obligation to a proved lemma or an admitted gap, check that the theorem's conditions actually
  hold in the regime the experiments use, hunt boundary counterexamples, and check the bound is
  non-vacuous at realistic scale. TRIGGER when the user has a theorem, proposition, bound, or
  convergence guarantee and asks whether it is sound, tight, or meaningful, or wants the theory
  section checked before submission. DO NOT TRIGGER for purely empirical papers or for
  numerical experiment design.
---

# Theory Rigor

Theory sections fail in review for reasons that are checkable in advance. Almost none of them
are "the proof has an error" — that is rare. The common causes are a theorem whose hypotheses
never hold in the experiments, a bound that is vacuous at realistic scale, and an assumption
doing load-bearing work that is never stated.

## 1. The assumption ledger

Every assumption, stated or implicit, in one table.

| # | Assumption | Where used | Load-bearing? | What breaks without it | Holds in experiments? |
|---|---|---|---|---|---|
| A1 | Loss is L-smooth | Thm 1, Lem 2 | yes | rate becomes O(1/√T) | unverified for the ReLU net used |

**Load-bearing** means the result fails without it, not merely that the proof used it. The
distinction matters: a convenience assumption can be relaxed in revision, a load-bearing one
defines the result's scope and belongs in its statement.

The last column is where theory papers lose. An assumption the experiments violate must be
stated as a limitation, not left for a reviewer to notice.

## 2. Proof-obligation tracking

Each theorem decomposes into obligations. Every obligation gets one status:

- `proved` — here, in full
- `cited` — proved elsewhere; **the citation must be verified to exist and to state what you
  say it states**
- `assumed` — an admitted gap. Legitimate when stated; fatal when discovered
- `sketched` — an argument that would need work to complete. Say so

Then: does each theorem's proof close? An obligation chain with a `sketched` step in the middle
is a conditional result and must be presented as one.

## 3. The vacuity check

**The most common theory-paper rejection.** Two questions:

**Does the hypothesis ever hold?** Instantiate every condition with the actual experimental
setup — real dimensions, real step sizes, real network. If the condition requires
`η < 1/L` and the experiments use `η = 0.1` with unmeasured `L`, the theorem is decorative
until you measure `L`.

**Is the conclusion non-vacuous at realistic scale?** Plug in real numbers. A generalization
bound that evaluates to "the error is at most 4.7" for a quantity in [0,1] states nothing. A
rate with a constant exponential in dimension says nothing at d=768. Compute it. Report the
value.

A vacuous-but-honest bound is publishable if the paper says what it does and does not
establish. An unevaluated bound presented as a guarantee is what draws the sharpest review.

## 4. Boundary counterexample hunt

Try to break your own statement before a reviewer does. Systematically:

- **Degenerate inputs** — zero, empty, singleton, all-identical, rank-deficient
- **Boundary parameters** — the exact edge of every stated range
- **Adversarial constructions** — the worst case for the quantity you bound
- **Limits** — dimension → ∞, step size → 0, batch → 1

Each surviving probe is evidence. Each failure is either a missing hypothesis (add it, narrow
the claim) or a broken theorem (find out now).

## 5. Constant and rate audit

- Are constants explicit, or hidden in `O(·)`? A hidden constant exponential in dimension makes
  a rate meaningless and is the classic place it hides.
- Is the rate **tight**, or only an upper bound? If only upper, say so — an upper bound does not
  establish that your method is faster than an alternative with a looser proven bound.
- Does the comparison to prior rates hold **under the same assumptions**? Comparing your rate
  under strong convexity to theirs under convexity is not a comparison.

## 6. Theory ↔ experiment binding

Most theory papers never do this, and it is the cheapest available credibility.

For each theorem, name **the experiment that tests its prediction**. Not an experiment showing
the method works — one showing the *predicted relationship* holds: the predicted rate appears in
the loss curve, the predicted threshold appears where predicted, the bound tracks the measured
quantity as a parameter varies.

Bind it on the spine: the theorem becomes an evidence anchor (`thm:1`) for a claim in
`claim-tree.json`, and the experiment testing it goes into `evidence-plan.json` with
`tests_claims` pointing at that claim.

**If no experiment could distinguish your theorem being true from it being false in this
setting, the theory and the empirics are two papers sharing a PDF.** Say so, or bind them.

---

## Output

```
## Theory audit — <paper>

### Assumption ledger
<the table>

### Proof obligations
| Theorem | Obligation | Status | Note |

### Vacuity check
| Theorem | Hypothesis holds in experiments? | Bound value at experimental scale | Verdict |

### Counterexample probes
| Probe | Result | Consequence |

### Constants and rates
### Theory ↔ experiment binding
| Theorem | Predicted relationship | Experiment that tests it | Status |

### Verdict
<sound / conditional / vacuous-as-stated / broken> — with the one change that most improves it
```

## Rules

- **Never assert a proof is correct without walking the obligations.** "The proof appears
  sound" is not an audit.
- **Compute the bound.** A bound whose value at experimental scale was never evaluated has not
  been checked.
- **A conditional result stated as conditional is publishable.** Stated as unconditional, it is
  the finding that costs the paper.
