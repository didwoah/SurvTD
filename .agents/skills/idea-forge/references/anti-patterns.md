# Anti-patterns: reject-favored compositions

**This is a guard, not a ban.** Three two-pattern compositions show an Oral rate at least 12
percentage points below the corpus baseline (58.4%) in the ICLR/ICML/NeurIPS analysis behind
the pattern vocabulary. They are not forbidden — each succeeds a third to a half of the time,
and sometimes a problem genuinely calls for one. What they require is that the documented
failure mode be **substantively mitigated in the mechanism**, before the candidate advances.

This is the only place a historical acceptance rate enters the design, and it enters at audit
time, never at generation time. Phase 2 must never choose or avoid a composition because of
its rate.

All three involve `heterogeneous_decomposition`. The empirical signal is that pairing "split
the population and treat the parts differently" with another constructive move stacks two
un-derived choices that then have to justify each other.

---

## 1. `heterogeneous_decomposition` + `self_supervised_signal_engineering`

*Oral 32.1%, Δ −26.3pp (n=53) — the corpus's strongest reject signal.*

**Failure mode.** "We made up the groups **and** we made up the labels." The decomposition is
one un-derived choice; manufacturing supervision per sub-population is a second. Reviewers read
the paper as two stacked heuristics, neither testable independently of the other.

**Required mitigation.** The decomposition criterion must be derivable from observed structure
or task-level supervision — not intuition. The manufactured signal must be the *unique* signal
that decomposition implies. And an ablation must hold the decomposition fixed while varying the
signal (or vice versa), showing the two are not conflated.

---

## 2. `heterogeneous_decomposition` + `structural_prior_encoding`

*Oral 45.2%, Δ −13.2pp (n=93) — the highest-volume case.*

**Failure mode.** "Which one is doing the work?" A structural prior layered on a decomposition
pipeline is asked to handle both the homogeneous core and the inter-group differences.
Reviewers cannot attribute the contribution to either axis.

**Required mitigation.** An ablation that disables the decomposition or the prior *in
isolation* must show a clear differential effect — not merely a smaller combined number. And
the prior must encode a property the decomposition does not already imply; otherwise the two
are the same job done twice.

---

## 3. `architectural_operator_substitution` + `heterogeneous_decomposition`

*Oral 46.2%, Δ −12.2pp (n=39).*

**Failure mode.** "The operator **is** the architecture." A more expressive operator asked to
also absorb multi-population heterogeneity stacks two tradeoffs into one design choice.
Reviewers consistently report the contribution as over-attributed to a single change.

**Required mitigation.** Demonstrate the operator's expressivity gain on *homogeneous* data
where decomposition is unnecessary, and the decomposition's effect on heterogeneous data with
the operator held fixed. Each leg's contribution must be independently identifiable.

---

## How the gauntlet applies this

1. Take the set of `gap_closure[].pattern` values in the candidate.
2. Test every two-element subset against the three compositions above.
3. On a match, judge whether the candidate's `core_mechanism` **substantively delivers** the
   required mitigation. Keyword presence is not delivery — the mitigation must be visible as an
   artifact in the mechanism, not as a sentence in the framing.
4. Matched, not delivered, and not insertable by revision → hard-floor `abandon`. Otherwise the
   candidate may advance with the mitigation surfaced as a reviewer concern.

## What this is not

- **Not exhaustive.** Five further combinations sit at −8 to −11pp; weaker signal, worth
  watching. `heterogeneous_decomposition + reframe_as_solvable_object` (47.4%, n=95) just misses
  the threshold.
- **Not the only source of risk.** Per-pattern failure modes in `ideation-patterns.md` operate a
  level below this; these are composition-level failures invisible in any single pattern.
- **Not a verdict.** A user whose problem genuinely matches one of these should proceed —
  informed about the failure mode, with the mitigation built in.
