# Attack catalog

The attacks a competent reviewer makes on an idea that has no results yet. Work the catalog
in order; record every attack that **lands**, with severity and answerability. An attack that
does not land is not worth reporting.

**Severity**
`fatal` — the idea does not survive it · `major` — survives but the paper is weakened
substantially · `minor` — a paragraph fixes it.

**Answerability**
`now` — the idea already contains the answer, it is just unstated · `with evidence` — an
experiment settles it · `unanswerable` — no experiment settles it, the framing must change.

---

## 1. Position attacks — is this worth doing?

**1.1 Soft bottleneck.** The problem is real but easy, or already handled by a method the
motivation does not mention. *Test: name the paper or practice that already handles it. If you
can, the attack lands.*

**1.2 Manufactured urgency.** The motivation asserts that a problem matters without anyone
suffering from it. *Test: name a specific person or system that pays a cost today. If the only
answer is "future systems might", it lands.*

**1.3 Solved at another level.** The problem disappears under a change that costs less than the
proposed method — more data, a bigger model, a different sampling rate. *This is the attack
that kills the most otherwise-clever ideas.*

**1.4 Scope inflation.** The framing claims a general phenomenon; the mechanism only addresses
one instance. *Test: does the method's applicability match the motivation's breadth?*

## 2. Mechanism attacks — is the method actually good?

**2.1 Equivalent to naive.** Stripped of vocabulary, the mechanism is what anyone would do.
*Construct the naive version independently — never use the author's, which is usually a
strawman — and diff.* `fatal` when it lands.

**2.2 The component that isn't load-bearing.** The mechanism has parts; removing one changes
nothing. *Test: for each component, what does its removal predict? A component with no
predicted loss is decoration and will be asked about.*

**2.3 Unstated precondition.** A step works only under a condition the idea never names, and
the condition is not generally true. *Test: read each step asking "when is this false?"*

**2.4 Circularity.** The mechanism needs, as input, something obtainable only after the problem
is solved. Common in self-supervision and evaluation designs.

**2.5 Complexity mistaken for depth.** The method is elaborate; the new *idea* inside it is
small. *Test: state the contribution in one sentence with no machinery. If nothing survives,
it lands.* (Michael Black: reviewers mistake technicality for novelty — so should the critic
not.)

**2.6 The borrowed tool.** The move is a textbook technique from a neighbouring field applied
directly. *This is not automatically fatal, but the paper must name the domain-specific
structure that made this instance unsolved, or state honestly that the contribution is
application-grade.*

## 3. Fit attacks — does the method address the stated problem?

**3.1 Adjacent problem.** The mechanism solves something near the stated gap. Very common when
the motivation was written after the method.

**3.2 Asserted bridge.** "Because our method does X, the gap closes." The step from X to
closure is claimed, not shown. *Test: is there a mechanism-level argument, or only a sentence?*

**3.3 Partial closure sold as full.** The method closes the gap in a regime that is not the
regime where the gap hurts.

## 4. Evidence attacks — could this be shown to be wrong?

**4.1 No falsifier.** No stated outcome would count against the claim.

**4.2 Tautological control.** The negative control intervenes on X and observes X. It tests a
definition, not a mechanism. *A real control lands on a downstream metric.*

**4.3 Invented numbers.** A threshold or gain figure appears with no derivation and no source.
*Every numeric bar needs `derived:` or `measured in <paper>`; anything else is fabrication and
the repair is to strike the number, not to soften the claim around it.*

**4.4 Unmeasurable claim.** The central assertion is about something no proposed instrument
measures.

**4.5 Success is guaranteed.** Every branch of the predicted outcome confirms the idea. *An
idea that cannot lose has not made a prediction.*

## 5. Positioning attacks — is the delta real?

**5.1 Delta by domain only.** The only difference from prior work is where it is applied.
Sometimes legitimate — but the paper must argue that the domain change is load-bearing, not
incidental.

**5.2 Delta by name.** The mechanism differs from prior work in vocabulary and not in
structure. *This is what the alias channel in `scoop-radar` exists to catch.*

**5.3 Superseded fix.** The proposed improvement was already implemented by an ancestor of the
method being improved. *The method-lineage tree catches this; a flat related-work list does
not.*

**5.4 Regression to a prior state.** The idea removes something a later method added for a
reason the idea does not address.

## 6. Ambition attacks — is this a paper?

**6.1 Fine but small.** Everything is correct and the contribution is one workshop-sized
increment. Say it plainly; this is the most commonly avoided honest verdict.

**6.2 Two half-papers.** The idea contains two contributions, neither developed enough.
*Symptom: the core claim will not fit in 25 words.*

**6.3 Infrastructure in a paper's clothing.** The work is a valuable system or dataset with a
research narrative bolted on. Not a criticism of the work — a criticism of the venue fit.

---

## Recording an attack

```
| # | Attack | Severity | Answerability | Where it lands |
|---|--------|----------|---------------|----------------|
| 2.1 | Stripped of the phase vocabulary, this is per-token advantage normalization | major | with evidence | core_mechanism, claim C0 |
```

**Do not fabricate attacks to fill the table.** An idea that survives the catalog with two
minor findings should be reported that way. Padding a critique with invented weaknesses is the
same failure as padding a paper with invented citations, and it teaches the author to
discount everything you say.
