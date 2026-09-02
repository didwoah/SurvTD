# The 15 ideation patterns

A vocabulary of research *moves*, induced from 1,947 ICLR / ICML / NeurIPS papers
(2021–2025) by Zhao et al., *ResearchStudio-Idea* (arXiv:2607.04439, MIT). Restated here
for use inside this skill; see `NOTICE.md` for attribution.

**How to use them.** These are diagnostic vocabulary, not labels a candidate must classify
into and not a menu ranked by acceptance rate. Phase 2.1 asks: *given the shape of this gap,
which operational signature structurally closes it?* Choosing a pattern because it appears
often in the corpus is exactly the failure mode these cards are meant to prevent — the
patterns are not mutually exclusive, most real papers compose two, and a candidate never
defends itself by naming one.

`n` is the count of corpus papers whose primary move was this pattern. It is reported for
transparency and must not be read as a prior on success.

---

### 1. Audit and Pivot an Assumption · `assumption_audit_and_pivot` · n=181

**Definition.** Locate the load-bearing implicit assumption a result, guarantee, or defense
rests on, then pivot on it: relax it to a weaker condition and re-prove, or violate it with a
constructed counterexample that breaks the system or unlocks new behavior.

**Signature.** identify the implicit assumption → relax it (weaker condition) or violate it
(counterexample / exploit) → re-derive the guarantee or demonstrate the new behavior

**When.** A result's strength or a system's safety hinges on an assumption real settings can
weaken, or an adversary can violate.

**Failure.** Pivoting on an assumption whose removal leaves the achievable outcome unchanged.
Relaxing a non-binding assumption yields an incremental paper; this is the single most common
rejection under this pattern. The assumption must be *load-bearing*: the existing result must
hold **because** of it.

---

### 2. Substitute the Operator or Representation · `architectural_operator_substitution` · n=109

**Definition.** Replace or relocate a costly operator, primitive, or intermediate
representation with a cheaper surrogate that provably preserves the essential property
(expressivity, sensitivity bound, curvature spectrum), breaking a complexity bottleneck.

**Signature.** identify an expensive operator → substitute a cheaper surrogate → prove it
preserves what matters

**When.** The cost bottleneck comes from an operator that can be approximated without losing
the property the system depends on.

**Failure.** Substituting without characterizing what is lost. "Comparable performance" is not
a preservation proof.

---

### 3. Liberate a Fixed Generative Component · `generative_process_redesign` · n=94

**Definition.** Recognize a conventionally-fixed component of an iterative or staged generative
procedure — the uninformative prior, fixed endpoints, unimodal step distribution, latent space,
conditioning granularity — as a free design variable, and redesign it.

**Signature.** find the component everyone inherits as fixed → treat it as free → redesign for
quality or efficiency

**When.** A pipeline carries a default that was never the actual constraint.

**Failure.** Freeing a component that was fixed for a reason nobody restates, then discovering
the reason in review.

---

### 4. Design a Confound-Isolating Diagnostic · `controlled_diagnostic_design` · n=86

**Definition.** Build an evaluation instrument that holds confounds fixed (source capability,
retrieval shortcuts, surface form) or systematically varies a hidden axis, so the measurement
reflects the property rather than an artifact.

**Signature.** name the confound → construct the controlled instrument → show the previously
reported effect changes under control

**When.** A widely reported result may be an artifact of how it was measured.

**Failure.** A diagnostic that isolates a confound nobody believed mattered. The payoff is
proportional to how much the field currently trusts the contaminated measurement.

---

### 5. Unify Heterogeneous Inputs into One Space · `unify_into_shared_representation` · n=82

**Definition.** Map modalities, tasks, or data types that are handled separately into one
representation where a single mechanism operates over all of them.

**Signature.** identify separately-handled input classes → construct a shared space → show one
mechanism now covers all

**When.** Parallel pipelines exist mainly for historical reasons and the sharing would transfer
signal.

**Failure.** Unification that averages away the distinctions the separate pipelines existed to
preserve. Show per-class performance, not only the aggregate.

---

### 6. Reframe as a Solvable Object · `reframe_as_solvable_object` · n=79

**Definition.** Recast an open, ill-posed problem as an instance of a class that already has
machinery — an optimization problem, a game, an inference problem, a known algebraic object.

**Signature.** restate the unsolved problem as a member of a solved class → import the
machinery → obtain the guarantee the original framing could not express

**When.** The obstacle is formulation, not capability.

**Failure.** The reframing is a relabeling: the hard part survives the translation intact, now
wearing new notation.

---

### 7. Manufacture the Supervisory Signal · `self_supervised_signal_engineering` · n=66

**Definition.** Construct supervision from structure already present in the data or the
process, where explicit labels are unavailable or unaffordable.

**Signature.** locate exploitable structure → derive a training signal from it → show it
substitutes for the missing supervision

**When.** The bottleneck is label cost or label impossibility, and the data has usable
structure.

**Failure.** Two un-derived choices stacked: an invented grouping plus an invented signal for
each group. Neither is testable without the other. See `anti-patterns.md`.

---

### 8. Encode Structure by Construction · `structural_prior_encoding` · n=61

**Definition.** Build a known invariance, symmetry, or constraint into the architecture so it
holds by construction rather than being learned or penalized.

**Signature.** name the property → encode it structurally → show it now holds exactly, and
what that buys

**When.** A property is currently learned approximately and its violations are costly.

**Failure.** Proposing the topology as prior without demonstrating it does identifiable work
over a standard alternative. This is the corpus's most reject-enriched sub-pattern.

---

### 9. Prove Equivalence to Unify · `algebraic_equivalence_unification` · n=59

**Definition.** Show that two methods believed distinct are the same object under a
transformation, then use the equivalence to transfer results or expose a shared limitation.

**Signature.** construct the mapping → prove equivalence → transfer a guarantee, or derive a
limitation both inherit

**When.** A field maintains parallel literatures with suspicious structural similarity.

**Failure.** An equivalence that holds only under conditions neither method actually operates in.

---

### 10. Decompose for Differentiated Treatment · `heterogeneous_decomposition` · n=47

**Definition.** Split a population treated uniformly into sub-populations with genuinely
different needs, and treat each accordingly.

**Signature.** show the population is heterogeneous in a way that matters → derive the split →
treat differentially → show uniform treatment was leaving value behind

**When.** Aggregate performance hides sub-populations that fail for distinct reasons.

**Failure.** The split criterion is intuition rather than observed structure. Appears in all
three documented reject-favored compositions — read `anti-patterns.md` before composing with it.

---

### 11. Decompose and Delegate to Solvers · `decompose_and_delegate` · n=42

**Definition.** Break a problem into sub-problems each of which has a mature dedicated solver,
and own the decomposition and the interfaces.

**Signature.** find the decomposition → delegate each part → show the composition beats the
monolith

**When.** A monolithic approach underuses solvers that are already excellent at the parts.

**Failure.** The contribution is plumbing. The decomposition itself must be non-obvious and the
interfaces must be where the difficulty lives.

---

### 12. Relax Discrete Search to Continuous · `relax_discrete_search_to_continuous` · n=35

**Definition.** Replace a combinatorial search with a continuous relaxation that gradient
methods can traverse, with a principled route back to discrete solutions.

**Signature.** formulate the relaxation → optimize continuously → recover a discrete solution →
bound the relaxation gap

**When.** The discrete formulation is the binding cost and the objective admits a meaningful
continuous extension.

**Failure.** No discretization guarantee. A relaxation whose rounding is unanalyzed has moved
the difficulty, not resolved it.

---

### 13. Adapt by Conditioning, Not Retraining · `adapt_via_conditioning` · n=18

**Definition.** Obtain new behavior by conditioning a fixed model — context, prompts, adapters,
steering — instead of updating weights.

**Signature.** identify the target behavior → find the conditioning channel → show it matches
or beats retraining at a fraction of the cost

**When.** Retraining is unaffordable, unsafe, or destroys capabilities you need.

**Failure.** Low corpus support; treat its statistics as thin. The usual weakness is not
establishing that retraining was ever the right comparison.

---

### 14. Characterize a Limit, Then Surpass It · `characterize_limit_then_surpass` · n=15

**Definition.** Prove precisely what current approaches cannot do, then construct a method
outside that characterization.

**Signature.** prove the limitation → show it is tight → construct the method that escapes the
premise → demonstrate the escape

**When.** A performance ceiling is treated as empirical when it is structural.

**Failure.** The characterization is a lower bound on a strawman formalization, and the escape
exploits the formalization rather than the real limit. Note: this pattern has the cleanest
cross-domain Oral signal in the corpus, and also the thinnest sample.

---

### 15. Design a Property-Targeting Pretext Objective · `targeted_self_supervised_objective` · n=15

**Definition.** Design a pretext task whose optimum provably requires the specific property you
want the representation to have.

**Signature.** name the property → construct the objective whose solution requires it → show
the learned representation has it

**When.** Generic pretraining yields representations that lack a property you need.

**Failure.** The pretext task's optimum does not actually require the property; a shortcut
solution exists and the model finds it.

---

## Composition

Most corpus papers execute **two** patterns, not one. Prefer a composition where the second
pattern earns its place by the removal test: *the anchor's story is incomplete without it.*
Common productive pairs share an intermediate object — e.g. `reframe_as_solvable_object` +
`assumption_audit_and_pivot` (the reframing is what makes the assumption visible), or
`characterize_limit_then_surpass` + `architectural_operator_substitution` (the limit proof
tells you which operator to replace).

Single-pattern candidates are legitimate and common; they require a written
`composition_note` defending why one move suffices. Three or more patterns usually means the
paper has not decided what it is.
