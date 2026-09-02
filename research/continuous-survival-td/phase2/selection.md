# Phase 2.1 — Gap selection and pattern choice (attempt 2)

**Entry point**: cold start, resumed at Phase 2. `phase0/` and `phase1/` are carried over
unchanged; `attempt_1/` holds the superseded candidate and its audit.

## Anchor gap (subtractive, at the shared ancestor of TCSR / DeepTCSR)

Every temporal-consistency survival method inherits one assumption from `Maystre2022`:
the transition between consecutive observations is an **exogenous, uniform unit step**
(`Δt = 1`). Two consequences are load-bearing, not incidental:

1. surviving one step yields a **1-bit** non-event signal, so evidence cannot scale with how
   much time actually elapsed; and
2. propagating the next-step distribution backwards requires renormalising by the survival
   probability — the `÷ S(Δt)` update, which diverges exactly in the high-risk regime
   (`S → 0`) the method exists to detect.

## Sibling gap (additive, at the leaf)

Terminal Monte-Carlo NLL (`Lee2019`, `Bleistein2024`) has no inter-observation consistency at
all, and pays for it in gradient variance and alarm jitter along long trajectories.

## Patterns chosen

| # | Pattern | What it does here |
|---|---|---|
| 1 | `assumption_audit_and_pivot` | The unit-step assumption is audited and relaxed to a continuous elapsed duration `Δt_j`, which turns the 1-bit non-event signal into an accumulated multi-bucket survival factor `S(Δt_j)`. |
| 2 | `architectural_operator_substitution` | The `÷ S(Δt)` renormalisation is *replaced* by a rightward shift `Φ_{+Δt}` composed with a categorical projection `Π`, and the substitution is defended by the property it preserves: non-expansiveness in the Cramér metric and exact conservation of unit probability mass. |

**Composition note.** The two share one intermediate object — the inter-observation transition
operator. Relaxing the unit-step assumption is what makes the continuous shift necessary, and
the shift is unusable without a projection back onto the fixed bin grid; auditing the
assumption without substituting the operator leaves the division explosion in place, which is
the residue `DeepTCSR2024` inherited.

**Deliberate change from attempt 1.** Attempt 1 paired the audit with
`reframe_as_solvable_object` ("dynamic survival is SMDP policy evaluation"). The archived
theorist report attacked exactly the failure mode that pattern documents — *the reframing is a
relabelling; the hard part survives the translation*. Naming the move as an operator
substitution puts the burden where it belongs: on what the replacement provably preserves.

**Negative constraints carried from `attempt_1/phase3/` (audit as constraint, not as text to
copy):** no "informative observation process" claim the mechanism does not estimate; contraction
stated at per-iteration strength with `θ⁻` frozen, not as global convergence; the `λ` ablation
and the `Δt` ablation must be *separate* controls, not one confounded arm; calibration (IPCW
Brier / IBS) reported alongside discrimination.

**Anti-pattern check.** Neither pattern is `heterogeneous_decomposition`; none of the three
documented reject-favoured compositions applies.
