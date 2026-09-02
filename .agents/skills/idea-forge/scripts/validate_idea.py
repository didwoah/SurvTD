#!/usr/bin/env python3
"""Deterministic gates for a candidate research idea (`candidate.json`).

These checks exist because every one of them is a place where a fluent model
silently degrades an idea: it softens the falsification it cannot meet, invents a
numeric bar to sound precise, writes a negative control that restates a
definition instead of testing a mechanism, or cites an ideation pattern it never
read.  Prose review does not reliably catch any of that; string comparison does.

    python3 validate_idea.py candidate.json
    python3 validate_idea.py candidate.json --baseline phase2/candidate.json

Exit 0 = every hard check passed (warnings may still be printed).
Exit 1 = at least one hard check failed.
Exit 2 = the file could not be read or parsed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# The 15 corpus-induced ideation patterns.  A candidate must name a pattern from
# this vocabulary; an invented pattern id means the generator never opened the card.
PATTERNS = {
    "assumption_audit_and_pivot",
    "architectural_operator_substitution",
    "generative_process_redesign",
    "controlled_diagnostic_design",
    "unify_into_shared_representation",
    "reframe_as_solvable_object",
    "self_supervised_signal_engineering",
    "structural_prior_encoding",
    "algebraic_equivalence_unification",
    "heterogeneous_decomposition",
    "decompose_and_delegate",
    "relax_discrete_search_to_continuous",
    "adapt_via_conditioning",
    "characterize_limit_then_surpass",
    "targeted_self_supervised_objective",
}

REQUIRED = [
    "title",
    "core_mechanism",
    "core_mechanism_steps",
    "gap_closure",
    "falsification_prediction",
    "load_bearing_variable",
    "negative_control",
    "compute_budget",
    "differentiation_from_lit",
    "signature_terms",
]

# A falsification has to predict a direction, otherwise it cannot be wrong.
# Stems carry their inflections explicitly: "the gap increases" must match just as
# "the gap will increase" does, and an earlier \b-anchored bare-stem list did not.
DIRECTION = re.compile(
    r"\b(?:"
    r"increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|widen(?:s|ed|ing)?|narrow(?:s|ed|ing|er)?|"
    r"shrink(?:s|ing)?|grow(?:s|ing)?|rise(?:s)?|rising|rose|fall(?:s|ing)?|fell|drop(?:s|ped|ping)?|"
    r"improv(?:e|es|ed|ing|ement)|degrad(?:e|es|ed|ing|ation)|collaps(?:e|es|ed|ing)|"
    r"recover(?:s|ed|ing|y)?|return(?:s|ed|ing)?|outperform(?:s|ed|ing)?|"
    r"higher|lower|larger|smaller|faster|slower|better|worse|flat|unchanged|no change"
    r")\b",
    re.I,
)
# A negative control has to intervene on something, not merely describe it.
CONTROL_VERB = re.compile(
    r"\b(permut\w*|shuffl\w*|randomi[sz]\w*|scrambl\w*|ablat\w*|remov\w*|replac\w*|"
    r"swap\w*|hold\w* (?:it |the \w+ )?(?:fixed|constant)|held (?:fixed|constant)|"
    r"disabl\w*|zero(?:ed|ing)? out|substitut\w*)\b",
    re.I,
)
# ...and it has to land somewhere observable.
OUTCOME = re.compile(
    r"\b(accuracy|error|loss|score|metric|performance|baseline|gain|f1|bleu|auc|"
    r"reward|success rate|win rate|perplexity|throughput|latency|regret|recall|precision)\b",
    re.I,
)
# Numeric outcome bars must carry provenance or they are fabrication.
NUMERIC_BAR = re.compile(
    r"(?:[<>≤≥]=?\s*\d+(?:\.\d+)?\s*%?|"
    r"\b\d+(?:\.\d+)?\s*(?:%|percentage points?|pp|points?|x|×)\b|"
    r"\bby (?:at least |more than |over )?\d+(?:\.\d+)?\b)"
)
PROVENANCE = re.compile(r"\b(derived:|measured in\b|reported in\b|from \w+ et al)", re.I)

# Method parameters must be named symbols with a default, not magic constants.
BARE_PARAM = re.compile(r"\b(?:top|first|last|every|each)\s+\d+\b|\b[<>≤≥]=?\s*\d+(?:\.\d+)?\b")


class Report:
    def __init__(self) -> None:
        self.failed = False

    def fail(self, check: str, detail: str) -> None:
        self.failed = True
        print(f"FAIL  {check}: {detail}")

    def warn(self, check: str, detail: str) -> None:
        print(f"WARN  {check}: {detail}")

    def ok(self, check: str, detail: str = "") -> None:
        print(f"ok    {check}{': ' + detail if detail else ''}")


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(t) > 2}


def check_required(c: dict, r: Report) -> None:
    missing = [f for f in REQUIRED if not c.get(f)]
    if missing:
        r.fail("required_fields", f"missing or empty: {', '.join(missing)}")
    else:
        r.ok("required_fields", f"{len(REQUIRED)} present")


def check_patterns(c: dict, r: Report) -> None:
    entries = c.get("gap_closure") or []
    if not isinstance(entries, list) or not entries:
        r.fail("pattern_vocabulary", "gap_closure is empty — no gap is being closed")
        return
    bad = []
    for i, e in enumerate(entries):
        pat = (e or {}).get("pattern", "")
        if pat not in PATTERNS:
            bad.append(f"gap_closure[{i}].pattern={pat!r}")
        if not (e or {}).get("how_closed"):
            bad.append(f"gap_closure[{i}].how_closed is empty")
    if bad:
        r.fail("pattern_vocabulary", "; ".join(bad))
    else:
        used = {e["pattern"] for e in entries}
        r.ok("pattern_vocabulary", f"{len(entries)} gap(s), {len(used)} distinct pattern(s)")
        if len(used) == 1 and not c.get("composition_note"):
            r.warn(
                "pattern_composition",
                "single-pattern candidate without a composition_note defending why one move suffices",
            )


def check_falsification(c: dict, r: Report) -> None:
    fp = c.get("falsification_prediction", "") or ""
    lbv = c.get("load_bearing_variable", "") or ""
    nc = c.get("negative_control", "") or ""

    if not DIRECTION.search(fp):
        r.fail(
            "falsification_direction",
            "no predicted direction — a prediction that names no direction cannot come out false",
        )
    else:
        r.ok("falsification_direction")

    if not lbv.strip():
        r.fail("load_bearing_variable", "not named")
    elif len(lbv.split()) > 40:
        r.warn("load_bearing_variable", "reads like a paragraph; it should name one quantity")
    else:
        r.ok("load_bearing_variable", lbv.strip()[:60])

    if not nc.strip():
        r.fail("negative_control", "absent")
        return
    problems = []
    if not CONTROL_VERB.search(nc):
        problems.append("names no intervention (permute / randomize / hold fixed / ablate ...)")
    if not OUTCOME.search(nc):
        problems.append("predicts no observable outcome — a control that changes nothing measurable tests nothing")
    # The classic self-confirming control: intervene on X so that X becomes zero.
    # That tests a definition, not a mechanism.  Require overlap with the variable
    # AND a downstream outcome, so "set C to 0, therefore C is 0" cannot pass.
    if lbv and not (tokens(lbv) & tokens(nc)):
        problems.append(f"does not touch the load-bearing variable ({lbv.strip()[:40]!r})")
    if problems:
        r.fail("negative_control", "; ".join(problems))
    else:
        r.ok("negative_control")


def check_numeric_provenance(c: dict, r: Report) -> None:
    fp = c.get("falsification_prediction", "") or ""
    bars = NUMERIC_BAR.findall(fp)
    if not bars:
        r.ok("numeric_provenance", "no numeric bar asserted")
        return
    if PROVENANCE.search(fp):
        r.ok("numeric_provenance", f"{len(bars)} bar(s), provenance marker present")
    else:
        r.fail(
            "numeric_provenance",
            f"numeric bar(s) {bars[:3]} with no 'derived:' or 'measured in <paper>' marker — "
            "an invented threshold is fabrication; strike the bar or source it",
        )


def check_parameters(c: dict, r: Report) -> None:
    steps = c.get("core_mechanism_steps") or []
    body = " ".join(steps) if isinstance(steps, list) else str(steps)
    hits = BARE_PARAM.findall(body)
    if hits:
        r.warn(
            "named_parameters",
            f"magic constant(s) {sorted(set(hits))[:4]} in the method steps — "
            "an unnamed quantity cannot be swept or graded; give it a symbol, a default, and a selection rule",
        )
    else:
        r.ok("named_parameters")


def check_differentiation(c: dict, r: Report) -> None:
    diffs = c.get("differentiation_from_lit") or []
    if not isinstance(diffs, list) or not diffs:
        r.fail("differentiation", "no delta against any retrieved paper")
        return
    thin = [
        i
        for i, d in enumerate(diffs)
        if not (d or {}).get("paper_id") or len((d or {}).get("delta", "")) < 25
    ]
    if thin:
        r.fail("differentiation", f"entries {thin} lack a paper_id or state only a token delta")
    else:
        r.ok("differentiation", f"{len(diffs)} grounded delta(s)")


def check_collision_terms(c: dict, r: Report) -> None:
    sig = c.get("signature_terms") or []
    alias = c.get("alias_terms") or []
    if len(sig) < 3:
        r.fail("signature_terms", f"{len(sig)} term(s); need >= 3 to drive the recent-collision channel")
    else:
        r.ok("signature_terms", f"{len(sig)} term(s)")
    if not alias:
        r.warn(
            "alias_terms",
            "empty — the renamed-ancestor blind spot is lexical, not temporal, "
            "so a wider date window cannot substitute for other communities' names for this mechanism",
        )
    else:
        r.ok("alias_terms", f"{len(alias)} term(s)")


def check_kill_switch(c: dict, baseline: dict | None, r: Report) -> None:
    if baseline is None:
        r.ok("kill_switch_integrity", "no baseline given — skipped")
        return
    drifted = [
        f
        for f in ("falsification_prediction", "compute_budget", "load_bearing_variable", "negative_control")
        if (c.get(f) or "") != (baseline.get(f) or "")
    ]
    if drifted:
        r.fail(
            "kill_switch_integrity",
            f"{', '.join(drifted)} changed since the baseline candidate — "
            "the fields that make the idea testable are locked; this is how a claim gets quietly softened",
        )
    else:
        r.ok("kill_switch_integrity", "byte-identical to baseline")


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", help="path to candidate.json")
    ap.add_argument("--baseline", help="earlier candidate.json to check kill-switch drift against")
    args = ap.parse_args()

    try:
        cand = load(args.candidate)
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot read {args.candidate}: {exc}", file=sys.stderr)
        return 2
    base = None
    if args.baseline:
        try:
            base = load(args.baseline)
        except Exception as exc:  # noqa: BLE001
            print(f"error: cannot read {args.baseline}: {exc}", file=sys.stderr)
            return 2

    r = Report()
    print(f"validate_idea: {args.candidate}\n")
    check_required(cand, r)
    check_patterns(cand, r)
    check_falsification(cand, r)
    check_numeric_provenance(cand, r)
    check_parameters(cand, r)
    check_differentiation(cand, r)
    check_collision_terms(cand, r)
    check_kill_switch(cand, base, r)

    print()
    if r.failed:
        print("VERDICT: fail — fix the FAIL lines above; do not edit a kill-switch field to make a check pass")
        return 1
    print("VERDICT: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
