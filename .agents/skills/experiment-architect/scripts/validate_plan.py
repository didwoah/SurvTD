#!/usr/bin/env python3
"""Bipartite and rigor gates for `evidence-plan.json` against `claim-tree.json`.

The central check is bipartite completeness, and it fails in *both* directions
on purpose:

  * a claim no experiment can falsify is an overclaim that nobody has noticed;
  * an experiment no claim depends on is page count.

Everything else here targets the weaknesses that dominate rejected ICLR/NeurIPS
submissions: undertuned baselines, missing ablations, absent negative controls,
and statistics decided after the numbers were seen.

    python3 validate_plan.py evidence-plan.json --tree claim-tree.json

Exit 0 = pass, 1 = at least one hard failure, 2 = unreadable input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

KINDS = {"main", "baseline", "ablation", "negative_control", "sensitivity", "scaling"}
MIN_SEEDS = 3
PREFERRED_SEEDS = 5

# "It improves results" is an aspiration.  A falsifier names an outcome that
# would count against the claim.
VACUOUS_FALSIFIER = re.compile(
    r"^\s*(it |the method |our method )?(does ?n[o']t |fails to |doesn't )?"
    r"(work|improve|help|perform|beat|win|show gains?)\b[\s.]*$",
    re.I,
)


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


def live_claim_ids(tree: dict) -> list[str]:
    core = tree.get("core_claim") or {}
    claims = ([core] if core.get("id") else []) + list(tree.get("sub_claims") or [])
    return [c["id"] for c in claims if c.get("status") != "withdrawn"]


def check_bipartite(plan: dict, tree: dict, r: Report) -> None:
    claim_ids = live_claim_ids(tree)
    exps = plan.get("experiments") or []
    if not exps:
        r.fail("bipartite", "no experiments in the plan")
        return

    covered: set[str] = set()
    for e in exps:
        eid = e.get("id", "?")
        tested = e.get("tests_claims") or []
        if not tested:
            r.fail(
                "orphan_experiment",
                f"{eid} tests no claim — an experiment that no claim depends on is page count, not evidence",
            )
            continue
        unknown = [c for c in tested if c not in claim_ids]
        if unknown:
            r.fail("dangling_claim_ref", f"{eid} references claim id(s) {unknown} absent from the tree")
        covered.update(c for c in tested if c in claim_ids)

    orphans = [c for c in claim_ids if c not in covered]
    if orphans:
        r.fail(
            "orphan_claim",
            f"claim(s) {orphans} have no experiment that could falsify them — "
            "either plan one or cut the claim; this is what an overclaim looks like before submission",
        )
    else:
        r.ok("bipartite", f"{len(claim_ids)} claim(s) ↔ {len(exps)} experiment(s), both directions covered")


def check_kinds_and_falsifiers(plan: dict, r: Report) -> None:
    exps = plan.get("experiments") or []
    ids = [e.get("id", f"[{i}]") for i, e in enumerate(exps)]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        r.fail("experiment_id_unique", f"duplicate ids: {sorted(dupes)}")

    for e in exps:
        eid = e.get("id", "?")
        kind = e.get("kind")
        if kind not in KINDS:
            r.fail("experiment_kind", f"{eid}: kind {kind!r} not in {sorted(KINDS)}")
        fals = (e.get("falsifies_if") or "").strip()
        if not fals:
            r.fail("falsifies_if", f"{eid}: absent — the experiment cannot come out against the claim")
        elif VACUOUS_FALSIFIER.match(fals) or len(fals.split()) < 4:
            r.fail(
                "falsifies_if",
                f"{eid}: {fals!r} is vacuous — name the concrete outcome that would count against the claim",
            )
    if exps and not r.failed:
        r.ok("falsifies_if", f"{len(exps)} experiment(s) carry a concrete falsifier")


def check_controls(plan: dict, tree: dict, r: Report) -> None:
    exps = plan.get("experiments") or []
    core_id = (tree.get("core_claim") or {}).get("id", "C0")

    ablations = [e for e in exps if e.get("kind") == "ablation" and core_id in (e.get("tests_claims") or [])]
    if not ablations:
        r.fail(
            "mechanism_isolating_ablation",
            f"no ablation tests the core claim {core_id} — without the experiment that removes exactly the "
            "claimed causal element, the contribution cannot be attributed to the mechanism",
        )
    else:
        r.ok("mechanism_isolating_ablation", f"{len(ablations)} ablation(s) on {core_id}")

    controls = [e for e in exps if e.get("kind") == "negative_control"]
    if not controls:
        r.fail(
            "negative_control",
            "no negative_control experiment — the intervention that should NOT work is what separates "
            "the mechanism from a confound",
        )
    else:
        r.ok("negative_control", f"{len(controls)} control(s)")

    baselines = [e for e in exps if e.get("kind") == "baseline"]
    if not baselines:
        r.warn("baseline_ladder", "no experiment tagged kind=baseline; confirm baselines live inside a main run")
    else:
        r.ok("baseline_ladder", f"{len(baselines)} baseline experiment(s)")


def check_statistics(plan: dict, r: Report) -> None:
    st = plan.get("statistics") or {}
    seeds = st.get("seeds")
    if not isinstance(seeds, int):
        r.fail("statistics_seeds", "statistics.seeds is not declared as an integer")
    elif seeds < MIN_SEEDS:
        r.fail("statistics_seeds", f"{seeds} seed(s) — below the {MIN_SEEDS} needed to say anything about spread")
    elif seeds < PREFERRED_SEEDS:
        r.warn("statistics_seeds", f"{seeds} seeds; {PREFERRED_SEEDS} is the defensible default")
    else:
        r.ok("statistics_seeds", f"{seeds} seeds")

    if not (st.get("variance_reported") or "").strip():
        r.fail("variance_reported", "not declared — a table of bare means cannot support a comparative claim")
    else:
        r.ok("variance_reported", st["variance_reported"][:60])

    if not (st.get("meaningful_delta") or "").strip():
        r.fail(
            "meaningful_delta",
            "not declared — deciding what counts as a real difference after seeing results is how "
            "noise becomes a contribution",
        )
    else:
        r.ok("meaningful_delta", str(st["meaningful_delta"])[:60])

    if st.get("declared_before_results") is not True:
        r.fail(
            "preregistration",
            "statistics.declared_before_results is not true — this plan cannot back a confirmatory claim",
        )
    else:
        r.ok("preregistration")

    parity = (plan.get("tuning_parity") or "").strip()
    if not parity:
        r.fail(
            "tuning_parity",
            "absent — 'you undertuned the baseline' is the most common reviewer attack on an empirical "
            "paper, and it is unanswerable after the runs are done",
        )
    elif len(parity.split()) < 6:
        r.warn("tuning_parity", "declared but thin; state the budget, the range, and the data each baseline got")
    else:
        r.ok("tuning_parity")


def check_budget(plan: dict, r: Report) -> None:
    exps = plan.get("experiments") or []
    if len(exps) > 1:
        missing = [e.get("id", "?") for e in exps if e.get("cut_order") is None]
        if missing:
            r.warn(
                "cut_order",
                f"{missing} have no cut_order — decide the drop order now, while the decision is still honest",
            )
        else:
            r.ok("cut_order", "drop order declared for every experiment")
    total = sum(e.get("cost_gpu_hours") or 0 for e in exps)
    if total:
        r.ok("cost_model", f"{total:g} GPU-hours planned across {len(exps)} experiment(s)")
    else:
        r.warn("cost_model", "no cost_gpu_hours declared; the plan cannot be checked against a budget")


def check_kill_criteria(plan: dict, tree: dict, r: Report) -> None:
    core_id = (tree.get("core_claim") or {}).get("id", "C0")
    kc = plan.get("kill_criteria") or []
    if not any(k.get("claim") == core_id and (k.get("abandon_if") or "").strip() for k in kc):
        r.fail(
            "kill_criteria",
            f"no abandon condition for the core claim {core_id} — a plan with no way to lose is not a test",
        )
    else:
        r.ok("kill_criteria", f"{len(kc)} criterion/criteria")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", help="path to evidence-plan.json")
    ap.add_argument("--tree", required=True, help="path to claim-tree.json")
    args = ap.parse_args()
    try:
        with open(args.plan, encoding="utf-8") as fh:
            plan = json.load(fh)
        with open(args.tree, encoding="utf-8") as fh:
            tree = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2

    r = Report()
    print(f"validate_plan: {args.plan} against {args.tree}\n")
    check_bipartite(plan, tree, r)
    check_kinds_and_falsifiers(plan, r)
    check_controls(plan, tree, r)
    check_statistics(plan, r)
    check_budget(plan, r)
    check_kill_criteria(plan, tree, r)

    print()
    if r.failed:
        print("VERDICT: fail")
        return 1
    print("VERDICT: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
