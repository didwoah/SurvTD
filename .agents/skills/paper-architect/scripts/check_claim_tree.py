#!/usr/bin/env python3
"""Structural gates for `claim-tree.json` — the spine every other skill addresses.

The invariants here are not style preferences.  Each one corresponds to a way
papers fail in review:

  * a core claim that will not fit in 25 words is two claims wearing one coat,
    and reviewers will attack the weaker half;
  * a sub-claim bound to two anchors is a claim whose evidence is spread thin
    enough that neither anchor establishes it;
  * a sub-claim bound to nothing is an overclaim that has not been noticed yet.

    python3 check_claim_tree.py claim-tree.json

Exit 0 = pass, 1 = at least one hard failure, 2 = unreadable input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

ANCHOR = re.compile(r"^(fig|tab|thm|app|eq):[A-Za-z0-9_.-]+$")
CLAIM_ID = re.compile(r"^C\d+$")
STRENGTHS = ["universal", "typical", "observed", "exploratory"]
STATUSES = {"planned", "supported", "conditional", "unsupported", "contradicted", "withdrawn"}
CORE_WORD_LIMIT = 25

# "We propose X" describes an activity, not a finding.  A contribution bullet has
# to be something the world could contradict.
UNFALSIFIABLE_OPENER = re.compile(
    r"^\s*(we (propose|present|introduce|develop|design|build|study|explore|investigate)|"
    r"this paper (proposes|presents|introduces))\b",
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


def anchors_of(claim: dict) -> list[str]:
    """Normalize the anchor field so a list, a comma list, and 'A and B' all read alike."""
    raw = claim.get("evidence_anchor")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    parts = re.split(r"\s*(?:,|;|\band\b|\+)\s*", str(raw).strip())
    return [p for p in parts if p]


def check_core(tree: dict, r: Report) -> None:
    core = tree.get("core_claim") or {}
    text = (core.get("text") or "").strip()
    if not text:
        r.fail("core_claim", "absent — the paper has no stated claim")
        return
    n = len(text.split())
    if n > CORE_WORD_LIMIT:
        r.fail(
            "core_claim_length",
            f"{n} words (limit {CORE_WORD_LIMIT}) — a core claim that will not fit is more than one claim; "
            "split it and promote the second half to a sub-claim",
        )
    else:
        r.ok("core_claim_length", f"{n} words")
    if core.get("id") != "C0":
        r.fail("core_claim_id", f"core claim id is {core.get('id')!r}, must be 'C0'")
    else:
        r.ok("core_claim_id")


def check_claims(tree: dict, r: Report) -> None:
    core = tree.get("core_claim") or {}
    subs = tree.get("sub_claims") or []
    everything = ([core] if core else []) + list(subs)

    ids: list[str] = []
    for i, cl in enumerate(everything):
        cid = cl.get("id", "")
        label = cid or f"sub_claims[{i - 1}]"
        if not CLAIM_ID.match(cid):
            r.fail("claim_id_format", f"{label}: id {cid!r} is not C<number>")
        ids.append(cid)

        # anchor: exactly one, well formed
        anc = anchors_of(cl)
        if not anc:
            if cl.get("status") != "withdrawn":
                r.fail(
                    "anchor_binding",
                    f"{label} is bound to no evidence anchor — cut the claim or plan an experiment for it",
                )
        elif len(anc) > 1:
            r.fail(
                "anchor_binding",
                f"{label} is bound to {len(anc)} anchors ({', '.join(anc)}) — "
                "a claim needing two anchors is two claims; split it",
            )
        elif not ANCHOR.match(anc[0]):
            r.fail("anchor_format", f"{label}: anchor {anc[0]!r} is not fig:/tab:/thm:/app:/eq:<id>")

        st = cl.get("strength")
        if st not in STRENGTHS:
            r.fail("strength", f"{label}: strength {st!r} not in {STRENGTHS}")
        stat = cl.get("status")
        if stat not in STATUSES:
            r.fail("status", f"{label}: status {stat!r} not in {sorted(STATUSES)}")

        if cl is not core:
            sup = cl.get("supports")
            if not sup:
                r.fail("supports", f"{label} supports nothing — a sub-claim that serves no claim is filler")
            elif sup not in [c.get("id") for c in everything]:
                r.fail("supports", f"{label} supports {sup!r}, which is not a claim in this tree")

    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        r.fail("claim_id_unique", f"duplicate ids: {sorted(dupes)}")
    elif ids:
        r.ok("claim_ids", f"{len(ids)} claim(s): {', '.join(ids)}")

    # A sub-claim must not outrank the claim it serves.
    by_id = {c.get("id"): c for c in everything}
    for cl in subs:
        sup = by_id.get(cl.get("supports"))
        if not sup or cl.get("strength") not in STRENGTHS or sup.get("strength") not in STRENGTHS:
            continue
        if STRENGTHS.index(cl["strength"]) < STRENGTHS.index(sup["strength"]):
            r.warn(
                "strength_monotonicity",
                f"{cl.get('id')} is stated at '{cl['strength']}' while the claim it supports "
                f"({sup.get('id')}) is only '{sup['strength']}' — the support cannot be stronger than the conclusion",
            )

    # Shared anchors are legal (one table can carry two claims) but worth surfacing.
    seen: dict[str, list[str]] = {}
    for cl in everything:
        for a in anchors_of(cl):
            seen.setdefault(a, []).append(cl.get("id", "?"))
    for anchor, owners in seen.items():
        if len(owners) > 1:
            r.warn("anchor_sharing", f"{anchor} carries {len(owners)} claims ({', '.join(owners)})")


def check_kill_switch(tree: dict, r: Report) -> None:
    ks = tree.get("kill_switch") or {}
    missing = [
        f
        for f in ("falsification_prediction", "load_bearing_variable", "negative_control", "compute_budget")
        if not (ks.get(f) or "").strip()
    ]
    if missing:
        r.fail("kill_switch", f"missing: {', '.join(missing)}")
    else:
        r.ok("kill_switch", "all four fields present")


def check_contributions(tree: dict, r: Report) -> None:
    bullets = tree.get("contributions") or []
    if not bullets:
        r.ok("contributions", "none declared — skipped")
        return
    bad = [b for b in bullets if UNFALSIFIABLE_OPENER.match(str(b))]
    if bad:
        r.fail(
            "contribution_falsifiability",
            f"{len(bad)} bullet(s) state an activity rather than a finding, e.g. {bad[0][:70]!r} — "
            "'X reduces Y by Z under condition W' can be wrong; 'we propose X' cannot",
        )
    else:
        r.ok("contribution_falsifiability", f"{len(bullets)} bullet(s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", help="path to claim-tree.json")
    args = ap.parse_args()
    try:
        with open(args.tree, encoding="utf-8") as fh:
            tree = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot read {args.tree}: {exc}", file=sys.stderr)
        return 2

    r = Report()
    print(f"check_claim_tree: {args.tree}\n")
    check_core(tree, r)
    check_claims(tree, r)
    check_kill_switch(tree, r)
    check_contributions(tree, r)

    print()
    if r.failed:
        print("VERDICT: fail")
        return 1
    print("VERDICT: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
