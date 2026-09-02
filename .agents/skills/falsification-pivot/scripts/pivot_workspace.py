#!/usr/bin/env python3
"""Deterministic workspace archiver and constraint extractor for falsification-pivot.

Usage:
    python3 pivot_workspace.py --workspace <path_to_project_dir> [--audit <path_to_audit.json>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path


def find_next_attempt_dir(workspace: Path) -> Path:
    existing = []
    for item in workspace.iterdir():
        if item.is_dir() and re.match(r"^attempt_(\d+)$", item.name):
            num = int(re.match(r"^attempt_(\d+)$", item.name).group(1))
            existing.append(num)
    next_num = (max(existing) + 1) if existing else 1
    return workspace / f"attempt_{next_num}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Path to project run directory")
    parser.add_argument("--audit", help="Path to evidence-audit.json (if available)")
    parser.add_argument("--diagnosis", help="Manual root cause diagnosis string")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    if not workspace.exists() or not workspace.is_dir():
        print(f"error: workspace {workspace} does not exist", file=sys.stderr)
        return 2

    attempt_dir = find_next_attempt_dir(workspace)
    print(f"[falsification-pivot] Creating archive: {attempt_dir.name}")
    attempt_dir.mkdir(parents=True, exist_ok=True)

    # Artifacts to move into attempt_N
    items_to_move = [
        "claim-tree.json",
        "evidence-plan.json",
        "preregistration.md",
        "abstract.md",
        "outline.md",
        "idea-card.md",
        "figure1_spec.json",
        "paper_experiments.md",
        "adversarial_stress_tests.md",
        "results",
        "phase2",
        "phase3",
    ]

    for item_name in items_to_move:
        src = workspace / item_name
        dst = attempt_dir / item_name
        if src.exists():
            if src.is_dir():
                shutil.move(str(src), str(dst))
            else:
                shutil.move(str(src), str(dst))
            print(f"  archived: {item_name} -> {attempt_dir.name}/{item_name}")

    # Extract or synthesize negative constraints
    audit_data = {}
    if args.audit and Path(args.audit).exists():
        with open(args.audit, encoding="utf-8") as fh:
            audit_data = json.load(fh)

    constraints = {
        "failed_attempt": attempt_dir.name,
        "falsified_claim": audit_data.get("falsified_claim", "C0"),
        "fatal_evidence": audit_data.get("fatal_evidence", "Kill criterion triggered during empirical evaluation"),
        "root_cause_diagnosis": args.diagnosis or audit_data.get("diagnosis", "Unstated confound or operator divergence identified"),
        "hard_negative_constraints": audit_data.get("negative_constraints", [
            "The next candidate must not reuse the operator that failed in this attempt",
            "The new mechanism must explicitly demonstrate immunity to the fatal evidence anchor"
        ])
    }

    constraints_path = attempt_dir / "negative_constraints.json"
    with open(constraints_path, "w", encoding="utf-8") as fh:
        json.dump(constraints, fh, indent=2)
    print(f"  created: {attempt_dir.name}/negative_constraints.json")

    # Reset clean phase2/ for the new iteration
    new_phase2 = workspace / "phase2"
    new_phase2.mkdir(exist_ok=True)
    readme_path = new_phase2 / "PIVOT_ACTIVE.md"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"# Active Pivot from {attempt_dir.name}\n\n"
            f"This workspace has been pivoted following falsification in `{attempt_dir.name}`.\n"
            f"Please read `{attempt_dir.name}/negative_constraints.json` before proposing `candidate.json`.\n"
            f"Phase 0 and Phase 1 context are preserved in `{workspace.name}/phase0` and `phase1`.\n"
        )
    print(f"  initialized clean workspace: {new_phase2.name}/ with PIVOT_ACTIVE.md")
    print(f"\n[falsification-pivot] SUCCESS: Run successfully pivoted to {attempt_dir.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
