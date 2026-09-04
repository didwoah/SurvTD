"""
Render `tier1_results.json` as Table 1: rows = arms, columns = cohort x landmark.

    python3 experiments/render_tier1_table.py [--metric c_td] [--json <path>]

Reporting rules this enforces rather than leaves to the writer:

* **Per landmark, never only the grid mean.** On C-MAPSS the mean is inflated by late
  landmarks where few units remain and the residual times are short.
* **`mean +- sd` over seeds**, with the seed count shown, because a cell backed by two
  surviving seeds is not the same evidence as one backed by five.
* **A below-chance `C^td` is printed as `!0.xxxx`, not as a score.** It is a defect
  alarm (D15, D16); the marker exists so it cannot be copied into a paper as a result.
* **Structurally undefined cells print `n/a` with a footnote**, never 0.500 — CoxSig at
  `L = 0` is undefined, not chance-level (D17).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARM_LABEL = {
    "km": "KM marginal (leak gate)",
    "landmark_cox": "Landmark Cox (van Houwelingen 2007)",
    "survtd": "SurvTD (ours)",
    "survtd_anchor_only": "SurvTD, alpha=1 anchor-only (ours)",
    "ddh": "Dynamic-DeepHit (Lee+ 2020)",
    "tcsr": "TCSR (Maystre & Russo 2022)",
    "tcsr_landmark": "TCSR, landmark arm",
    "coxsig": "CoxSig (Bleistein+ 2024)",
    "ncde": "NCDE (Kidger+ 2020)",
    "deeptcsr_tcn": "DeepTCSR tcn (Vargas Vieyra & Frossard 2024)",
}
ORDER = list(ARM_LABEL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="experiments/results/tier1/tier1_results.json")
    ap.add_argument("--metric", default="c_td", choices=("c_td", "ibs", "auc", "c_ipcw"))
    args = ap.parse_args()

    path = args.json if os.path.isabs(args.json) else os.path.join(ROOT, args.json)
    with open(path) as f:
        blob = json.load(f)

    # (cohort, landmark) -> arm -> [values over seeds]
    cells = defaultdict(lambda: defaultdict(list))
    undefined = defaultdict(lambda: defaultdict(list))
    errors = defaultdict(list)
    for key, entry in blob["results"].items():
        cohort, arm, seed = key.split("|")
        if "metrics" not in entry:
            errors[(cohort, arm)].append(f"{seed}: {entry.get('error', '?')}")
            continue
        for lm, m in entry["metrics"].items():
            landmark = lm.split(",")[0]
            v = m.get(args.metric)
            if v is None or not np.isfinite(v):
                undefined[(cohort, landmark)][arm].append(m.get("reason", "degenerate"))
            else:
                cells[(cohort, landmark)][arm].append(float(v))

    columns = sorted(cells, key=lambda c: (c[0], float(c[1].split("=")[1])))
    arms = [a for a in ORDER if any(a in cells[c] or a in undefined[c] for c in columns)]
    arms += sorted({a for c in columns for a in cells[c]} - set(arms))

    width = max(len(ARM_LABEL.get(a, a)) for a in arms) + 2
    head = f"{'arm':<{width}}" + "".join(f"{c[0][:6]} {c[1]:>10}  " for c in columns)
    print(f"\nTable 1 -- {args.metric}   (mean +- sd over seeds; n in brackets)")
    print(head)
    print("-" * len(head))

    footnotes = []
    for arm in arms:
        row = f"{ARM_LABEL.get(arm, arm):<{width}}"
        for c in columns:
            vals = cells[c].get(arm, [])
            if not vals:
                reasons = undefined[c].get(arm, [])
                row += f"{'n/a':>18}  "
                if reasons:
                    footnotes.append(f"  {arm} @ {c[0]} {c[1]}: {reasons[0][:110]}")
                continue
            v = np.asarray(vals)
            flag = "!" if args.metric == "c_td" and v.mean() < 0.5 else " "
            row += f"{flag}{v.mean():.4f}+-{v.std():.4f}[{len(v)}]  "
        print(row)

    if args.metric == "c_td":
        print("\n'!' marks a BELOW-CHANCE cell: a defect alarm, not a result (D15, D16).")
    if footnotes:
        print("\nUndefined / degenerate cells:")
        for f in dict.fromkeys(footnotes):
            print(f)
    if errors:
        print("\nCells that errored:")
        for (cohort, arm), msgs in errors.items():
            print(f"  {cohort}|{arm}: {len(msgs)} seed(s) -- {msgs[0][:120]}")

    alarms = blob.get("below_chance_alarms", [])
    if alarms:
        print(f"\n{len(alarms)} below-chance (arm, seed, landmark) cells recorded in the run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
