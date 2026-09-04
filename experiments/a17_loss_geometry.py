"""
A-17: which loss geometry does each term of the objective want? (decided-after-results)

The anchor-geometry diagnostic (`deviation_log.md` D-anchor) attributed +0.0998 of the
Cohort 1 deficit to loss geometry and -0.0026 to grid expansion. This measures the
geometry of both terms directly, on the SAME training path KC5 used, so the two
reference numbers below are comparable cell for cell.

alpha separates the terms with no new hyperparameter:
  * alpha = 1  ->  TD term inactive; the cell measures ANCHOR geometry alone
  * alpha = 0  ->  anchor inactive;   the cell measures TD geometry alone

This is deliberately not a weighted blend of the three losses. Their scales differ by
~10x, so blending would turn alpha into a scale knob rather than a convex weight, and
with the effect we are chasing (0.025) smaller than the seed noise (SD 0.04-0.10) a
blend's result could not be attributed to any term. Two axes, one change per cell.

PRE-DECLARED CRITERIA (fixed before running):
  * ANCHOR axis: a cell clears if mean C^td >= 0.60. That confirms the -0.0998
    attribution. Person-Period on the same raw irregular visits reached 0.6315, which
    is the ceiling this axis can reasonably reach; the Cramer anchor sits at 0.5318.
  * TD axis: absolute level will be low at alpha = 0 regardless, so the criteria are
    (a) improvement over the Cramer TD cell at 0.5353, and (b) reduction in the
    across-seed SD, which KC5 measured at 0.0841 against the anchor arm's 0.0407.
    The TD term multiplying variance by 4.3x while moving the mean +0.0036 is the
    finding this axis has to explain.
  * Anything that clears neither is reported as such. No cell is re-run with a
    different setting to make it clear.

Nothing here revives the preregistered outcome: KC3 and KC5 both fired and both stand.
Even a clean sweep of the anchor axis projects alpha = 1 to ~0.63, which is parity with
Person-Period (0.6367) and Dynamic-DeepHit (0.6461), not the +0.025 over them that C_3
requires.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.data.cohorts import COHORTS
from src.training.trainer import get_device
from experiments.run_track_b import train_and_eval_survtd

# Measured on the same training path; see experiments/results/kc5_anchor_equivalence/
REF = {"anchor_cramer": 0.5318, "td_cramer": 0.5353,
       "anchor_cramer_sd": 0.0407, "td_cramer_sd": 0.0841,
       "pp_raw_mle": 0.6315}
ANCHOR_GATE = 0.60

ARMS = [
    # (label, alpha, anchor_loss, td_loss)
    ("anchor/logit_cramer", 1.0, "logit_cramer", "cramer"),
    ("anchor/ce",           1.0, "ce",           "cramer"),
    ("td/logit_cramer",     0.0, "cramer",       "logit_cramer"),
    ("td/ce",               0.0, "cramer",       "ce"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 101112])
    ap.add_argument("--cohort", type=str, default="synthetic_icu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--output_dir", type=str, default="experiments/results/a17_loss_geometry")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    spec = COHORTS[args.cohort]
    raw = out / "a17_raw.json"
    rec = {label: {} for label, *_ in ARMS}

    def flush():
        raw.write_text(json.dumps({"cohort": args.cohort, "epochs": args.epochs,
                                   "seeds": args.seeds, "reference": REF,
                                   "arms": {l: {"alpha": a, "anchor_loss": al, "td_loss": tl}
                                            for l, a, al, tl in ARMS},
                                   "records": rec}, indent=2))

    print("=" * 80)
    print(f"  A-17 LOSS GEOMETRY  |  cohort={args.cohort}  device={device}  epochs={args.epochs}")
    print(f"  reference (same path): anchor/cramer {REF['anchor_cramer']:.4f}  "
          f"td/cramer {REF['td_cramer']:.4f}  PP-raw(MLE) {REF['pp_raw_mle']:.4f}")
    print("=" * 80)

    for seed in args.seeds:
        cd = spec.load(seed=seed)
        for label, alpha, aloss, tloss in ARMS:
            t0 = time.time()
            _, c, auc = train_and_eval_survtd(
                cd, spec, "full", alpha, args.epochs, args.batch_size, args.lr, device,
                seed=seed, anchor_loss=aloss, td_loss=tloss)
            rec[label][str(seed)] = {"c_td": c, "auc": auc,
                                     "wall_clock_s": round(time.time() - t0, 1)}
            print(f"  seed {seed:>7} | {label:<20} C_td {c:.4f}  AUC {auc:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            flush()

    print("-" * 80)
    summary = {}
    for label, alpha, aloss, tloss in ARMS:
        v = np.array([rec[label][s]["c_td"] for s in rec[label]
                      if not np.isnan(rec[label][s]["c_td"])])
        if v.size < 3:
            continue
        axis = "anchor" if alpha == 1.0 else "td"
        base, base_sd = REF[f"{axis}_cramer"], REF[f"{axis}_cramer_sd"]
        summary[label] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                          "n": int(v.size), "axis": axis,
                          "delta_vs_cramer": float(v.mean() - base),
                          "sd_ratio_vs_cramer": float(v.std(ddof=1) / base_sd)}
        gate = ("PASS" if v.mean() >= ANCHOR_GATE else "fail") if axis == "anchor" else (
            "improved" if v.mean() > base else "not improved")
        print(f"  {label:<20} {v.mean():.4f} +- {v.std(ddof=1):.4f}   "
              f"vs cramer {v.mean()-base:+.4f}   SD ratio {v.std(ddof=1)/base_sd:.2f}x   -> {gate}")
    print("=" * 80)
    (out / "a17_summary.json").write_text(json.dumps(
        {"anchor_gate": ANCHOR_GATE, "reference": REF, "summary": summary}, indent=2))
    print(f"Saved -> {out / 'a17_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
