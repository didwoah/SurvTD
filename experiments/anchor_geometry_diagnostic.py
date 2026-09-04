"""
Diagnostic: why does SurvTD's anchor-only arm trail Person-Period by ~0.10?

The two arms share a backbone (`build_backbone("grud", d, 64, 2, 0.1)`), a head
(`DiscreteHazardHead`), a bin convention and a prediction target (residual time
R_j = tte - t_j), and NEITHER uses the TD term. Yet:

    SurvTD anchor-only (alpha = 1)   0.5318 +- 0.0407     (KC5, 5 seeds)
    Person-Period                    0.6367 +- 0.0759     (primary run, 5 seeds)

Exactly two differences remain:

  (a) LOSS GEOMETRY. The anchor is a squared Cramer / censored-CRPS on the CDF,
      `delta_s * sum_k (F(k) - 1{k >= k_j})^2`. Person-Period is a masked binary
      cross-entropy on the per-bin hazards. Both are proper, but squared Cramer is
      L2 on the CDF -- its gradient is linear in the residual and is dominated by
      the population-level curve shape -- while BCE's gradient diverges on
      confident errors and separates subjects harder. That would predict exactly
      what is observed: acceptable IBS, poor C^td.

  (b) SUPERVISION DENSITY. Person-Period trains on `expand_to_regular_grid`
      (grid_step = 1.0 h on synthetic_icu against a mean inter-visit gap of 2.99 h,
      so roughly 3x the rows). The anchor trains on the raw irregular visits.

This script isolates them by running Person-Period BOTH ways through one code path:
with the preregistered grid expansion, and on the raw irregular visits -- the
latter being, model for model, precisely "the alpha = 1 arm with an MLE anchor".

INTERPRETATION RULE, FIXED BEFORE RUNNING (the step A-16 skipped):

  * If PP-raw stays near PP-expanded (delta < 0.03) and both sit near 0.63, then
    grid expansion is not the explanation and (a) LOSS GEOMETRY is. Swapping the
    anchor to a likelihood is then the indicated change, and the alpha = 1 arm
    should reach ~0.63.
  * If PP-raw falls to near the anchor-only arm (~0.53), then (b) SUPERVISION
    DENSITY is the explanation, the Cramer anchor is exonerated, and swapping the
    loss would gain nothing -- the deficit is the irregular sampling itself.
  * Anything in between is not a clean answer and must be reported as such rather
    than rounded toward whichever story is preferred.

Person-Period WITH expansion remains the preregistered Rung 1 baseline. The raw
variant is a diagnostic and is never a baseline substitute.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.data.cohorts import COHORTS
from src.data.dataset import expand_to_regular_grid
from src.models.baselines.person_period import PersonPeriodModel
from src.evaluation.landmark import evaluate_landmarked
from src.training.trainer import train_model, get_device

# Reference values these are compared against, both 5-seed means on synthetic_icu.
REF_PP_EXPANDED = 0.6367      # primary run, experiments/results/preregistered_primary_2026-09-04/
REF_ANCHOR_ONLY = 0.5318      # KC5, experiments/results/kc5_anchor_equivalence/


def run_arm(cohort_data, spec, expand: bool, epochs, batch_size, lr, device):
    train_data, val_data, test_data = cohort_data.train, cohort_data.val, cohort_data.test
    if expand:
        step = getattr(spec, "person_period_grid_step", 1.0)
        train_data = expand_to_regular_grid(train_data, grid_step=step)
        val_data = expand_to_regular_grid(val_data, grid_step=step)
        test_data = expand_to_regular_grid(test_data, grid_step=step)

    model = PersonPeriodModel(
        input_dim=cohort_data.input_dim, hidden_dim=64,
        num_bins=spec.num_bins, delta_s=spec.delta_s, include_overflow=True,
    )
    if hasattr(model.backbone, "set_empirical_mean"):
        model.backbone.set_empirical_mean(cohort_data.x_mean)

    trained = train_model(
        model=model, train_dataset=train_data, val_dataset=val_data,
        val_spec=spec.landmark_spec, delta_s=spec.delta_s, model_type="person_period",
        lr=lr, batch_size=batch_size, epochs=epochs, patience=5, device=device, verbose=False,
    )
    metrics = evaluate_landmarked(
        trained, train_data, test_data, spec.landmark_spec, spec.delta_s, device, strict=False)
    c = [m["c_td"] for m in metrics.values() if not np.isnan(m["c_td"])]
    a = [m["auc"] for m in metrics.values() if not np.isnan(m["auc"])]
    i = [m["ibs"] for m in metrics.values() if not np.isnan(m["ibs"])]
    return (float(np.mean(c)) if c else float("nan"),
            float(np.mean(a)) if a else float("nan"),
            float(np.mean(i)) if i else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 101112])
    ap.add_argument("--cohort", type=str, default="synthetic_icu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--output_dir", type=str, default="experiments/results/anchor_geometry_diagnostic")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    spec = COHORTS[args.cohort]
    raw_path = out / "diagnostic_raw.json"
    records = {}

    def flush():
        raw_path.write_text(json.dumps({
            "cohort": args.cohort, "epochs": args.epochs, "seeds": args.seeds,
            "ref_pp_expanded": REF_PP_EXPANDED, "ref_anchor_only": REF_ANCHOR_ONLY,
            "records": records,
        }, indent=2))

    print("=" * 78)
    print(f"  ANCHOR GEOMETRY DIAGNOSTIC  |  cohort={args.cohort}  device={device}")
    print(f"  Person-Period WITH grid expansion  vs  on RAW irregular visits")
    print("=" * 78)

    for seed in args.seeds:
        cohort_data = spec.load(seed=seed)
        cell = {}
        for label, expand in (("expanded", True), ("raw", False)):
            t0 = time.time()
            c, a, i = run_arm(cohort_data, spec, expand, args.epochs, args.batch_size, args.lr, device)
            cell[label] = {"c_td": c, "auc": a, "ibs": i, "wall_clock_s": round(time.time() - t0, 1)}
            print(f"  seed {seed:>7} | PP {label:>8} | C_td {c:.4f}  AUC {a:.4f}  IBS {i:.4f}  "
                  f"({cell[label]['wall_clock_s']:.0f}s)")
            records[str(seed)] = cell
            flush()
        print(f"  seed {seed:>7} | expansion effect {cell['expanded']['c_td'] - cell['raw']['c_td']:+.4f}")

    usable = [s for s in map(str, args.seeds) if s in records and "raw" in records[s]]
    exp = np.array([records[s]["expanded"]["c_td"] for s in usable])
    raw = np.array([records[s]["raw"]["c_td"] for s in usable])

    print("-" * 78)
    print(f"  PP expanded (preregistered Rung 1) : {exp.mean():.4f} +- {exp.std(ddof=1):.4f}"
          f"   (primary run: {REF_PP_EXPANDED:.4f})")
    print(f"  PP raw  (= alpha=1 with MLE anchor) : {raw.mean():.4f} +- {raw.std(ddof=1):.4f}")
    print(f"  SurvTD anchor-only (Cramer, KC5)    : {REF_ANCHOR_ONLY:.4f}")
    print("-" * 78)
    d_expansion = float(exp.mean() - raw.mean())
    d_geometry = float(raw.mean() - REF_ANCHOR_ONLY)
    print(f"  attributable to grid expansion : {d_expansion:+.4f}")
    print(f"  attributable to loss geometry  : {d_geometry:+.4f}")

    if abs(d_expansion) < 0.03 and raw.mean() > REF_ANCHOR_ONLY + 0.05:
        verdict = "LOSS_GEOMETRY"
        msg = ("The Cramer anchor is the deficit. Swapping it for a likelihood is "
               "the indicated change; alpha=1 should reach ~0.63.")
    elif abs(d_geometry) < 0.03:
        verdict = "SUPERVISION_DENSITY"
        msg = ("Grid expansion explains the gap. The Cramer anchor is exonerated and "
               "swapping the loss would gain nothing.")
    else:
        verdict = "MIXED"
        msg = ("Neither cause dominates. Report as inconclusive; do not round toward "
               "either story.")
    print("=" * 78)
    print(f"  DIAGNOSTIC VERDICT: {verdict}")
    print(f"  {msg}")
    print("=" * 78)

    (out / "diagnostic_verdict.json").write_text(json.dumps({
        "verdict": verdict, "message": msg,
        "pp_expanded_mean": float(exp.mean()), "pp_expanded_sd": float(exp.std(ddof=1)),
        "pp_raw_mean": float(raw.mean()), "pp_raw_sd": float(raw.std(ddof=1)),
        "ref_anchor_only": REF_ANCHOR_ONLY, "ref_pp_expanded": REF_PP_EXPANDED,
        "delta_grid_expansion": d_expansion, "delta_loss_geometry": d_geometry,
        "seeds": usable,
        "pp_expanded_c_td": exp.tolist(), "pp_raw_c_td": raw.tolist(),
    }, indent=2))
    print(f"Saved -> {out / 'diagnostic_verdict.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
