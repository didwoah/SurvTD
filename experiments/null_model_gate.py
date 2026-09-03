"""
Leakage gate: does the landmark harness give an uninformative model chance-level scores?

This runs BEFORE any training and is the check that would have caught the last-visit
leak. If an uninformative predictor scores well, the evaluation is broken and nothing
downstream of it means anything.

Two design corrections, both found by running an earlier version of this gate
----------------------------------------------------------------------------
1. **An untrained network is NOT a null model.** Random weights give a random but
   non-degenerate linear projection of the features, so on a cohort where a feature
   nearly determines the outcome (tumour size in the tumour cohort) a single random
   initialization reached AUC 0.882. The expectation over initializations is 0.5, but
   one draw is not. The gate therefore averages over `--inits` random seeds and tests
   the MEAN, reporting the spread so a wide one is visible.

2. **An absolute Brier ceiling is the wrong guard.** An untrained network has per-bin
   hazard ~0.5, so its survival decays like 0.5^k and it effectively predicts
   immediate failure for everyone -- legitimately scoring ~1.0 on survivors. The
   earlier version asserted IBS <= 0.30 inside the metric and tripped on exactly
   that. The correct yardstick is the covariate-free Kaplan-Meier reference, which is
   reported here alongside.

Usage:
    python experiments/diagnostics/null_model_gate.py [--inits 10]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from src.evaluation.landmark import (
    DegenerateLandmarkError,
    LandmarkSpec,
    evaluate_landmarked,
    km_marginal_reference,
    landmark_labels,
)
from src.evaluation.metrics import (
    concordance_antolini,
    integrated_brier,
    make_structured,
)
from src.models.survtd import SurvTDModel

# Single source of truth: the declared specs live in src/data/cohorts.py so the gate
# and the experiments cannot drift apart.
from src.data.cohorts import COHORTS as COHORT_SPECS


# Decision rule. An earlier version compared the mean null-model AUC to a fixed
# 0.08 band, which ignores Monte-Carlo error: with 6 initializations and ~50
# at-risk subjects the standard error of the mean is 0.05-0.07, so a mean of 0.58
# was flagged while sitting within 1.2-1.7 standard errors of chance.
#
# The correct statement is "the mean is within Monte-Carlo error of 0.5". The band
# is NOT widened until things pass -- that would be the behaviour this whole effort
# exists to prevent. Instead the error is reduced by using more initializations,
# and a floor keeps the criterion from becoming vacuous when the spread is tiny.
FLOOR = 0.05         # minimum band, so a near-zero spread cannot make this vacuous
N_SIGMA = 2.0        # ~95% under normality


def km_reference_scores(train_ds, test_ds, spec, landmark):
    """Covariate-free KM predictor: the honest yardstick for both C^td and IBS."""
    ref = km_marginal_reference(train_ds, landmark, spec)
    train_labels = landmark_labels(train_ds, landmark, spec)
    # Apply the KM curve, fit on train, to the TEST at-risk set.
    from src.evaluation.landmark import predict_landmark  # noqa: F401  (docs the pairing)

    test_labels = landmark_labels(test_ds, landmark, spec)
    if test_labels.size == 0 or int(test_labels["event"].sum()) < 2:
        raise DegenerateLandmarkError("KM reference: test at-risk set too small")
    surv = np.tile(ref.surv[0], (test_labels.size, 1))
    return {
        "c_td": concordance_antolini(surv, ref.grid, test_labels["time"],
                                     test_labels["event"].astype(float)),
        "ibs": integrated_brier(train_labels, test_labels, surv, ref.grid),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cohort", type=str, default="all")
    args = ap.parse_args()

    device = torch.device("cpu")   # deterministic and fast enough for forward passes
    names = list(COHORT_SPECS) if args.cohort == "all" else [args.cohort]
    report, failures = {}, []

    for name in names:
        cfg = COHORT_SPECS[name]
        data = cfg.load(args.seed)
        train, val, test, in_dim, x_mean = (
            data.train, data.val, data.test, data.input_dim, data.x_mean)
        spec, ds, K = cfg.landmark_spec, cfg.delta_s, cfg.num_bins
        print(f"\n=== {name} (delta_s={ds}, K={K}) ===")

        # KM-marginal reference, per landmark.
        km_rows = {}
        for L in spec.landmarks:
            try:
                km_rows[L] = km_reference_scores(train, test, spec, L)
                print(f"  KM-marginal   L={L:>7.1f}  C_td={km_rows[L]['c_td']:.3f}  "
                      f"IBS={km_rows[L]['ibs']:.3f}")
            except (DegenerateLandmarkError, AssertionError) as exc:
                km_rows[L] = {"reason": str(exc)}
                print(f"  KM-marginal   L={L:>7.1f}  n/a: {str(exc)[:70]}")

        # Untrained models, averaged over initializations.
        collected = {}
        for i in range(args.inits):
            torch.manual_seed(1000 + i)
            model = SurvTDModel(input_dim=in_dim, hidden_dim=32, num_bins=K, delta_s=ds)
            model.backbone.set_empirical_mean(x_mean)
            res = evaluate_landmarked(model, train, test, spec, ds, device)
            for key, row in res.items():
                collected.setdefault(key, []).append(row)

        cohort_report = {"km_marginal": {str(k): v for k, v in km_rows.items()}, "null_model": {}}
        for key, rows in sorted(collected.items()):
            L, delta = key
            usable = [r for r in rows if "reason" not in r]
            if not usable:
                print(f"  null model    L={L:>7.1f} d={delta:<7.1f} n/a: "
                      f"{rows[0].get('reason', 'unknown')[:60]}")
                cohort_report["null_model"][f"{L}_{delta}"] = {"reason": rows[0].get("reason")}
                continue

            auc = np.array([r["auc"] for r in usable])
            c_td = np.array([r["c_td"] for r in usable])
            ibs = np.array([r["ibs"] for r in usable])
            sem = float(auc.std(ddof=1) / np.sqrt(len(auc))) if len(auc) > 1 else 0.0
            band = max(FLOOR, N_SIGMA * sem)
            ok = abs(auc.mean() - 0.5) <= band

            print(f"  null model    L={L:>7.1f} d={delta:<7.1f} "
                  f"AUC={auc.mean():.3f}+-{sem:.3f}se (band {band:.3f})  "
                  f"C_td={c_td.mean():.3f}+-{c_td.std():.3f}  "
                  f"IBS={ibs.mean():.3f}  n={usable[0]['n_at_risk']:<4d} "
                  f"ev={usable[0]['n_events']:<4d} {'PASS' if ok else 'LEAK?'}")

            cohort_report["null_model"][f"{L}_{delta}"] = {
                "auc_mean": round(float(auc.mean()), 4),
                "auc_sem": round(sem, 4),
                "auc_band": round(band, 4),
                "c_td_mean": round(float(c_td.mean()), 4),
                "ibs_mean": round(float(ibs.mean()), 4),
                "n_at_risk": usable[0]["n_at_risk"],
                "n_events": usable[0]["n_events"],
                "n_inits": len(usable),
                "pass": bool(ok),
            }
            if not ok:
                failures.append(
                    f"{name} L={L} d={delta}: mean null-model AUC {auc.mean():.3f} is "
                    f"{abs(auc.mean()-0.5):.3f} from chance, outside the "
                    f"{band:.3f} Monte-Carlo band"
                )

        report[name] = cohort_report

    out_dir = os.path.join("experiments", "results", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "null_model_gate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"floor": FLOOR, "n_sigma": N_SIGMA, "inits": args.inits, "report": report,
                   "failures": failures}, f, indent=2)

    print("\n" + "=" * 72)
    if failures:
        print("GATE FAILED -- an uninformative predictor scores above chance:")
        for f_ in failures:
            print("  -", f_)
        print("\nDo not train or report anything until this is understood.")
    else:
        print("GATE PASSED: uninformative predictors score at chance on every")
        print("cohort x landmark. The landmark harness is not leaking the outcome.")
    print(f"saved: {path}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
