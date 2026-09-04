"""
Kill Criterion 5 — Anchor-Equivalence Kill (preregistration §6.6).

    "If full SurvTD fails to outperform the anchor-only arm (alpha = 1, identical
     backbone without TD consistency) by at least 0.015 in C-index, the temporal-
     difference consistency mechanism carries no empirical value, and claims C_0
     and C_3 are falsified."

This criterion has NEVER executed. It is wired into `run_track_b.py:190-233`, but
Track B trains six arms per seed (~6.7 h) and has not been run; the session that
produced the Cohort 1 benchmark died inside Track A. HANDOVER names C_0 as "the
likeliest failure" and Kill Criterion 5 as "the most likely way the paper dies", so
it is worth the ~2.2 h to isolate it rather than waiting on the full Track B.

Adjudication follows preregistration §5 / A-08: mean paired delta >= 0.015 AND the
1000-sample paired bootstrap 95% CI lower bound strictly above 0.

Configuration is the PREREGISTERED one -- default initialization, alpha = 0.0 for
the full arm, the 5 preregistered seeds -- so this is a primary result, not an A-16
exploratory one, with a single declared exception:

  `es_warmup = 5` (A-16) is in effect. It is applied IDENTICALLY to both arms and
  can only help the arm that cold-starts, which is the full alpha = 0 arm, not the
  anchor-only arm. A falsification under this setting is therefore conservative:
  the TD arm received the benefit and still lost. Were the criterion to pass, the
  caveat would apply and must be reported.
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
from src.evaluation.stats import compute_paired_bootstrap_ci
from src.training.trainer import get_device
from experiments.run_track_b import train_and_eval_survtd

KC5_THRESHOLD = 0.015


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 101112])
    ap.add_argument("--cohort", type=str, default="synthetic_icu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--alpha_full", type=float, default=0.0,
                    help="alpha for the full arm; 0.0 is what the primary Cohort 1 run used")
    ap.add_argument("--output_dir", type=str, default="experiments/results/kc5_anchor_equivalence")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    spec = COHORTS[args.cohort]
    json_path = out / "kc5_raw.json"

    print("=" * 76)
    print(f"  KILL CRITERION 5 — Anchor-Equivalence  |  cohort={args.cohort}  device={device}")
    print(f"  full arm alpha={args.alpha_full}  vs  anchor-only alpha=1.0  |  "
          f"epochs={args.epochs}  seeds={args.seeds}")
    print("=" * 76)

    records = {}

    def flush():
        json_path.write_text(json.dumps({
            "cohort": args.cohort,
            "alpha_full": args.alpha_full,
            "alpha_anchor_only": 1.0,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "init": "default (preregistered)",
            "es_warmup": 5,
            "records": records,
        }, indent=2))

    for seed in args.seeds:
        cohort_data = spec.load(seed=seed)
        cell = {}

        t0 = time.time()
        _, c_full, auc_full = train_and_eval_survtd(
            cohort_data, spec, "full", args.alpha_full,
            args.epochs, args.batch_size, args.lr, device, seed=seed)
        cell["full"] = {"c_td": c_full, "auc": auc_full, "wall_clock_s": round(time.time() - t0, 1)}
        print(f"  seed {seed:>7} | full (alpha={args.alpha_full})   C_td = {c_full:.4f}  "
              f"AUC = {auc_full:.4f}  ({cell['full']['wall_clock_s']:.0f}s)")
        records[str(seed)] = cell
        flush()

        t0 = time.time()
        _, c_anchor, auc_anchor = train_and_eval_survtd(
            cohort_data, spec, "full", 1.0,
            args.epochs, args.batch_size, args.lr, device, seed=seed)
        cell["anchor_only"] = {"c_td": c_anchor, "auc": auc_anchor,
                               "wall_clock_s": round(time.time() - t0, 1)}
        print(f"  seed {seed:>7} | anchor-only (alpha=1) C_td = {c_anchor:.4f}  "
              f"AUC = {auc_anchor:.4f}  ({cell['anchor_only']['wall_clock_s']:.0f}s)")
        print(f"  seed {seed:>7} | paired delta          {c_full - c_anchor:+.4f}")
        flush()

    usable = [s for s in map(str, args.seeds)
              if s in records and "anchor_only" in records[s]
              and not np.isnan(records[s]["full"]["c_td"])
              and not np.isnan(records[s]["anchor_only"]["c_td"])]
    full = np.array([records[s]["full"]["c_td"] for s in usable])
    anchor = np.array([records[s]["anchor_only"]["c_td"] for s in usable])

    if full.size < 3:
        raise SystemExit(f"only {full.size} usable seeds; KC5 needs at least 3")

    delta, ci_lo, ci_hi, se = compute_paired_bootstrap_ci(full.tolist(), anchor.tolist())
    falsified = (delta < KC5_THRESHOLD) or (ci_lo <= 0.0)

    print("-" * 76)
    print(f"{'seed':>9} {'full':>9} {'anchor-only':>13} {'delta':>9}")
    for k, s in enumerate(usable):
        print(f"{s:>9} {full[k]:>9.4f} {anchor[k]:>13.4f} {full[k]-anchor[k]:>+9.4f}")
    print("-" * 76)
    print(f"  full        : {full.mean():.4f} +- {full.std(ddof=1):.4f}")
    print(f"  anchor-only : {anchor.mean():.4f} +- {anchor.std(ddof=1):.4f}")
    print(f"  paired delta: {delta:+.4f}   95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]   SE {se:.4f}")
    print(f"  requirement : delta >= {KC5_THRESHOLD} AND CI lower bound > 0")
    print("=" * 76)
    print(f"  KILL CRITERION 5: {'FALSIFIED' if falsified else 'UPHELD'}")
    if falsified:
        print("  -> The TD consistency term carries no empirical value over the same")
        print("     backbone with the same supervision. C_0 and C_3 are falsified.")
    print("=" * 76)

    verdict = {
        "test_id": "Kill Criterion 5",
        "claim": "C0, C3",
        "cohort": args.cohort,
        "alpha_full": args.alpha_full,
        "seeds": usable,
        "full_c_td": full.tolist(),
        "anchor_only_c_td": anchor.tolist(),
        "mean_full": float(full.mean()),
        "mean_anchor_only": float(anchor.mean()),
        "paired_delta": float(delta),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "se": float(se),
        "threshold": KC5_THRESHOLD,
        "verdict": "FALSIFIED" if falsified else "UPHELD",
        "caveat": ("es_warmup=5 (A-16) applied identically to both arms; it can only "
                   "help the cold-starting full arm, so a falsification is conservative"),
    }
    (out / "kc5_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"Saved -> {out / 'kc5_verdict.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
