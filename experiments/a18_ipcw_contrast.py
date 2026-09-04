"""
A-18: does IPCW-weighting the TRAINING loss help or hurt? (decided-after-results)

A-04/D11 weight the anchor by `1/G_hat(c)` on censored trajectories. That is the
correct construction for an ESTIMATOR -- it is what makes IBS and Uno's cumulative
dynamic AUC unbiased under right censoring -- but it was carried into the TRAINING
objective by analogy and never justified there. The two uses are not the same: an
estimator wants the bias removed, while an optimiser pays for it in variance.

Measured on synthetic_icu seed 42: 214 of 300 training trajectories are censored and
carry weights of 1.0-10.0 (median 1.72, p90 7.35, capped at 10). The top 10% of
weighted subjects hold 34.3% of the total weight. Dynamic-DeepHit has no such
weighting, and `censored_crps_anchor`'s own module already floors G at 0.05 because,
in its words, "a single subject could then dominate an entire metric".

This also matters for a second reason. The `anchor/ce` arm's loss is bit-identical to
Dynamic-DeepHit's L1 (verified: max |diff| = 0.00e+00 per visit over 7 visits, both
branches), so any performance gap between them is not the loss. Three candidates
remain: IPCW, the reduction convention (DeepHit takes a flat mean over visit rows and
so weights a 31-visit subject 15.5x a 2-visit subject; SurvTD averages within subject
then across subjects), and the absent ranking term. Turning IPCW off makes this arm
match DeepHit exactly on the first axis, isolating it.

Arms, both at alpha = 1 with IPCW disabled, against the IPCW-on cells:
    anchor/cramer  vs  the KC5 reference 0.5318 +- 0.0407
    anchor/ce      vs  the A-17b cell

`cramer` is included deliberately: if IPCW is a general variance source it has been
depressing every SurvTD arm measured so far, not just `ce`, and that is a materially
different finding from "the ce arm has a quirk".

PRE-DECLARED READING (fixed before running):
  * If both arms improve by a similar margin -> IPCW is a general cost on the training
    objective, and every SurvTD number reported so far carries it.
  * If only `ce` improves -> the interaction is specific to the likelihood geometry.
  * If neither improves -> IPCW is exonerated and the DeepHit gap is the reduction
    convention or the ranking term; A-19 would test the reduction next.
  * Improvement here does NOT license removing IPCW from the reported pipeline: it is
    preregistered in A-04, and dropping it re-opens the censoring bias it was added to
    correct. Any change would be a separate, argued amendment.
"""

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.data.cohorts import COHORTS
from src.training.trainer import get_device
from experiments.run_track_b import train_and_eval_survtd

REF = {"anchor/cramer": 0.5318, "anchor/cramer_sd": 0.0407}
ARMS = [("anchor/cramer  ipcw=off", "cramer"), ("anchor/ce      ipcw=off", "ce")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 101112])
    ap.add_argument("--cohort", type=str, default="synthetic_icu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--output_dir", type=str, default="experiments/results/a18_ipcw_contrast")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = get_device(); spec = COHORTS[args.cohort]
    rec = {lab: {} for lab, _ in ARMS}
    raw = out / "a18_raw.json"

    def flush():
        raw.write_text(json.dumps({"cohort": args.cohort, "epochs": args.epochs,
                                   "seeds": args.seeds, "use_ipcw": False,
                                   "reference": REF, "records": rec}, indent=2))

    print("=" * 78)
    print(f"  A-18 IPCW CONTRAST (use_ipcw=False)  |  {args.cohort}  {device}")
    print(f"  reference, IPCW on: anchor/cramer {REF['anchor/cramer']:.4f} "
          f"+- {REF['anchor/cramer_sd']:.4f}")
    print("=" * 78)

    for seed in args.seeds:
        cd = spec.load(seed=seed)
        for lab, aloss in ARMS:
            t0 = time.time()
            _, c, auc = train_and_eval_survtd(
                cd, spec, "full", 1.0, args.epochs, 16, 0.001, device,
                seed=seed, anchor_loss=aloss, use_ipcw=False)
            rec[lab][str(seed)] = {"c_td": c, "auc": auc,
                                   "wall_clock_s": round(time.time() - t0, 1)}
            print(f"  seed {seed:>7} | {lab:<26} C_td {c:.4f}  AUC {auc:.4f}", flush=True)
            flush()

    print("-" * 78)
    summ = {}
    for lab, _ in ARMS:
        v = np.array([rec[lab][s]["c_td"] for s in rec[lab]
                      if not np.isnan(rec[lab][s]["c_td"])])
        if v.size < 3:
            continue
        summ[lab] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "n": int(v.size)}
        print(f"  {lab:<26} {v.mean():.4f} +- {v.std(ddof=1):.4f}")
    if "anchor/cramer  ipcw=off" in summ:
        d = summ["anchor/cramer  ipcw=off"]["mean"] - REF["anchor/cramer"]
        print(f"\n  cramer: IPCW off - IPCW on = {d:+.4f}")
    print("=" * 78)
    (out / "a18_summary.json").write_text(json.dumps({"reference": REF, "summary": summ}, indent=2))
    print(f"Saved -> {out / 'a18_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
