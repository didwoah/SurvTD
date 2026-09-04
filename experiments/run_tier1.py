"""
Tier-1 benchmark runner: arms x datasets x seeds, one scorer, incremental flush.

    python3 experiments/run_tier1.py --cohorts pbc framingham --seeds 42 123 456

Every arm produces a `(N_test, n_landmarks, n_horizons)` conditional-survival matrix
and is scored by `src/evaluation/curve_scoring.evaluate_curves`. Nothing in this file
computes a metric; that is the point. The vendored repos each ship their own C-index
and Brier code, and scoring each column of Table 1 with whatever its own repo happened
to use would make the columns incomparable.

Results are flushed to JSON after **every** (cohort, arm, seed) cell. Two hours of
compute has already been lost once in this project by living only in stdout.

Reading the output
------------------
* Metrics are reported **per landmark**, never only as a grid mean. On C-MAPSS the mean
  is inflated by late landmarks where few units remain and the residual times are short.
* A cell can legitimately be `null`: CoxSig is structurally undefined at `L = 0` (D17),
  and a landmark can fail `min_at_risk` / `min_events`. Those carry a `reason`.
* **A below-chance `C^td` is a defect alarm, not a result** (D15, D16). This runner
  flags them on stdout as it goes; do not write them into a table without tracing them.
* Wall clock is recorded per cell. C_3 and Kill Criterion 3 called for *compute-matched*
  baselines; running each under its authors' own optimiser makes that unachievable by
  construction, so the preregistration's compute matching is replaced by compute
  reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.cohorts import COHORTS                                  # noqa: E402
from src.evaluation.curve_scoring import evaluate_curves              # noqa: E402
from src.evaluation.worker_format import build_worker_bundle          # noqa: E402

# name -> (script, cwd, input-flag, output-flag, extra args). The two flag spellings
# are the vendored repos' own; they are not worth churning for consistency.
WORKERS = {
    "coxsig":        ("baselines/signature_survival/coxsig_worker.py",
                      "baselines/signature_survival", "--input_pt", "--output_json", []),
    "ncde":          ("baselines/signature_survival/ncde_worker.py",
                      "baselines/signature_survival", "--input_pt", "--output_json", []),
    "ddh":           ("baselines/dynamic_deephit_pytorch/ddh_worker.py",
                      "baselines/dynamic_deephit_pytorch", "--input_pt", "--output_json", []),
    "deeptcsr_tcn":  ("baselines/deep_tcsr/deeptcsr_worker.py",
                      "baselines/deep_tcsr", "--input_pt", "--output_json", []),
    "tcsr":          ("baselines/tcsr/tcsr_worker.py",
                      "baselines/tcsr", "--data", "--out", ["--arm", "tcsr"]),
    "tcsr_landmark": ("baselines/tcsr/tcsr_worker.py",
                      "baselines/tcsr", "--data", "--out", ["--arm", "landmark"]),
}

# In-process arms. `alpha_anchor` is SurvTD's mixing weight between the anchor loss
# and the TD loss: 1.0 is the anchor-only control that answers Q1 ("does TD help at
# all?"), and the configured default is the full method.
#
# NOTE for anyone reading this beside a `tdsurv` result: SurvTD's `lam` and TCSR's
# `lambda_` run in OPPOSITE directions. SurvTD's lam = 1 is Monte Carlo; tdsurv's
# lambda_ = 1 is landmarking and 0 is pure TD. The `tcsr` / `tcsr_landmark` arm names
# above exist so no raw lambda ever has to be read off this table.
IN_PROCESS = {
    "km": None,
    "landmark_cox": None,
    "survtd": dict(alpha_anchor=0.5),
    "survtd_anchor_only": dict(alpha_anchor=1.0),
}

DEFAULT_ARMS = ["km", "landmark_cox", "survtd", "survtd_anchor_only",
                "ddh", "tcsr", "tcsr_landmark", "coxsig", "ncde", "deeptcsr_tcn"]


def km_curves(bundle, wb):
    """Marginal Kaplan-Meier: the same curve for every subject.

    Its only job is to score exactly 0.500. It is the leak gate, kept as an arm so the
    gate runs on every cohort and seed rather than once by hand.
    """
    from sksurv.nonparametric import kaplan_meier_estimator

    y = wb.surv_labels_train
    t, s = kaplan_meier_estimator(y[:, 1].astype(bool), y[:, 0])
    n = wb.paths_test.shape[0]
    out = np.ones((n, len(wb.pred_times), len(wb.eval_times)))
    for j, L in enumerate(wb.pred_times):
        s_at_L = float(np.interp(L, t, s, left=1.0, right=float(s[-1])))
        ahead = np.interp(L + wb.eval_times, t, s, left=1.0, right=float(s[-1]))
        out[:, j, :] = np.clip(ahead / max(s_at_L, 1e-12), 0.0, 1.0)
    return np.minimum.accumulate(out, axis=-1)


def survtd_curves(cd, spec, seed, arm_cfg, eval_times, epochs):
    """Train SurvTD on the cohort's NATIVE irregular visits and read curves off it.

    Deliberately not the forward-filled worker bundle. The claim under test is about
    irregular Delta-t, and projecting onto a regular grid is precisely what removes
    that signal -- so the method is evaluated on the protocol its claim is about, and
    `curves_from_model` puts the result on the same residual grid the worker arms are
    scored on. The two protocols are reported side by side and never mixed inside a
    column; `worker_format.py` states the same split from the other side.
    """
    import torch as _torch
    from src.evaluation.curve_scoring import curves_from_model
    from src.models.survtd import SurvTDModel
    from src.training.trainer import get_device, train_model

    _torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    model = SurvTDModel(input_dim=cd.input_dim, hidden_dim=64,
                        num_bins=spec.num_bins, delta_s=spec.delta_s,
                        alpha_anchor=arm_cfg["alpha_anchor"], include_overflow=True)
    if hasattr(model.backbone, "set_empirical_mean"):
        model.backbone.set_empirical_mean(cd.x_mean)

    trained = train_model(model=model, train_dataset=cd.train, val_dataset=cd.val,
                          val_spec=spec.landmark_spec, delta_s=spec.delta_s,
                          model_type="survtd",
                          # `ablation_mode` is the RETURN-CONSTRUCTION axis (A1/A2/
                          # count-geometric); it has no "anchor_only" value and never
                          # had one. The anchor-only control is `alpha_anchor = 1.0`,
                          # which is what actually zeroes the TD term at the loss.
                          ablation_mode="full",
                          alpha_anchor=arm_cfg["alpha_anchor"],
                          epochs=epochs, patience=5, device=device, verbose=False)
    return curves_from_model(trained, cd.test, spec.landmark_spec,
                             spec.delta_s, eval_times, device)


def run_worker(arm, bundle_path, out_json, epochs, timeout):
    script, cwd, in_flag, out_flag, extra = WORKERS[arm]
    cmd = [sys.executable, os.path.join(ROOT, script),
           in_flag, bundle_path, out_flag, out_json] + extra
    if arm not in ("tcsr", "tcsr_landmark"):
        cmd += ["--epochs", str(epochs), "--emit_curves"]
    subprocess.run(cmd, cwd=os.path.join(ROOT, cwd), check=True, timeout=timeout)
    with open(out_json) as f:
        return np.asarray(json.load(f)["surv_curves"], dtype=float)


def run_cell(cohort_name, arm, seed, epochs, timeout, tmpdir):
    """One (cohort, arm, seed) cell -> per-landmark metrics."""
    spec = COHORTS[cohort_name]
    cd = spec.load(seed=seed)
    wb = build_worker_bundle(cd, spec)

    if arm == "km":
        curves = km_curves(None, wb)
    elif arm == "landmark_cox":
        from src.models.baselines.authentic.landmark_cox import landmark_cox_curves
        curves = landmark_cox_curves(cd.train, cd.test, spec.landmark_spec,
                                     wb.eval_times)
    elif arm in IN_PROCESS:
        curves = survtd_curves(cd, spec, seed, IN_PROCESS[arm], wb.eval_times, epochs)
    else:
        path = os.path.join(tmpdir, f"{cohort_name}_{seed}.pt")
        if not os.path.exists(path):
            wb.save(path)
        curves = run_worker(arm, path,
                            os.path.join(tmpdir, f"{cohort_name}_{arm}_{seed}.json"),
                            epochs, timeout)

    # Score on the cohort's original clock. The `WorkerBundle` FIELDS are unscaled;
    # `worker_format.save` divides the time axes by `time_scale` on the way to disk so
    # CoxSig's level-2 signatures do not overflow, and only the subprocess ever sees
    # that scaled copy. Curve VALUES are unitless probabilities and column `k` still
    # means `eval_times[k]`, so the in-memory axes are the right ones to score against
    # and nothing needs converting back. (Scaling them again here is exactly the bug
    # that put PBC2's second landmark at 194400 days on this runner's first run.)
    # `wb.surv_labels_test` is built by iterating `cohort_data.test` in order, which is
    # the same order `landmark_cox_curves` and `curves_from_model` iterate, so one
    # label array serves every arm.
    res = evaluate_curves(curves, wb.surv_labels_test,
                          wb.pred_times, wb.eval_times,
                          cd.train, spec.landmark_spec)
    return {f"L={k[0]:g},H={k[1]:g}": v for k, v in sorted(res.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["pbc", "framingham", "cmapss"])
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 101112])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="experiments/results/tier1/tier1_results.json")
    # A SurvTD cell on Framingham costs 14x what it costs on PBC2, so a run can span
    # hours and a killed process should not throw away what it already paid for.
    ap.add_argument("--resume", action="store_true",
                    help="keep cells already present in --out and skip re-running them")
    args = ap.parse_args()

    out_path = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results, alarms = {}, []
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            prior = json.load(f)
        # Only successful cells are kept. A cell that errored is retried, since the
        # usual cause is the process being killed part-way rather than the arm failing.
        results = {k: v for k, v in prior.get("results", {}).items() if "metrics" in v}
        alarms = list(prior.get("below_chance_alarms", []))
        print(f"resuming: {len(results)} cell(s) already complete in {out_path}")
    tmpdir = tempfile.mkdtemp(prefix="tier1_")

    def flush():
        with open(out_path, "w") as f:
            json.dump({"config": vars(args), "results": results,
                       "below_chance_alarms": alarms}, f, indent=2)

    for cohort in args.cohorts:
        for arm in args.arms:
            if arm not in IN_PROCESS and arm not in WORKERS:
                raise SystemExit(f"unknown arm {arm!r}")
            for seed in args.seeds:
                key = f"{cohort}|{arm}|{seed}"
                if key in results:
                    print(f"{key:44s} (cached)", flush=True)
                    continue
                t0 = time.time()
                try:
                    cell = run_cell(cohort, arm, seed, args.epochs, args.timeout, tmpdir)
                    entry = {"metrics": cell, "wall_clock_s": round(time.time() - t0, 1)}
                    for lm, m in cell.items():
                        c = m.get("c_td")
                        if c is not None and np.isfinite(c) and c < 0.5:
                            alarm = f"{key} {lm}: c_td={c:.4f}"
                            alarms.append(alarm)
                            print(f"  !! BELOW CHANCE (defect alarm, not a result): {alarm}")
                    tds = [m["c_td"] for m in cell.values() if np.isfinite(m.get("c_td", np.nan))]
                    print(f"{key:44s} c_td={['%.4f' % v for v in tds]} "
                          f"({entry['wall_clock_s']}s)", flush=True)
                except Exception as exc:
                    entry = {"error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc(),
                             "wall_clock_s": round(time.time() - t0, 1)}
                    print(f"{key:44s} FAILED: {type(exc).__name__}: {exc}", flush=True)
                results[key] = entry
                flush()

    print(f"\nwritten to {out_path}")
    if alarms:
        print(f"\n{len(alarms)} below-chance cell(s) -- trace each before citing "
              f"anything from this file:")
        for a in alarms:
            print(f"  {a}")
    # KM must score exactly 0.500 wherever it ran.
    for key, entry in results.items():
        if key.split("|")[1] == "km" and "metrics" in entry:
            for lm, m in entry["metrics"].items():
                if np.isfinite(m.get("c_td", np.nan)) and abs(m["c_td"] - 0.5) > 1e-9:
                    print(f"\nLEAK GATE FAILED: {key} {lm} c_td={m['c_td']:.10f} != 0.5")
                    return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
