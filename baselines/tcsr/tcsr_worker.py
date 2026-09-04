"""
Subprocess worker for TCSR (Maystre & Russo, *Temporally-Consistent Survival
Analysis*, NeurIPS 2022), running the vendored Spotify `tdsurv` package unmodified.

Same contract as the other workers in `baselines/*/`: read a `.pt` bundle, run the
upstream model, write JSON. Isolation is by `cwd` and `sys.path`, because `tdsurv`
needs JAX while the parent project runs PyTorch, and because several vendored trees
expose a top-level `src` package that would shadow the project's own.

Why it exists
-------------
TCSR is the direct predecessor of this project's claim -- temporal-difference learning
for survival -- and until now it was vendored with **zero call sites**. It is the only
baseline that answers "does duration-awareness beat survival TD that ignores Δt?",
which is the question the paper is for.

Two upstream properties that matter and are preserved
-----------------------------------------------------
1. **`lambda_` runs the opposite way to SurvTD's.** In `tdsurv`, `lambda_ = 1.0` is
   landmarking (no bootstrap) and `lambda_ = 0.0` is pure TD; SurvTD's `lam = 1`
   reduces to Monte Carlo. The authors' own notebook
   (`notebooks/pbc2-experiments.ipynb`) fits TD with `lambda_=0.0, n_iters=30`.
   Reporting the two side by side without stating this would invert the reader's
   understanding of every ablation, so the CLI takes `--arm` names rather than a raw
   lambda.
2. **There is no sequence encoder.** `models.py` offers `Linear`, `Tabular`, `CoxPH`
   and `BetaGeom`, all per-period parameter vectors over the current state. That is
   the authors' claim, not an oversight: TD consistency is argued to help *even with a
   static per-step head*. Left exactly as it is.

Output is **conditional survival curves, not metrics**, so that every arm in the
benchmark is scored by one scorer rather than by whichever metric its own repo shipped.
TCSR needs no conditional renormalisation for this: it is a per-period model, so
`survival_curve(state_at_L)` is already survival over residual periods from L.
"""

import argparse
import json
import os
import sys

import numpy as np

# jaxlib 0.10.2 expects numpy >= 2.0's `numpy.dtypes.StringDType`; this project pins
# numpy 1.26.4. Same shim `baselines/deep_tcsr/deeptcsr_worker.py:7-10` uses. Verified
# not to be cosmetic: under it, the authors' own PBC2 protocol reproduces finite
# parameters, a monotone survival curve and C-index 0.83.
if not hasattr(np.dtypes, "StringDType"):
    class StringDType:            # noqa: D401
        def __init__(self, *a, **k):
            pass
    np.dtypes.StringDType = StringDType

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

import torch  # noqa: E402
from tdsurv import CoxPH, Linear, Tabular, unroll  # noqa: E402

MODELS = {"coxph": CoxPH, "linear": Linear, "tabular": Tabular}

# The authors' three arms, named so `lambda_`'s inverted convention cannot be misread.
# Values follow `notebooks/pbc2-experiments.ipynb`.
ARMS = {
    "initial_state": dict(unroll=False, lambda_=1.0, l2=0.1),
    "landmark":      dict(unroll=True,  lambda_=1.0, l2=0.1),
    "tcsr":          dict(unroll=True,  lambda_=0.0, l2=0.0, n_iters=30),
}


def paths_to_period_grid(paths, surv_labels, sampling_times):
    """Worker `.pt` format -> `tdsurv`'s `(seqs, ts, cs)`.

    `paths` is `(N, T, D)` with channel 0 carrying time; `tdsurv` wants states on a
    fixed period grid plus a terminal period index and a censoring flag.

    `ts` follows the authors' convention, pinned in
    `experiments/unit_tests/test_pbc2_literature_fidelity.py`: the index of the period
    in which the trajectory ends -- the last observed period for a censored subject,
    one step past it for an event.
    """
    seqs = np.asarray(paths[..., 1:], dtype=np.float64)          # drop the time channel
    horizon = seqs.shape[1]
    times = np.asarray(surv_labels[:, 0], dtype=float)
    events = np.asarray(surv_labels[:, 1], dtype=float) > 0.5

    grid = np.asarray(sampling_times, dtype=float)
    last_obs = np.searchsorted(grid, times, side="right") - 1
    last_obs = np.clip(last_obs, 0, horizon - 1)

    ts = np.where(events, np.minimum(last_obs + 1, horizon - 1), last_obs).astype(int)
    cs = ~events
    return seqs, ts, cs


def conditional_curves(model, seqs, pred_idx, n_eval):
    """(N, |pred_idx|, n_eval) conditional survival, read off the per-period model.

    TCSR predicts from the state at a period, so `survival_curve(state_at_L)` is
    already survival over residual periods measured from L -- no `S(L+r)/S(L)`
    renormalisation, unlike the absolute-axis baselines.
    """
    n = seqs.shape[0]
    out = np.zeros((n, len(pred_idx), n_eval), dtype=float)
    for j, p in enumerate(pred_idx):
        curve = np.asarray(model.survival_curve(seqs[:, p, :]))   # (N, horizon + 1)
        for k in range(n_eval):
            col = min(k + 1, curve.shape[1] - 1)
            out[:, j, k] = curve[:, col]
    return np.clip(out, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="torch.save'd bundle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="tcsr", choices=sorted(ARMS))
    ap.add_argument("--model", default="coxph", choices=sorted(MODELS))
    args = ap.parse_args()

    data = torch.load(args.data, weights_only=False)
    tr_seqs, tr_ts, tr_cs = paths_to_period_grid(
        data["paths_train"], data["surv_labels_train"], data["sampling_times"])
    te_seqs, te_ts, te_cs = paths_to_period_grid(
        data["paths_test"], data["surv_labels_test"], data["sampling_times"])

    horizon, n_feats = tr_seqs.shape[1], tr_seqs.shape[2]
    cfg = dict(ARMS[args.arm])
    do_unroll = cfg.pop("unroll")

    model = MODELS[args.model](horizon=horizon, n_feats=n_feats)
    fit_seqs, fit_ts, fit_cs = (
        unroll(tr_seqs, tr_ts, tr_cs) if do_unroll else (tr_seqs, tr_ts, tr_cs))
    model.fit(fit_seqs, fit_ts, fit_cs, **cfg)

    grid = np.asarray(data["sampling_times"], dtype=float)
    pred_times = np.asarray(data["pred_times"], dtype=float)
    eval_times = np.asarray(data["eval_times"], dtype=float)
    pred_idx = np.clip(np.searchsorted(grid, pred_times, side="right") - 1,
                       0, horizon - 1)

    curves = conditional_curves(model, te_seqs, pred_idx, len(eval_times))

    params = np.asarray(model.params)
    with open(args.out, "w") as f:
        json.dump({
            "model": "tcsr",
            "arm": args.arm,
            "head": args.model,
            "lambda_": cfg.get("lambda_"),
            "lambda_convention": "tdsurv: 1.0 = landmarking, 0.0 = pure TD "
                                 "(inverted relative to SurvTD's lam)",
            "surv_curves": curves.tolist(),      # (N_test, n_pred_times, n_eval_times)
            "pred_times": pred_times.tolist(),
            "eval_times": eval_times.tolist(),
            "surv_labels_test": np.asarray(data["surv_labels_test"]).tolist(),
            "params_finite": bool(np.all(np.isfinite(params))),
            "params_absmax": float(np.abs(params).max()),
            "horizon": int(horizon),
            "n_feats": int(n_feats),
        }, f)


if __name__ == "__main__":
    main()
