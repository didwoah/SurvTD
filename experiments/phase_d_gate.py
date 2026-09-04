"""
Phase D gate: the DDH port and the DDH worker must agree on PBC2.

Why this check and not another
------------------------------
D15 -- 100% of events trained as censored -- survived five audits, two leak gates, a
null-model gate and 26 unit tests. Every one of those checked a component against its
own specification. None compared two independent implementations of the same thing.

The port (`src/models/baselines/authentic/ddh.py`, verified bit-identical to upstream:
40,428 parameters, forward and `total_loss` diff 0.000e+00) and the worker
(`baselines/dynamic_deephit_pytorch/ddh_worker.py`, running the vendored package in a
subprocess) share **nothing** but the data bundle and the scorer. So if they agree,
three things are validated at once that no unit test reaches:

  * the cohort -> worker-format converter (`src/evaluation/worker_format.py`)
  * the worker contract (`--emit_curves`, the curve axes, the label convention)
  * the single scorer (`src/evaluation/curve_scoring.py`)

and if they disagree, the fault is localised to whichever side moves.

What "agree" means here
-----------------------
Not equality. The two run in different processes with different RNG streams, so they
differ by optimiser noise. The gate is therefore: **the port's C^td must land inside
the worker's own run-to-run spread**, with that spread measured rather than assumed --
`--n_worker_runs` repeats of the worker on the identical bundle. A port that had, say,
the D15 defect would not sit near the top of that spread; it would sit at 0.5.

Everything else is held identical by construction: the port is fed the same
forward-filled bundle the worker reads, discretised the same way, with the worker's own
hyperparameters (`WORKER_CFG` below, transcribed from `ddh_worker.py`), and both are
scored by `evaluate_curves`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.cohorts import COHORTS                                  # noqa: E402
from src.evaluation.curve_scoring import evaluate_curves              # noqa: E402
from src.evaluation.worker_format import build_worker_bundle          # noqa: E402
from src.models.baselines.authentic.ddh import (                      # noqa: E402
    AuthenticDynamicDeepHit,
    authentic_ddh_total_loss,
)

# Transcribed from `baselines/dynamic_deephit_pytorch/ddh_worker.py`. If the worker
# changes, this must change with it -- the gate is only meaningful while both sides
# run the same configuration.
WORKER_CFG = dict(num_bins=40, hidden_rnn=32, layers_rnn=1, typ="LSTM", risks=1,
                  lr=1e-3, weight_decay=1e-4, epochs=25, batch_size=16,
                  alpha=0.5, beta=0.5, sigma=0.1, grad_clip=1.0)

WORKER = os.path.join(ROOT, "baselines/dynamic_deephit_pytorch/ddh_worker.py")
WORKER_CWD = os.path.join(ROOT, "baselines/dynamic_deephit_pytorch")


def _bins(surv_labels_train, surv_labels_test, num_bins):
    """The worker's discretisation, `ddh_worker.py:48-51`."""
    max_tte = max(np.max(surv_labels_train[:, 0]), np.max(surv_labels_test[:, 0])) * 1.05
    edges = np.linspace(0, max_tte, num_bins + 1)
    return edges, edges[1] - edges[0]


def port_curves(bundle, seed: int, cfg=WORKER_CFG) -> np.ndarray:
    """Train the port on the bundle and return (N_test, n_pred, n_eval) curves.

    A transcription of the worker's train and predict blocks onto the port's API. The
    NaN masking, the bin arithmetic and the conditional renormalisation
    `S(L + dt) / S(L)` are the worker's, kept line-for-line so that a disagreement
    isolates the model code rather than the harness around it.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    paths_tr = bundle["paths_train"].numpy()
    paths_te = bundle["paths_test"].numpy()
    y_tr, y_te = bundle["surv_labels_train"], bundle["surv_labels_test"]
    pred_times = np.asarray(bundle["pred_times"], dtype=float)
    eval_times = np.asarray(bundle["eval_times"], dtype=float)

    feat_tr, times_tr = paths_tr[:, :, 1:], paths_tr[:, :, 0]
    feat_te, times_te = paths_te[:, :, 1:], paths_te[:, :, 0]
    n_tr, _, d_feat = feat_tr.shape
    n_te = feat_te.shape[0]

    edges, delta_s = _bins(y_tr, y_te, cfg["num_bins"])
    t_bins = np.clip(np.digitize(y_tr[:, 0], edges) - 1, 0, cfg["num_bins"] - 1)
    e_tr = y_tr[:, 1].astype(int)

    # D16: truncate training sequences at the LANDMARKS, exactly as the worker now
    # does. Truncating at `tte` instead makes sequence length a near-perfect label
    # proxy that is constant at prediction time; both sides scored below chance on
    # Framingham until this was repaired.
    xs, ts_, es_ = [], [], []
    for L in pred_times:
        for i in np.nonzero(y_tr[:, 0] > L)[0]:
            xi = np.copy(feat_tr[i])
            xi[times_tr[i] > L, :] = np.nan
            xs.append(xi)
            ts_.append(t_bins[i])
            es_.append(e_tr[i])
    x_tr = np.asarray(xs, dtype=np.float32)
    t_bins = np.asarray(ts_)
    e_tr = np.asarray(es_)
    n_tr = len(x_tr)

    model = AuthenticDynamicDeepHit(input_dim=d_feat, output_dim=cfg["num_bins"],
                                    layers_rnn=cfg["layers_rnn"],
                                    hidden_rnn=cfg["hidden_rnn"],
                                    typ=cfg["typ"], risks=cfg["risks"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])

    xt = torch.from_numpy(x_tr).float()
    tt = torch.from_numpy(t_bins).long()
    et = torch.from_numpy(e_tr).int()

    model.train()
    for _ in range(cfg["epochs"]):
        perm = np.random.permutation(n_tr)
        for b in range(0, n_tr, cfg["batch_size"]):
            idx = perm[b:b + cfg["batch_size"]]
            opt.zero_grad()
            loss = authentic_ddh_total_loss(model, xt[idx], tt[idx], et[idx],
                                            cfg["alpha"], cfg["beta"], cfg["sigma"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

    model.eval()
    curves = np.zeros((n_te, len(pred_times), len(eval_times)))
    with torch.no_grad():
        for j, pt in enumerate(pred_times):
            x_sub = np.copy(feat_te)
            for i in range(n_te):
                x_sub[i, times_te[i] > pt, :] = np.nan
            _, outcomes = model(torch.from_numpy(x_sub).float())
            surv_all = np.clip(1.0 - np.cumsum(outcomes[0].numpy(), axis=-1), 1e-5, 1.0)

            pt_bin = min(max(int(round(pt / delta_s)), 0), cfg["num_bins"] - 1)
            s_at_pt = surv_all[:, pt_bin]
            for k, dt in enumerate(eval_times):
                t_bin = min(max(int(round((pt + dt) / delta_s)), 0), cfg["num_bins"] - 1)
                curves[:, j, k] = np.clip(surv_all[:, t_bin] / np.maximum(s_at_pt, 1e-4),
                                          0.0, 1.0)
    return curves


def run_worker(bundle_path: str, out_json: str, epochs: int) -> np.ndarray:
    subprocess.run([sys.executable, WORKER, "--input_pt", bundle_path,
                    "--output_json", out_json, "--epochs", str(epochs),
                    "--emit_curves"],
                   cwd=WORKER_CWD, check=True)
    with open(out_json) as f:
        return np.asarray(json.load(f)["surv_curves"], dtype=float)


def score(curves, bundle, cohort, spec, time_scale) -> dict:
    """Both sides through the same scorer, on the cohort's ORIGINAL clock.

    The bundle's axes are divided by `time_scale` (see `worker_format.save`); curve
    VALUES are unitless, so multiplying the axes back is the whole conversion.
    """
    labels = np.array(bundle["surv_labels_test"], dtype=float, copy=True)
    labels[:, 0] *= time_scale
    return evaluate_curves(curves, labels,
                           np.asarray(bundle["pred_times"]) * time_scale,
                           np.asarray(bundle["eval_times"]) * time_scale,
                           cohort.train, spec.landmark_spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="pbc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_worker_runs", type=int, default=3)
    ap.add_argument("--n_port_runs", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = COHORTS[args.cohort]
    cohort = spec.load(seed=args.seed)
    wb = build_worker_bundle(cohort, spec)

    tmp = tempfile.mkdtemp(prefix="phase_d_")
    bundle_path = os.path.join(tmp, "bundle.pt")
    wb.save(bundle_path)
    bundle = torch.load(bundle_path, weights_only=False)

    def c_tds(curves):
        r = score(curves, bundle, cohort, spec, wb.time_scale)
        return {f"L={k[0]:g}": v["c_td"] for k, v in sorted(r.items())}, r

    worker, port = [], []
    for i in range(args.n_worker_runs):
        cur = run_worker(bundle_path, os.path.join(tmp, f"w{i}.json"),
                         WORKER_CFG["epochs"])
        m, _ = c_tds(cur)
        worker.append(m)
        print(f"worker run {i}: {m}", flush=True)
    for i in range(args.n_port_runs):
        m, _ = c_tds(port_curves(bundle, seed=args.seed + i))
        port.append(m)
        print(f"port   run {i}: {m}", flush=True)

    keys = sorted(worker[0])
    verdict, report = True, {"cohort": args.cohort, "seed": args.seed,
                             "config": WORKER_CFG, "landmarks": {}}
    print(f"\n{'landmark':>12} {'worker range':>22} {'port range':>22}  verdict")
    for k in keys:
        w = np.array([m[k] for m in worker])
        p = np.array([m[k] for m in port])
        # The port must lie inside the worker's measured spread, widened by that
        # spread again -- two independent optimiser streams, so the port's own noise
        # is of the same order and a strict containment test would fail on noise alone.
        pad = max(w.max() - w.min(), 1e-3)
        lo, hi = w.min() - pad, w.max() + pad
        ok = bool(np.all((p >= lo) & (p <= hi)))
        # Below-chance is a defect alarm on either side, whatever the agreement says.
        chance = bool(np.all(w > 0.5 + 1e-9) and np.all(p > 0.5 + 1e-9))
        verdict &= ok and chance
        report["landmarks"][k] = {
            "worker": w.tolist(), "port": p.tolist(),
            "accept_interval": [lo, hi], "inside": ok, "above_chance": chance}
        print(f"{k:>12} [{w.min():.4f}, {w.max():.4f}]  ->  [{p.min():.4f}, {p.max():.4f}]"
              f"  {'PASS' if ok and chance else 'FAIL'}")

    report["pass"] = verdict
    print(f"\nPhase D gate: {'PASS' if verdict else 'FAIL'}")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"written to {args.out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
