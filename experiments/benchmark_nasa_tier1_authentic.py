"""
Authentic Tier 1 NASA C-MAPSS FD001 Benchmark.

All baseline models run STRICTLY on their original author codebases:
1. CoxSig: baselines/signature_survival (Path Signature + Cox, Bleistein et al. 2023)
2. NCDE: baselines/signature_survival (ControlledResNet Neural CDE, Bleistein et al. 2023)
3. DeepTCSR: baselines/deep_tcsr (TCN in Haiku/JAX, Vargas Vieyra & Frossard 2024 / Maystre & Russo 2022)
4. Dynamic-DeepHit: baselines/dynamic_deephit_pytorch (LSTM + Temporal Attention, Lee et al. 2020)
5. SurvTD (Ours): src/models/survtd.py (Continuous Semi-Markov Renewal TD + GRU-D)

Zero data leakage: StandardScaler fitted strictly on train units.
Identical evaluation metric: official score("c_index", ...) and score("bs", ...) from CoxSig.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.nasa_protocol_loader import get_nasa_splits
from src.evaluation.coxsig_evaluator import setup_coxsig_horizons
from src.models.survtd import SurvTDModel

# Import official CoxSig evaluation score function via importlib to avoid root src namespace collision
import importlib.util
spec = importlib.util.spec_from_file_location('sig_utils', os.path.join(ROOT_DIR, "baselines/signature_survival/src/utils.py"))
sig_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sig_utils)
score = sig_utils.score


def run_worker(worker_path: str, cwd: str, input_dict: dict, epochs: int = 25) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_pt = os.path.join(tmpdir, "input.pt")
        output_json = os.path.join(tmpdir, "output.json")
        torch.save(input_dict, input_pt)

        cmd = [
            sys.executable, worker_path,
            "--input_pt", input_pt,
            "--output_json", output_json,
            "--epochs", str(epochs)
        ]
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Worker {worker_path} failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")

        with open(output_json, "r") as f:
            data = json.load(f)

        return {
            "dynamic_cindex": np.array(data["dynamic_cindex"]),
            "dynamic_bs": np.array(data["dynamic_bs"]),
            "t0_cindex": float(data["t0_cindex"]),
            "t0_bs": float(data["t0_bs"])
        }


def resolve_survtd_alpha(name: str):
    """Model name -> SurvTD anchor weight, or None if this is not a SurvTD arm.

    `alpha_anchor` mixes the Cramer anchor loss against the TD loss. 1.0 zeroes the TD
    term and is the anchor-only control that answers Q1 ("does the TD term contribute
    anything?"); 0.0 drops the anchor and leaves TD alone; 0.5 is the configured
    default and is what `survtd` has always meant in this file.

    Until this entry the NASA benchmark had no anchor-only arm at all -- the weight was
    hardcoded at 0.5 -- so NASA was the one cohort where the paper's headline was
    measured without the control that PBC2 and Framingham both run.
    """
    if name == "survtd":
        return 0.5
    if name == "survtd_anchor_only":
        return 1.0
    if name.startswith("survtd_a"):
        alpha = float(name[len("survtd_a"):])
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha out of [0, 1] in model {name!r}")
        return alpha
    return None


def run_survtd_native(splits: dict, pred_times: np.ndarray, eval_times: np.ndarray, epochs: int = 25, seed: int = 42, alpha_anchor: float = 0.5) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"

    train_units = splits['train_units']
    df_train = splits['df_train']
    test_units = splits['test_units']
    df_test = splits['df_test']
    feat_cols = splits['feat_cols']

    # Build trajectory sequences
    def get_trajs(df, units):
        samples = []
        for u in units:
            sub = df[df['id'] == u].sort_values('times')
            times = sub['times_scaled'].values.astype(np.float32)
            features = sub[feat_cols].values.astype(np.float32)
            tte = float(sub['tte_scaled'].iloc[0])
            event = int(sub['event'].iloc[0])

            dts = np.zeros_like(times)
            dts[1:] = times[1:] - times[:-1]
            dts[0] = times[0] if times[0] > 0 else 0.01

            samples.append({
                "x": torch.from_numpy(features),
                "dts": torch.from_numpy(dts),
                "times": times,
                "tte": tte,
                "event": event
            })
        return samples

    train_samples = get_trajs(df_train, train_units)
    test_samples = get_trajs(df_test, test_units)

    model = SurvTDModel(
        input_dim=len(feat_cols),
        hidden_dim=32,
        num_bins=40,
        delta_s=0.1,
        backbone_type="grud",
        num_layers=1,
        dropout=0.1,
        lam=0.6,
        alpha_anchor=alpha_anchor,
        include_overflow=True
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        perm = np.random.permutation(len(train_samples))
        for idx in perm:
            s = train_samples[idx]
            x = s['x']
            dts = s['dts']
            tte = s['tte']
            event = s['event']

            optimizer.zero_grad()
            events_seq = torch.zeros(x.shape[0])
            if event == 1:
                events_seq[-1] = 1.0

            out = model.compute_loss_trajectory(
                x=x, dts=dts, events=events_seq, tte=tte, tau_event=tte
            )
            loss = out[0] if isinstance(out, tuple) else out
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.update_target_network()

    model.eval()

    # Evaluation
    N_test = len(test_samples)
    n_pts = len(pred_times)
    n_evs = len(eval_times)
    cindex_matrix = np.zeros((n_pts, n_evs))
    bs_matrix = np.zeros((n_pts, n_evs))

    surv_labels_test = np.array([[s['tte'], s['event']] for s in test_samples])

    with torch.no_grad():
        for j in range(n_pts):
            pt = pred_times[j]
            cond_surv = np.zeros((N_test, n_evs))

            surv_times_test = surv_labels_test[:, 0]
            surv_inds_test = surv_labels_test[:, 1]
            idx_sel = surv_times_test >= pt
            alive_indices = np.where(idx_sel)[0]

            for i in alive_indices:
                s = test_samples[i]
                times = s['times']
                valid_mask = times <= pt
                if not np.any(valid_mask):
                    valid_mask[0] = True

                x_sub = s['x'][valid_mask].unsqueeze(0)
                dts_sub = s['dts'][valid_mask].unsqueeze(0)

                _, survival, _, _ = model.forward(x_sub, dts_sub)
                surv_curve = survival[0, -1].numpy()

                for k in range(n_evs):
                    dt = eval_times[k]
                    bin_idx = min(max(int(round(dt / model.delta_s)), 0), len(surv_curve) - 1)
                    cond_surv[i, k] = surv_curve[bin_idx]

            surv_times_test = surv_labels_test[:, 0]
            surv_inds_test = surv_labels_test[:, 1]
            idx_sel = surv_times_test >= pt
            surv_times_ = surv_times_test[idx_sel] - pt
            surv_inds_ = surv_inds_test[idx_sel]
            surv_labels_ = np.array([surv_times_, surv_inds_]).T
            surv_preds_ = cond_surv[idx_sel]

            bs_matrix[j] = score("bs", surv_labels_, surv_labels_, surv_preds_, eval_times)
            cindex_matrix[j] = score("c_index", surv_labels_, surv_labels_, surv_preds_, eval_times)

    # Static t=0
    from lifelines.utils import concordance_index
    exp_times = np.zeros(N_test)
    with torch.no_grad():
        for i in range(N_test):
            s = test_samples[i]
            x_0 = s['x'][:1].unsqueeze(0)
            dt_0 = s['dts'][:1].unsqueeze(0)
            _, survival, _, _ = model.forward(x_0, dt_0)
            surv_curve = survival[0, 0].numpy()
            exp_times[i] = float(np.sum(surv_curve) * model.delta_s)

    t0_cindex = float(concordance_index(surv_labels_test[:, 0], exp_times, surv_labels_test[:, 1]))
    t0_bs = float(np.nanmean(bs_matrix[0]))

    return {
        "dynamic_cindex": cindex_matrix,
        "dynamic_bs": bs_matrix,
        "t0_cindex": t0_cindex,
        "t0_bs": t0_bs
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 101112])
    parser.add_argument("--models", nargs="+", type=str, default=["coxsig", "ncde", "deeptcsr", "ddh", "survtd"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output_json", type=str, default="experiments/results/nasa_tier1_authentic/benchmark_results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    all_results = {m: {"dynamic_cindex": [], "dynamic_bs": [], "t0_cindex": [], "t0_bs": []} for m in args.models}

    workers = {
        "coxsig": (os.path.join(ROOT_DIR, "baselines/signature_survival/coxsig_worker.py"), os.path.join(ROOT_DIR, "baselines/signature_survival")),
        "ncde": (os.path.join(ROOT_DIR, "baselines/signature_survival/ncde_worker.py"), os.path.join(ROOT_DIR, "baselines/signature_survival")),
        "deeptcsr": (os.path.join(ROOT_DIR, "baselines/deep_tcsr/deeptcsr_worker.py"), os.path.join(ROOT_DIR, "baselines/deep_tcsr")),
        "ddh": (os.path.join(ROOT_DIR, "baselines/dynamic_deephit_pytorch/ddh_worker.py"), os.path.join(ROOT_DIR, "baselines/dynamic_deephit_pytorch"))
    }

    for seed in args.seeds:
        print(f"\n=======================================================")
        print(f"   RUNNING AUTHENTIC BENCHMARK ON SEED {seed}")
        print(f"=======================================================")
        splits = get_nasa_splits(seed=seed, test_ratio=0.2)
        surv_labels_train = splits['surv_labels_train']
        pred_times, eval_times = setup_coxsig_horizons(surv_labels_train)

        worker_input = {
            'paths_train': splits['paths_train'],
            'surv_labels_train': splits['surv_labels_train'],
            'paths_test': splits['paths_test'],
            'surv_labels_test': splits['surv_labels_test'],
            'pred_times': pred_times,
            'eval_times': eval_times,
            'sampling_times': splits['sampling_times']
        }

        for m in args.models:
            print(f"--> Running Model: {m.upper()} (Authentic Codebase)...", flush=True)
            t0 = time.time()
            alpha = resolve_survtd_alpha(m)
            if alpha is not None:
                res = run_survtd_native(splits, pred_times, eval_times,
                                        epochs=args.epochs, seed=seed, alpha_anchor=alpha)
            else:
                w_path, w_cwd = workers[m]
                res = run_worker(w_path, w_cwd, worker_input, epochs=args.epochs)
            elapsed = time.time() - t0

            mean_dyn_c = float(np.nanmean(res['dynamic_cindex']))
            mean_dyn_bs = float(np.nanmean(res['dynamic_bs']))
            print(f"    Finished in {elapsed:.1f}s | Dyn C-index: {mean_dyn_c:.4f} | Dyn Brier: {mean_dyn_bs:.4f} | t0 C-index: {res['t0_cindex']:.4f}")

            all_results[m]["dynamic_cindex"].append(mean_dyn_c)
            all_results[m]["dynamic_bs"].append(mean_dyn_bs)
            all_results[m]["t0_cindex"].append(res["t0_cindex"])
            all_results[m]["t0_bs"].append(res["t0_bs"])

    # Compute summary
    summary = {}
    for m in args.models:
        summary[m] = {
            "dynamic_cindex_mean": float(np.mean(all_results[m]["dynamic_cindex"])),
            "dynamic_cindex_std": float(np.std(all_results[m]["dynamic_cindex"])),
            "dynamic_bs_mean": float(np.mean(all_results[m]["dynamic_bs"])),
            "dynamic_bs_std": float(np.std(all_results[m]["dynamic_bs"])),
            "t0_cindex_mean": float(np.mean(all_results[m]["t0_cindex"])),
            "t0_cindex_std": float(np.std(all_results[m]["t0_cindex"])),
            "t0_bs_mean": float(np.mean(all_results[m]["t0_bs"])),
            "t0_bs_std": float(np.std(all_results[m]["t0_bs"])),
            "raw_dynamic_cindex": all_results[m]["dynamic_cindex"],
            "raw_dynamic_bs": all_results[m]["dynamic_bs"],
            "raw_t0_cindex": all_results[m]["t0_cindex"],
            "raw_t0_bs": all_results[m]["t0_bs"],
        }

    final_payload = {
        "summary": summary,
        "seeds": args.seeds,
        "epochs": args.epochs
    }

    with open(args.output_json, "w") as f:
        json.dump(final_payload, f, indent=2)

    print("\n=======================================================")
    print("           AUTHENTIC BENCHMARK COMPLETE                ")
    print("=======================================================")
    for m in args.models:
        s = summary[m]
        print(f"{m.upper():12s} | Dyn C-index: {s['dynamic_cindex_mean']:.4f} ± {s['dynamic_cindex_std']:.4f} | Dyn Brier: {s['dynamic_bs_mean']:.4f} ± {s['dynamic_bs_std']:.4f} | t0 C-index: {s['t0_cindex_mean']:.4f} ± {s['t0_cindex_std']:.4f}")

if __name__ == "__main__":
    main()
