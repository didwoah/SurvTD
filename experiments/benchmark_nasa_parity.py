"""
Unified NASA C-MAPSS FD001 Benchmark Evaluation: CoxSig vs DeepTCSR vs SurvTD.

Evaluates under:
1. CoxSig-style Dynamic Landmark Protocol (pred_times x eval_times)
2. Static Initial-State Protocol (t=0, for direct DeepTCSR Table 1 comparison)
Strictly leak-free StandardScaler fitted on training split only.
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure root is on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.nasa_protocol_loader import get_nasa_splits, RELEVANT_FEATURES
from src.evaluation.coxsig_evaluator import (
    setup_coxsig_horizons,
    evaluate_coxsig_at_landmarks,
    compute_coxsig_cindex,
    compute_coxsig_brier
)
from src.models.survtd import SurvTDModel
from src.models.baselines.deeptcsr_clamped import DeepTCSRClampedModel


class NASATrajectoryDataset(Dataset):
    """PyTorch dataset for variable-length engine sensor sequences."""
    def __init__(self, df: any, units: np.ndarray, feat_cols: List[str]):
        self.samples = []
        for u in units:
            sub = df[df['id'] == u].sort_values('times')
            times = sub['times_scaled'].values.astype(np.float32)
            features = sub[feat_cols].values.astype(np.float32)
            tte = float(sub['tte_scaled'].iloc[0])
            event = int(sub['event'].iloc[0])
            
            # Inter-observation durations
            dts = np.zeros_like(times)
            dts[1:] = times[1:] - times[:-1]
            dts[0] = times[0] if times[0] > 0 else 0.01

            self.samples.append({
                "unit": u,
                "x": torch.from_numpy(features),      # (L, D)
                "dts": torch.from_numpy(dts),         # (L,)
                "times": torch.from_numpy(times),     # (L,)
                "tte": tte,
                "event": event
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def run_coxsig_evaluation(
    splits: dict,
    pred_times: np.ndarray,
    eval_times: np.ndarray
) -> Dict[str, any]:
    """Train and evaluate CoxSig using an isolated subprocess to prevent namespace collision."""
    import subprocess
    import tempfile
    coxsig_dir = os.path.join(ROOT_DIR, "baselines", "signature_survival")
    worker_script = os.path.join(coxsig_dir, "coxsig_worker.py")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_pt = os.path.join(tmpdir, "input.pt")
        output_json = os.path.join(tmpdir, "output.json")

        torch.save({
            "paths_train": splits['paths_train'],
            "surv_labels_train": splits['surv_labels_train'],
            "paths_test": splits['paths_test'],
            "surv_labels_test": splits['surv_labels_test'],
            "pred_times": pred_times,
            "eval_times": eval_times
        }, input_pt)

        cmd = [sys.executable, worker_script, "--input_pt", input_pt, "--output_json", output_json]
        res = subprocess.run(cmd, cwd=coxsig_dir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"CoxSig worker failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")

        with open(output_json, "r") as f:
            data = json.load(f)

        return {
            "dynamic_cindex": np.array(data["dynamic_cindex"]),
            "dynamic_bs": np.array(data["dynamic_bs"]),
            "t0_cindex": data["t0_cindex"],
            "t0_bs": data["t0_bs"]
        }



def train_pytorch_model(
    model: nn.Module,
    train_dataset: NASATrajectoryDataset,
    epochs: int = 25,
    lr: float = 1e-3,
    device: str = "cpu"
) -> nn.Module:
    """Train SurvTD or DeepTCSR in PyTorch with gradient updates."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        perm = np.random.permutation(len(train_dataset))
        epoch_loss = 0.0
        for idx in perm:
            sample = train_dataset[idx]
            x = sample['x'].to(device)
            dts = sample['dts'].to(device)
            tte = sample['tte']
            event = sample['event']

            optimizer.zero_grad()
            events_seq = torch.zeros(x.shape[0], device=device)
            if event == 1:
                events_seq[-1] = 1.0

            out = model.compute_loss_trajectory(
                x=x,
                dts=dts,
                events=events_seq,
                tte=tte,
                tau_event=tte
            )
            loss = out[0] if isinstance(out, tuple) else out
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if hasattr(model, "update_target_network"):
                model.update_target_network()

            epoch_loss += loss.item()

    model.eval()
    return model


def evaluate_pytorch_model_coxsig_protocol(
    model: nn.Module,
    test_dataset: NASATrajectoryDataset,
    pred_times: np.ndarray,
    eval_times: np.ndarray,
    device: str = "cpu"
) -> Dict[str, any]:
    """Evaluate PyTorch model under both dynamic landmarking and static t=0."""
    surv_labels_test = np.zeros((len(test_dataset), 2))
    for i, s in enumerate(test_dataset):
        surv_labels_test[i, 0] = s['tte']
        surv_labels_test[i, 1] = s['event']

    def predict_cond_surv(pred_time: float, horizons: np.ndarray, alive_indices: np.ndarray) -> np.ndarray:
        preds = np.zeros((len(alive_indices), len(horizons)), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for k, orig_idx in enumerate(alive_indices):
                sample = test_dataset[orig_idx]
                times = sample['times'].numpy()

                # Truncate sequence up to pred_time
                valid_mask = times <= pred_time
                if not np.any(valid_mask):
                    valid_mask[0] = True
                
                x_sub = sample['x'][valid_mask].to(device)
                dts_sub = sample['dts'][valid_mask].to(device)

                # Forward pass
                # model outputs: hazard, survival, pmf, cdf of shape (1, L, K)
                _, survival, _, _ = model.forward(x_sub.unsqueeze(0), dts_sub.unsqueeze(0))
                surv_curve = survival[0, -1].cpu().numpy()  # conditional survival from pred_time

                # Map eval_times to bin indices
                delta_s = getattr(model, "delta_s", 0.1)
                for h_idx, h in enumerate(horizons):
                    bin_idx = int(round(h / delta_s))
                    bin_idx = min(max(bin_idx, 0), len(surv_curve) - 1)
                    preds[k, h_idx] = surv_curve[bin_idx]

        return preds

    # 1. Dynamic landmark evaluation
    dynamic_res = evaluate_coxsig_at_landmarks(
        surv_labels_test=surv_labels_test,
        pred_times=pred_times,
        eval_times=eval_times,
        predict_cond_surv_fn=predict_cond_surv
    )

    # 2. Static t=0 evaluation via Lifelines Concordance Index (DeepTCSR protocol)
    from lifelines.utils import concordance_index
    exp_times = np.zeros(len(test_dataset))
    model.eval()
    delta_s = getattr(model, "delta_s", 0.1)
    with torch.no_grad():
        for i, sample in enumerate(test_dataset):
            x_0 = sample['x'][:1].to(device)
            dt_0 = sample['dts'][:1].to(device)
            _, survival, _, _ = model.forward(x_0.unsqueeze(0), dt_0.unsqueeze(0))
            surv_curve = survival[0, 0].cpu().numpy()
            exp_times[i] = float(np.sum(surv_curve) * delta_s)

    try:
        t0_cindex = float(concordance_index(surv_labels_test[:, 0], exp_times, surv_labels_test[:, 1]))
    except Exception:
        t0_cindex = 0.5
    t0_bs = float(np.nanmean(dynamic_res['bs'][0]))

    return {
        "dynamic_cindex": dynamic_res['c_index'],
        "dynamic_bs": dynamic_res['bs'],
        "t0_cindex": t0_cindex,
        "t0_bs": t0_bs
    }



def run_benchmark(
    seeds: List[int] = (42, 123, 456, 789, 101112),
    models: List[str] = ("coxsig", "deeptcsr", "survtd"),
    epochs: int = 25,
    output_dir: str = "experiments/results/nasa_parity"
):
    os.makedirs(output_dir, exist_ok=True)
    all_results = {m: {"dynamic_cindex": [], "dynamic_bs": [], "t0_cindex": [], "t0_bs": []} for m in models}

    for seed in seeds:
        print(f"\n==========================================")
        print(f"   RUNNING BENCHMARK ON SEED {seed}")
        print(f"==========================================")
        splits = get_nasa_splits(seed=seed, test_ratio=0.2)
        surv_labels_train = splits['surv_labels_train']
        pred_times, eval_times = setup_coxsig_horizons(surv_labels_train)
        print(f"Pred times (quantiles): {pred_times.round(3)}")
        print(f"Eval times (residual):  {eval_times.round(3)}")

        train_ds = NASATrajectoryDataset(splits['df_train'], splits['train_units'], splits['feat_cols'])
        test_ds = NASATrajectoryDataset(splits['df_test'], splits['test_units'], splits['feat_cols'])

        # 1. CoxSig
        if "coxsig" in models:
            print("\n--- Running CoxSig ---")
            t_start = time.time()
            res_cox = run_coxsig_evaluation(splits, pred_times, eval_times)
            elapsed = time.time() - t_start
            print(f"CoxSig completed in {elapsed:.1f}s | t=0 C-index: {res_cox['t0_cindex']:.4f} | Dynamic C-index: {np.nanmean(res_cox['dynamic_cindex']):.4f}")
            for k in res_cox:
                all_results["coxsig"][k].append(res_cox[k])

        # 2. DeepTCSR
        if "deeptcsr" in models:
            print("\n--- Running DeepTCSR ---")
            t_start = time.time()
            dtcsr_model = DeepTCSRClampedModel(
                input_dim=len(splits['feat_cols']),
                hidden_dim=32,
                num_bins=40,
                delta_s=0.1,
                backbone_type="grud",
                num_layers=1,
                dropout=0.1
            )
            dtcsr_model = train_pytorch_model(dtcsr_model, train_ds, epochs=epochs, lr=1e-3)
            res_dtcsr = evaluate_pytorch_model_coxsig_protocol(dtcsr_model, test_ds, pred_times, eval_times)
            elapsed = time.time() - t_start
            print(f"DeepTCSR completed in {elapsed:.1f}s | t=0 C-index: {res_dtcsr['t0_cindex']:.4f} | Dynamic C-index: {np.nanmean(res_dtcsr['dynamic_cindex']):.4f}")
            for k in res_dtcsr:
                all_results["deeptcsr"][k].append(res_dtcsr[k])

        # 3. SurvTD
        if "survtd" in models:
            print("\n--- Running SurvTD ---")
            t_start = time.time()
            survtd_model = SurvTDModel(
                input_dim=len(splits['feat_cols']),
                hidden_dim=32,
                num_bins=40,
                delta_s=0.1,
                backbone_type="grud",
                num_layers=1,
                dropout=0.1,
                lam=0.6,
                alpha_anchor=0.5
            )
            survtd_model = train_pytorch_model(survtd_model, train_ds, epochs=epochs, lr=1e-3)
            res_survtd = evaluate_pytorch_model_coxsig_protocol(survtd_model, test_ds, pred_times, eval_times)
            elapsed = time.time() - t_start
            print(f"SurvTD completed in {elapsed:.1f}s | t=0 C-index: {res_survtd['t0_cindex']:.4f} | Dynamic C-index: {np.nanmean(res_survtd['dynamic_cindex']):.4f}")
            for k in res_survtd:
                all_results["survtd"][k].append(res_survtd[k])

    # Consolidated summary
    summary = {}
    print("\n" + "=" * 80)
    print("       FINAL NASA FD001 BENCHMARK SUMMARY (5 SEEDS)")
    print("=" * 80)
    print(f"{'Model':<12} | {'t=0 C-index':<16} | {'t=0 Brier':<16} | {'Dynamic C-index':<16} | {'Dynamic Brier':<16}")
    print("-" * 84)

    for m in models:
        t0_ci = np.array(all_results[m]["t0_cindex"])
        t0_bs = np.array(all_results[m]["t0_bs"])
        dyn_ci = np.array([np.nanmean(arr) for arr in all_results[m]["dynamic_cindex"]])
        dyn_bs = np.array([np.nanmean(arr) for arr in all_results[m]["dynamic_bs"]])
        summary[m] = {
            "t0_cindex_mean": float(np.mean(t0_ci)),
            "t0_cindex_std": float(np.std(t0_ci)),
            "t0_bs_mean": float(np.mean(t0_bs)),
            "t0_bs_std": float(np.std(t0_bs)),
            "dynamic_cindex_mean": float(np.mean(dyn_ci)),
            "dynamic_cindex_std": float(np.std(dyn_ci)),
            "dynamic_bs_mean": float(np.mean(dyn_bs)),
            "dynamic_bs_std": float(np.std(dyn_bs)),
            "raw_t0_cindex": [float(x) for x in t0_ci],
            "raw_t0_bs": [float(x) for x in t0_bs],
            "raw_dynamic_cindex": [float(x) for x in dyn_ci],
            "raw_dynamic_bs": [float(x) for x in dyn_bs]
        }
        print(f"{m:<12} | {np.mean(t0_ci):.4f} ± {np.std(t0_ci):.4f} | {np.mean(t0_bs):.4f} ± {np.std(t0_bs):.4f} | {np.mean(dyn_ci):.4f} ± {np.std(dyn_ci):.4f} | {np.mean(dyn_bs):.4f} ± {np.std(dyn_bs):.4f}")
    print("=" * 80)

    out_file = os.path.join(output_dir, "parity_results.json")
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "seeds": list(seeds), "epochs": epochs}, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 101112])
    parser.add_argument("--models", nargs="+", default=["coxsig", "deeptcsr", "survtd"])
    args = parser.parse_args()

    run_benchmark(seeds=args.seeds, models=args.models, epochs=args.epochs)
