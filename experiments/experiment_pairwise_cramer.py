"""
Experimental Evaluation: SurvTD with Continuous First-Order Stochastic Dominance (FSD) Pairwise Ranking Loss.

Evaluates on NASA C-MAPSS FD001:
- Arm 1: SurvTD-Pure (beta_rank = 0.0)
- Arm 2: SurvTD-Rank (beta_rank = 0.1)
- Arm 3: SurvTD-Rank (beta_rank = 0.5)

Measures:
1. Dynamic Landmark C-index & Dynamic Brier Score (CoxSig protocol)
2. Static t=0 C-index & Static Brier Score (DeepTCSR protocol)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
from src.operators.pairwise_cramer import compute_batch_pairwise_cramer_loss


class NASATrajectoryDataset(Dataset):
    def __init__(self, df: any, units: np.ndarray, feat_cols: list):
        self.samples = []
        for u in units:
            sub = df[df['id'] == u].sort_values('times')
            times = sub['times_scaled'].values.astype(np.float32)
            features = sub[feat_cols].values.astype(np.float32)
            tte = float(sub['tte_scaled'].iloc[0])
            event = int(sub['event'].iloc[0])

            dts = np.zeros_like(times)
            dts[1:] = times[1:] - times[:-1]
            dts[0] = times[0] if times[0] > 0 else 0.01

            self.samples.append({
                "unit": u,
                "x": torch.from_numpy(features),
                "dts": torch.from_numpy(dts),
                "times": torch.from_numpy(times),
                "tte": tte,
                "event": event
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_survtd_with_rank(
    train_dataset: NASATrajectoryDataset,
    input_dim: int,
    beta_rank: float = 0.0,
    margin: float = 0.0,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42
) -> nn.Module:
    """Train SurvTD with optional First-Order Stochastic Dominance (FSD) pairwise ranking loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SurvTDModel(
        input_dim=input_dim,
        hidden_dim=32,
        num_bins=40,
        delta_s=0.1,
        backbone_type="grud",
        num_layers=1,
        dropout=0.1,
        lam=0.6,
        alpha_anchor=0.5,
        include_overflow=True
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    N_train = len(train_dataset)

    for epoch in range(epochs):
        perm = np.random.permutation(N_train)

        for b_start in range(0, N_train, batch_size):
            b_indices = perm[b_start:b_start + batch_size]
            batch_samples = [train_dataset[i] for i in b_indices]

            optimizer.zero_grad()
            total_survtd_loss = torch.tensor(0.0, device=device)
            patient_evals = []

            for s in batch_samples:
                x = s['x'].to(device)
                dts = s['dts'].to(device)
                tte = s['tte']
                event = s['event']

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
                loss_i = out[0] if isinstance(out, tuple) else out
                total_survtd_loss = total_survtd_loss + loss_i

                # Forward pass online network for CDF extraction
                if beta_rank > 0.0:
                    _, survival, _, _ = model.forward(x.unsqueeze(0), dts.unsqueeze(0))
                    cdf = 1.0 - survival.squeeze(0)  # (L, K)
                    times = s['times']
                    patient_evals.append({
                        'cdf_last': cdf[-1],
                        'cdf_init': cdf[0],
                        'tte': tte,
                        'last_time': float(times[-1].item()),
                        'event': event
                    })

            mean_survtd_loss = total_survtd_loss / len(batch_samples)

            if beta_rank > 0.0 and len(patient_evals) > 1:
                rank_loss, num_pairs = compute_batch_pairwise_cramer_loss(
                    patient_evals=patient_evals,
                    delta_s=model.delta_s,
                    margin=margin,
                    device=device
                )
                total_loss = mean_survtd_loss + beta_rank * rank_loss
            else:
                total_loss = mean_survtd_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if hasattr(model, "update_target_network"):
                model.update_target_network()

    model.eval()
    return model


def evaluate_survtd(
    model: nn.Module,
    test_dataset: NASATrajectoryDataset,
    pred_times: np.ndarray,
    eval_times: np.ndarray,
    device: str = "cpu"
) -> dict:
    """Evaluates SurvTD under both dynamic landmarking and static t=0."""
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

                valid_mask = times <= pred_time
                if not np.any(valid_mask):
                    valid_mask[0] = True

                x_sub = sample['x'][valid_mask].to(device)
                dts_sub = sample['dts'][valid_mask].to(device)

                _, survival, _, _ = model.forward(x_sub.unsqueeze(0), dts_sub.unsqueeze(0))
                surv_curve = survival[0, -1].cpu().numpy()

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

    # 2. Static t=0 evaluation
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
        "dynamic_cindex": float(np.nanmean(dynamic_res['c_index'])),
        "dynamic_bs": float(np.nanmean(dynamic_res['bs'])),
        "t0_cindex": float(t0_cindex),
        "t0_bs": float(t0_bs)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    beta_configs = [
        ("SurvTD-Pure (beta=0.0)", 0.0),
        ("SurvTD-Rank (beta=0.1)", 0.1),
        ("SurvTD-Rank (beta=0.5)", 0.5),
        ("SurvTD-Rank (beta=1.0)", 1.0)
    ]

    all_results = {name: [] for name, _ in beta_configs}

    print("=" * 80)
    print("SURVTD CONTINUOUS FIRST-ORDER STOCHASTIC DOMINANCE (FSD) PAIRWISE EXPERIMENT")
    print(f"Seeds: {args.seeds} | Epochs: {args.epochs}")
    print("=" * 80)

    for s in args.seeds:
        print(f"\n>>> Running Seed {s}...")
        splits = get_nasa_splits(seed=s, test_ratio=0.2)
        train_ds = NASATrajectoryDataset(splits['df_train'], splits['train_units'], splits['feat_cols'])
        test_ds = NASATrajectoryDataset(splits['df_test'], splits['test_units'], splits['feat_cols'])
        input_dim = len(splits['feat_cols'])
        pred_times, eval_times = setup_coxsig_horizons(splits['surv_labels_train'])

        for name, beta in beta_configs:
            t0 = time.time()
            model = train_survtd_with_rank(
                train_dataset=train_ds,
                input_dim=input_dim,
                beta_rank=beta,
                epochs=args.epochs,
                device=args.device,
                seed=s
            )
            eval_metrics = evaluate_survtd(model, test_ds, pred_times, eval_times, device=args.device)
            elapsed = time.time() - t0
            all_results[name].append(eval_metrics)

            print(f"  [{name}] Dyn C-idx: {eval_metrics['dynamic_cindex']:.4f} | Dyn Brier: {eval_metrics['dynamic_bs']:.4f} | "
                  f"t=0 C-idx: {eval_metrics['t0_cindex']:.4f} | Elapsed: {elapsed:.1f}s")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY ACROSS SEEDS:")
    print("=" * 80)
    print(f"{'Model Configuration':<28} | {'Dynamic C-index':<18} | {'Dynamic Brier':<18} | {'t=0 C-index':<18}")
    print("-" * 80)

    for name, _ in beta_configs:
        dyn_c = [m['dynamic_cindex'] for m in all_results[name]]
        dyn_b = [m['dynamic_bs'] for m in all_results[name]]
        t0_c = [m['t0_cindex'] for m in all_results[name]]

        dyn_c_str = f"{np.mean(dyn_c):.4f} +/- {np.std(dyn_c):.4f}"
        dyn_b_str = f"{np.mean(dyn_b):.4f} +/- {np.std(dyn_b):.4f}"
        t0_c_str = f"{np.mean(t0_c):.4f} +/- {np.std(t0_c):.4f}"

        print(f"{name:<28} | {dyn_c_str:<18} | {dyn_b_str:<18} | {t0_c_str:<18}")

    out_path = os.path.join(ROOT_DIR, "experiments", "results", "pairwise_fsd_experiment_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw results to: {out_path}")


if __name__ == "__main__":
    main()
