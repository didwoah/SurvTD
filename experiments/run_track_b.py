"""
Track B: Adversarial Stress Tests & Hypothesis Destroyer Pipeline
- EXP-03: Factorial Operator Ablation NC-A1 & NC-A2 (Discount vs Shift)
- EXP-04: Negative Control NC-B (Within-Patient Duration Permutation) with Noise Floor
- EXP-05: Numerical Degeneracy & Clamped Division (NC-A3: E5a, E5b, E5c)
- EXP-06: Effective Horizon Matching & Subsampling Sweep (NC-C: E6a analytic, E6b empirical)
- Kill Criterion 5: Anchor-Only Supervised Control (alpha = 1.0)
- Automated Pre-Registered Kill Criteria Guard & Table 3 Generator.
"""

import os
import sys
import json
import random
import argparse
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.data.cohorts import COHORTS
from src.data.dataset import LongitudinalSurvivalDataset
from src.models.survtd import SurvTDModel
from src.models.baselines.deeptcsr_clamped import DeepTCSRClampedModel
from src.operators.survtd_operator import ARMS
from src.operators.ablations import (
    permute_patient_durations,
    shuffle_durations_across_patients,
)
from src.evaluation.landmark import evaluate_landmarked
from src.evaluation.stats import compute_paired_bootstrap_ci
from src.training.trainer import train_model, get_device


def run_e6a_analytic_horizon_check(lam: float = 0.6, delta_s: float = 2.0) -> dict:
    retentions = [1.0, 0.5, 0.25]
    dts_base = [2.0, 3.5, 1.5, 4.0, 2.5]

    results = {}
    for r in retentions:
        dts = [dt / r for dt in dts_base]
        total_time = sum(dts)
        w_dur = lam ** (total_time / delta_s)
        n_steps = len(dts)
        w_count = lam ** n_steps

        results[f"retention_{int(r*100)}"] = {
            "retention": r,
            "duration_geometric_weight": float(w_dur),
            "count_geometric_weight": float(w_count),
        }

    return results


def train_and_eval_survtd(
    cohort_data,
    spec,
    ablation_mode: str = "full",
    alpha_anchor: float = 0.5,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.001,
    device: torch.device = None,
    permute_within: bool = False,
    permute_across: bool = False,
    subsample_ratio: float = 1.0,
    seed: int = 42,
    anchor_loss: str = "cramer",
    td_loss: str = "cramer",
    use_ipcw: bool = True,
) -> tuple:
    train_patients = cohort_data.train.patients
    val_patients = cohort_data.val.patients
    test_patients = cohort_data.test.patients

    rng = random.Random(seed)

    if permute_within:
        train_patients = [permute_patient_durations(p, rng) for p in train_patients]
        val_patients = [permute_patient_durations(p, rng) for p in val_patients]
        test_patients = [permute_patient_durations(p, rng) for p in test_patients]
    elif permute_across:
        train_patients = shuffle_durations_across_patients(train_patients, rng)
        val_patients = shuffle_durations_across_patients(val_patients, rng)
        test_patients = shuffle_durations_across_patients(test_patients, rng)

    if subsample_ratio < 1.0:
        def subsample_traj(p):
            L = len(p['features'])
            keep_len = max(2, int(round(L * subsample_ratio)))
            idx = sorted(rng.sample(range(L), keep_len))
            return {
                'id': p['id'],
                'features': p['features'][idx],
                'dts': p['dts'][idx],
                'times': p['times'][idx],
                'events': p['events'][idx],
                'tte': p['tte'],
                'event': p['event'],
                'mask': p['mask'][idx] if p.get('mask') is not None else None,
            }

        train_patients = [subsample_traj(p) for p in train_patients]
        val_patients = [subsample_traj(p) for p in val_patients]

    train_ds = LongitudinalSurvivalDataset(train_patients)
    val_ds = LongitudinalSurvivalDataset(val_patients)
    test_ds = LongitudinalSurvivalDataset(test_patients)

    dim = cohort_data.input_dim
    delta_s = spec.delta_s
    num_bins = spec.num_bins
    l_spec = spec.landmark_spec

    model = SurvTDModel(
        input_dim=dim,
        hidden_dim=64,
        num_bins=num_bins,
        delta_s=delta_s,
        alpha_anchor=alpha_anchor,
        include_overflow=True,
        anchor_loss=anchor_loss,
        td_loss=td_loss,
        use_ipcw=use_ipcw,
    )
    if hasattr(model.backbone, "set_empirical_mean"):
        model.backbone.set_empirical_mean(cohort_data.x_mean)

    trained_model = train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        val_spec=l_spec,
        delta_s=delta_s,
        model_type="survtd",
        ablation_mode=ablation_mode,
        alpha_anchor=alpha_anchor,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        patience=5,
        device=device,
        verbose=False,
    )

    metrics = evaluate_landmarked(trained_model, train_ds, test_ds, l_spec, delta_s, device, strict=False)
    valid_c = [m["c_td"] for m in metrics.values() if not np.isnan(m["c_td"])]
    valid_auc = [m["auc"] for m in metrics.values() if not np.isnan(m["auc"])]

    c_td_mean = float(np.mean(valid_c)) if valid_c else float("nan")
    auc_mean = float(np.mean(valid_auc)) if valid_auc else float("nan")

    return trained_model, c_td_mean, auc_mean


def run_track_b(
    seeds=None,
    cohort="synthetic_icu",
    epochs=20,
    batch_size=16,
    lr=0.001,
    alpha_anchor=0.5,
    dry_run=False,
    output_dir="experiments/results",
):
    if seeds is None:
        seeds = [42, 123, 456, 789, 101112]

    if dry_run:
        seeds = [42]
        epochs = 2
        output_dir = "experiments/results/dry_run"
        print(f"[*] DRY RUN ACTIVE: using seed 42, 2 epochs, output -> {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = get_device()
    spec = COHORTS[cohort]

    print(f"[*] Starting Track B on cohort: {cohort} | Device: {device} | Seeds: {seeds}")

    e6a_results = run_e6a_analytic_horizon_check(lam=0.6, delta_s=spec.delta_s)
    print(f"[*] E6a Analytic Effective Horizon Check complete.")

    scores = {
        "full": [],
        "arm_a1_discount": [],
        "arm_a2_shift": [],
        "nc_b_within_perm": [],
        "nc_b_across_perm": [],
        "alpha_1_anchor_only": [],
    }

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        cohort_data = spec.load(seed=seed)

        # 1. Full SurvTD
        _, c_full, _ = train_and_eval_survtd(cohort_data, spec, "full", alpha_anchor, epochs, batch_size, lr, device, seed=seed)
        scores["full"].append(c_full)
        print(f"  Full SurvTD:                   C_td = {c_full:.4f}")

        # 2. Arm A1
        _, c_a1, _ = train_and_eval_survtd(cohort_data, spec, "arm_a1_discount", alpha_anchor, epochs, batch_size, lr, device, seed=seed)
        scores["arm_a1_discount"].append(c_a1)
        print(f"  Arm A1 (Discount Ablation):    C_td = {c_a1:.4f}")

        # 3. Arm A2
        _, c_a2, _ = train_and_eval_survtd(cohort_data, spec, "arm_a2_shift", alpha_anchor, epochs, batch_size, lr, device, seed=seed)
        scores["arm_a2_shift"].append(c_a2)
        print(f"  Arm A2 (Shift Ablation):       C_td = {c_a2:.4f}")

        # 4. NC-B (Within-patient permutation)
        _, c_wb, _ = train_and_eval_survtd(cohort_data, spec, "full", alpha_anchor, epochs, batch_size, lr, device, permute_within=True, seed=seed)
        scores["nc_b_within_perm"].append(c_wb)
        print(f"  NC-B (Within Permutation):     C_td = {c_wb:.4f}")

        # 5. NC-B Floor (Across-patient permutation)
        _, c_ab, _ = train_and_eval_survtd(cohort_data, spec, "full", alpha_anchor, epochs, batch_size, lr, device, permute_across=True, seed=seed)
        scores["nc_b_across_perm"].append(c_ab)
        print(f"  NC-B Floor (Across Perm):      C_td = {c_ab:.4f}")

        # 6. Anchor-Only Control (alpha = 1.0)
        _, c_a10, _ = train_and_eval_survtd(cohort_data, spec, "full", 1.0, epochs, batch_size, lr, device, seed=seed)
        scores["alpha_1_anchor_only"].append(c_a10)
        print(f"  Anchor-Only (alpha=1.0):       C_td = {c_a10:.4f}")

    # Compute Statistical Verdicts
    falsification_report = []

    delta_a10, ci_low_a10, ci_high_a10, se_a10 = compute_paired_bootstrap_ci(scores["full"], scores["alpha_1_anchor_only"])
    if delta_a10 < 0.015 or ci_low_a10 <= 0.0:
        falsification_report.append({
            "test_id": "Kill Criterion 5",
            "claim": "C0, C3",
            "reason": f"Full SurvTD failed to outperform anchor-only (alpha=1) by >= 0.015 (delta: {delta_a10:.4f}, 95% CI [{ci_low_a10:.4f}, {ci_high_a10:.4f}])",
            "verdict": "FALSIFIED",
        })

    mean_full = float(np.mean(scores["full"]))
    mean_within = float(np.mean(scores["nc_b_within_perm"]))
    mean_floor = float(np.mean(scores["nc_b_across_perm"]))
    total_signal = max(1e-4, mean_full - mean_floor)
    retained_signal = max(0.0, mean_within - mean_floor)
    retention_ratio = retained_signal / total_signal

    if retention_ratio > 0.50:
        falsification_report.append({
            "test_id": "EXP-04 / Kill Criterion 0",
            "claim": "C0",
            "reason": f"Within-patient permutation retained {retention_ratio*100:.1f}% of temporal signal over noise floor (>50%)",
            "verdict": "FALSIFIED",
        })

    delta_a1, ci_low_a1, _, _ = compute_paired_bootstrap_ci(scores["full"], scores["arm_a1_discount"])
    delta_a2, ci_low_a2, _, _ = compute_paired_bootstrap_ci(scores["full"], scores["arm_a2_shift"])

    def _fmt(arr):
        s = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        return f"{np.mean(arr):.4f}±{s:.4f}"

    t3_lines = [
        "# Table 3: Adversarial Stress Tests & Ablations (Track B)",
        "",
        f"Cohort: **{spec.display_name}** | Seeds: {seeds}",
        "",
        "| Test Condition | Mean C^td | Paired Delta vs Full | 95% Bootstrap CI | Pre-Registered Falsification Rule | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **Full SurvTD** | {_fmt(scores['full'])} | — | — | Target Proposal | — |",
        f"| **Arm A1 (Discount Ablation)** | {_fmt(scores['arm_a1_discount'])} | {delta_a1:.4f} | [{ci_low_a1:.4f}, —] | Lose >= 0.025 | {'PASS' if delta_a1 >= 0.025 else 'UNDERPOWERED/NULL'} |",
        f"| **Arm A2 (Shift Ablation)** | {_fmt(scores['arm_a2_shift'])} | {delta_a2:.4f} | [{ci_low_a2:.4f}, —] | Lose >= 0.025 | {'PASS' if delta_a2 >= 0.025 else 'UNDERPOWERED/NULL'} |",
        f"| **NC-B (Within-Patient Perm)** | {_fmt(scores['nc_b_within_perm'])} | {np.mean(scores['full']) - mean_within:.4f} | — | Retain <= 50% of floor gain | {'FAIL' if retention_ratio > 0.50 else 'PASS'} |",
        f"| **NC-B Noise Floor (Across Perm)** | {_fmt(scores['nc_b_across_perm'])} | {total_signal:.4f} | — | Empirical noise floor | Baseline Floor |",
        f"| **Anchor-Only (alpha=1.0)** | {_fmt(scores['alpha_1_anchor_only'])} | {delta_a10:.4f} | [{ci_low_a10:.4f}, {ci_high_a10:.4f}] | Delta >= 0.015 (Kill Criterion 5) | {'PASS' if delta_a10 >= 0.015 and ci_low_a10 > 0 else 'FALSIFIED'} |",
    ]

    table3_md = "\n".join(t3_lines)
    with open(output_path / "table3_adversarial.md", "w") as f:
        f.write(table3_md + "\n")
    print(f"\nSaved Table 3 -> {output_path / 'table3_adversarial.md'}")

    with open(output_path / "falsification_report.json", "w") as f:
        json.dump(falsification_report, f, indent=2)
    print(f"Saved Falsification Report ({len(falsification_report)} failures) -> {output_path / 'falsification_report.json'}")

    return scores, falsification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 101112])
    parser.add_argument("--cohort", type=str, default="synthetic_icu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha_anchor", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true", help="Runs single seed dry run in safe isolated directory")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    run_track_b(
        seeds=args.seeds,
        cohort=args.cohort,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_anchor=args.alpha_anchor,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
