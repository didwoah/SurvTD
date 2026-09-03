"""
Track A: Paper Experiments Pipeline (EXP-01, EXP-02, EXP-07)
- EXP-01: Multi-Cohort Grand Benchmark Evaluation (Synthetic ICU, NASA C-MAPSS, PBC, Tumor Growth)
- EXP-02: Baseline Ladder Parity (KM Reference, Person-Period, Dynamic-DeepHit, DeepTCSR Clamped, SurvTD)
- EXP-07: Clinical Bedside Alarm Fatigue & Utility Evaluation (Synthetic ICU)
- Generates Table 1 and Table 2 in Markdown and CSV/JSON formats.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.data.cohorts import COHORTS
from src.models.survtd import SurvTDModel
from src.models.baselines.person_period import PersonPeriodModel
from src.models.baselines.dynamic_deephit import DynamicDeepHitModel
from src.models.baselines.deeptcsr_clamped import DeepTCSRClampedModel

from src.evaluation.landmark import (
    evaluate_landmarked,
    km_marginal_reference,
    landmark_labels,
    LandmarkSpec,
    DegenerateLandmarkError,
)
from src.evaluation.metrics import (
    concordance_antolini,
    integrated_brier,
    make_structured,
)
from src.evaluation.alarm_fatigue import (
    calibrate_threshold_for_ppv,
    evaluate_alarm_fatigue,
    compute_decision_curve_analysis,
)
from src.evaluation.stats import (
    compute_bootstrap_ci,
    compute_paired_bootstrap_ci,
    paired_wilcoxon_test,
)
from src.training.trainer import train_model, get_device


def evaluate_km_reference(train_dataset, test_dataset, spec: LandmarkSpec) -> dict:
    results = {}
    for landmark in spec.landmarks:
        try:
            train_labels = landmark_labels(train_dataset, landmark, spec)
            test_labels = landmark_labels(test_dataset, landmark, spec)
            if train_labels.size == 0 or int(train_labels["event"].sum()) < 2:
                continue
            if test_labels.size == 0 or int(test_labels["event"].sum()) < 2:
                continue

            ref = km_marginal_reference(train_dataset, landmark, spec)
            surv = np.tile(ref.surv[0], (test_labels.size, 1))
            c_td = concordance_antolini(surv, ref.grid, test_labels["time"], test_labels["event"].astype(float))
            ibs = integrated_brier(train_labels, test_labels, surv, ref.grid)

            for delta in spec.horizons:
                d = float(delta)
                results[(float(landmark), d)] = {
                    "c_td": c_td,
                    "auc": 0.500,
                    "ibs": ibs,
                    "n_at_risk": len(test_labels),
                    "n_events": int(test_labels["event"].sum()),
                }
        except (DegenerateLandmarkError, AssertionError):
            for delta in spec.horizons:
                results[(float(landmark), float(delta))] = {
                    "c_td": 0.500, "auc": 0.500, "ibs": float("nan"),
                    "n_at_risk": 0, "n_events": 0,
                }
    return results


def train_and_evaluate_model(
    method_name: str,
    cohort_data,
    spec,
    epochs: int,
    batch_size: int,
    lr: float,
    alpha_anchor: float,
    device: torch.device,
    verbose: bool = False,
) -> tuple:
    train_data = cohort_data.train
    val_data = cohort_data.val
    test_data = cohort_data.test
    dim = cohort_data.input_dim
    delta_s = spec.delta_s
    num_bins = spec.num_bins
    l_spec = spec.landmark_spec

    if method_name == "km":
        metrics = evaluate_km_reference(train_data, test_data, l_spec)
        return None, metrics

    if method_name == "person_period":
        model = PersonPeriodModel(
            input_dim=dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s,
            include_overflow=True
        )
    elif method_name == "dynamic_deephit":
        model = DynamicDeepHitModel(
            input_dim=dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s
        )
    elif method_name == "deeptcsr":
        model = DeepTCSRClampedModel(
            input_dim=dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s,
            alpha_anchor=alpha_anchor, include_overflow=True
        )
    elif method_name == "survtd":
        model = SurvTDModel(
            input_dim=dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s,
            alpha_anchor=alpha_anchor, include_overflow=True
        )
    else:
        raise ValueError(f"Unknown method {method_name}")

    if hasattr(model, "backbone") and hasattr(model.backbone, "set_empirical_mean"):
        model.backbone.set_empirical_mean(cohort_data.x_mean)

    trained_model = train_model(
        model=model,
        train_dataset=train_data,
        val_dataset=val_data,
        val_spec=l_spec,
        delta_s=delta_s,
        model_type=method_name,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        patience=5,
        device=device,
        alpha_anchor=alpha_anchor,
        verbose=verbose,
    )

    metrics = evaluate_landmarked(
        trained_model, train_data, test_data, l_spec, delta_s, device, strict=False
    )
    return trained_model, metrics


def extract_patient_trajectories(model, dataset, delta_s, device):
    model.eval()
    trajectories_risk = []
    trajectories_times = []
    patient_events = []

    with torch.no_grad():
        for p in dataset:
            x = p['features'].unsqueeze(0).to(device)
            dts = p['dts'].unsqueeze(0).to(device)
            mask = p['mask'].unsqueeze(0).to(device) if p['mask'] is not None else None
            times = p['times'].cpu().numpy()
            event = float(p['event'])

            _, _, _, cdf = model(x, dts, mask)
            cdf = cdf.squeeze(0).cpu().numpy()

            k_eval = min(cdf.shape[-1] - 1, max(1, cdf.shape[-1] // 2))
            risk = cdf[:, k_eval]

            trajectories_risk.append(risk)
            trajectories_times.append(times)
            patient_events.append(event)

    return trajectories_risk, trajectories_times, patient_events


def run_track_a(
    seeds=None,
    cohorts=None,
    methods=None,
    epochs=20,
    batch_size=16,
    lr=0.001,
    alpha_anchor=0.5,
    dry_run=False,
    output_dir="experiments/results",
):
    if seeds is None:
        seeds = [42, 123, 456, 789, 101112]
    if cohorts is None:
        cohorts = ["synthetic_icu", "cmapss", "pbc", "tumor"]
    if methods is None:
        methods = ["km", "person_period", "dynamic_deephit", "deeptcsr", "survtd"]

    if dry_run:
        seeds = [42]
        epochs = 2
        output_dir = "experiments/results/dry_run"
        print(f"[*] DRY RUN ACTIVE: using seed 42, 2 epochs, output -> {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"[*] Starting Track A on device: {device} | Seeds: {seeds}")

    table1_raw = {}
    table2_raw = {}

    for cohort_name in cohorts:
        spec = COHORTS[cohort_name]
        print(f"\n=======================================================")
        print(f" COHORT: {spec.display_name} ({cohort_name})")
        print(f"=======================================================")

        for seed in seeds:
            print(f"\n--- Loading Seed {seed} ---")
            cohort_data = spec.load(seed=seed)

            for method in methods:
                t0 = time.time()
                trained_model, metrics = train_and_evaluate_model(
                    method_name=method,
                    cohort_data=cohort_data,
                    spec=spec,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    alpha_anchor=alpha_anchor,
                    device=device,
                    verbose=False,
                )
                elapsed = time.time() - t0

                valid_c = [m["c_td"] for m in metrics.values() if not np.isnan(m["c_td"])]
                valid_auc = [m["auc"] for m in metrics.values() if not np.isnan(m["auc"])]
                valid_ibs = [m["ibs"] for m in metrics.values() if not np.isnan(m["ibs"])]

                c_td_mean = float(np.mean(valid_c)) if valid_c else float("nan")
                auc_mean = float(np.mean(valid_auc)) if valid_auc else float("nan")
                ibs_mean = float(np.mean(valid_ibs)) if valid_ibs else float("nan")

                print(f"[{method:>15s}] Seed {seed:6d} | C_td: {c_td_mean:.4f} | AUC: {auc_mean:.4f} | IBS: {ibs_mean:.4f} ({elapsed:.1f}s)")

                key = (cohort_name, method)
                table1_raw.setdefault(key, {})[seed] = {
                    "c_td": c_td_mean,
                    "auc": auc_mean,
                    "ibs": ibs_mean,
                }

                if cohort_name == "synthetic_icu" and trained_model is not None:
                    val_risks, _, val_events = extract_patient_trajectories(trained_model, cohort_data.val, spec.delta_s, device)
                    test_risks, test_times, test_events = extract_patient_trajectories(trained_model, cohort_data.test, spec.delta_s, device)

                    val_thresh = calibrate_threshold_for_ppv(val_risks, val_events, target_ppv=0.30)
                    af_metrics = evaluate_alarm_fatigue(
                        trajectories_risk=test_risks,
                        trajectories_times=test_times,
                        patient_events=test_events,
                        calibrated_threshold=val_thresh,
                    )
                    table2_raw.setdefault(method, {})[seed] = af_metrics

    # Save JSON
    json_path = output_path / "table1_benchmarks_raw.json"
    serializable_raw = {f"{c}_{m}": {str(s): v for s, v in seeds_dict.items()} for (c, m), seeds_dict in table1_raw.items()}
    with open(json_path, "w") as f:
        json.dump(serializable_raw, f, indent=2)

    # Format Table 1 Markdown
    md_lines = [
        "# Table 1: Multi-Cohort Dynamic Survival Performance (EXP-01 & EXP-02)",
        "",
        "Metrics: Landmarked Antolini $C^{td}$ / Uno Dynamic AUC / Integrated Brier Score (IBS) (mean ± SD across seeds).",
        "",
        "| Method | " + " | ".join([f"{c} (C / AUC / IBS)" for c in cohorts]) + " |",
        "| :--- | " + " | ".join([":---:" for _ in cohorts]) + " |",
    ]

    for method in methods:
        row = [f"**{method}**"]
        for cohort in cohorts:
            seeds_dict = table1_raw.get((cohort, method), {})
            c_vals = [v["c_td"] for v in seeds_dict.values() if not np.isnan(v["c_td"])]
            a_vals = [v["auc"] for v in seeds_dict.values() if not np.isnan(v["auc"])]
            i_vals = [v["ibs"] for v in seeds_dict.values() if not np.isnan(v["ibs"])]

            if c_vals and a_vals and i_vals:
                c_str = f"{np.mean(c_vals):.3f}±{np.std(c_vals):.3f}" if len(c_vals) > 1 else f"{np.mean(c_vals):.3f}"
                a_str = f"{np.mean(a_vals):.3f}±{np.std(a_vals):.3f}" if len(a_vals) > 1 else f"{np.mean(a_vals):.3f}"
                i_str = f"{np.mean(i_vals):.3f}±{np.std(i_vals):.3f}" if len(i_vals) > 1 else f"{np.mean(i_vals):.3f}"
                cell = f"{c_str} / {a_str} / {i_str}"
            else:
                cell = "n/a"
            row.append(cell)
        md_lines.append("| " + " | ".join(row) + " |")

    table1_md = "\n".join(md_lines)
    with open(output_path / "table1_benchmarks.md", "w") as f:
        f.write(table1_md + "\n")
    print(f"\nSaved Table 1 -> {output_path / 'table1_benchmarks.md'}")

    if table2_raw:
        t2_lines = [
            "# Table 2: Bedside Alarm Fatigue Evaluation on Synthetic ICU (EXP-07)",
            "",
            "Operating point calibrated to 0.30 PPV on validation split. Metrics reported on out-of-sample test split.",
            "",
            "| Method | Calibrated Thresh | False Alert Rate (/day) | Unstable Window Rate (/day) | Jitter Patient Fraction |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for method, s_dict in table2_raw.items():
            ths = [v["calibrated_threshold"] for v in s_dict.values()]
            fas = [v["false_alert_rate_per_day"] for v in s_dict.values()]
            jits = [v["unstable_window_rate_per_day"] for v in s_dict.values()]
            j_fracs = [v["jitter_patient_fraction"] for v in s_dict.values()]

            th_str = f"{np.mean(ths):.3f}±{np.std(ths):.3f}" if len(ths) > 1 else f"{np.mean(ths):.3f}"
            fa_str = f"{np.mean(fas):.3f}±{np.std(fas):.3f}" if len(fas) > 1 else f"{np.mean(fas):.3f}"
            jit_str = f"{np.mean(jits):.3f}±{np.std(jits):.3f}" if len(jits) > 1 else f"{np.mean(jits):.3f}"
            jf_str = f"{np.mean(j_fracs):.3f}±{np.std(j_fracs):.3f}" if len(j_fracs) > 1 else f"{np.mean(j_fracs):.3f}"

            t2_lines.append(f"| **{method}** | {th_str} | {fa_str} | {jit_str} | {jf_str} |")

        table2_md = "\n".join(t2_lines)
        with open(output_path / "table2_alarm_fatigue.md", "w") as f:
            f.write(table2_md + "\n")
        print(f"Saved Table 2 -> {output_path / 'table2_alarm_fatigue.md'}")

    return table1_raw, table2_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 101112])
    parser.add_argument("--cohorts", type=str, nargs="+", default=["synthetic_icu", "cmapss", "pbc", "tumor"])
    parser.add_argument("--methods", type=str, nargs="+", default=["km", "person_period", "dynamic_deephit", "deeptcsr", "survtd"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha_anchor", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true", help="Runs single seed dry run in safe isolated directory")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    run_track_a(
        seeds=args.seeds,
        cohorts=args.cohorts,
        methods=args.methods,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_anchor=args.alpha_anchor,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
