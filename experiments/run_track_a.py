"""
Track A: Paper Experiments Pipeline (EXP-01, EXP-02, EXP-07)
- EXP-01: Multi-Cohort Grand Benchmark Evaluation (Synthetic ICU, NASA C-MAPSS, PBC, Tumor Growth)
- EXP-02: Baseline Ladder Parity (KM Reference, Person-Period, Dynamic-DeepHit, DeepTCSR Clamped, SurvTD)
- EXP-07: Clinical Bedside Alarm Fatigue & Utility Evaluation (Synthetic ICU)
- Pre-Registered Kill Criteria: Kill Criterion 3 (Baseline Margin >= 0.025) & Kill Criterion 4 (Alarm Reduction >= 25%)
- Generates Table 1, Table 2, and falsification reports.
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
from src.data.dataset import expand_to_regular_grid
from src.models.survtd import SurvTDModel
from src.models.hazard_head import apply_hazard_prior_init
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
    init_mode: str = "default",
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
        return None, metrics, {"init": "n/a"}

    if method_name == "person_period":
        grid_step = getattr(spec, "person_period_grid_step", 1.0)
        train_data = expand_to_regular_grid(train_data, grid_step=grid_step)
        val_data = expand_to_regular_grid(val_data, grid_step=grid_step)
        test_data = expand_to_regular_grid(test_data, grid_step=grid_step)

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

    # Initialization scheme, applied IDENTICALLY to every neural arm. Applying it to
    # SurvTD alone would hand the proposal a free advantage over the baselines and
    # make Kill Criterion 3 unfalsifiable; all four arms expose the same
    # DiscreteHazardHead at `.head`, so parity is enforceable rather than hoped for.
    # Fit on `train_data` only -- note this is the person-period expanded split when
    # method_name == 'person_period', which is the split that model actually trains on.
    init_record = apply_hazard_prior_init(model, init_mode, train_data, delta_s)

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
    return trained_model, metrics, init_record


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
    init_mode="default",
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
    print(f"[*] Starting Track A on device: {device} | Seeds: {seeds} | Init: {init_mode}")

    # Incremental result checkpoint. The previous version wrote table1 exactly once,
    # after every cohort had finished; when the session died inside cohort 2 roughly
    # two hours of compute survived only as stdout. Results are now flushed after
    # every (cohort, seed, method) so an interrupted run loses at most one cell.
    json_path = output_path / "table1_benchmarks_raw.json"

    def _flush_raw():
        payload = {
            f"{c}_{m}": {str(s_): v for s_, v in d.items()}
            for (c, m), d in table1_raw.items()
        }
        with open(json_path, "w") as fh:
            json.dump({"init_mode": init_mode, "alpha_anchor": alpha_anchor,
                       "epochs": epochs, "seeds": seeds, "results": payload}, fh, indent=2)
        if table2_raw:
            with open(output_path / "table2_alarm_fatigue_raw.json", "w") as fh:
                json.dump({str(k): {str(s_): v for s_, v in d.items()}
                           for k, d in table2_raw.items()}, fh, indent=2)

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
                trained_model, metrics, init_record = train_and_evaluate_model(
                    method_name=method,
                    cohort_data=cohort_data,
                    spec=spec,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    alpha_anchor=alpha_anchor,
                    device=device,
                    verbose=False,
                    init_mode=init_mode,
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
                    "wall_clock_s": round(elapsed, 1),
                    "init": init_record,
                }
                _flush_raw()

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
                    _flush_raw()

    _flush_raw()

    # Format Table 1 Markdown (with sample std ddof=1)
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
                c_std = np.std(c_vals, ddof=1) if len(c_vals) > 1 else 0.0
                a_std = np.std(a_vals, ddof=1) if len(a_vals) > 1 else 0.0
                i_std = np.std(i_vals, ddof=1) if len(i_vals) > 1 else 0.0
                c_str = f"{np.mean(c_vals):.3f}±{c_std:.3f}" if len(c_vals) > 1 else f"{np.mean(c_vals):.3f}"
                a_str = f"{np.mean(a_vals):.3f}±{a_std:.3f}" if len(a_vals) > 1 else f"{np.mean(a_vals):.3f}"
                i_str = f"{np.mean(i_vals):.3f}±{i_std:.3f}" if len(i_vals) > 1 else f"{np.mean(i_vals):.3f}"
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

            th_std = np.std(ths, ddof=1) if len(ths) > 1 else 0.0
            fa_std = np.std(fas, ddof=1) if len(fas) > 1 else 0.0
            jit_std = np.std(jits, ddof=1) if len(jits) > 1 else 0.0
            jf_std = np.std(j_fracs, ddof=1) if len(j_fracs) > 1 else 0.0

            th_str = f"{np.mean(ths):.3f}±{th_std:.3f}" if len(ths) > 1 else f"{np.mean(ths):.3f}"
            fa_str = f"{np.mean(fas):.3f}±{fa_std:.3f}" if len(fas) > 1 else f"{np.mean(fas):.3f}"
            jit_str = f"{np.mean(jits):.3f}±{jit_std:.3f}" if len(jits) > 1 else f"{np.mean(jits):.3f}"
            jf_str = f"{np.mean(j_fracs):.3f}±{jf_std:.3f}" if len(j_fracs) > 1 else f"{np.mean(j_fracs):.3f}"

            t2_lines.append(f"| **{method}** | {th_str} | {fa_str} | {jit_str} | {jf_str} |")

        table2_md = "\n".join(t2_lines)
        with open(output_path / "table2_alarm_fatigue.md", "w") as f:
            f.write(table2_md + "\n")
        print(f"Saved Table 2 -> {output_path / 'table2_alarm_fatigue.md'}")

    # Adjudicate Track A Pre-Registered Kill Criteria (Kill Criterion 3 & 4)
    falsification_report_a = []

    # Kill Criterion 3: SurvTD must beat DeepTCSR and Dynamic-DeepHit by >= 0.025 in C_td / AUC
    for cohort in cohorts:
        survtd_c = [v["c_td"] for v in table1_raw.get((cohort, "survtd"), {}).values() if not np.isnan(v["c_td"])]
        deeptcsr_c = [v["c_td"] for v in table1_raw.get((cohort, "deeptcsr"), {}).values() if not np.isnan(v["c_td"])]
        deephit_c = [v["c_td"] for v in table1_raw.get((cohort, "dynamic_deephit"), {}).values() if not np.isnan(v["c_td"])]

        if survtd_c and deeptcsr_c and deephit_c:
            m_survtd = float(np.mean(survtd_c))
            m_deeptcsr = float(np.mean(deeptcsr_c))
            m_deephit = float(np.mean(deephit_c))

            delta_tcsr = m_survtd - m_deeptcsr
            delta_hit = m_survtd - m_deephit

            if delta_tcsr < 0.025:
                falsification_report_a.append({
                    "test_id": "Kill Criterion 3 (DeepTCSR Clamped)",
                    "cohort": cohort,
                    "claim": "C3",
                    "reason": f"SurvTD ({m_survtd:.4f}) failed to achieve >= 0.025 margin over DeepTCSR ({m_deeptcsr:.4f}) on {cohort} (delta: {delta_tcsr:.4f})",
                    "verdict": "FALSIFIED",
                })
            if delta_hit < 0.025:
                falsification_report_a.append({
                    "test_id": "Kill Criterion 3 (Dynamic-DeepHit)",
                    "cohort": cohort,
                    "claim": "C3",
                    "reason": f"SurvTD ({m_survtd:.4f}) failed to achieve >= 0.025 margin over Dynamic-DeepHit ({m_deephit:.4f}) on {cohort} (delta: {delta_hit:.4f})",
                    "verdict": "FALSIFIED",
                })

    # Kill Criterion 4: Bedside alarm fatigue reduction >= 25%
    if "survtd" in table2_raw and "dynamic_deephit" in table2_raw:
        survtd_fa = float(np.mean([v["false_alert_rate_per_day"] for v in table2_raw["survtd"].values()]))
        hit_fa = float(np.mean([v["false_alert_rate_per_day"] for v in table2_raw["dynamic_deephit"].values()]))
        survtd_jit = float(np.mean([v["unstable_window_rate_per_day"] for v in table2_raw["survtd"].values()]))
        hit_jit = float(np.mean([v["unstable_window_rate_per_day"] for v in table2_raw["dynamic_deephit"].values()]))

        fa_reduction = (hit_fa - survtd_fa) / max(1e-4, hit_fa)
        jit_reduction = (hit_jit - survtd_jit) / max(1e-4, hit_jit)

        if fa_reduction < 0.25 or jit_reduction < 0.25:
            falsification_report_a.append({
                "test_id": "Kill Criterion 4",
                "cohort": "synthetic_icu",
                "claim": "C4",
                "reason": f"SurvTD failed to reduce false alert rate (reduction: {fa_reduction*100:.1f}%) or jitter (reduction: {jit_reduction*100:.1f}%) by >= 25% over Dynamic-DeepHit",
                "verdict": "FALSIFIED",
            })

    with open(output_path / "falsification_report_track_a.json", "w") as f:
        json.dump(falsification_report_a, f, indent=2)
    print(f"Saved Track A Falsification Report ({len(falsification_report_a)} violations) -> {output_path / 'falsification_report_track_a.json'}")

    return table1_raw, table2_raw, falsification_report_a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 101112])
    parser.add_argument("--cohorts", type=str, nargs="+", default=["synthetic_icu", "cmapss", "pbc", "tumor"])
    parser.add_argument("--methods", type=str, nargs="+", default=["km", "person_period", "dynamic_deephit", "deeptcsr", "survtd"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alpha_anchor", type=float, default=0.5)
    parser.add_argument("--init", type=str, default="default",
                        choices=["default", "optimistic", "km_prior"],
                        help="Hazard-head initialization, applied identically to every neural arm")
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
        init_mode=args.init,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
