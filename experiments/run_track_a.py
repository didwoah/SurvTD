"""
Track A: Paper Experiments Pipeline (EXP-01, EXP-02, EXP-07)
- EXP-01: Multi-Cohort Grand Benchmark Evaluation (NASA C-MAPSS, PBC, Tumor Growth, Sepsis-3)
- EXP-02: Baseline Ladder Parity (Person-Period, Dynamic-DeepHit, DeepTCSR Clamped, SurvTD)
- EXP-07: Clinical Bedside Alarm Fatigue & Utility Evaluation (Sepsis-3)
- Generates Table 1 and Table 2 in Markdown and CSV formats.
"""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch


from src.models.survtd import SurvTDModel
from src.models.baselines.person_period import PersonPeriodModel
from src.models.baselines.dynamic_deephit import DynamicDeepHitModel
from src.models.baselines.deeptcsr_clamped import DeepTCSRClampedModel

from src.data.cmapss_loader import load_cmapss_downsampled
from src.data.pbc_loader import load_pbc
from src.data.tumor_loader import generate_tumor_growth_cohort
from src.data.sepsis_loader import generate_sepsis_icu_cohort

from src.evaluation.metrics import (
    compute_concordance_td,
    compute_time_dependent_auc,
    compute_integrated_brier_score
)
from src.evaluation.alarm_fatigue import evaluate_alarm_fatigue, compute_decision_curve_analysis
from src.evaluation.stats import compute_bootstrap_ci, paired_wilcoxon_test
from src.training.trainer import train_model, get_device


def evaluate_dataset_model(model, test_dataset, model_type, delta_s, max_horizon, device):
    """
    Evaluates a trained model on a test cohort and returns C-index, AUC, IBS, and risk trajectories.
    """
    model.eval()
    risk_scores = []
    survival_curves = []
    trajectories_risk = []
    trajectories_times = []
    event_times = []
    event_indicators = []

    with torch.no_grad():
        for p in test_dataset:
            x = p['features'].unsqueeze(0).to(device)
            dts = p['dts'].unsqueeze(0).to(device)
            mask = p['mask'].unsqueeze(0).to(device) if p['mask'] is not None else None
            tte = float(p['tte'])
            event = float(p['event'])
            times = p['times'].cpu().numpy()

            _, survival, pmf, cdf = model(x, dts, mask)
            cdf = cdf.squeeze(0).cpu().numpy()          # (L, K)
            survival = survival.squeeze(0).cpu().numpy()  # (L, K)

            # Trajectory evaluation
            # Representative risk score at terminal visit or median visit
            risk_score = float(cdf[-1, min(len(cdf[-1]) // 2, len(cdf[-1]) - 1)])
            risk_scores.append(risk_score)
            survival_curves.append(survival[-1])

            # Patient-level longitudinal risk at target horizon
            h_idx = min(int(round(max_horizon * 0.5 / delta_s)), cdf.shape[-1] - 1)
            trajectories_risk.append(cdf[:, h_idx])
            trajectories_times.append(times)

            event_times.append(tte)
            event_indicators.append(event)

    risk_arr = np.array(risk_scores)
    time_arr = np.array(event_times)
    event_arr = np.array(event_indicators)

    # 1. C-index
    c_index = compute_concordance_td(risk_arr, time_arr, event_arr)

    # 2. Time-Dependent AUC at median follow-up
    eval_horizon = float(np.median(time_arr))
    auc = compute_time_dependent_auc(risk_arr, time_arr, event_arr, eval_horizon)

    # 3. Integrated Brier Score
    eval_grid = np.linspace(eval_horizon * 0.2, eval_horizon * 1.5, 8)
    eval_grid = eval_grid[eval_grid < max_horizon]
    if len(eval_grid) == 0:
        eval_grid = np.array([eval_horizon])
    ibs = compute_integrated_brier_score(survival_curves, eval_grid, time_arr, event_arr, delta_s)

    return {
        'c_index': c_index,
        'auc': auc,
        'ibs': ibs,
        'trajectories_risk': trajectories_risk,
        'trajectories_times': trajectories_times,
        'event_indicators': event_arr
    }


def run_track_a(epochs: int = 15, seeds: list = [0, 1, 2, 3, 4], dry_run: bool = False, output_dir: str = "experiments/results"):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"=== [Track A] Starting Paper Experiments on {device} ===")

    if dry_run:
        epochs = 2
        seeds = [0]
        print(">> Running in DRY RUN mode (2 epochs, 1 seed)")

    benchmarks = ["mimic_sepsis", "cmapss", "pbc", "tumor_growth"]
    methods = ["person_period", "dynamic_deephit", "deeptcsr", "survtd"]

    # Table 1: Multi-cohort benchmark storage
    results_tab1 = {b: {m: {'c_index': [], 'auc': [], 'ibs': []} for m in methods} for b in benchmarks}
    alarm_results_tab2 = {m: {'false_alert_rate': [], 'alert_jitter': []} for m in methods}

    for b_idx, b_name in enumerate(benchmarks):
        print(f"\n[{b_idx + 1}/4] Benchmarking Cohort: {b_name.upper()}")

        for s_idx, seed in enumerate(seeds):
            print(f"  -> Seed {seed} ({s_idx + 1}/{len(seeds)})")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Load dataset
            if b_name == "cmapss":
                train_set, test_set, in_dim, max_h = load_cmapss_downsampled(seed=seed, max_units=25 if dry_run else None)
                delta_s = 5.0
                num_bins = 30
            elif b_name == "pbc":
                train_set, test_set, in_dim, max_h = load_pbc(seed=seed)
                delta_s = 100.0
                num_bins = 30
            elif b_name == "tumor_growth":
                train_set, test_set, in_dim, max_h = generate_tumor_growth_cohort(n_samples=60 if dry_run else 250, seed=seed)
                delta_s = 0.5
                num_bins = 25
            elif b_name == "mimic_sepsis":
                train_set, test_set, in_dim, max_h = generate_sepsis_icu_cohort(n_patients=60 if dry_run else 300, seed=seed)
                delta_s = 2.5
                num_bins = 30

            for m_name in methods:
                # Instantiate model
                if m_name == "survtd":
                    model = SurvTDModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s)
                elif m_name == "deeptcsr":
                    model = DeepTCSRClampedModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s)
                elif m_name == "dynamic_deephit":
                    model = DynamicDeepHitModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s)
                elif m_name == "person_period":
                    model = PersonPeriodModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s)

                trained = train_model(
                    model, train_set, model_type=m_name, epochs=epochs,
                    batch_size=16, lr=0.002, device=device, verbose=False
                )

                metrics = evaluate_dataset_model(trained, test_set, m_name, delta_s, max_h, device)
                results_tab1[b_name][m_name]['c_index'].append(metrics['c_index'])
                results_tab1[b_name][m_name]['auc'].append(metrics['auc'])
                results_tab1[b_name][m_name]['ibs'].append(metrics['ibs'])

                # EXP-07: Clinical alarm fatigue evaluation on MIMIC Sepsis
                if b_name == "mimic_sepsis":
                    alarm_metrics = evaluate_alarm_fatigue(
                        metrics['trajectories_risk'],
                        metrics['trajectories_times'],
                        metrics['event_indicators'],
                        target_ppv=0.30,
                        window_hours=6.0
                    )
                    alarm_results_tab2[m_name]['false_alert_rate'].append(alarm_metrics['false_alert_rate_per_day'])
                    alarm_results_tab2[m_name]['alert_jitter'].append(alarm_metrics['alert_jitter_count'])

    # Format Table 1 (Multi-Cohort Dynamic Survival Performance)
    table1_md = "# Table 1: Multi-Cohort Dynamic Survival Performance (EXP-01 & EXP-02)\n\n"
    table1_md += "| Method | MIMIC-IV Sepsis-3 (C / AUC / IBS) | NASA C-MAPSS 50% (C / AUC / IBS) | PBC Trial (C / AUC / IBS) | Tumor Growth ODE (C / AUC / IBS) |\n"
    table1_md += "| :--- | :---: | :---: | :---: | :---: |\n"

    for m in methods:
        m_label = "SurvTD (Ours)" if m == "survtd" else ("DeepTCSR (Clamped)" if m == "deeptcsr" else ("Dynamic-DeepHit" if m == "dynamic_deephit" else "Person-Period (1h)"))
        row = f"| **{m_label}** |"
        for b in benchmarks:
            c_mean = np.mean(results_tab1[b][m]['c_index'])
            auc_mean = np.mean(results_tab1[b][m]['auc'])
            ibs_mean = np.mean(results_tab1[b][m]['ibs'])
            row += f" {c_mean:.3f} / {auc_mean:.3f} / {ibs_mean:.3f} |"
        table1_md += row + "\n"

    # Save Table 1
    tab1_path = os.path.join(output_dir, "table1_benchmarks.md")
    with open(tab1_path, "w", encoding="utf-8") as f:
        f.write(table1_md)
    print(f"\n>> Saved: {tab1_path}")

    # Format Table 2 (Clinical Bedside Alarm Fatigue EXP-07)
    table2_md = "# Table 2: Clinical Bedside Alarm Fatigue & Utility on MIMIC-IV Sepsis-3 (EXP-07)\n\n"
    table2_md += "| Method | False Alert Episode Rate (per pt-day @ 0.30 PPV) | Alert Jitter Count (6h Window Osc.) | Relative Jitter Reduction vs DDH |\n"
    table2_md += "| :--- | :---: | :---: | :---: |\n"

    ddh_jitter = max(1e-4, np.mean(alarm_results_tab2['dynamic_deephit']['alert_jitter']))

    for m in methods:
        m_label = "SurvTD (Ours)" if m == "survtd" else ("DeepTCSR (Clamped)" if m == "deeptcsr" else ("Dynamic-DeepHit" if m == "dynamic_deephit" else "Person-Period (1h)"))
        fa_mean = np.mean(alarm_results_tab2[m]['false_alert_rate'])
        jit_mean = np.mean(alarm_results_tab2[m]['alert_jitter'])
        rel_red = (1.0 - (jit_mean / ddh_jitter)) * 100.0 if ddh_jitter > 0 else 0.0
        table2_md += f"| **{m_label}** | {fa_mean:.2f} | {jit_mean:.1f} | {rel_red:+.1f}% |\n"

    tab2_path = os.path.join(output_dir, "table2_alarm_fatigue.md")
    with open(tab2_path, "w", encoding="utf-8") as f:
        f.write(table2_md)
    print(f">> Saved: {tab2_path}")

    print("\n=== [Track A] Execution Complete ===")
    return results_tab1, alarm_results_tab2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run_track_a(epochs=args.epochs, dry_run=args.dry_run)
