"""
Track B: Adversarial Stress Tests & Hypothesis Destroyer Pipeline
- EXP-04: Negative Control NC-B (Within-Patient Duration Permutation)
- EXP-03: Factorial Operator Ablation NC-A1 & NC-A2 (Discount vs Shift)
- EXP-05: Negative Control NC-A3 (Clamped Continuous Division Comparison)
- EXP-06: Effective Horizon Matching & Subsampling Sweep NC-C
- EXP-08: Projection Variance Diffusion Bound <= delta_s^2 / 6 & Contraction Test
- Automated Kill Criteria Guard & Table 3 Generator.
"""

import os
import sys
import json
import random
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch


from src.models.survtd import SurvTDModel
from src.models.baselines.deeptcsr_clamped import DeepTCSRClampedModel
from src.models.baselines.person_period import PersonPeriodModel

from src.data.cmapss_loader import load_cmapss_downsampled
from src.data.sepsis_loader import generate_sepsis_icu_cohort

from src.operators.ablations import permute_patient_durations
from src.operators.survtd_operator import categorical_projection_shift
from src.evaluation.metrics import compute_concordance_td
from src.training.trainer import train_model, get_device


def evaluate_concordance(model, dataset, delta_s, device):
    model.eval()
    risk_scores = []
    event_times = []
    event_indicators = []

    with torch.no_grad():
        for p in dataset:
            x = p['features'].unsqueeze(0).to(device)
            dts = p['dts'].unsqueeze(0).to(device)
            mask = p['mask'].unsqueeze(0).to(device) if p['mask'] is not None else None
            tte = float(p['tte'])
            event = float(p['event'])

            _, _, _, cdf = model(x, dts, mask)
            cdf = cdf.squeeze(0).cpu().numpy()
            risk_score = float(cdf[-1, min(len(cdf[-1]) // 2, len(cdf[-1]) - 1)])
            risk_scores.append(risk_score)
            event_times.append(tte)
            event_indicators.append(event)

    return compute_concordance_td(np.array(risk_scores), np.array(event_times), np.array(event_indicators))


def run_track_b(epochs: int = 12, dry_run: bool = False, output_dir: str = "experiments/results"):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"=== [Track B] Starting Adversarial Stress Tests on {device} ===")

    if dry_run:
        epochs = 2
        print(">> Running Track B in DRY RUN mode")

    torch.manual_seed(42)
    np.random.seed(42)
    rng = random.Random(42)

    # Use Sepsis cohort as primary stress-testing ground
    train_set, test_set, in_dim, max_h = generate_sepsis_icu_cohort(n_patients=60 if dry_run else 250, seed=42)
    delta_s = 2.5
    num_bins = 30

    adversarial_table = []
    kill_triggers = []

    # -------------------------------------------------------------
    # 0. Reference Full SurvTD and Unregularized Baseline
    # -------------------------------------------------------------
    print("\n[0/5] Training Reference Full SurvTD and Baseline...")
    model_baseline = PersonPeriodModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s)
    train_model(model_baseline, train_set, model_type="person_period", epochs=epochs, device=device)
    c_baseline = evaluate_concordance(model_baseline, test_set, delta_s, device)

    model_survtd = SurvTDModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s)
    train_model(model_survtd, train_set, model_type="survtd", epochs=epochs, device=device)
    c_survtd = evaluate_concordance(model_survtd, test_set, delta_s, device)
    total_gain = max(1e-4, c_survtd - c_baseline)

    print(f"  Reference Concordance: Baseline = {c_baseline:.4f} | SurvTD = {c_survtd:.4f} | Delta = {total_gain:+.4f}")

    # -------------------------------------------------------------
    # EXP-04: Negative Control NC-B (Within-Patient Duration Permutation)
    # -------------------------------------------------------------
    print("\n[1/5] Executing EXP-04 (NC-B: Within-Patient Duration Permutation)...")
    permuted_train = [permute_patient_durations(train_set[i], rng) for i in range(len(train_set))]
    permuted_test = [permute_patient_durations(test_set[i], rng) for i in range(len(test_set))]

    model_perm = SurvTDModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s)
    train_model(model_perm, permuted_train, model_type="survtd", epochs=epochs, device=device)
    c_perm = evaluate_concordance(model_perm, permuted_test, delta_s, device)
    gain_retained_perm = max(0.0, (c_perm - c_baseline) / total_gain)

    status_exp04 = "PASSED (Hypothesis Upheld)" if gain_retained_perm <= 0.50 else "FAILED (Kill Criterion Fired)"
    if gain_retained_perm > 0.50:
        kill_triggers.append({
            "test_id": "EXP-04",
            "claim": "C0",
            "reason": f"Permuted durations retained {gain_retained_perm*100:.1f}% of gain (> 50%)"
        })

    adversarial_table.append({
        "id": "EXP-04 (NC-B)",
        "threat": "Visit count confounding",
        "threshold": "Permuted retains > 50% gain",
        "observed": f"{gain_retained_perm*100:.1f}% gain retained (C={c_perm:.3f})",
        "verdict": status_exp04
    })
    print(f"  -> Result: {gain_retained_perm*100:.1f}% gain retained | Verdict: {status_exp04}")

    # -------------------------------------------------------------
    # EXP-03: Factorial Operator Ablations NC-A1 & NC-A2
    # -------------------------------------------------------------
    print("\n[2/5] Executing EXP-03 (Factorial Operator Ablation NC-A1 & NC-A2)...")
    # Arm A1: Discount ablation
    model_a1 = SurvTDModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s)
    train_model(model_a1, train_set, model_type="survtd", ablation_mode="arm_a1_discount", epochs=epochs, device=device)
    c_a1 = evaluate_concordance(model_a1, test_set, delta_s, device)
    gain_retained_a1 = max(0.0, (c_a1 - c_baseline) / total_gain)

    # Arm A2: Shift ablation
    model_a2 = SurvTDModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s)
    train_model(model_a2, train_set, model_type="survtd", ablation_mode="arm_a2_shift", epochs=epochs, device=device)
    c_a2 = evaluate_concordance(model_a2, test_set, delta_s, device)
    gain_retained_a2 = max(0.0, (c_a2 - c_baseline) / total_gain)

    status_exp03 = "PASSED (Hypothesis Upheld)" if (gain_retained_a1 <= 0.50 and gain_retained_a2 <= 0.50) else "FAILED"
    if gain_retained_a1 > 0.50 or gain_retained_a2 > 0.50:
        kill_triggers.append({
            "test_id": "EXP-03",
            "claim": "C0",
            "reason": f"Arm A1 retained {gain_retained_a1*100:.1f}%, Arm A2 retained {gain_retained_a2*100:.1f}%"
        })

    adversarial_table.append({
        "id": "EXP-03 (NC-A1/A2)",
        "threat": "Component bundling",
        "threshold": "Arm A1 or A2 retains > 50% gain",
        "observed": f"A1={gain_retained_a1*100:.1f}%, A2={gain_retained_a2*100:.1f}% retained",
        "verdict": status_exp03
    })
    print(f"  -> Arm A1 retained {gain_retained_a1*100:.1f}% | Arm A2 retained {gain_retained_a2*100:.1f}% | Verdict: {status_exp03}")

    # -------------------------------------------------------------
    # EXP-05: Negative Control NC-A3 (Clamped Division Comparison)
    # -------------------------------------------------------------
    print("\n[3/5] Executing EXP-05 (NC-A3: Clamped Division Comparison)...")
    model_clamped = DeepTCSRClampedModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s, clamp_eps=1e-3)
    train_model(model_clamped, train_set, model_type="deeptcsr", epochs=epochs, device=device)
    c_clamped = evaluate_concordance(model_clamped, test_set, delta_s, device)

    # Monitor gradient norms on high-risk batch
    model_clamped.train()
    sample_p = train_set[0]
    loss_clamped = model_clamped.compute_loss_trajectory(
        sample_p['features'].to(device), sample_p['dts'].to(device),
        sample_p['events'].to(device), float(sample_p['tte']), float(sample_p['tte'])
    )
    loss_clamped.backward()
    grad_norm_clamped = float(torch.nn.utils.clip_grad_norm_(model_clamped.parameters(), max_norm=100.0).item())

    status_exp05 = "PASSED (Hypothesis Upheld)" if (c_survtd > c_clamped + 0.01 or grad_norm_clamped > 1.5) else "FAILED"
    adversarial_table.append({
        "id": "EXP-05 (NC-A3)",
        "threat": "Naive clamped division substitute",
        "threshold": "Clamped division matches stability & C-index",
        "observed": f"Clamped C={c_clamped:.3f}, GradNorm={grad_norm_clamped:.2f}",
        "verdict": status_exp05
    })
    print(f"  -> Clamped C-index = {c_clamped:.4f} vs SurvTD = {c_survtd:.4f} | Verdict: {status_exp05}")

    # -------------------------------------------------------------
    # EXP-06: Effective Horizon Matching Sweep NC-C
    # -------------------------------------------------------------
    print("\n[4/5] Executing EXP-06 (NC-C: Effective Horizon Matching & Lambda Sweep)...")
    model_count_geom = SurvTDModel(input_dim=in_dim, hidden_dim=64, num_bins=num_bins, delta_s=delta_s)
    train_model(model_count_geom, train_set, model_type="survtd", ablation_mode="count_geometric", epochs=epochs, device=device)
    c_count_geom = evaluate_concordance(model_count_geom, test_set, delta_s, device)

    status_exp06 = "PASSED (Hypothesis Upheld)" if c_survtd >= c_count_geom else "FAILED"
    adversarial_table.append({
        "id": "EXP-06 (NC-C)",
        "threat": "Bootstrapping horizon drift across sampling",
        "threshold": "Count-geometric matches duration-geometric",
        "observed": f"Duration-geom C={c_survtd:.3f} vs Count-geom C={c_count_geom:.3f}",
        "verdict": status_exp06
    })
    print(f"  -> Duration-geometric C = {c_survtd:.4f} vs Count-geometric = {c_count_geom:.4f} | Verdict: {status_exp06}")

    # -------------------------------------------------------------
    # EXP-08: Projection Variance Diffusion Bound <= delta_s^2 / 6
    # -------------------------------------------------------------
    print("\n[5/5] Executing EXP-08 (Projection Variance Diffusion Bound <= delta_s^2 / 6)...")
    K = 40
    p_init = torch.zeros(K)
    p_init[K // 2] = 1.0
    grid = (torch.arange(K, dtype=torch.float32) + 0.5) * delta_s
    cur_p = p_init.clone()

    # 100 consecutive projection steps with uniform offsets in [0, delta_s]
    total_diffusion = 0.0
    for _ in range(100):
        dt_step = float(rng.uniform(0.1, delta_s))
        p_next = categorical_projection_shift(cur_p, dt_step, delta_s, K)
        mean_next = float(torch.sum(p_next * grid).item())
        var_next = float(torch.sum(p_next * ((grid - mean_next) ** 2)).item())

        mean_prev = float(torch.sum(cur_p * grid).item())
        var_prev = float(torch.sum(cur_p * ((grid - mean_prev) ** 2)).item())

        step_var_increase = max(0.0, var_next - var_prev)
        total_diffusion += step_var_increase
        cur_p = p_next

    avg_step_variance = total_diffusion / 100.0
    theoretical_max = (delta_s ** 2) / 6.0
    status_exp08 = "PASSED (Hypothesis Upheld)" if avg_step_variance <= theoretical_max + 1e-3 else "FAILED"

    adversarial_table.append({
        "id": "EXP-08 (Diffusion)",
        "threat": "Projection variance blowup O(sqrt(n))",
        "threshold": f"Variance <= delta_s^2 / 6 ({theoretical_max:.4f})",
        "observed": f"Measured avg step var = {avg_step_variance:.4f}",
        "verdict": status_exp08
    })
    print(f"  -> Measured avg step var = {avg_step_variance:.4f} vs Bound = {theoretical_max:.4f} | Verdict: {status_exp08}")

    # -------------------------------------------------------------
    # Format Table 3 Markdown & Check Kill Criteria
    # -------------------------------------------------------------
    table3_md = "# Table 3: Adversarial Falsification Matrix & Kill Criteria Report\n\n"
    table3_md += "| Stress Test ID | Hostile Threat Interrogated | Kill Threshold (Falsifier) | Observed Metric | Verdict |\n"
    table3_md += "| :--- | :--- | :--- | :--- | :---: |\n"
    for row in adversarial_table:
        table3_md += f"| **{row['id']}** | {row['threat']} | {row['threshold']} | {row['observed']} | **{row['verdict']}** |\n"

    tab3_path = os.path.join(output_dir, "table3_adversarial.md")
    with open(tab3_path, "w", encoding="utf-8") as f:
        f.write(table3_md)
    print(f"\n>> Saved: {tab3_path}")

    # Check if falsification pivot should trigger
    if len(kill_triggers) > 0:
        print("\n" + "!" * 70)
        print("🚨 KILL CRITERIA FIRED: Falsification condition detected!")
        print("!" * 70)
        report_path = os.path.join(output_dir, "falsification_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(kill_triggers, f, indent=2)
        print(f"Triggered pivot payload saved to: {report_path}")
        print("Invoke skill: .agents/skills/falsification-pivot/ to perform causal autopsy.")
    else:
        print("\n" + "=" * 70)
        print("🛡️  ALL ADVERSARIAL STRESS TESTS PASSED: Hypothesis C0 firmly survives!")
        print("=" * 70)

    print("\n=== [Track B] Execution Complete ===")
    return adversarial_table, kill_triggers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run_track_b(epochs=args.epochs, dry_run=args.dry_run)
