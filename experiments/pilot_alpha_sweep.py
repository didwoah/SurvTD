"""
Pilot Alpha Sweep on Validation Split (Seed 42, Synthetic ICU)
- Pre-registers and locks alpha_anchor in {0.0, 0.25, 0.5, 0.75, 1.0}
- Purely evaluated on validation split to avoid test leakage.
- Output: Best alpha_anchor to be frozen across all Track A & Track B runs.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.data.cohorts import COHORTS
from src.models.survtd import SurvTDModel
from src.evaluation.landmark import evaluate_landmarked
from src.training.trainer import train_model, get_device


def run_alpha_sweep(cohort_name="synthetic_icu", seed=42, epochs=10, batch_size=16, lr=0.001):
    device = get_device()
    spec = COHORTS[cohort_name]
    cohort_data = spec.load(seed=seed)
    
    candidates = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("=" * 70)
    print(f"  🔍 SurvTD Pilot Alpha Sweep (Cohort: {cohort_name}, Seed: {seed})")
    print(f"  Candidate Alphas: {candidates} | Epochs: {epochs} | Device: {device}")
    print("=" * 70)

    results = {}

    for alpha in candidates:
        print(f"\n---> Evaluating alpha = {alpha:.2f} <---")
        model = SurvTDModel(
            input_dim=cohort_data.input_dim,
            hidden_dim=64,
            num_bins=spec.num_bins,
            delta_s=spec.delta_s,
            alpha_anchor=alpha,
            include_overflow=True,
        )
        if hasattr(model.backbone, "set_empirical_mean"):
            model.backbone.set_empirical_mean(cohort_data.x_mean)

        trained_model = train_model(
            model=model,
            train_dataset=cohort_data.train,
            val_dataset=cohort_data.val,
            val_spec=spec.landmark_spec,
            delta_s=spec.delta_s,
            model_type="survtd",
            alpha_anchor=alpha,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            patience=3,
            device=device,
            verbose=False,
        )

        # Evaluate strictly on validation split (out-of-sample from training, zero test touch)
        val_metrics = evaluate_landmarked(
            trained_model, cohort_data.train, cohort_data.val, spec.landmark_spec, spec.delta_s, device, strict=False
        )
        valid_c = [m["c_td"] for m in val_metrics.values() if not np.isnan(m["c_td"])]
        mean_c_td = float(np.mean(valid_c)) if valid_c else float("nan")

        print(f"Result: alpha = {alpha:.2f}  ==>  Validation Landmarked C_td = {mean_c_td:.4f}")
        results[str(alpha)] = mean_c_td

    best_alpha = max(results.keys(), key=lambda k: results[k])
    print("\n" + "=" * 70)
    print(f"  🏆 PILOT SWEEP WINNER: alpha = {best_alpha} (Val C_td = {results[best_alpha]:.4f})")
    print(f"  This value is now pre-registered and locked across all benchmarks.")
    print("=" * 70)

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/pilot_alpha_sweep.json", "w") as f:
        json.dump({
            "cohort": cohort_name,
            "seed": seed,
            "candidates": results,
            "best_alpha": float(best_alpha),
            "best_val_c_td": results[best_alpha],
        }, f, indent=2)

    return float(best_alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=str, default="synthetic_icu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    run_alpha_sweep(args.cohort, args.seed, args.epochs)
