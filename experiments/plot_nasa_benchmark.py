"""
Publication-Quality Plotting Script for NASA C-MAPSS FD001 Benchmark.
Generates 5-model comparative figure conforming to figure-smith guidelines:
- Colorblind-safe Okabe-Ito palette
- Direct error bar rendering
- Camera-ready PDF vector and high-DPI PNG
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# figure-smith plot styling
sys.path.insert(0, '/Users/yangjaemo/Desktop/SurvTD/.agents/skills/figure-smith/scripts')
try:
    from plot_style import use, PALETTE, savefig
    use(column='double', height_ratio=0.45, base_font=9.0)
except ImportError:
    pass

def generate_benchmark_figure(
    results_json: str = "experiments/results/nasa_parity/parity_results.json",
    pdf_out: str = "figures/fig_nasa_benchmark.pdf",
    png_out: str = "figures/fig_nasa_benchmark.png"
):
    with open(results_json, "r") as f:
        data = json.load(f)

    summary = data["summary"]
    
    # Preferred display order
    model_keys = ["coxsig", "deeptcsr", "ddh", "survtd", "survtd_v2"]
    labels = [
        "CoxSig\n(NeurIPS 23)",
        "DeepTCSR\n(EPFL 24)",
        "DDH\n(Lee 20)",
        "SurvTD-v1\n(Ours)",
        "SurvTD-v2\n(Ours, BLA+CHHM)"
    ]

    # Filter only available models
    active_keys = []
    active_labels = []
    for k, l in zip(model_keys, labels):
        if k in summary:
            active_keys.append(k)
            active_labels.append(l)

    dyn_means = [summary[k]["dynamic_cindex_mean"] for k in active_keys]
    dyn_stds = [summary[k]["dynamic_cindex_std"] for k in active_keys]

    t0_means = [summary[k]["t0_cindex_mean"] for k in active_keys]
    t0_stds = [summary[k]["t0_cindex_std"] for k in active_keys]

    os.makedirs(os.path.dirname(pdf_out), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Okabe-Ito palette: Sky Blue, Orange, Reddish Purple, Bluish Green, Vermillion
    palette_map = {
        "coxsig": "#56B4E9",
        "deeptcsr": "#E69F00",
        "ddh": "#CC79A7",
        "survtd": "#009E73",
        "survtd_v2": "#D55E00"
    }
    colors = [palette_map.get(k, "#999999") for k in active_keys]

    # Subplot 1: Dynamic Landmark C-index
    ax1 = axes[0]
    bars1 = ax1.bar(
        active_labels, dyn_means, yerr=dyn_stds, capsize=4,
        color=colors, alpha=0.9, width=0.55, edgecolor="black", linewidth=0.8
    )
    ax1.set_ylabel("Dynamic Landmark C-index", fontsize=9)
    ax1.set_ylim(0.45, 1.05)
    ax1.set_title("(a) Dynamic Landmark Evaluation ($p_t \\in [10\\%, 20\\%, 40\\%]$)", fontsize=9.5, pad=8)
    ax1.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    for bar, m, s in zip(bars1, dyn_means, dyn_stds):
        ax1.text(
            bar.get_x() + bar.get_width()/2, min(m + s + 0.015, 1.02),
            f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=7.0
        )

    # Subplot 2: Static t=0 C-index
    ax2 = axes[1]
    bars2 = ax2.bar(
        active_labels, t0_means, yerr=t0_stds, capsize=4,
        color=colors, alpha=0.9, width=0.55, edgecolor="black", linewidth=0.8
    )
    ax2.set_ylabel("Static C-index ($t=0$)", fontsize=9)
    ax2.set_ylim(0.25, 0.75)
    ax2.set_title("(b) Static $t=0$ Lifespan Ranking", fontsize=9.5, pad=8)
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="Chance (0.50)")
    ax2.legend(loc="lower left", fontsize=7.5)
    for bar, m, s in zip(bars2, t0_means, t0_stds):
        ax2.text(
            bar.get_x() + bar.get_width()/2, m + s + 0.015,
            f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=7.0
        )

    plt.tight_layout()
    try:
        from plot_style import savefig
        savefig(pdf_out)
    except Exception:
        plt.savefig(pdf_out, bbox_inches="tight")
    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    print(f"Generated {pdf_out} and {png_out}")

if __name__ == "__main__":
    generate_benchmark_figure()
