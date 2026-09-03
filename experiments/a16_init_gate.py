"""
A-16 initialization gate (decided-after-results; see deviation_log.md §4 A-16).

Adjudicates the pre-declared proceed/stop decision on whether the hazard-head
initialization defect explains SurvTD's Kill Criterion 3 failure, BEFORE committing
to a ~1 day full re-run of Tracks A and B.

Gate (both conditions required):
  G1. SurvTD mean landmarked C^td over the 5 preregistered seeds >= 0.60
  G2. corr(SurvTD C^td, cohort difficulty) across seeds > 0

G2 is the load-bearing one. A mean can rise by luck; tracking cohort difficulty is
direct evidence the model is learning the data rather than emitting optimization
noise. In the primary run SurvTD scored corr = -0.211 against Dynamic-DeepHit while
Person-Period scored +0.988, i.e. every arm except SurvTD tracked learnability.

Difficulty proxy: the PRIMARY run's per-seed Dynamic-DeepHit C^td. `spec.load(seed)`
is deterministic, so seed n indexes the same cohort in both runs, and DeepHit's
per-seed scores correlate with Person-Period's at r = 0.988 -- they measure how much
signal seed n's cohort contains. Using the primary run's values keeps the gate to a
SurvTD-only run (~1 h) instead of a full five-arm cohort (~2 h).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Per-seed C^td from the preregistered primary run (default init, alpha = 0.0),
# transcribed from experiments/results/preregistered_primary_2026-09-04/cohort1_raw_run.log
PRIMARY_SEEDS = [42, 123, 456, 789, 101112]
PRIMARY = {
    "km":              [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
    "person_period":   [0.5308, 0.7289, 0.5977, 0.6488, 0.6772],
    "dynamic_deephit": [0.5570, 0.7112, 0.6151, 0.6717, 0.6754],
    "deeptcsr":        [0.5705, 0.4497, 0.5390, 0.4420, 0.5562],
    "survtd":          [0.5831, 0.5076, 0.5192, 0.5172, 0.6409],
}

G1_THRESHOLD = 0.60


def load_run(path: Path, method: str = "survtd", cohort: str = "synthetic_icu"):
    payload = json.loads(path.read_text())
    results = payload.get("results", payload)
    entry = results.get(f"{cohort}_{method}")
    if entry is None:
        raise SystemExit(f"no {cohort}_{method} in {path}")
    scores, seeds = [], []
    for seed in PRIMARY_SEEDS:
        cell = entry.get(str(seed))
        if cell is None or np.isnan(cell["c_td"]):
            continue
        seeds.append(seed)
        scores.append(cell)
    return payload, seeds, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str,
                    default="experiments/results/exploratory_init_gate/table1_benchmarks_raw.json")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"gate results not found: {path}")

    payload, seeds, cells = load_run(path)
    if len(seeds) < 3:
        raise SystemExit(f"only {len(seeds)} usable seeds; gate needs at least 3")

    c_td = np.array([c["c_td"] for c in cells])
    auc = np.array([c["auc"] for c in cells])
    ibs = np.array([c["ibs"] for c in cells])
    idx = [PRIMARY_SEEDS.index(s) for s in seeds]

    difficulty = np.array([PRIMARY["dynamic_deephit"][i] for i in idx])
    baseline = np.array([PRIMARY["survtd"][i] for i in idx])

    mean_c = float(c_td.mean())
    sd_c = float(c_td.std(ddof=1)) if c_td.size > 1 else 0.0
    corr = float(np.corrcoef(c_td, difficulty)[0, 1])
    corr_before = float(np.corrcoef(baseline, difficulty)[0, 1])

    g1 = mean_c >= G1_THRESHOLD
    g2 = corr > 0.0
    verdict = "PROCEED" if (g1 and g2) else "STOP"

    print("=" * 72)
    print(f"  A-16 INITIALIZATION GATE  |  init={payload.get('init_mode')}  "
          f"alpha={payload.get('alpha_anchor')}  epochs={payload.get('epochs')}")
    print("=" * 72)
    print(f"{'seed':>8} {'C_td (A-16)':>12} {'C_td (primary)':>15} {'delta':>9} "
          f"{'difficulty':>11} {'IBS':>8}")
    for k, s in enumerate(seeds):
        print(f"{s:>8} {c_td[k]:>12.4f} {baseline[k]:>15.4f} "
              f"{c_td[k] - baseline[k]:>+9.4f} {difficulty[k]:>11.4f} {ibs[k]:>8.4f}")

    print("-" * 72)
    print(f"  SurvTD C_td : {mean_c:.4f} +- {sd_c:.4f}   "
          f"(primary: {baseline.mean():.4f})   delta {mean_c - baseline.mean():+.4f}")
    print(f"  SurvTD AUC  : {auc.mean():.4f}    IBS: {ibs.mean():.4f}")
    print(f"  vs DeepHit (primary, default init): {mean_c - difficulty.mean():+.4f}")
    print("-" * 72)
    print(f"  G1  mean C_td >= {G1_THRESHOLD:.2f}      : {mean_c:.4f}  -> {'PASS' if g1 else 'FAIL'}")
    print(f"  G2  corr(SurvTD, difficulty) > 0 : {corr:+.3f}  "
          f"(primary: {corr_before:+.3f})  -> {'PASS' if g2 else 'FAIL'}")
    print("=" * 72)
    print(f"  VERDICT: {verdict}")
    if verdict == "PROCEED":
        print("  -> Re-select alpha over 3 seeds under km_prior, then run Track B")
        print("     (Kill Criterion 5 has never executed) before the full Track A.")
    else:
        print("  -> The initialization diagnosis is rejected. The preregistered")
        print("     primary outcome stands as final: C_3 falsified on Cohort 1.")
    print("=" * 72)

    out = path.parent / "a16_gate_verdict.json"
    out.write_text(json.dumps({
        "verdict": verdict,
        "g1_mean_c_td": {"value": mean_c, "threshold": G1_THRESHOLD, "pass": bool(g1)},
        "g2_corr_difficulty": {"value": corr, "primary": corr_before, "pass": bool(g2)},
        "seeds": seeds,
        "c_td": c_td.tolist(),
        "auc": auc.tolist(),
        "ibs": ibs.tolist(),
        "primary_c_td": baseline.tolist(),
        "difficulty_proxy": difficulty.tolist(),
        "init_mode": payload.get("init_mode"),
        "alpha_anchor": payload.get("alpha_anchor"),
    }, indent=2))
    print(f"Saved -> {out}")
    return 0 if verdict == "PROCEED" else 1


if __name__ == "__main__":
    sys.exit(main())
