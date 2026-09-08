#!/usr/bin/env bash
# Overnight stream A -- NASA alpha ablation, then the synthetic ICU table.
#
# Ordered cheapest-and-most-informative first, because an unattended run is a
# prefix of a plan, not a plan: whatever the wall clock reaches is what you get.
#
# NASA first. It is the one cohort where SurvTD's headline (0.9538) was measured
# WITHOUT the anchor-only control that PBC2 and Framingham both run, and both of those
# answered "the TD term contributes nothing". Seed 42 already came back 0.9822 for
# alpha = 1.0 against 0.9712 for alpha = 0.5, so this grid is the decisive experiment
# in the project right now and it costs ~75 min.
#
# `benchmark_nasa_tier1_authentic.py` writes its output ONLY after every seed and
# model finishes, so it is invoked once per (arm, seed) into its own file. A kill then
# costs one cell instead of the run. `merge_nasa_alpha.py` assembles them.
set -u
cd "$(dirname "$0")/.."
mkdir -p experiments/results/nasa_alpha_ablation/cells experiments/results/tier1/logs

SEEDS="42 123 456 789 101112"
ALPHAS="survtd_a0.0 survtd_a0.25 survtd_a0.5 survtd_a0.75 survtd_a1.0"
EPOCHS=15          # matches final_5seeds_authentic_benchmark.json; do not change

echo "=== [A1/A3] NASA alpha ablation: 5 alphas x 5 seeds, ${EPOCHS} epochs ==="
for arm in $ALPHAS; do
  for seed in $SEEDS; do
    out="experiments/results/nasa_alpha_ablation/cells/${arm}_seed${seed}.json"
    if [ -f "$out" ]; then echo "  (cached) ${arm} seed=${seed}"; continue; fi
    echo "  running ${arm} seed=${seed} at $(date +%H:%M:%S)"
    python3 -u experiments/benchmark_nasa_tier1_authentic.py \
        --models "$arm" --seeds "$seed" --epochs "$EPOCHS" \
        --output_json "$out" 2>&1 | grep -E "Finished|Error|Traceback" || true
  done
done
python3 experiments/merge_nasa_alpha.py || true
echo "=== [A1/A3] done at $(date) ==="

# The synthetic ICU table. Ten arms for the main table plus the three alpha grid
# points that are not already covered (a0.5 == survtd, a1.0 == survtd_anchor_only,
# so running them again would only buy a duplicate column).
echo "=== [A2/A3] synthetic ICU: 13 arms x 5 seeds ==="
python3 -u experiments/run_tier1.py \
    --cohorts synthetic_icu \
    --arms km landmark_cox survtd survtd_anchor_only ddh tcsr tcsr_landmark \
           coxsig ncde deeptcsr_tcn survtd_a0.0 survtd_a0.25 survtd_a0.75 \
    --seeds $SEEDS --epochs 25 --resume \
    --out experiments/results/tier1/tier1_synthetic_icu.json
echo "=== [A2/A3] done at $(date) ==="

# Whatever is left of the budget goes to the second simulated cohort, which has never
# been run at all. Pure bonus -- if the clock runs out here nothing above is affected.
echo "=== [A3/A3] tumor cohort (bonus) ==="
python3 -u experiments/run_tier1.py \
    --cohorts tumor \
    --arms km landmark_cox survtd survtd_anchor_only ddh tcsr tcsr_landmark \
           coxsig ncde deeptcsr_tcn \
    --seeds $SEEDS --epochs 25 --resume \
    --out experiments/results/tier1/tier1_tumor.json
echo "=== STREAM-A-COMPLETE at $(date) ==="
