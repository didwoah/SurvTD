#!/usr/bin/env bash
# Overnight stream B -- the alpha ablation on the two real cohorts.
#
# Only the three grid points that are not already measured are run: a0.5 is the
# `survtd` arm and a1.0 is `survtd_anchor_only`, both already complete on PBC2 and
# Framingham at 5 seeds, so re-running them would buy a duplicate column and cost
# ~3 hours. `--resume` means a kill costs one cell.
#
# PBC2 first (249 subjects, ~8 min per cell) and Framingham second (3,547 train
# subjects, ~18 min per cell). If the budget runs out it runs out in Framingham,
# which is the cohort with the least to learn: its SurvTD and anchor-only arms already
# agree to four decimal places.
set -u
cd "$(dirname "$0")/.."
mkdir -p experiments/results/tier1/logs

SEEDS="42 123 456 789 101112"
NEW_ALPHAS="survtd_a0.0 survtd_a0.25 survtd_a0.75"

echo "=== [B1/B2] PBC2 alpha ablation: 3 alphas x 5 seeds ==="
python3 -u experiments/run_tier1.py \
    --cohorts pbc --arms $NEW_ALPHAS \
    --seeds $SEEDS --epochs 25 --resume \
    --out experiments/results/tier1/tier1_pbc_alpha.json
echo "=== [B1/B2] done at $(date) ==="

echo "=== [B2/B2] Framingham alpha ablation: 3 alphas x 5 seeds ==="
python3 -u experiments/run_tier1.py \
    --cohorts framingham --arms $NEW_ALPHAS \
    --seeds $SEEDS --epochs 25 --resume \
    --out experiments/results/tier1/tier1_framingham_alpha.json
echo "=== STREAM-B-COMPLETE at $(date) ==="
