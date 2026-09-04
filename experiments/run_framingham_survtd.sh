#!/usr/bin/env bash
# Framingham's SurvTD arms, started once the baseline block frees a core.
#
# These are the expensive cells in the whole benchmark: Framingham's train+val is
# 3,547 subjects against PBC2's 249, so one cell costs ~14x a PBC2 cell (~2 h against
# ~8.4 min) and the ten cells here are ~20 h of wall clock. --resume means a killed
# process costs one cell rather than the run.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "tier1_framingham_baselines" >/dev/null; do sleep 60; done
echo "baselines done; starting Framingham SurvTD arms at $(date)"
python3 -u experiments/run_tier1.py --cohorts framingham \
    --arms survtd survtd_anchor_only --seeds 42 123 456 789 101112 --epochs 25 \
    --resume --out experiments/results/tier1/tier1_framingham_survtd.json
echo "FRAMINGHAM-SURVTD-COMPLETE"
