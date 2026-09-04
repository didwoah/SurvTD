#!/usr/bin/env bash
# Everything left after the Phase D gate passed, in one sequence so the machine is
# never oversubscribed. Each stage writes its own file and the Tier-1 runner flushes
# after every (cohort, arm, seed), so an interruption costs one cell, not the run.
set -u
cd "$(dirname "$0")/.."
mkdir -p experiments/results/nasa_tier1_authentic experiments/results/tier1

echo "=== [1/2] NASA DDH re-run under the D16 repair ==========================="
# The DDH cell in nasa_tier1_authentic/final_5seeds_authentic_benchmark.json (0.7599
# +- 0.1307) came from the worker under the sequence-length leak. Re-run just that arm.
python3 -u experiments/benchmark_nasa_tier1_authentic.py \
    --models ddh --seeds 42 123 456 789 101112 --epochs 25 \
    --output_json experiments/results/nasa_tier1_authentic/ddh_D16_repaired.json \
    2>&1 | grep -Ev "^\s*$"
echo "=== [1/2] exit=$? ==="

echo "=== [2/2] Tier-1: 3 cohorts x 10 arms x 5 seeds ========================="
python3 -u experiments/run_tier1.py \
    --cohorts pbc framingham cmapss \
    --seeds 42 123 456 789 101112 \
    --epochs 25 \
    --out experiments/results/tier1/tier1_results.json
echo "=== [2/2] exit=$? ==="
echo "ALL-STAGES-COMPLETE"
