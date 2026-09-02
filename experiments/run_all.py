"""
SurvTD Master Experimental Pipeline (Track A -> Track B Sequential Execution)
- Runs Track A (EXP-01, EXP-02, EXP-07)
- Runs Track B (EXP-03, EXP-04, EXP-05, EXP-06, EXP-08)
- Formats Table 1, Table 2, Table 3 and prints the comprehensive validation report.
"""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_track_a import run_track_a
from experiments.run_track_b import run_track_b



def main():
    parser = argparse.ArgumentParser(description="SurvTD Master Pipeline Runner")
    parser.add_argument("--track", choices=["all", "a", "b"], default="all", help="Which track to run")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--dry_run", action="store_true", help="Run fast 1-2 epoch smoke test")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    print("=" * 80)
    print("  🚀 SurvTD (Survival Temporal Difference Learning) Master Pipeline")
    print("=" * 80)
    print(f"• Mode: {'DRY RUN' if args.dry_run else 'FULL BENCHMARK'}")
    print(f"• Selected Track: {args.track.upper()}")
    print(f"• Output Directory: {args.output_dir}\n")

    tab1_results, tab2_results = None, None
    tab3_results, kill_triggers = None, None

    # Execute Track A
    if args.track in ["all", "a"]:
        print(">>> [PHASE 1/2] RUNNING TRACK A: PUBLICATION BENCHMARKS <<<")
        tab1_results, tab2_results = run_track_a(
            epochs=args.epochs,
            dry_run=args.dry_run,
            output_dir=args.output_dir
        )

    # Execute Track B
    if args.track in ["all", "b"]:
        print("\n>>> [PHASE 2/2] RUNNING TRACK B: ADVERSARIAL STRESS TESTS <<<")
        tab3_results, kill_triggers = run_track_b(
            epochs=args.epochs,
            dry_run=args.dry_run,
            output_dir=args.output_dir
        )

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  🏁 ALL REQUESTED EXPERIMENTS COMPLETED IN {elapsed:.1f} SECONDS")
    print("=" * 80)

    # Summary of generated tables
    print("\nGenerated Artifacts:")
    if os.path.exists(os.path.join(args.output_dir, "table1_benchmarks.md")):
        print(f"• Table 1: {os.path.join(args.output_dir, 'table1_benchmarks.md')}")
    if os.path.exists(os.path.join(args.output_dir, "table2_alarm_fatigue.md")):
        print(f"• Table 2: {os.path.join(args.output_dir, 'table2_alarm_fatigue.md')}")
    if os.path.exists(os.path.join(args.output_dir, "table3_adversarial.md")):
        print(f"• Table 3: {os.path.join(args.output_dir, 'table3_adversarial.md')}")


if __name__ == "__main__":
    main()
