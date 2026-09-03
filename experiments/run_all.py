"""
SurvTD Master Experimental Pipeline (Track A -> Track B Sequential Execution)
- Runs Track A (EXP-01, EXP-02, EXP-07, Kill Criteria 3 & 4)
- Runs Track B (EXP-03, EXP-04, EXP-05, EXP-06, Kill Criteria 0 & 5)
- Unifies Table 1, Table 2, Table 3 and compiles master falsification report.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_track_a import run_track_a
from experiments.run_track_b import run_track_b


def main():
    parser = argparse.ArgumentParser(description="SurvTD Master Pipeline Runner")
    parser.add_argument("--track", choices=["all", "a", "b"], default="all", help="Which track to run")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--alpha_anchor", type=float, default=0.5, help="Alpha anchor weight")
    parser.add_argument("--dry_run", action="store_true", help="Run fast 1-2 epoch smoke test in isolated dir")
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    actual_output_dir = "experiments/results/dry_run" if args.dry_run else args.output_dir
    os.makedirs(actual_output_dir, exist_ok=True)
    start_time = time.time()

    print("=" * 80)
    print("  🚀 SurvTD (Survival Temporal Difference Learning) Master Pipeline")
    print("=" * 80)
    print(f"• Mode: {'DRY RUN' if args.dry_run else 'FULL BENCHMARK'}")
    print(f"• Selected Track: {args.track.upper()}")
    print(f"• Output Directory: {actual_output_dir}\n")

    master_falsifications = []

    # Execute Track A
    if args.track in ["all", "a"]:
        print(">>> [PHASE 1/2] RUNNING TRACK A: PUBLICATION BENCHMARKS <<<")
        _, _, fals_a = run_track_a(
            epochs=args.epochs,
            alpha_anchor=args.alpha_anchor,
            dry_run=args.dry_run,
            output_dir=actual_output_dir
        )
        master_falsifications.extend(fals_a)

    # Execute Track B
    if args.track in ["all", "b"]:
        print("\n>>> [PHASE 2/2] RUNNING TRACK B: ADVERSARIAL STRESS TESTS <<<")
        _, fals_b = run_track_b(
            epochs=args.epochs,
            alpha_anchor=args.alpha_anchor,
            dry_run=args.dry_run,
            output_dir=actual_output_dir
        )
        master_falsifications.extend(fals_b)

    # Save Master Falsification Report
    report_path = Path(actual_output_dir) / "falsification_report.json"
    with open(report_path, "w") as f:
        json.dump(master_falsifications, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  🏁 ALL EXPERIMENTS COMPLETED IN {elapsed:.1f} SECONDS")
    print(f"  📊 Pre-Registered Falsifications Triggered: {len(master_falsifications)}")
    print("=" * 80)
    if master_falsifications:
        print("\n[!] WARNING: The following pre-registered falsification criteria fired:")
        for f_item in master_falsifications:
            print(f"  • [{f_item.get('test_id', 'Unknown')}] {f_item.get('reason', '')}")
    else:
        print("\n[✓] ALL PRE-REGISTERED CRITERIA UPHELD: Zero falsifications triggered.")


if __name__ == "__main__":
    main()
