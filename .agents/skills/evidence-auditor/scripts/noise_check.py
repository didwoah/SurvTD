#!/usr/bin/env python3
"""Decide whether a reported gain is larger than the seed noise it sits in.

This is the arithmetic behind the most common way an empirical claim fails in
review: the mean moved, the spread was never shown, and the difference turns out
to be inside the run-to-run variation.  The tool takes long-format results and
answers one question per comparison — does this delta survive resampling?

Input CSV (header required; `setting` optional, `seed` optional but strongly advised):

    method,setting,seed,value
    vanilla,128k,0,41.2
    vanilla,128k,1,40.5
    ps-grpo,128k,0,44.9
    ps-grpo,128k,1,45.6

Usage:

    python3 noise_check.py results/main.csv --baseline vanilla --meaningful-delta 0.8

Exit 0 = every comparison clears the noise floor.
Exit 1 = at least one comparison is inside the noise (or underpowered).
Exit 2 = unreadable input.  Pass --report-only to always exit 0.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from collections import defaultdict

BOOTSTRAP_N = 10000


def bootstrap_ci(
    treat: list[float], base: list[float], rng: random.Random, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI for the difference of means.

    Paired when the two arms have the same seed count — pairing removes the
    seed-level variance the two arms share, which is exactly the variance that
    otherwise hides a real effect (and manufactures a fake one).
    """
    deltas = []
    paired = len(treat) == len(base) and len(treat) > 1
    n = len(treat)
    for _ in range(BOOTSTRAP_N):
        if paired:
            idx = [rng.randrange(n) for _ in range(n)]
            d = statistics.fmean(treat[i] for i in idx) - statistics.fmean(base[i] for i in idx)
        else:
            t = [rng.choice(treat) for _ in treat]
            b = [rng.choice(base) for _ in base]
            d = statistics.fmean(t) - statistics.fmean(b)
        deltas.append(d)
    deltas.sort()
    lo = deltas[int((alpha / 2) * BOOTSTRAP_N)]
    hi = deltas[min(int((1 - alpha / 2) * BOOTSTRAP_N), BOOTSTRAP_N - 1)]
    return lo, hi


def spread(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("--baseline", required=True, help="value of the `method` column to compare against")
    ap.add_argument("--meaningful-delta", type=float, default=0.0,
                    help="the effect size declared in evidence-plan.json BEFORE results existed")
    ap.add_argument("--lower-is-better", action="store_true", help="for loss / error / latency metrics")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed, so the verdict is reproducible")
    ap.add_argument("--report-only", action="store_true", help="always exit 0")
    args = ap.parse_args()

    try:
        with open(args.csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot read {args.csv_path}: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"error: {args.csv_path} has no data rows", file=sys.stderr)
        return 2
    if "method" not in rows[0] or "value" not in rows[0]:
        print("error: CSV needs at least `method` and `value` columns", file=sys.stderr)
        return 2

    # (setting, method) -> ordered values, ordered by seed when seeds are present
    # so the paired bootstrap lines up matching runs.
    buckets: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for i, row in enumerate(rows):
        try:
            val = float(row["value"])
        except (TypeError, ValueError):
            print(f"warn: row {i + 2} has non-numeric value {row.get('value')!r}, skipped", file=sys.stderr)
            continue
        try:
            seed = float(row.get("seed") or i)
        except ValueError:
            seed = i
        buckets[(row.get("setting", "") or "-", row["method"])].append((seed, val))

    settings = sorted({s for s, _ in buckets})
    methods = sorted({m for _, m in buckets})
    if args.baseline not in methods:
        print(f"error: baseline {args.baseline!r} not among methods {methods}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    sign = -1.0 if args.lower_is_better else 1.0
    print(f"noise_check: {args.csv_path}   baseline={args.baseline}   "
          f"meaningful_delta={args.meaningful_delta}   "
          f"direction={'lower is better' if args.lower_is_better else 'higher is better'}\n")
    header = f"{'setting':<12} {'method':<16} {'n':>3} {'mean':>9} {'std':>7} {'delta':>8} {'95% CI':>18}  verdict"
    print(header)
    print("-" * len(header))

    problems = 0
    for setting in settings:
        base_pairs = sorted(buckets.get((setting, args.baseline), []))
        base = [v for _, v in base_pairs]
        if not base:
            print(f"{setting:<12} {'(no baseline rows)':<16}")
            continue
        print(f"{setting:<12} {args.baseline:<16} {len(base):>3} {statistics.fmean(base):>9.3f} "
              f"{spread(base):>7.3f} {'—':>8} {'—':>18}  baseline")
        for method in methods:
            if method == args.baseline:
                continue
            pairs = sorted(buckets.get((setting, method), []))
            treat = [v for _, v in pairs]
            if not treat:
                continue
            delta = (statistics.fmean(treat) - statistics.fmean(base)) * sign
            n_eff = min(len(treat), len(base))
            if n_eff < args.min_seeds:
                verdict, ci_txt = "UNDERPOWERED", "—"
                problems += 1
            else:
                lo, hi = bootstrap_ci(treat, base, rng)
                lo, hi = sorted((lo * sign, hi * sign))
                ci_txt = f"[{lo:+.3f}, {hi:+.3f}]"
                excludes_zero = lo > 0 or hi < 0
                clears_bar = abs(delta) >= args.meaningful_delta
                if excludes_zero and clears_bar and delta > 0:
                    verdict = "supported"
                elif excludes_zero and delta < 0:
                    verdict = "WORSE-THAN-BASELINE"
                    problems += 1
                elif not excludes_zero:
                    verdict = "WITHIN-NOISE"
                    problems += 1
                else:
                    verdict = "BELOW-DECLARED-DELTA"
                    problems += 1
            print(f"{'':<12} {method:<16} {len(treat):>3} {statistics.fmean(treat):>9.3f} "
                  f"{spread(treat):>7.3f} {delta:>+8.3f} {ci_txt:>18}  {verdict}")
        print()

    if problems:
        print(f"{problems} comparison(s) do not support a claim as stated.")
        print("  WITHIN-NOISE          → the delta is inside run-to-run variation; report it as a tie")
        print("  BELOW-DECLARED-DELTA  → real but smaller than the effect size you pre-registered")
        print("  UNDERPOWERED          → too few seeds to say anything; add seeds, do not add adjectives")
        print("  WORSE-THAN-BASELINE   → the sign is against you")
    else:
        print("All comparisons clear the noise floor and the declared effect size.")
    return 0 if (args.report_only or not problems) else 1


if __name__ == "__main__":
    sys.exit(main())
