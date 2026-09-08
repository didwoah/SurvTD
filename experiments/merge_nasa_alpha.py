#!/usr/bin/env python3
"""Assemble the per-cell NASA alpha-ablation files into one summary.

`benchmark_nasa_tier1_authentic.py` flushes only after every seed and model, so the
overnight driver runs it once per (arm, seed) into `cells/`. This collects whatever
finished -- a partial grid summarises fine, with `n` per row saying how partial.
"""
import glob
import json
import os
import re

import numpy as np

CELLS = "experiments/results/nasa_alpha_ablation/cells"
OUT = "experiments/results/nasa_alpha_ablation/alpha_ablation_summary.json"
PAT = re.compile(r"(?P<arm>survtd_a[0-9.]+)_seed(?P<seed>\d+)\.json$")


def main():
    raw = {}
    for f in sorted(glob.glob(os.path.join(CELLS, "*.json"))):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:          # killed mid-write
            print(f"  skipping unreadable {os.path.basename(f)}")
            continue
        summ = d.get("summary", {})
        if not summ:
            continue
        s = next(iter(summ.values()))          # one model per file
        raw.setdefault(m["arm"], {})[m["seed"]] = {
            "dynamic_cindex": s["raw_dynamic_cindex"][0],
            "dynamic_bs": s["raw_dynamic_bs"][0],
            "t0_cindex": s["raw_t0_cindex"][0],
        }

    summary = {}
    for arm, seeds in sorted(raw.items(), key=lambda kv: float(kv[0][len("survtd_a"):])):
        dc = [v["dynamic_cindex"] for v in seeds.values()]
        bs = [v["dynamic_bs"] for v in seeds.values()]
        t0 = [v["t0_cindex"] for v in seeds.values()]
        summary[arm] = {
            "alpha_anchor": float(arm[len("survtd_a"):]),
            "n_seeds": len(dc),
            "seeds": sorted(seeds, key=int),
            "dynamic_cindex_mean": float(np.mean(dc)),
            "dynamic_cindex_std": float(np.std(dc, ddof=1)) if len(dc) > 1 else 0.0,
            "dynamic_bs_mean": float(np.mean(bs)),
            "t0_cindex_mean": float(np.mean(t0)),
            "per_seed": {s: seeds[s]["dynamic_cindex"] for s in sorted(seeds, key=int)},
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"summary": summary, "note":
               "alpha_anchor mixes the Cramer anchor against the TD loss: 1.0 zeroes "
               "the TD term (anchor-only control), 0.0 drops the anchor. Protocol is "
               "FD001 + CoxSig score(), 15 epochs, matching "
               "final_5seeds_authentic_benchmark.json."},
              open(OUT, "w"), indent=2)

    print(f"\n{'arm':18s} {'alpha':>6s} {'dyn C-index':>20s} {'t0 C':>8s}  n")
    for arm, s in summary.items():
        print(f"{arm:18s} {s['alpha_anchor']:>6.2f} "
              f"{s['dynamic_cindex_mean']:>10.4f} +- {s['dynamic_cindex_std']:.4f} "
              f"{s['t0_cindex_mean']:>8.4f}  {s['n_seeds']}")

    # Q1 on NASA, paired by seed, against the anchor-only control.
    ctrl = raw.get("survtd_a1.0", {})
    base = raw.get("survtd_a0.5", {})
    common = sorted(set(ctrl) & set(base), key=int)
    if common:
        d = np.array([base[s]["dynamic_cindex"] - ctrl[s]["dynamic_cindex"]
                      for s in common])
        sd = d.std(ddof=1) if len(d) > 1 else 0.0
        t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else float("nan")
        print(f"\nQ1 on NASA  SurvTD(0.5) - anchor-only(1.0): "
              f"{d.mean():+.4f} +- {sd:.4f}  {int((d > 0).sum())}/{len(d)} seeds  t={t:+.2f}")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
