import os
import json
import numpy as np

DIR = "experiments/results/nasa_tier1_authentic"
files = [
    os.path.join(DIR, "benchmark_results.json"),
    os.path.join(DIR, "part_123_456.json"),
    os.path.join(DIR, "part_789_101112.json")
]

models = ["coxsig", "ncde", "deeptcsr", "ddh", "survtd"]
merged_raw = {m: {"dynamic_cindex": [], "dynamic_bs": [], "t0_cindex": [], "t0_bs": []} for m in models}
all_seeds = []

for fpath in files:
    if os.path.exists(fpath):
        with open(fpath) as f:
            d = json.load(f)
        all_seeds.extend(d.get("seeds", []))
        for m in models:
            if m in d.get("summary", {}):
                m_sum = d["summary"][m]
                merged_raw[m]["dynamic_cindex"].extend(m_sum.get("raw_dynamic_cindex", []))
                merged_raw[m]["dynamic_bs"].extend(m_sum.get("raw_dynamic_bs", []))
                merged_raw[m]["t0_cindex"].extend(m_sum.get("raw_t0_cindex", []))
                merged_raw[m]["t0_bs"].extend(m_sum.get("raw_t0_bs", []))

summary = {}
for m in models:
    summary[m] = {
        "dynamic_cindex_mean": float(np.mean(merged_raw[m]["dynamic_cindex"])),
        "dynamic_cindex_std": float(np.std(merged_raw[m]["dynamic_cindex"])),
        "dynamic_bs_mean": float(np.mean(merged_raw[m]["dynamic_bs"])),
        "dynamic_bs_std": float(np.std(merged_raw[m]["dynamic_bs"])),
        "t0_cindex_mean": float(np.mean(merged_raw[m]["t0_cindex"])),
        "t0_cindex_std": float(np.std(merged_raw[m]["t0_cindex"])),
        "t0_bs_mean": float(np.mean(merged_raw[m]["t0_bs"])),
        "t0_bs_std": float(np.std(merged_raw[m]["t0_bs"])),
        "raw_dynamic_cindex": merged_raw[m]["dynamic_cindex"],
        "raw_dynamic_bs": merged_raw[m]["dynamic_bs"],
        "raw_t0_cindex": merged_raw[m]["t0_cindex"],
        "raw_t0_bs": merged_raw[m]["t0_bs"]
    }

final_output = {
    "summary": summary,
    "seeds": all_seeds,
    "epochs": 15
}

out_file = os.path.join(DIR, "final_5seeds_authentic_benchmark.json")
with open(out_file, "w") as f:
    json.dump(final_output, f, indent=2)

print("Successfully merged 5 seeds into", out_file)
for m in models:
    s = summary[m]
    print(f"{m.upper():12s} | Dyn C-index: {s['dynamic_cindex_mean']:.4f} ± {s['dynamic_cindex_std']:.4f} | Dyn Brier: {s['dynamic_bs_mean']:.4f} ± {s['dynamic_bs_std']:.4f} | t0 C-index: {s['t0_cindex_mean']:.4f} ± {s['t0_cindex_std']:.4f}")
