"""Summarize A-17b from the incremental raw file, so a run stopped early is still
reportable. Reference cells are KC5's, measured on the same training path."""
import json, sys
from pathlib import Path
import numpy as np

REF = {"anchor": (0.5318, 0.0407), "td": (0.5353, 0.0841)}
GATE = 0.60

p = Path(sys.argv[1] if len(sys.argv) > 1
         else "experiments/results/a17_loss_geometry/a17_raw.json")
d = json.loads(p.read_text())
rec, arms = d["records"], d["arms"]

print("=" * 84)
print(f"  A-17b LOSS GEOMETRY  |  {d['cohort']}  epochs={d['epochs']}")
print(f"  reference (KC5 path): anchor/cramer {REF['anchor'][0]:.4f}+-{REF['anchor'][1]:.4f}   "
      f"td/cramer {REF['td'][0]:.4f}+-{REF['td'][1]:.4f}")
print("=" * 84)
out = {}
for lab, meta in arms.items():
    seeds = sorted(rec.get(lab, {}), key=lambda x: int(x))
    v = np.array([rec[lab][s]["c_td"] for s in seeds if not np.isnan(rec[lab][s]["c_td"])])
    if v.size == 0:
        continue
    axis = "anchor" if meta["alpha"] == 1.0 else "td"
    base, base_sd = REF[axis]
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    print(f"  {lab:<22} n={v.size}  {v.mean():.4f} +- {sd:.4f}   "
          f"vs cramer {v.mean()-base:+.4f}   SD ratio {sd/base_sd:.2f}x   "
          f"seeds {[round(x,4) for x in v]}")
    out[lab] = {"n": int(v.size), "mean": float(v.mean()), "sd": sd, "axis": axis,
                "delta_vs_cramer": float(v.mean()-base), "sd_ratio": float(sd/base_sd),
                "per_seed": v.tolist(), "seeds": seeds[:v.size]}
print("-" * 84)
for axis, name in (("anchor", "ANCHOR"), ("td", "TD")):
    cells = {k: x for k, x in out.items() if x["axis"] == axis}
    if not cells:
        continue
    spread = max(c["mean"] for c in cells.values()) - min(c["mean"] for c in cells.values())
    best = max(cells, key=lambda k: cells[k]["mean"])
    if axis == "anchor":
        v = f"{'PASS' if cells[best]['mean'] >= GATE else 'fail'} (gate {GATE})"
    else:
        v = ("improved" if cells[best]["mean"] > REF['td'][0] else "not improved") + \
            f"; best SD ratio {min(c['sd_ratio'] for c in cells.values()):.2f}x"
    print(f"  {name:<7} best={best.split('/')[-1]:<14} spread across geometries {spread:.4f}   -> {v}")
print("=" * 84)
Path(p.parent / "a17_summary_partial.json").write_text(
    json.dumps({"anchor_gate": GATE, "reference": REF, "summary": out}, indent=2))
print(f"Saved -> {p.parent / 'a17_summary_partial.json'}")
