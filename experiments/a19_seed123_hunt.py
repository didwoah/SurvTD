"""
A-19: why does anchor/ce collapse on seed 123? A 2x2 on the two variables this
amendment chain introduced, run on the one seed where the failure is large.

anchor/ce's loss is bit-identical to Dynamic-DeepHit's L1 (verified per visit, both
branches, max |diff| = 0.00e+00), yet on seed 123 it scores 0.3329 against DeepHit's
0.7112 and Person-Period-raw's 0.7069 -- below chance, on the cohort where every other
arm finds the most signal. A proper scoring rule cannot invert a ranking; this is a
defect signature, and the project's own history says so (Person-Period scored 0.263
before repair for exactly this kind of reason).

Two candidates were introduced by this amendment chain and are separable:
  * IPCW (A-04/D11): weights up to 10x, on the 71% of trajectories that are censored,
    rising with censoring time -- i.e. concentrated on the long survivors. DeepHit and
    Person-Period have none.
  * The A-17b normalization constant (ce x 1.8102): before it, `ce` was the LEAST
    clipped arm (initial ||g|| 4.98, clip-free by epoch 3); matching it to `cramer`
    pushed it to 12.2 and kept it clipped longer. That is a variable I introduced, so
    it has to be falsifiable on its own rather than assumed harmless.

Seed 123 only, alpha = 1, 20 epochs. Reference cells on this seed:
    anchor/cramer (KC5)  0.5185      anchor/ce (A-17b)  0.3329
    Person-Period raw    0.7069      Dynamic-DeepHit    0.7112
"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.cohorts import COHORTS
from src.training.trainer import get_device
from experiments.run_track_b import train_and_eval_survtd

SEED = 123
REF = {"anchor/cramer (ipcw on, KC5)": 0.5185, "anchor/ce (ipcw on, scale 1.81)": 0.3329,
       "Person-Period raw (MLE)": 0.7069, "Dynamic-DeepHit": 0.7112}
CELLS = [
    ("ce      ipcw=off  scale=1.81", "ce",     False, 1.8102),
    ("ce      ipcw=on   scale=1.00", "ce",     True,  1.0),
    ("ce      ipcw=off  scale=1.00", "ce",     False, 1.0),
    ("cramer  ipcw=off  scale=1.00", "cramer", False, 1.0),
]

def main():
    out = Path("experiments/results/a19_seed123_hunt"); out.mkdir(parents=True, exist_ok=True)
    device = get_device(); spec = COHORTS["synthetic_icu"]; cd = spec.load(seed=SEED)
    print("=" * 74)
    print(f"  A-19 SEED {SEED} HUNT  |  alpha=1, 20 epochs, device={device}")
    for k, v in REF.items():
        print(f"    reference  {k:<34} {v:.4f}")
    print("=" * 74)
    rec = {}
    for label, aloss, ipcw, scale in CELLS:
        t0 = time.time()
        _, c, auc = train_and_eval_survtd(
            cd, spec, "full", 1.0, 20, 16, 0.001, device, seed=SEED,
            anchor_loss=aloss, use_ipcw=ipcw, anchor_scale=scale)
        rec[label] = {"c_td": c, "auc": auc, "anchor_loss": aloss,
                      "use_ipcw": ipcw, "anchor_scale": scale,
                      "wall_clock_s": round(time.time() - t0, 1)}
        print(f"  {label:<30} C_td {c:.4f}  AUC {auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        (out / "a19_raw.json").write_text(json.dumps({"seed": SEED, "reference": REF,
                                                      "records": rec}, indent=2))
    print("-" * 74)
    base = REF["anchor/ce (ipcw on, scale 1.81)"]
    for label in ("ce      ipcw=off  scale=1.81", "ce      ipcw=on   scale=1.00",
                  "ce      ipcw=off  scale=1.00"):
        if label in rec:
            print(f"  {label:<30} delta vs 0.3329 = {rec[label]['c_td'] - base:+.4f}")
    if "cramer  ipcw=off  scale=1.00" in rec:
        print(f"  {'cramer ipcw=off':<30} delta vs 0.5185 = "
              f"{rec['cramer  ipcw=off  scale=1.00']['c_td'] - REF['anchor/cramer (ipcw on, KC5)']:+.4f}")
    print("=" * 74)
    return 0

if __name__ == "__main__":
    sys.exit(main())
