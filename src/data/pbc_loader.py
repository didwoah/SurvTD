"""
Primary Biliary Cirrhosis (PBC2) Longitudinal Clinical Trial Loader.

Real data: 312 subjects, irregular visit schedules, dynamic biomarkers.

Defects repaired here
---------------------
* **Competing risks were silently mishandled (X-04).** `label` has three levels,
  `{0.0: 143, 1.0: 140, 2.0: 29}`, where 2 is transplant. The old loader stored
  `event = 2.0` verbatim, then `if event > 0.5` treated transplant as a DEATH in the
  training loss, while `metrics.py`'s `event_indicators == 1` treated the same
  subject as CENSORED in evaluation. Train and eval disagreed on 9% of the cohort.
  Transplant is now declared as censoring at the transplant time; the alternative is
  exposed via `competing_risk_policy` but must be declared before a run.
* **Leakage (X-08).** Imputation used whole-dataframe medians and standardization
  whole-dataframe statistics, both before the split. Both are now fit on train only.
* **No validation split.** Now a subject-level 60/20/20.
* **`mask` was None.** It is now built from the PRE-imputation NaN pattern, so the
  backbone can tell a measured value from an imputed one. This is the only cohort
  where the missingness mask comes from real data.
* **`max_horizon` was wrong (X-07/D-22).** It used `df['tte'].quantile(0.95)`
  computed pre-split; it is now a train-split quantile.
* Docstring claimed ">55% right-censoring"; the measured event rate is 0.62, i.e.
  38% censored (X-09).

Scale note: the cohort's residual time (tte minus last visit) has median 36 while
tte has median 328 and max 744. The delta_s = 100 used previously made the bin width
~3x the median quantity being predicted and left ~75% of a 30-bin grid structurally
empty; see cohorts.py for the corrected grid (X-05).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from src.data.dataset import LongitudinalSurvivalDataset
from src.data.preprocessing import (
    apply_feature_stats,
    empirical_feature_mean,
    fit_feature_stats,
    subject_level_split,
)

DEFAULT_CSV = "baselines/signature_survival/competing_methods/Dynamic_DeepHit/data/pbc2_cleaned.csv"

FEATURE_COLS = [
    "drug", "age", "sex", "ascites", "hepatomegaly", "spiders", "edema",
    "serBilir", "serChol", "albumin", "alkaline", "SGOT", "platelets",
    "prothrombin", "histologic",
]

# PBC2 `label`: 0 = censored alive, 1 = death, 2 = transplant.
LABEL_CENSORED, LABEL_DEATH, LABEL_TRANSPLANT = 0.0, 1.0, 2.0


def load_pbc(
    csv_path: str = DEFAULT_CSV,
    seed: int = 42,
    competing_risk_policy: str = "censor",
    fracs: tuple = (0.6, 0.2, 0.2),
):
    """
    Returns (train, val, test, input_dim, max_horizon, x_mean).

    Args:
        competing_risk_policy:
            'censor' (default, declared): transplant is right-censoring at the
                transplant time. The event of interest is death.
            'event': transplant counts as the event. Only valid if declared in the
                preregistration before the run.
    """
    if competing_risk_policy not in ("censor", "event"):
        raise ValueError(f"unknown competing_risk_policy {competing_risk_policy!r}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"PBC dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Missingness mask from the RAW pattern, before any imputation.
    observed = df[FEATURE_COLS].notna().to_numpy(dtype=np.float32)
    df = df.copy()
    df["_row"] = np.arange(len(df))

    patients_raw = []
    for pid in sorted(df["id"].unique()):
        p = df[df["id"] == pid].sort_values("times")
        rows = p["_row"].to_numpy()
        times = p["times"].to_numpy(dtype=np.float32)
        label = float(p["label"].iloc[0])
        tte = float(p["tte"].iloc[0])

        if label == LABEL_DEATH:
            event = 1.0
        elif label == LABEL_TRANSPLANT:
            event = 1.0 if competing_risk_policy == "event" else 0.0
        else:
            event = 0.0

        dts = np.diff(times, prepend=np.float32(0.0)).astype(np.float32)
        dts[0] = max(float(times[0]), 1.0)
        dts = np.where(dts <= 0, 1.0, dts).astype(np.float32)

        patients_raw.append({
            "id": int(pid),
            "_rows": rows,
            "times": torch.tensor(times, dtype=torch.float32),
            "dts": torch.tensor(dts, dtype=torch.float32),
            "events": torch.zeros(len(times), dtype=torch.float32),
            "mask": torch.tensor(observed[rows], dtype=torch.float32),
            "tte": tte,
            "event": event,
            "label_raw": label,
        })

    tr_idx, va_idx, te_idx = subject_level_split(len(patients_raw), seed=seed, fracs=fracs)

    # Imputation medians from the TRAIN split's rows only.
    train_rows = np.concatenate([patients_raw[i]["_rows"] for i in tr_idx])
    medians = df.iloc[train_rows][FEATURE_COLS].median()
    filled = df[FEATURE_COLS].fillna(medians).to_numpy(dtype=np.float32)

    for p in patients_raw:
        p["features"] = torch.tensor(filled[p["_rows"]], dtype=torch.float32)
        del p["_rows"]

    train = [patients_raw[i] for i in tr_idx]
    val = [patients_raw[i] for i in va_idx]
    test = [patients_raw[i] for i in te_idx]

    stats = fit_feature_stats(train)
    train, val, test = (apply_feature_stats(s, stats) for s in (train, val, test))
    x_mean = empirical_feature_mean(train)

    max_horizon = float(np.quantile([p["tte"] for p in train], 0.95))

    return (
        LongitudinalSurvivalDataset(train),
        LongitudinalSurvivalDataset(val),
        LongitudinalSurvivalDataset(test),
        len(FEATURE_COLS),
        max_horizon,
        x_mean,
    )
