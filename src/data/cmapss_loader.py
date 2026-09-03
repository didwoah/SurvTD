"""
NASA C-MAPSS irregular degradation loader (real data).

Implements the 50% Poisson cycle-downsampling protocol to induce continuous temporal
irregularity on turbofan degradation trajectories.

Defects repaired here
---------------------
* **Standardization leaked across the split.** Sensor means and standard deviations
  were computed over the whole file, including the units that become the test split.
  They are now fit on the train split's rows only.
* **`max_horizon` returned the wrong quantity (D-22).** It was
  `df['cycle'].quantile(0.95)`, a percentile of pooled observation CYCLE INDICES
  across all rows including test -- not of time-to-event. It is now a train-split
  quantile of tte.
* **There was no censoring and the failure cycle leaked (X-07).** Every unit has
  `event = 1.0` and the trajectory runs all the way to failure, so the last
  observation *is* the event. Administrative censoring at a declared horizon is now
  available via `admin_censor_time`, which creates genuine censoring out of real data
  with nothing synthetic added. Under the landmark protocol the same effect is
  obtained at `L + Delta`.
* **No validation split**, and **`mask` was None**. Both fixed; C-MAPSS has no
  missingness, so the mask is explicitly all-ones rather than implicitly absent.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from src.data.dataset import LongitudinalSurvivalDataset
from src.data.preprocessing import (
    administratively_censor,
    apply_feature_stats,
    empirical_feature_mean,
    fit_feature_stats,
    subject_level_split,
)

DEFAULT_DIR = "baselines/signature_survival/data_loader/NASA"


def load_cmapss_downsampled(
    data_dir: str = DEFAULT_DIR,
    dataset_id: int = 2,
    poisson_rate: float = 2.0,
    seed: int = 42,
    max_units: int = None,
    admin_censor_time: float = None,
    fracs: tuple = (0.6, 0.2, 0.2),
):
    """
    Returns (train, val, test, input_dim, max_horizon, x_mean).

    Args:
        admin_censor_time: if set, units still running at this cycle become
            right-censored there and later observations are dropped. Leave None when
            the landmark protocol supplies the administrative censoring instead.
    """
    rng = np.random.default_rng(seed)
    feature_cols = ["setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    col_names = ["id", "cycle"] + feature_cols

    train_file = os.path.join(data_dir, f"train_FD00{dataset_id}.txt")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"NASA C-MAPSS file not found at: {train_file}")

    df = pd.read_csv(train_file, sep=r"\s+", header=None, names=col_names)

    # Drop near-constant sensors. Computed on the full file, which is acceptable:
    # it is a structural property of the instrumentation, not a fitted statistic.
    sensor_cols = [c for c in feature_cols if df[c].std() > 1e-4]

    max_cycles = df.groupby("id")["cycle"].max().to_dict()
    unit_ids = sorted(df["id"].unique())
    if max_units is not None:
        unit_ids = unit_ids[:max_units]

    patients = []
    for uid in unit_ids:
        unit_df = df[df["id"] == uid].sort_values("cycle").reset_index(drop=True)
        n_rows = len(unit_df)

        # Poisson downsampling to induce irregular observation gaps.
        sampled = [0]
        cur = 0
        while True:
            cur += int(rng.poisson(lam=poisson_rate)) + 1
            if cur < n_rows - 1:
                sampled.append(cur)
            else:
                sampled.append(n_rows - 1)
                break

        sub = unit_df.iloc[sampled].reset_index(drop=True)
        cycles = sub["cycle"].to_numpy(dtype=np.float32)
        feats = sub[sensor_cols].to_numpy(dtype=np.float32)

        dts = np.diff(cycles, prepend=cycles[0]).astype(np.float32)
        dts[0] = float(cycles[0]) if cycles[0] > 0 else 1.0

        patients.append({
            "id": int(uid),
            "features": torch.tensor(feats, dtype=torch.float32),
            "dts": torch.tensor(dts, dtype=torch.float32),
            "times": torch.tensor(cycles, dtype=torch.float32),
            "events": torch.zeros(len(cycles), dtype=torch.float32),
            # C-MAPSS is fully instrumented: no missingness. Explicit, not None, so
            # the backbone's mask path is exercised uniformly across cohorts.
            "mask": torch.ones((len(cycles), len(sensor_cols)), dtype=torch.float32),
            "tte": float(max_cycles[uid]),
            "event": 1.0,
        })

    tr_idx, va_idx, te_idx = subject_level_split(len(patients), seed=seed, fracs=fracs)
    train = [patients[i] for i in tr_idx]
    val = [patients[i] for i in va_idx]
    test = [patients[i] for i in te_idx]

    if admin_censor_time is not None:
        train, val, test = (
            administratively_censor(s, float(admin_censor_time)) for s in (train, val, test)
        )

    stats = fit_feature_stats(train)
    train, val, test = (apply_feature_stats(s, stats) for s in (train, val, test))
    x_mean = empirical_feature_mean(train)

    # Time-to-event quantile from the train split, not a pooled cycle-index quantile.
    max_horizon = float(np.quantile([p["tte"] for p in train], 0.95))

    return (
        LongitudinalSurvivalDataset(train),
        LongitudinalSurvivalDataset(val),
        LongitudinalSurvivalDataset(test),
        len(sensor_cols),
        max_horizon,
        x_mean,
    )
