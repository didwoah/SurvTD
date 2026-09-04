"""
Framingham Heart Study loader.

4,434 participants, at most three scheduled exams, 18 covariates. Its role in Table 1
is scale and regularity, not irregularity: PBC2's test split is 63 subjects, which is
where most of the between-seed variance in this project's results has come from, and
Framingham's is ~880. If the duration-awareness claim is real it should hold up here
without winning, because the exams are *scheduled* -- this is the cohort where SurvTD
has the least Delta-t signal to exploit.

Provenance
----------
`baselines/dynamic_deephit_pytorch/DeepSurvivalMachines/dsm/datasets/framingham.csv`,
the copy bundled with Deep Survival Machines and used by the Dynamic-DeepHit PyTorch
port -- the same "read the authors' own file" policy as `pbc_loader.py:48`.

The feature set is upstream's exactly (`dsm/datasets.py:83-91`): 10 "categorical" plus
8 numeric columns. Upstream builds them with `pd.get_dummies(dat_cat)`, which on this
file is a **no-op** -- every one of those columns is already numeric dtype, so
get_dummies passes them through unchanged and the design matrix is 18 wide, not wider.
Verified rather than assumed; `test_framingham_literature_fidelity.py` pins it.

Two upstream steps deliberately NOT copied
------------------------------------------
`dsm/datasets.py:96-97` runs `SimpleImputer` and `StandardScaler` over the **whole**
file before any split. That is the leak this project repaired as X-08 for PBC2, and
copying it here to "match upstream" would reintroduce it. Imputation medians and
standardisation statistics are fit on the training split only, as in `pbc_loader.py`.
The feature *set* is upstream's; the feature *scaling* is this project's.

Upstream also stores `time = TIMEDTH - TIME`, i.e. residual time per row. This loader
keeps the absolute clock (`tte = TIMEDTH`, `times = TIME`) because that is the contract
`LongitudinalSurvivalDataset` and every arm's adapter are written against.

Cohort properties that change how results must be read
------------------------------------------------------
* **Censoring is entirely administrative and at a single time.** Every censored
  subject has `TIMEDTH == 8766` (24 years); there is exactly one censoring time in the
  cohort. The censoring KM is therefore a step function that stays at 1.0 until 8766,
  so IPCW weights are ~1 over any horizon inside the follow-up and the IPCW IBS is
  close to an unweighted Brier score. That is a property of the data, not a bug, but a
  reader comparing IBS across cohorts has to know it.
* **Residual time is long**: median tte minus last exam is 4,376 days, against PBC2's
  36. `delta_s` is a year here, not a month; see `cohorts.py`.
* Event rate 0.3496 = 1550/4434.

`events` (per visit) is left all-zero, matching the other four loaders. Under D15 that
array is no longer read for the event flag by anything -- the three sites that did now
derive it from `tau_event` -- and introducing a fifth convention here would be worse
than keeping the shared one.
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

DEFAULT_CSV = ("baselines/dynamic_deephit_pytorch/DeepSurvivalMachines/dsm/"
               "datasets/framingham.csv")

# Upstream's split, kept in upstream's order so the design matrix is column-comparable.
CATEGORICAL_COLS = ["SEX", "CURSMOKE", "DIABETES", "BPMEDS", "educ",
                    "PREVCHD", "PREVAP", "PREVMI", "PREVSTRK", "PREVHYP"]
NUMERIC_COLS = ["TOTCHOL", "AGE", "SYSBP", "DIABP",
                "CIGPDAY", "BMI", "HEARTRTE", "GLUCOSE"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

ID_COL, TIME_COL, TTE_COL, EVENT_COL = "RANDID", "TIME", "TIMEDTH", "DEATH"


def load_framingham(
    csv_path: str = DEFAULT_CSV,
    seed: int = 42,
    fracs: tuple = (0.6, 0.2, 0.2),
):
    """Returns (train, val, test, input_dim, max_horizon, x_mean)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Framingham dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS + [ID_COL, TIME_COL, TTE_COL, EVENT_COL]
               if c not in df.columns]
    if missing:
        raise ValueError(f"Framingham CSV is missing columns: {missing}")

    # Missingness mask from the RAW pattern, before any imputation. Real, unlike the
    # simulated cohorts: BPMEDS is 5.1% missing and GLUCOSE 12.4%.
    observed = df[FEATURE_COLS].notna().to_numpy(dtype=np.float32)
    df = df.copy()
    df["_row"] = np.arange(len(df))

    patients_raw = []
    for pid in sorted(df[ID_COL].unique()):
        p = df[df[ID_COL] == pid].sort_values(TIME_COL)
        rows = p["_row"].to_numpy()
        times = p[TIME_COL].to_numpy(dtype=np.float32)

        # Both are constant within a subject in this file; asserted, not assumed.
        tte = float(p[TTE_COL].iloc[0])
        event = float(p[EVENT_COL].iloc[0])
        if p[TTE_COL].nunique() != 1 or p[EVENT_COL].nunique() != 1:
            raise ValueError(f"subject {pid}: {TTE_COL}/{EVENT_COL} vary within subject")
        # A visit at or after the event time would leak the outcome into the covariates.
        if times[-1] >= tte:
            raise ValueError(f"subject {pid}: last exam {times[-1]} >= tte {tte}")

        dts = np.diff(times, prepend=np.float32(0.0)).astype(np.float32)
        dts[0] = max(float(times[0]), 1.0)          # exam 1 is TIME == 0 for everyone
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
        })

    tr_idx, va_idx, te_idx = subject_level_split(len(patients_raw), seed=seed, fracs=fracs)

    # Imputation medians from the TRAIN split's rows only (X-08).
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
