"""
NASA C-MAPSS Irregular Degradation Dataset Loader:
- Implements the 50% Poisson cycle downsampling protocol (mean interval 2 cycles)
  to induce continuous temporal irregularity on turbofan engine degradation trajectories.
"""

import os
import random
import numpy as np
import pandas as pd
import torch

from src.data.dataset import LongitudinalSurvivalDataset


def load_cmapss_downsampled(
    data_dir: str = "baselines/signature_survival/data_loader/NASA",
    dataset_id: int = 1,
    poisson_rate: float = 2.0,
    seed: int = 42,
    max_units: int = None
):
    """
    Loads NASA C-MAPSS FD00{dataset_id} and applies Poisson downsampling.
    """
    rng = np.random.default_rng(seed)
    features_col_name = ['setting1', 'setting2', 'setting3'] + [f"s{i}" for i in range(1, 22)]
    col_names = ['id', 'cycle'] + features_col_name

    train_file = os.path.join(data_dir, f"train_FD00{dataset_id}.txt")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"NASA C-MAPSS file not found at: {train_file}")

    df = pd.read_csv(train_file, sep=r'\s+', header=None, names=col_names)

    # Filter out near-constant sensors
    sensor_cols = []
    for col in features_col_name:
        if df[col].std() > 1e-4:
            sensor_cols.append(col)

    # Normalize sensor features
    mean_vals = df[sensor_cols].mean()
    std_vals = df[sensor_cols].std().replace(0, 1.0)
    df[sensor_cols] = (df[sensor_cols] - mean_vals) / std_vals

    # Compute max cycle (TTE) for each engine
    max_cycles = df.groupby('id')['cycle'].max().to_dict()

    unit_ids = sorted(df['id'].unique())
    if max_units is not None:
        unit_ids = unit_ids[:max_units]

    patients = []

    for uid in unit_ids:
        unit_df = df[df['id'] == uid].sort_values('cycle').reset_index(drop=True)
        total_cycles = max_cycles[uid]

        # 50% Poisson downsampling: skip cycles with Poisson steps
        sampled_indices = [0]
        cur_idx = 0
        N = len(unit_df)
        while True:
            step = rng.poisson(lam=poisson_rate) + 1  # at least 1 cycle step
            cur_idx += step
            if cur_idx < N - 1:
                sampled_indices.append(cur_idx)
            else:
                sampled_indices.append(N - 1)  # include final cycle before failure
                break

        sampled_df = unit_df.iloc[sampled_indices].reset_index(drop=True)
        cycles = sampled_df['cycle'].values.astype(np.float32)
        feats = sampled_df[sensor_cols].values.astype(np.float32)

        # Elapsed dt in cycles
        dts = np.diff(cycles, prepend=cycles[0])
        dts[0] = 1.0  # initial baseline dt

        # Events: C-MAPSS engines run until failure (event = 1.0)
        events = np.zeros(len(cycles), dtype=np.float32)
        events[-1] = 1.0

        patients.append({
            'id': int(uid),
            'features': torch.tensor(feats, dtype=torch.float32),
            'dts': torch.tensor(dts, dtype=torch.float32),
            'times': torch.tensor(cycles, dtype=torch.float32),
            'events': torch.tensor(events, dtype=torch.float32),
            'tte': float(total_cycles),
            'event': 1.0,
            'mask': None
        })

    # Train/Test Split (80% / 20%)
    n_total = len(patients)
    n_train = int(0.8 * n_total)
    perm = rng.permutation(n_total)

    train_patients = [patients[i] for i in perm[:n_train]]
    test_patients = [patients[i] for i in perm[n_train:]]

    return (
        LongitudinalSurvivalDataset(train_patients),
        LongitudinalSurvivalDataset(test_patients),
        len(sensor_cols),
        float(df['cycle'].quantile(0.95))
    )
