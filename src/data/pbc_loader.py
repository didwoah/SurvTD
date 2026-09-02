"""
Primary Biliary Cirrhosis (PBC) Longitudinal Clinical Trial Loader:
- 312 patients with irregular visit schedules and >55% right-censoring.
- Dynamic biomarkers: bilirubin, albumin, alkaline phosphatase, SGOT, platelets, etc.
"""

import os
import numpy as np
import pandas as pd
import torch

from src.data.dataset import LongitudinalSurvivalDataset


def load_pbc(
    csv_path: str = "baselines/signature_survival/competing_methods/Dynamic_DeepHit/data/pbc2_cleaned.csv",
    seed: int = 42
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"PBC dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Feature columns
    feature_cols = [
        'drug', 'age', 'sex', 'ascites', 'hepatomegaly', 'spiders', 'edema',
        'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets',
        'prothrombin', 'histologic'
    ]

    # Impute missing values with column median
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    # Standardize continuous variables
    mean_vals = df[feature_cols].mean()
    std_vals = df[feature_cols].std().replace(0, 1.0)
    df[feature_cols] = (df[feature_cols] - mean_vals) / std_vals

    patients = []
    patient_ids = sorted(df['id'].unique())

    for pid in patient_ids:
        p_df = df[df['id'] == pid].sort_values('times').reset_index(drop=True)
        times = p_df['times'].values.astype(np.float32)
        tte = float(p_df['tte'].iloc[0])
        event = float(p_df['label'].iloc[0])
        feats = p_df[feature_cols].values.astype(np.float32)

        # Elapsed dt
        dts = np.diff(times, prepend=0.0)
        dts[0] = max(1.0, times[0]) if len(times) > 0 and times[0] > 0 else 1.0
        # Replace zero dts with minimal epsilon
        dts = np.where(dts <= 0, 1.0, dts)

        events = np.zeros(len(times), dtype=np.float32)
        if event > 0.5:
            events[-1] = 1.0

        patients.append({
            'id': int(pid),
            'features': torch.tensor(feats, dtype=torch.float32),
            'dts': torch.tensor(dts, dtype=torch.float32),
            'times': torch.tensor(times, dtype=torch.float32),
            'events': torch.tensor(events, dtype=torch.float32),
            'tte': tte,
            'event': event,
            'mask': None
        })

    # 80/20 train/test split
    rng = np.random.default_rng(seed)
    n_total = len(patients)
    n_train = int(0.8 * n_total)
    perm = rng.permutation(n_total)

    train_patients = [patients[i] for i in perm[:n_train]]
    test_patients = [patients[i] for i in perm[n_train:]]

    max_horizon = float(df['tte'].quantile(0.95))

    return (
        LongitudinalSurvivalDataset(train_patients),
        LongitudinalSurvivalDataset(test_patients),
        len(feature_cols),
        max_horizon
    )
