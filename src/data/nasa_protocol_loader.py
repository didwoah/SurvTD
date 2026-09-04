"""
NASA C-MAPSS FD001 Literature Parity Loader.

Constructs the 200-engine dataset (100 run-to-failure + 100 cut-off) matching CoxSig & DeepTCSR,
with strictly LEAK-FREE preprocessing (Scaler fitted only on train split).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Optional
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


RELEVANT_FEATURES = [
    'setting1', 'setting2', 's2', 's3', 's4', 's7',
    's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21'
]


def load_raw_nasa_fd001(
    data_dir: str = "baselines/signature_survival/data_loader/NASA"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load train_FD001.txt and test_FD001.txt and merge into 200 units.
    """
    features_col_name = ['setting1', 'setting2', 'setting3'] + [f"s{i}" for i in range(1, 22)]
    col_names = ['id', 'times'] + features_col_name

    train_path = os.path.join(data_dir, "train_FD001.txt")
    test_path = os.path.join(data_dir, "test_FD001.txt")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"NASA data not found at {data_dir}")

    df_train = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
    df_test = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)

    # Label: Train = 1 (event observed), Test = 0 (right-censored at cutoff)
    df_train['tte'] = df_train.groupby('id')['times'].transform('max')
    df_train['event'] = 1

    df_test['tte'] = df_test.groupby('id')['times'].transform('max')
    df_test['event'] = 0
    # Offset test IDs to 101-200
    df_test['id'] = df_test['id'] + df_train['id'].max()

    # Zero-base times (cycle 1 -> 0)
    df_train['times'] = df_train['times'] - 1.0
    df_train['tte'] = df_train['tte'] - 1.0
    df_test['times'] = df_test['times'] - 1.0
    df_test['tte'] = df_test['tte'] - 1.0

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all = df_all[['id', 'times', 'tte', 'event'] + RELEVANT_FEATURES]

    return df_all, RELEVANT_FEATURES


def get_nasa_splits(
    seed: int = 42,
    test_ratio: float = 0.2,
    time_scale: float = 100.0,
    data_dir: str = "baselines/signature_survival/data_loader/NASA"
) -> Dict[str, any]:
    """
    Produce 80/20 train/test split with strict leak-free StandardScaler.
    
    Returns
    -------
    dict with:
      - paths_train, paths_test : torch.Tensor of shape (n_samples, n_sampling_times, 1 + n_features)
      - surv_labels_train, surv_labels_test : np.ndarray of shape (n_samples, 2) [tte, event]
      - sampling_times : np.ndarray
      - train_units, test_units : list of unit IDs
      - scaler : fitted StandardScaler
      - df_train, df_test : scaled dataframes
    """
    df_all, feat_cols = load_raw_nasa_fd001(data_dir)
    all_units = np.unique(df_all['id'].values)
    n_units = len(all_units)  # 200

    # 1. Split unit IDs
    rng = np.random.default_rng(seed)
    n_train = int((1.0 - test_ratio) * n_units)
    train_units = rng.choice(all_units, size=n_train, replace=False)
    test_units = np.array([u for u in all_units if u not in train_units])

    df_train = df_all[df_all['id'].isin(train_units)].copy()
    df_test = df_all[df_all['id'].isin(test_units)].copy()

    # 2. Strict leak-free normalization: fit scaler ONLY on train
    scaler = StandardScaler()
    df_train[feat_cols] = scaler.fit_transform(df_train[feat_cols])
    df_test[feat_cols] = scaler.transform(df_test[feat_cols])

    # 3. Apply time scale
    df_train['times_scaled'] = df_train['times'] / time_scale
    df_train['tte_scaled'] = df_train['tte'] / time_scale
    df_test['times_scaled'] = df_test['times'] / time_scale
    df_test['tte_scaled'] = df_test['tte'] / time_scale

    # Combined sampling grid across train and test for CoxSig / continuous paths
    all_times = np.concatenate([
        np.zeros(1),
        np.unique(np.concatenate([df_train['times_scaled'].values, df_test['times_scaled'].values]))
    ])
    sampling_times = np.unique(all_times)
    n_sampling_times = len(sampling_times)
    n_feats = len(feat_cols)

    def build_path_tensor(df: pd.DataFrame, units: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        n_sub = len(units)
        X = np.zeros((n_sub, n_sampling_times, 1 + n_feats), dtype=np.float32)
        surv_labels = np.zeros((n_sub, 2), dtype=np.float32)

        for i, u in enumerate(units):
            sub_df = df[df['id'] == u]
            surv_labels[i, 0] = sub_df['tte_scaled'].iloc[0]
            surv_labels[i, 1] = sub_df['event'].iloc[0]

            # Forward-fill / back-fill along sampling_times
            grid_df = pd.DataFrame({'id': u, 'times_scaled': sampling_times})
            merged = pd.merge(grid_df, sub_df, how='left', on=['id', 'times_scaled'])
            merged = merged.ffill().bfill()
            X[i, :, 0] = merged['times_scaled'].values
            X[i, :, 1:] = merged[feat_cols].values

        return torch.from_numpy(X), surv_labels

    paths_train, surv_labels_train = build_path_tensor(df_train, train_units)
    paths_test, surv_labels_test = build_path_tensor(df_test, test_units)

    return {
        "paths_train": paths_train,
        "surv_labels_train": surv_labels_train,
        "paths_test": paths_test,
        "surv_labels_test": surv_labels_test,
        "sampling_times": sampling_times,
        "train_units": train_units,
        "test_units": test_units,
        "scaler": scaler,
        "df_train": df_train,
        "df_test": df_test,
        "time_scale": time_scale,
        "feat_cols": feat_cols
    }
