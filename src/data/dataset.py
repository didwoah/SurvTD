"""
Unified Time-Series Survival Dataset and Collate Functions:
Supports irregular observation intervals, variable sequence lengths, and censoring indicators.
"""

import math

import numpy as np
import torch
from torch.utils.data import Dataset


class LongitudinalSurvivalDataset(Dataset):
    """
    In-memory list of patient trajectories:
    Each item is a dictionary with keys:
    - 'id': int or str
    - 'features': Tensor (L, D)
    - 'dts': Tensor (L,)
    - 'times': Tensor (L,)
    - 'events': Tensor (L,)
    - 'tte': float
    - 'event': float (0 = censored, 1 = event)
    - 'mask': optional Tensor (L, D)
    """
    def __init__(self, patient_list):
        self.patients = patient_list

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        return self.patients[idx]


def collate_patient_batch(batch):
    """
    Pads variable-length sequences to max sequence length in batch.
    """
    max_len = max(p['features'].shape[0] for p in batch)
    feat_dim = batch[0]['features'].shape[1]
    B = len(batch)

    padded_features = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    padded_dts = torch.zeros(B, max_len, dtype=torch.float32)
    padded_times = torch.zeros(B, max_len, dtype=torch.float32)
    padded_events = torch.zeros(B, max_len, dtype=torch.float32)
    seq_lengths = torch.zeros(B, dtype=torch.long)
    ttes = torch.zeros(B, dtype=torch.float32)
    overall_events = torch.zeros(B, dtype=torch.float32)
    mask = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)

    for i, p in enumerate(batch):
        L = p['features'].shape[0]
        seq_lengths[i] = L
        padded_features[i, :L] = p['features']
        padded_dts[i, :L] = p['dts']
        padded_times[i, :L] = p['times']
        padded_events[i, :L] = p['events']
        ttes[i] = float(p['tte'])
        overall_events[i] = float(p['event'])
        if 'mask' in p and p['mask'] is not None:
            mask[i, :L] = p['mask']
        else:
            mask[i, :L] = 1.0

    return {
        'features': padded_features,
        'dts': padded_dts,
        'times': padded_times,
        'events': padded_events,
        'seq_lengths': seq_lengths,
        'ttes': ttes,
        'overall_events': overall_events,
        'mask': mask,
        'raw_batch': batch
    }


def expand_to_regular_grid(dataset, grid_step: float, max_steps: int = 512):
    """
    Forward-fill each irregular trajectory onto a uniform grid of width `grid_step`.

    This is the Rung 1 definition in preregistration section 3 -- "person-period
    expanded discrete hazard model evaluated on a 1-hour regular grid (forward-fill
    interpolation)" -- which the shipped code described in its docstring but never
    performed. Implementing it rather than dropping the claim is the conservative
    choice: it makes the naive baseline stronger, not weaker.

    Semantics:
      * grid points are t = grid_step, 2*grid_step, ... up to the last observation
      * each grid point carries the most recent observation at or before it
        (forward fill); grid points before the first observation carry it too
      * `mask` is 1 only where a real observation lands on that grid point, so a
        missingness-aware backbone can tell carried-forward values from fresh ones
      * `tte`, `event` and the subject id are untouched

    Args:
        dataset: iterable of patient dicts (see LongitudinalSurvivalDataset)
        grid_step: grid width, in the cohort's time unit
        max_steps: safety cap, so a long follow-up with a small step cannot explode
    Returns:
        LongitudinalSurvivalDataset of expanded trajectories
    """
    if grid_step <= 0:
        raise ValueError(f"grid_step must be positive, got {grid_step}")

    out = []
    for p in dataset:
        times = p['times'].detach().cpu().numpy()
        feats = p['features'].detach().cpu().numpy()
        if times.size == 0:
            out.append(p)
            continue

        n = int(min(max(1, math.ceil(float(times[-1]) / grid_step)), max_steps))
        grid = np.arange(1, n + 1, dtype=np.float32) * grid_step

        # Index of the most recent observation at or before each grid point.
        src = np.searchsorted(times, grid, side='right') - 1
        src = np.clip(src, 0, len(times) - 1)

        grid_feats = feats[src]
        # A grid point is a fresh observation when some real time falls in
        # (t - grid_step, t].
        fresh = np.zeros(n, dtype=np.float32)
        hit = np.clip(np.ceil(times / grid_step).astype(int) - 1, 0, n - 1)
        fresh[hit] = 1.0

        dts = np.full(n, grid_step, dtype=np.float32)

        expanded = dict(p)
        expanded['features'] = torch.tensor(grid_feats, dtype=torch.float32)
        expanded['times'] = torch.tensor(grid, dtype=torch.float32)
        expanded['dts'] = torch.tensor(dts, dtype=torch.float32)
        # Interval event flags are derived from residual times downstream, so a
        # zero vector here is correct and avoids re-encoding the event location.
        expanded['events'] = torch.zeros(n, dtype=torch.float32)
        expanded['mask'] = torch.tensor(
            np.repeat(fresh[:, None], feats.shape[1], axis=1), dtype=torch.float32
        )
        out.append(expanded)

    return LongitudinalSurvivalDataset(out)
