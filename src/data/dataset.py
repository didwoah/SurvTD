"""
Unified Time-Series Survival Dataset and Collate Functions:
Supports irregular observation intervals, variable sequence lengths, and censoring indicators.
"""

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
