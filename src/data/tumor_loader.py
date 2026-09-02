"""
Synthetic Biophysical ODE Tumor Growth Generator:
- Non-linear Gompertzian tumor proliferation with stochastic perturbations.
- Irregular observation timestamps and threshold-crossing survival events.
"""

import math
import numpy as np
import torch

from src.data.dataset import LongitudinalSurvivalDataset


def generate_tumor_growth_cohort(
    n_samples: int = 400,
    end_time: float = 10.0,
    lethal_threshold: float = 2.0,
    seed: int = 42
):
    rng = np.random.default_rng(seed)
    patients = []

    for i in range(n_samples):
        # Sample irregular observation visit count and times
        n_visits = rng.integers(4, 12)
        visit_times = np.sort(rng.uniform(0.1, end_time, size=n_visits))
        visit_times = np.unique(np.round(visit_times, 2))
        n_visits = len(visit_times)

        # Gompertzian dynamics: dy/dt = r * y * log(K_cap / y) + noise
        r = rng.normal(0.5, 0.1)
        K_cap = 3.5
        y = rng.uniform(0.3, 0.8)

        y_trajectory = []
        dts = []
        prev_t = 0.0
        event_time = None

        for t in visit_times:
            dt = max(0.1, t - prev_t)
            dts.append(dt)
            # ODE integration step
            growth = r * y * max(0.01, math.log(max(1.01, K_cap / max(0.01, y)))) * dt
            noise = rng.normal(0, 0.05 * math.sqrt(dt))
            y = max(0.05, y + growth + noise)
            y_trajectory.append(y)

            if y >= lethal_threshold and event_time is None:
                event_time = float(t)
            prev_t = t

        y_arr = np.array(y_trajectory, dtype=np.float32)
        dts_arr = np.array(dts, dtype=np.float32)

        # Feature matrix: [y, dy/dt_approx, sin(t), cos(t)]
        dy = np.diff(y_arr, prepend=y_arr[0]) / np.maximum(dts_arr, 0.05)
        feats = np.stack([y_arr, dy, np.sin(visit_times), np.cos(visit_times)], axis=1)

        has_event = 1.0 if event_time is not None else 0.0
        tte = event_time if event_time is not None else float(end_time)

        # Interval event indicator
        events = np.zeros(n_visits, dtype=np.float32)
        if has_event > 0.5:
            # Mark the interval where lethal threshold was reached
            for k_idx, t_k in enumerate(visit_times):
                if t_k >= tte:
                    events[k_idx] = 1.0
                    break

        patients.append({
            'id': i,
            'features': torch.tensor(feats, dtype=torch.float32),
            'dts': torch.tensor(dts_arr, dtype=torch.float32),
            'times': torch.tensor(visit_times, dtype=torch.float32),
            'events': torch.tensor(events, dtype=torch.float32),
            'tte': float(tte),
            'event': has_event,
            'mask': None
        })

    n_train = int(0.8 * n_samples)
    train_patients = patients[:n_train]
    test_patients = patients[n_train:]

    return (
        LongitudinalSurvivalDataset(train_patients),
        LongitudinalSurvivalDataset(test_patients),
        4,
        float(end_time)
    )
