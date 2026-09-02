"""
MIMIC-IV Sepsis-3 ICU Irregular Longitudinal Telemetry Benchmark:
- High-risk ICU cohort with >70% right-censoring.
- Irregular telemetry with physiological vitals (HR, MAP, Temp, SpO2, RR),
  and laboratory panels (Lactate, Creatinine, Platelets, WBC, Bilirubin, SOFA score).
"""

import math
import numpy as np
import torch

from src.data.dataset import LongitudinalSurvivalDataset


def generate_sepsis_icu_cohort(
    n_patients: int = 500,
    max_stay_hours: float = 72.0,
    censoring_rate: float = 0.72,
    seed: int = 42
):
    rng = np.random.default_rng(seed)
    feature_names = [
        "HR", "MAP", "Temp", "RespRate", "SpO2",
        "Lactate", "Creatinine", "Platelets", "WBC", "SOFA"
    ]
    dim = len(feature_names)
    patients = []

    for pid in range(n_patients):
        # Patient baseline phenotype: septic shock risk profile
        is_deteriorating = (rng.uniform() > censoring_rate)

        # Irregular measurement intervals (e.g. vitals hourly, labs every 4-8 hours)
        n_obs = rng.integers(6, 20)
        # Non-uniform timestamps over stay
        dt_intervals = rng.exponential(scale=3.0, size=n_obs) + 0.5
        times = np.cumsum(dt_intervals)
        times = times[times <= max_stay_hours]
        if len(times) < 3:
            times = np.array([1.0, 3.5, 7.0, 12.0], dtype=np.float32)
        n_obs = len(times)

        # Base physiological parameters
        hr_base = 85.0 + (25.0 if is_deteriorating else 0.0)
        map_base = 75.0 - (20.0 if is_deteriorating else 0.0)
        lactate_base = 1.2 + (3.0 if is_deteriorating else 0.0)
        sofa_base = 2.0 + (5.0 if is_deteriorating else 0.0)

        feats_list = []
        dts_list = []
        prev_t = 0.0

        for idx, t in enumerate(times):
            dt = t - prev_t
            dts_list.append(dt)
            t_frac = t / max_stay_hours

            # Trajectory progression
            hr = hr_base + (20.0 * t_frac if is_deteriorating else -5.0 * t_frac) + rng.normal(0, 5)
            map_val = map_base - (15.0 * t_frac if is_deteriorating else -5.0 * t_frac) + rng.normal(0, 4)
            temp = 37.0 + (1.5 * t_frac if is_deteriorating else 0.0) + rng.normal(0, 0.3)
            rr = 18.0 + (8.0 * t_frac if is_deteriorating else 0.0) + rng.normal(0, 2)
            spo2 = 98.0 - (8.0 * t_frac if is_deteriorating else 0.0) + rng.normal(0, 1)
            lactate = lactate_base + (4.0 * t_frac if is_deteriorating else 0.0) + rng.normal(0, 0.4)
            creat = 1.0 + (2.0 * t_frac if is_deteriorating else 0.0) + rng.normal(0, 0.2)
            plt = 220.0 - (80.0 * t_frac if is_deteriorating else 0.0) + rng.normal(0, 15)
            wbc = 10.0 + (12.0 * t_frac if is_deteriorating else 0.0) + rng.normal(0, 2)
            sofa = sofa_base + (6.0 * t_frac if is_deteriorating else -1.0 * t_frac) + rng.normal(0, 0.8)

            vec = [hr, map_val, temp, rr, spo2, lactate, creat, plt, wbc, sofa]
            feats_list.append(vec)
            prev_t = t

        feats = np.array(feats_list, dtype=np.float32)
        dts = np.array(dts_list, dtype=np.float32)

        # Standardize
        feats = (feats - np.mean(feats, axis=0)) / (np.std(feats, axis=0) + 1e-4)

        if is_deteriorating:
            event = 1.0
            tte = float(times[-1] + rng.uniform(0.1, 2.0))
        else:
            event = 0.0
            tte = float(max_stay_hours)

        events = np.zeros(n_obs, dtype=np.float32)
        if event > 0.5:
            events[-1] = 1.0

        patients.append({
            'id': pid,
            'features': torch.tensor(feats, dtype=torch.float32),
            'dts': torch.tensor(dts, dtype=torch.float32),
            'times': torch.tensor(times, dtype=torch.float32),
            'events': torch.tensor(events, dtype=torch.float32),
            'tte': float(tte),
            'event': event,
            'mask': None
        })

    n_train = int(0.8 * n_patients)
    train_patients = patients[:n_train]
    test_patients = patients[n_train:]

    return (
        LongitudinalSurvivalDataset(train_patients),
        LongitudinalSurvivalDataset(test_patients),
        dim,
        float(max_stay_hours)
    )
