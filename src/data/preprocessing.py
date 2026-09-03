"""
Shared cohort preprocessing: splits, standardization, censoring, missingness.

Every loader duplicated this logic and each got a different part of it wrong:

  * `sepsis_loader.py:78` standardized each subject against their OWN trajectory,
    forcing every subject's feature mean to 0 (measured max |mean| = 3e-5). The
    signal that determines the label there is precisely the BETWEEN-subject level
    offset (HR +25, MAP -20, lactate +3, SOFA +5), so per-subject z-scoring erased
    exactly the thing being predicted.
  * `pbc_loader.py:32-38` imputed with whole-dataframe medians and standardized with
    whole-dataframe statistics, both BEFORE the split -- test data informing the
    training transform.
  * No loader produced a validation split, so there was nowhere honest to select a
    hyperparameter or an early-stopping epoch.
  * Every loader set `mask: None`, which left GRU-D's decayed-imputation branch dead
    code and its `x_mean` buffer permanently zero.

Statistics are fit on the TRAIN split only and applied to all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature mean and standard deviation, fit on the training split."""

    mean: np.ndarray
    std: np.ndarray

    def as_tensor_mean(self) -> torch.Tensor:
        return torch.tensor(self.mean, dtype=torch.float32)


def fit_feature_stats(patients: Sequence[dict], eps: float = 1e-6) -> FeatureStats:
    """
    Pool every observation of every training subject and take a COHORT-level mean and
    standard deviation.

    Cohort-level, not per-subject: the whole point is to preserve between-subject
    level differences, since those carry the risk signal.
    """
    if len(patients) == 0:
        raise ValueError("cannot fit feature statistics on an empty split")

    stacked = np.concatenate(
        [p["features"].detach().cpu().numpy() for p in patients], axis=0
    )
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return FeatureStats(mean=mean.astype(np.float64), std=std.astype(np.float64))


def apply_feature_stats(patients: Sequence[dict], stats: FeatureStats) -> list[dict]:
    """Standardize in place-ish (returns new dicts, leaves inputs untouched)."""
    out = []
    for p in patients:
        q = dict(p)
        f = p["features"].detach().cpu().numpy().astype(np.float64)
        q["features"] = torch.tensor((f - stats.mean) / stats.std, dtype=torch.float32)
        out.append(q)
    return out


def subject_level_split(
    n: int,
    seed: int,
    fracs: tuple = (0.6, 0.2, 0.2),
) -> tuple:
    """
    Disjoint train/val/test index lists, split by SUBJECT.

    Returns (train_idx, val_idx, test_idx). The previous loaders produced only
    train/test, which is why no honest model selection was possible.
    """
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError(f"fracs must sum to 1, got {fracs}")
    if n < 3:
        raise ValueError(f"need at least 3 subjects to make three splits, got {n}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    # Guarantee every split is non-empty even for small cohorts.
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))

    return (
        perm[:n_train].tolist(),
        perm[n_train:n_train + n_val].tolist(),
        perm[n_train + n_val:].tolist(),
    )


def empirical_feature_mean(patients: Sequence[dict]) -> torch.Tensor:
    """
    Mean feature vector for GRU-D's decayed imputation.

    `set_empirical_mean` was never called anywhere, so `x_mean` stayed all zeros.
    Note the ordering dependency: because every loader also passed `mask=None`,
    `x_hat = x` identically and `x_mean` was never even read -- fixing the mean
    without fixing the mask changes nothing.
    """
    return torch.tensor(
        np.concatenate([p["features"].detach().cpu().numpy() for p in patients], axis=0).mean(axis=0),
        dtype=torch.float32,
    )


def ensure_mask(patients: Sequence[dict]) -> list[dict]:
    """Replace `mask: None` with an all-observed mask, so the field is never None."""
    out = []
    for p in patients:
        q = dict(p)
        if q.get("mask") is None:
            q["mask"] = torch.ones_like(q["features"])
        out.append(q)
    return out


def administratively_censor(patients: Sequence[dict], horizon: float) -> list[dict]:
    """
    Administrative censoring at a fixed horizon: subjects whose event falls beyond it
    become right-censored there, and observations after it are dropped.

    This is how genuine censoring is introduced into C-MAPSS and the tumor cohort
    without inventing anything -- both had event rate 1.00, i.e. they were not
    survival problems at all.
    """
    out = []
    for p in patients:
        q = dict(p)
        times = p["times"].detach().cpu().numpy()
        keep = times <= horizon
        if keep.sum() < 1:
            keep[0] = True

        q["features"] = p["features"][: int(keep.sum())]
        q["times"] = p["times"][: int(keep.sum())]
        q["dts"] = p["dts"][: int(keep.sum())]
        q["events"] = p["events"][: int(keep.sum())]
        if p.get("mask") is not None:
            q["mask"] = p["mask"][: int(keep.sum())]

        if float(p["tte"]) > horizon:
            q["tte"] = float(horizon)
            q["event"] = 0.0
            q["events"] = torch.zeros_like(q["events"])
        out.append(q)
    return out


def validate_cohort(
    name: str,
    splits: dict,
    *,
    require_censoring: bool = True,
    synthetic: bool = False,
) -> dict:
    """
    Cheap automated guards. Five of the defects found in this codebase would have
    been caught here, which is why this runs at the top of every experiment rather
    than living in a notebook.

    Raises AssertionError on a violation; returns a report dict otherwise.
    """
    report = {"cohort": name, "splits": {}}
    problems = []

    for split_name, patients in splits.items():
        ev = np.array([float(p["event"]) for p in patients])
        tte = np.array([float(p["tte"]) for p in patients])
        lens = np.array([len(p["dts"]) for p in patients])

        # Per-subject feature mean: if this is ~0 the between-subject level signal
        # has been standardized away (the sepsis defect, measured at 3e-5).
        per_subj = np.array([
            np.abs(p["features"].detach().cpu().numpy().mean(axis=0)).max()
            for p in patients
        ])

        n_masked = sum(1 for p in patients if p.get("mask") is None)
        mask_density = float(np.mean([
            float(p["mask"].float().mean()) for p in patients if p.get("mask") is not None
        ])) if n_masked < len(patients) else float("nan")

        info = {
            "n": len(patients),
            "event_rate": round(float(ev.mean()), 4),
            "tte_median": round(float(np.median(tte)), 3),
            "n_distinct_censored_tte": int(len(np.unique(tte[ev == 0]))),
            "n_censored": int((ev == 0).sum()),
            "L_mean": round(float(lens.mean()), 2),
            "max_abs_subject_mean": round(float(per_subj.max()), 6),
            "mask_density": None if np.isnan(mask_density) else round(mask_density, 4),
        }
        report["splits"][split_name] = info

        if n_masked > 0:
            problems.append(f"{name}/{split_name}: {n_masked} subjects have mask=None")

        if require_censoring and not (0.10 <= info["event_rate"] <= 0.90):
            problems.append(
                f"{name}/{split_name}: event rate {info['event_rate']} outside [0.10, 0.90]"
                " -- with no censoring this is not a survival problem"
            )

        if synthetic and info["n_censored"] >= 5 and info["n_distinct_censored_tte"] < 3:
            problems.append(
                f"{name}/{split_name}: only {info['n_distinct_censored_tte']} distinct "
                f"censored tte across {info['n_censored']} censored subjects -- a "
                f"degenerate censoring distribution empties the AUC control set"
            )

        if info["max_abs_subject_mean"] < 0.05:
            problems.append(
                f"{name}/{split_name}: max per-subject feature mean is "
                f"{info['max_abs_subject_mean']:.2e}, i.e. features were standardized "
                f"PER SUBJECT and the between-subject level signal is gone"
            )

    report["problems"] = problems
    if problems:
        raise AssertionError(
            f"cohort validation failed for {name}:\n  - " + "\n  - ".join(problems)
        )
    return report


def schedule_outcome_correlation(patients: Sequence[dict]) -> float:
    """
    corr(number of observations, tte) among subjects with an event.

    Reported rather than asserted. This correlation is INTRINSIC to longitudinal
    survival data -- sicker subjects die sooner and are therefore observed fewer
    times -- and measures 0.843 in the real PBC2 data. It cannot be engineered
    away, and an earlier diagnosis in this effort was wrong to claim that fixing
    the generator would neutralise EXP-04. The load-bearing fix for that is
    landmarking, where every at-risk subject shares one clock and the model sees
    only the observations up to the landmark.

    What the generator fix does address is the far more degenerate coupling
    `tte = times[-1] + U(0.1, 2.0)`, which made tte an almost deterministic
    function of the last observation time.
    """
    ev = np.array([float(p["event"]) for p in patients])
    if (ev == 1).sum() < 3:
        return float("nan")
    n_obs = np.array([len(p["dts"]) for p in patients], dtype=float)[ev == 1]
    tte = np.array([float(p["tte"]) for p in patients], dtype=float)[ev == 1]
    if n_obs.std() < 1e-9 or tte.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(n_obs, tte)[0, 1])
