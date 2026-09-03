"""
Statistical Significance and Bootstrap Uncertainty Utilities:
- 1000-sample test bootstrap 95% confidence intervals.
- Paired Wilcoxon signed-rank test across evaluation seeds.
"""

import numpy as np
from scipy import stats


def compute_bootstrap_ci(values: list or np.ndarray, n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple:
    """
    Computes empirical bootstrap confidence interval.
    Returns:
        (mean, ci_lower, ci_upper, standard_error)
    """
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = []

    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1.0 - alpha))
    mean = float(np.mean(arr))
    se = float(np.std(boot_means))

    return mean, lower, upper, se


def paired_wilcoxon_test(scores_a: list, scores_b: list) -> tuple:
    """
    Paired Wilcoxon signed-rank test between two methods across seeds or paired test samples.
    """
    diffs = np.array(scores_a) - np.array(scores_b)
    if np.all(diffs == 0):
        return 0.0, 1.0
    stat, p_val = stats.wilcoxon(scores_a, scores_b)
    return float(stat), float(p_val)


def compute_paired_bootstrap_ci(scores_a: list or np.ndarray, scores_b: list or np.ndarray, n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple:
    """
    Computes empirical bootstrap confidence interval for paired differences (scores_a - scores_b).
    Mandated by deviation log A-08 for adjudicating meaningful deltas.
    Returns:
        (mean_diff, ci_lower, ci_upper, standard_error)
    """
    diffs = np.array(scores_a, dtype=np.float64) - np.array(scores_b, dtype=np.float64)
    return compute_bootstrap_ci(diffs, n_bootstraps=n_bootstraps, ci=ci, seed=seed)
