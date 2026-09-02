"""
Standard Dynamic Survival Analysis Metrics:
- Time-Dependent Concordance Index (C^td)
- Cumulative / Dynamic Time-Dependent AUC at target horizons
- Integrated Brier Score (IBS) with IPCW weighting
"""

import math
import numpy as np
import torch
from lifelines import KaplanMeierFitter


def compute_concordance_td(risk_scores: np.ndarray, event_times: np.ndarray, event_indicators: np.ndarray) -> float:
    """
    Computes Harrell's C-index between dynamic risk scores and event times.
    Higher risk score -> shorter survival time.
    """
    n = len(risk_scores)
    concordant = 0
    permissible = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Check permissibility
            if event_times[i] < event_times[j] and event_indicators[i] == 1:
                permissible += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    tied += 1
            elif event_times[j] < event_times[i] and event_indicators[j] == 1:
                permissible += 1
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] == risk_scores[i]:
                    tied += 1

    if permissible == 0:
        return 0.5
    return float((concordant + 0.5 * tied) / permissible)


def compute_time_dependent_auc(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    eval_horizon: float
) -> float:
    """
    Cumulative/Dynamic AUC evaluated at eval_horizon.
    Cases: experienced event before or at eval_horizon (event_times <= eval_horizon, event=1)
    Controls: survived past eval_horizon (event_times > eval_horizon)
    """
    cases = (event_times <= eval_horizon) & (event_indicators == 1)
    controls = (event_times > eval_horizon)

    n_cases = np.sum(cases)
    n_controls = np.sum(controls)

    if n_cases == 0 or n_controls == 0:
        return 0.5

    case_scores = risk_scores[cases]
    ctrl_scores = risk_scores[controls]

    # Mann-Whitney U statistic
    concordant = 0
    tied = 0
    for cs in case_scores:
        concordant += np.sum(cs > ctrl_scores)
        tied += np.sum(cs == ctrl_scores)

    auc = (concordant + 0.5 * tied) / (n_cases * n_controls)
    return float(auc)


def compute_integrated_brier_score(
    predicted_survival_curves: list,
    eval_times: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    delta_s: float
) -> float:
    """
    Integrated Brier Score (IBS) with Kaplan-Meier censoring IPCW estimator.
    """
    n = len(event_times)
    # Fit censoring distribution G(t) using Kaplan-Meier on reversed indicator
    kmf = KaplanMeierFitter()
    kmf.fit(event_times, event_observed=(1 - event_indicators))

    brier_scores = []

    for t_eval in eval_times:
        bs_t = 0.0
        n_valid = 0

        # Censoring weight G(t_eval)
        g_t = max(kmf.predict(t_eval), 0.01)

        for i in range(n):
            t_i = event_times[i]
            e_i = event_indicators[i]

            # Index on grid for predicted survival probability S_hat_i(t_eval)
            s_curve = predicted_survival_curves[i]
            k_bin = min(int(math.floor(t_eval / delta_s)), len(s_curve) - 1)
            s_pred = float(s_curve[k_bin])

            if t_i <= t_eval and e_i == 1:
                # Event before t_eval: outcome Y(t) = 0
                g_ti = max(kmf.predict(t_i), 0.01)
                weight = 1.0 / g_ti
                bs_t += weight * (s_pred ** 2)
                n_valid += 1
            elif t_i > t_eval:
                # Survived past t_eval: outcome Y(t) = 1
                weight = 1.0 / g_t
                bs_t += weight * ((1.0 - s_pred) ** 2)
                n_valid += 1

        if n_valid > 0:
            brier_scores.append(bs_t / n_valid)

    if len(brier_scores) == 0:
        return 0.25
    return float(np.mean(brier_scores))
