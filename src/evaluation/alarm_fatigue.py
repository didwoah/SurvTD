"""
Clinical Bedside Alarm Fatigue & Utility Evaluation Suite (EXP-07):
1. False Alert Episode Rate per patient-day at clinically matched 0.30 PPV.
2. Alert Jitter Rate: Rate of unstable 6h windows per patient-day and fraction of jittering patients.
3. Decision Curve Analysis (DCA): Net Benefit curves across clinical risk thresholds.

Amended per deviation log A-13 and A-14:
- Threshold is calibrated strictly on validation data and applied out-of-sample to test data.
- Jitter rate counts non-overlapping or sliding window instability normalized per patient-day,
  avoiding the single-break bug that collapsed count to patient fraction.
"""

from __future__ import annotations
import numpy as np


def calibrate_threshold_for_ppv(
    risk_predictions: list[np.ndarray],
    patient_events: list[float],
    target_ppv: float = 0.30
) -> float:
    """
    Finds the risk threshold that yields the closest Positive Predictive Value (PPV) to target_ppv.
    Must be called on VALIDATION split to avoid test leakage.
    """
    if len(risk_predictions) == 0:
        return 0.5

    all_scores = np.concatenate([np.asarray(p_scores, dtype=float) for p_scores in risk_predictions if len(p_scores) > 0])
    if len(all_scores) == 0:
        return 0.5

    candidate_thresholds = np.unique(np.quantile(all_scores, np.linspace(0.05, 0.95, 50)))
    if len(candidate_thresholds) == 0:
        return 0.5

    best_thresh = float(candidate_thresholds[len(candidate_thresholds) // 2])
    best_diff = float("inf")

    for th in candidate_thresholds:
        tp = 0
        fp = 0
        for scores, event in zip(risk_predictions, patient_events):
            scores_arr = np.asarray(scores, dtype=float)
            if len(scores_arr) == 0:
                continue
            alert_fired = bool(np.any(scores_arr >= th))
            if alert_fired:
                if float(event) > 0.5:
                    tp += 1
                else:
                    fp += 1
        if (tp + fp) > 0:
            ppv = tp / (tp + fp)
            diff = abs(ppv - target_ppv)
            if diff < best_diff:
                best_diff = diff
                best_thresh = float(th)

    return best_thresh


def evaluate_alarm_fatigue(
    trajectories_risk: list[np.ndarray],
    trajectories_times: list[np.ndarray],
    patient_events: list[float],
    calibrated_threshold: float = None,
    target_ppv: float = 0.30,
    window_hours: float = 6.0
) -> dict:
    """
    Evaluates alarm fatigue metrics with out-of-sample calibrated threshold.

    Args:
        trajectories_risk: list of 1D numpy arrays containing risk scores over time
        trajectories_times: list of 1D numpy arrays containing observation timestamps (hours)
        patient_events: list of binary event indicators (1 = event, 0 = censored)
        calibrated_threshold: pre-calibrated threshold (from validation split). If None, calibrated here.
        target_ppv: target PPV for calibration if threshold is None
        window_hours: sliding window duration for alert jitter detection

    Returns:
        dict with:
        - 'calibrated_threshold': float
        - 'false_alert_rate_per_day': float
        - 'unstable_window_rate_per_day': float
        - 'jitter_patient_fraction': float
        - 'total_patient_days': float
    """
    if calibrated_threshold is None:
        threshold = calibrate_threshold_for_ppv(trajectories_risk, patient_events, target_ppv)
    else:
        threshold = float(calibrated_threshold)

    total_false_alert_episodes = 0
    total_unstable_windows = 0
    total_jitter_patients = 0
    total_followup_hours = 0.0
    n_valid_patients = 0

    for risks, times, event in zip(trajectories_risk, trajectories_times, patient_events):
        risks_arr = np.asarray(risks, dtype=float)
        times_arr = np.asarray(times, dtype=float)
        L = len(risks_arr)
        if L == 0:
            continue

        n_valid_patients += 1
        patient_hours = max(1.0, float(times_arr[-1] - times_arr[0])) if L > 1 else 1.0
        total_followup_hours += patient_hours

        # 1. Alert State Sequence
        alert_states = (risks_arr >= threshold).astype(int)

        # Count false alert episodes on non-event / censored patients
        if float(event) <= 0.5:
            # Transitions from 0 -> 1 count as a new alert episode
            episodes = int(np.sum((alert_states[1:] == 1) & (alert_states[:-1] == 0)))
            if alert_states[0] == 1:
                episodes += 1
            total_false_alert_episodes += episodes

        # 2. Alert Jitter: count unstable 6h windows without double-counting,
        # plus track whether patient experienced any jitter.
        patient_has_jitter = False
        patient_unstable_windows = 0
        last_counted_t = -float("inf")

        for i in range(L - 1):
            t_start = times_arr[i]
            if t_start < last_counted_t + window_hours:
                continue  # enforce non-overlapping windows to avoid double-counting
            in_window = (times_arr >= t_start) & (times_arr <= t_start + window_hours)
            sub_states = alert_states[in_window]
            if len(sub_states) >= 2:
                flips = int(np.sum(sub_states[1:] != sub_states[:-1]))
                if flips >= 2:
                    patient_unstable_windows += 1
                    patient_has_jitter = True
                    last_counted_t = t_start

        total_unstable_windows += patient_unstable_windows
        if patient_has_jitter:
            total_jitter_patients += 1

    total_patient_days = max(1.0, total_followup_hours / 24.0)
    false_alert_rate_per_day = float(total_false_alert_episodes / total_patient_days)
    unstable_window_rate_per_day = float(total_unstable_windows / total_patient_days)
    jitter_patient_fraction = float(total_jitter_patients / max(1, n_valid_patients))

    return {
        'calibrated_threshold': threshold,
        'false_alert_rate_per_day': false_alert_rate_per_day,
        'unstable_window_rate_per_day': unstable_window_rate_per_day,
        'jitter_patient_fraction': jitter_patient_fraction,
        'total_patient_days': total_patient_days,
        'n_patients': n_valid_patients,
    }


def compute_decision_curve_analysis(
    risk_scores: np.ndarray,
    events: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.1, 0.5, 9)
) -> dict:
    """
    Computes Net Benefit across threshold probabilities for Decision Curve Analysis.
    Evaluated on a specific landmark-horizon risk score rather than leaking max over trajectory.
    """
    risks = np.asarray(risk_scores, dtype=float)
    evs = np.asarray(events, dtype=float)
    N = len(risks)
    if N == 0:
        return {'thresholds': thresholds.tolist(), 'net_benefits': [0.0] * len(thresholds)}

    net_benefits = []
    for pt in thresholds:
        tp = np.sum((risks >= pt) & (evs > 0.5))
        fp = np.sum((risks >= pt) & (evs <= 0.5))
        nb = (tp / N) - (fp / N) * (pt / (1.0 - pt))
        net_benefits.append(float(nb))

    return {
        'thresholds': thresholds.tolist(),
        'net_benefits': net_benefits
    }
