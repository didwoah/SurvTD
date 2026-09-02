"""
Clinical Bedside Alarm Fatigue & Utility Evaluation Suite (EXP-07):
1. False Alert Episode Rate per patient-day at clinically matched 0.30 PPV.
2. Alert Jitter Count: High-frequency state oscillations crossing the decision threshold in 6h windows.
3. Decision Curve Analysis (DCA): Net Benefit curves across clinical risk thresholds.
"""

import numpy as np


def calibrate_threshold_for_ppv(risk_predictions: list, patient_events: list, target_ppv: float = 0.30) -> float:
    """
    Finds the risk threshold that yields the closest Positive Predictive Value (PPV) to target_ppv.
    """
    all_scores = np.concatenate([p_scores for p_scores in risk_predictions])
    candidate_thresholds = np.quantile(all_scores, np.linspace(0.1, 0.95, 50))

    best_thresh = float(candidate_thresholds[len(candidate_thresholds) // 2])
    best_diff = 1.0

    for th in candidate_thresholds:
        tp = 0
        fp = 0
        for scores, event in zip(risk_predictions, patient_events):
            alert_fired = np.any(scores >= th)
            if alert_fired:
                if event == 1.0:
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
    trajectories_risk: list,
    trajectories_times: list,
    patient_events: list,
    target_ppv: float = 0.30,
    window_hours: float = 6.0
) -> dict:
    """
    Args:
        trajectories_risk: list of 1D numpy arrays containing risk scores over time for each patient
        trajectories_times: list of 1D numpy arrays containing observation timestamps (hours)
        patient_events: list of binary event indicators (1 = event, 0 = censored)
    Returns:
        dict with:
        - 'calibrated_threshold': float
        - 'false_alert_rate_per_day': float
        - 'alert_jitter_count': int
        - 'total_patient_days': float
    """
    threshold = calibrate_threshold_for_ppv(trajectories_risk, patient_events, target_ppv)

    total_false_alert_episodes = 0
    total_jitter_crossings = 0
    total_followup_hours = 0.0

    for risks, times, event in zip(trajectories_risk, trajectories_times, patient_events):
        L = len(risks)
        if L == 0:
            continue

        patient_hours = max(1.0, float(times[-1] - times[0])) if L > 1 else 1.0
        total_followup_hours += patient_hours

        # 1. Alert State Sequence
        alert_states = (risks >= threshold).astype(int)

        # Count false alert episodes (episodes where an alert fired on a censored/non-event patient)
        if event == 0.0:
            # Transitions from 0 -> 1 count as a new alert episode
            episodes = np.sum((alert_states[1:] == 1) & (alert_states[:-1] == 0))
            if alert_states[0] == 1:
                episodes += 1
            total_false_alert_episodes += episodes

        # 2. Alert Jitter Count: oscillations within sliding window
        for i in range(L - 1):
            t_start = times[i]
            # Find indices within window [t_start, t_start + window_hours]
            in_window = (times >= t_start) & (times <= t_start + window_hours)
            sub_states = alert_states[in_window]
            if len(sub_states) > 2:
                # Count threshold crossings (state flips)
                flips = np.sum(sub_states[1:] != sub_states[:-1])
                if flips >= 2:
                    total_jitter_crossings += 1
                    break  # count once per window

    total_patient_days = max(1.0, total_followup_hours / 24.0)
    false_alert_rate_per_day = float(total_false_alert_episodes / total_patient_days)

    return {
        'calibrated_threshold': threshold,
        'false_alert_rate_per_day': false_alert_rate_per_day,
        'alert_jitter_count': int(total_jitter_crossings),
        'total_patient_days': total_patient_days
    }


def compute_decision_curve_analysis(
    trajectories_risk: list,
    patient_events: list,
    thresholds: np.ndarray = np.linspace(0.1, 0.5, 9)
) -> dict:
    """
    Computes Net Benefit across threshold probabilities for Decision Curve Analysis.
    """
    N = len(trajectories_risk)
    max_risks = np.array([np.max(r) if len(r) > 0 else 0.0 for r in trajectories_risk])
    events = np.array(patient_events)

    net_benefits = []
    for pt in thresholds:
        tp = np.sum((max_risks >= pt) & (events == 1))
        fp = np.sum((max_risks >= pt) & (events == 0))
        nb = (tp / N) - (fp / N) * (pt / (1.0 - pt))
        net_benefits.append(float(nb))

    return {
        'thresholds': thresholds.tolist(),
        'net_benefits': net_benefits
    }
