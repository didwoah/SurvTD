"""
CoxSig-style Dynamic Evaluation Protocol.

Directly adapted from CoxSig (Bleistein et al., 2023, baselines/signature_survival/src/utils.py & coxsig.py)
to provide 100% parity with published benchmark results while ensuring leak-free preprocessing.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Union, Sequence, Optional
import numpy as np
from sksurv.metrics import brier_score, cumulative_dynamic_auc


def convert_surv_label_structarray(surv_label: np.ndarray) -> np.recarray:
    """Convert (time, event) array to sksurv structured array."""
    n_samples = surv_label.shape[0]
    struct_list = []
    for i in range(n_samples):
        struct_list.append((bool(surv_label[i, 1]), float(surv_label[i, 0])))

    return np.rec.array(
        struct_list,
        dtype=[('indicator', bool), ('time', np.float32)]
    )


def compute_coxsig_cindex(
    surv_time: np.ndarray,
    surv_ind: np.ndarray,
    estimates: np.ndarray,
    eval_times: np.ndarray
) -> np.ndarray:
    """
    Compute time-dependent C-index exactly as implemented in CoxSig utils.py.
    
    Parameters
    ----------
    surv_time : shape (n_samples,)
        Observed residual survival times (T - pred_time).
    surv_ind : shape (n_samples,)
        Event indicators (1 for failure, 0 for censored).
    estimates : shape (n_samples, n_eval_times)
        Predicted conditional survival probabilities at each eval_time.
    eval_times : shape (n_eval_times,)
        Residual time horizons evaluated.
    """
    n_samples = estimates.shape[0]
    n_eval_time = len(eval_times)
    results = np.zeros(n_eval_time)

    for k in range(n_eval_time):
        eval_time = eval_times[k]
        A = np.zeros((n_samples, n_samples))
        Q = np.zeros((n_samples, n_samples))
        N_t = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            A[i, np.where(surv_time[i] < surv_time)] = 1
            # Lower survival estimate means higher risk
            Q[i, np.where(-estimates[i, k] > -estimates[:, k])] = 1

            if (surv_time[i] <= eval_time) and (surv_ind[i] == 1):
                N_t[i, :] = 1

        Num = np.sum(((A) * N_t) * Q)
        Den = np.sum((A) * N_t)

        if Num == 0 and Den == 0:
            results[k] = np.nan
        else:
            results[k] = float(Num / Den)

    return results


def compute_coxsig_brier(
    surv_time: np.ndarray,
    surv_ind: np.ndarray,
    estimates: np.ndarray,
    eval_times: np.ndarray
) -> np.ndarray:
    """Compute Brier Score as implemented in CoxSig utils.py."""
    n_eval_time = len(eval_times)
    results = np.zeros(n_eval_time)
    for k in range(n_eval_time):
        eval_time = eval_times[k]
        at_risk = surv_time > eval_time
        results[k] = np.mean(
            at_risk * (1.0 - estimates[:, k]) ** 2 +
            ~at_risk * surv_ind * (estimates[:, k] ** 2)
        )
    return results


def setup_coxsig_horizons(
    surv_labels: np.ndarray,
    quantile_pred_times: Sequence[float] = (0.1, 0.2, 0.4),
    n_eval_times: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct pred_times and eval_times matching CoxSig's NASA setup.
    """
    events_mask = surv_labels[:, 1] == 1
    tte = surv_labels[events_mask, 0] if np.any(events_mask) else surv_labels[:, 0]
    
    pred_times = np.quantile(tte, quantile_pred_times)
    eval_times = []
    for k in range(n_eval_times):
        delta = max(np.quantile(tte, np.array(quantile_pred_times) + (k + 1) * 0.05) - pred_times)
        eval_times.append(delta)
    eval_times = np.array(eval_times)
    return pred_times, eval_times


def evaluate_coxsig_at_landmarks(
    surv_labels_test: np.ndarray,
    pred_times: np.ndarray,
    eval_times: np.ndarray,
    predict_cond_surv_fn
) -> Dict[str, np.ndarray]:
    """
    Unified evaluator executing CoxSig's landmark evaluation loop across all pred_times and eval_times.
    
    Parameters
    ----------
    surv_labels_test : np.ndarray, shape (n_test, 2)
        (time, event) for test samples.
    pred_times : np.ndarray, shape (n_pred_times,)
        Prediction landmark times.
    eval_times : np.ndarray, shape (n_eval_times,)
        Residual evaluation window lengths.
    predict_cond_surv_fn : callable(pred_time, eval_times, alive_idx) -> np.ndarray of shape (n_alive, n_eval_times)
        Function predicting S(pred_time + eval_time | history up to pred_time) for alive test samples.
        
    Returns
    -------
    dict with keys 'c_index', 'bs', each having shape (n_pred_times, n_eval_times)
    """
    n_pred_times = len(pred_times)
    n_eval_times = len(eval_times)
    
    cindex_matrix = np.zeros((n_pred_times, n_eval_times))
    bs_matrix = np.zeros((n_pred_times, n_eval_times))
    
    surv_times = surv_labels_test[:, 0]
    surv_inds = surv_labels_test[:, 1]
    
    for j, pred_time in enumerate(pred_times):
        # Filter individuals still alive at prediction time
        alive_idx = np.where(surv_times >= pred_time)[0]
        if len(alive_idx) == 0:
            cindex_matrix[j, :] = np.nan
            bs_matrix[j, :] = np.nan
            continue
            
        # Residual times on relative clock
        res_times = surv_times[alive_idx] - pred_time
        res_inds = surv_inds[alive_idx]
        
        # Query model for conditional survival estimates at eval_times
        cond_surv_preds = predict_cond_surv_fn(pred_time, eval_times, alive_idx)
        assert cond_surv_preds.shape == (len(alive_idx), n_eval_times), \
            f"Expected shape ({len(alive_idx)}, {n_eval_times}), got {cond_surv_preds.shape}"
            
        cindex_matrix[j, :] = compute_coxsig_cindex(res_times, res_inds, cond_surv_preds, eval_times)
        bs_matrix[j, :] = compute_coxsig_brier(res_times, res_inds, cond_surv_preds, eval_times)
        
    return {
        "c_index": cindex_matrix,
        "bs": bs_matrix,
        "pred_times": pred_times,
        "eval_times": eval_times
    }
