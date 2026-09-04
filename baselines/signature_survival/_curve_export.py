"""
Shared curve export for the CoxSig and NCDE workers.

Project glue, not upstream code. Both models expose
`predict_survival(path, pred_times)`, which returns a **list** with one entry per
prediction time, each `(n_samples, n_remaining_grid)` and already conditioned on
survival to that landmark (`coxsig.py:232` divides by `surv_pred_at_t`). What it does
not do is put those curves on a residual-time grid, which is what the benchmark's
single scorer needs.

The absolute-time axis is recovered from the array length rather than by repeating
upstream's index arithmetic. Upstream slices `surv_preds[:, j, t_pred_id + 1:]`, which
always runs to the end of the axis, so a returned curve of width `M` covers the last
`M` points of `sampling_times[1:]` -- i.e. `sampling_times[-M:]`. Deriving it this way
is immune to an off-by-one in `t_pred_id`, which would otherwise shift every curve by
one grid step and be completely silent in the metrics.
"""

import numpy as np


def curves_on_residual_grid(predict_survival_fn, paths, sampling_times,
                            pred_times, eval_times):
    """(N, n_pred, n_eval) conditional survival on the residual-time grid.

    Args:
        predict_survival_fn: the model's own `predict_survival(path, pred_times)`
        sampling_times: the shared absolute clock the bundle was built on
        eval_times: residual horizons, measured from each landmark
    """
    sampling_times = np.asarray(sampling_times, dtype=float)
    pred_times = np.asarray(pred_times, dtype=float)
    eval_times = np.asarray(eval_times, dtype=float)

    per_landmark = predict_survival_fn(paths, pred_times)
    n = np.asarray(per_landmark[0]).shape[0]
    out = np.zeros((n, len(pred_times), len(eval_times)), dtype=float)

    for j, pred_time in enumerate(pred_times):
        curve = np.asarray(per_landmark[j], dtype=float)          # (N, M)
        m = curve.shape[1]
        abs_times = sampling_times[1:][-m:] if m else sampling_times[-1:]
        query = pred_time + eval_times
        for i in range(n):
            # flat-left at 1.0 (nothing before the landmark), flat-right at the last
            # known value, matching src/evaluation/landmark.py::interp_survival
            out[i, j] = np.interp(query, abs_times, curve[i],
                                  left=1.0, right=float(curve[i][-1]))

    return np.clip(np.minimum.accumulate(out, axis=-1), 0.0, 1.0)
