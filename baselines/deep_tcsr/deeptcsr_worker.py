import os
import sys
import json
import argparse
import numpy as np

if not hasattr(np.dtypes, 'StringDType'):
    class StringDType:
        def __init__(self, *a, **k): pass
    np.dtypes.StringDType = StringDType

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import torch
from lifelines.utils import concordance_index

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)
# Add signature_survival for identical score function
sys.path.insert(0, os.path.abspath(os.path.join(CUR_DIR, "../signature_survival")))
from src.utils import score

from networks import TCN
from deep_lambda_cox import bce_logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    # Emit conditional survival curves instead of this repo's own metrics, so every arm
    # in the benchmark is scored by one scorer. The metric path is kept behind the flag
    # so the existing NASA parity results stay reproducible.
    parser.add_argument("--emit_curves", action="store_true")
    args = parser.parse_args()

    data = torch.load(args.input_pt)
    paths_train = data['paths_train'].numpy()       # (N_train, T, 1+D)
    surv_labels_train = data['surv_labels_train']   # (N_train, 2) [tte, event]
    paths_test = data['paths_test'].numpy()         # (N_test, T, 1+D)
    surv_labels_test = data['surv_labels_test']     # (N_test, 2)
    pred_times = data['pred_times']
    eval_times = data['eval_times']

    # Feats: (N, D, T) for TCN
    times_train = paths_train[:, :, 0]
    feat_train = np.transpose(paths_train[:, :, 1:], (0, 2, 1)) # (N, D, T)
    times_test = paths_test[:, :, 0]
    feat_test = np.transpose(paths_test[:, :, 1:], (0, 2, 1))   # (N, D, T)

    N_train, D_feat, T_len = feat_train.shape
    num_bins = 40
    max_tte = max(np.max(surv_labels_train[:, 0]), np.max(surv_labels_test[:, 0])) * 1.05
    bin_edges = np.linspace(0, max_tte, num_bins + 1)
    delta_s = bin_edges[1] - bin_edges[0]

    # Setup TCN model from DeepTCSR
    def forward_fn(x, is_training=True):
        tcn = TCN(
            num_inputs=D_feat,
            num_channels=[32, 32],
            kernel_size=2,
            dilation_factor=2,
            dropout=0.1 if is_training else 0.0,
            seed=42
        )
        h = tcn(x) # (B, 32, T)
        h_perm = jnp.transpose(h, (0, 2, 1)) # (B, T, 32)
        logits = hk.Linear(num_bins)(h_perm)  # (B, T, num_bins)
        return logits

    init_fn, apply_fn = hk.transform(forward_fn)
    rng = jax.random.PRNGKey(42)
    dummy_x = jnp.zeros((1, D_feat, T_len))
    params = init_fn(rng, dummy_x)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    # Targets: for each visit t_l, target hazard is 1 at event bin, 0 before
    t_train_bins = np.clip(np.digitize(surv_labels_train[:, 0], bin_edges) - 1, 0, num_bins - 1)
    e_train = surv_labels_train[:, 1].astype(int)

    targets_train = np.zeros((N_train, T_len, num_bins), dtype=np.float32)
    masks_train = np.zeros((N_train, T_len), dtype=np.float32)
    for i in range(N_train):
        tte_i = surv_labels_train[i, 0]
        valid_steps = times_train[i] <= tte_i
        masks_train[i, valid_steps] = 1.0
        for l in np.where(valid_steps)[0]:
            tau = times_train[i, l]
            res_bin = np.clip(int(round((tte_i - tau) / delta_s)), 0, num_bins - 1)
            if e_train[i] == 1:
                targets_train[i, l, res_bin] = 1.0

    @jax.jit
    def loss_fn(p, x, y_tgt, m, rng_step):
        logits = apply_fn(p, rng_step, x, is_training=True)
        loss_elem = bce_logits(y_tgt, logits) # (B, T, num_bins)
        loss_masked = jnp.sum(loss_elem, axis=-1) * m # (B, T)
        return jnp.sum(loss_masked) / jnp.maximum(jnp.sum(m), 1.0)

    @jax.jit
    def train_step(p, opt_st, x, y_tgt, m, rng_step):
        loss, grads = jax.value_and_grad(loss_fn)(p, x, y_tgt, m, rng_step)
        updates, opt_st = optimizer.update(grads, opt_st, p)
        p = optax.apply_updates(p, updates)
        return p, opt_st, loss

    x_jnp = jnp.array(feat_train)
    y_jnp = jnp.array(targets_train)
    m_jnp = jnp.array(masks_train)

    for ep in range(args.epochs):
        perm = np.random.permutation(N_train)
        for b in range(0, N_train, args.batch_size):
            idx = perm[b:b + args.batch_size]
            rng, subkey = jax.random.split(rng)
            params, opt_state, _ = train_step(
                params, opt_state,
                x_jnp[idx], y_jnp[idx], m_jnp[idx],
                subkey
            )

    # Evaluation
    N_test = paths_test.shape[0]
    n_pts = len(pred_times)
    n_evs = len(eval_times)
    cindex_matrix = np.zeros((n_pts, n_evs))
    bs_matrix = np.zeros((n_pts, n_evs))
    # (N, n_pred, n_eval) for EVERY test subject: the at-risk filter belongs to
    # `curve_scoring.predictions_from_curves`, so that all arms share one risk set by
    # construction rather than by five workers agreeing on `idx_sel`.
    all_curves = np.zeros((N_test, n_pts, n_evs))

    test_x_jnp = jnp.array(feat_test)
    test_logits = np.array(apply_fn(params, rng, test_x_jnp, is_training=False)) # (N_test, T, num_bins)
    test_hazards = 1.0 / (1.0 + np.exp(-test_logits)) # sigmoid
    test_surv_curves = np.cumprod(1.0 - np.clip(test_hazards, 0.0, 0.99), axis=-1) # (N_test, T, num_bins)

    for j in range(n_pts):
        pt = pred_times[j]
        cond_surv = np.zeros((N_test, n_evs))
        for i in range(N_test):
            obs_indices = np.where(times_test[i] <= pt)[0]
            last_obs = obs_indices[-1] if len(obs_indices) > 0 else 0
            surv_i = test_surv_curves[i, last_obs]
            for k in range(n_evs):
                dt = eval_times[k]
                h_bin = min(max(int(round(dt / delta_s)), 0), num_bins - 1)
                cond_surv[i, k] = surv_i[h_bin]

        all_curves[:, j, :] = cond_surv
        if args.emit_curves:
            continue

        surv_times_test = surv_labels_test[:, 0]
        surv_inds_test = surv_labels_test[:, 1]
        idx_sel = surv_times_test >= pt
        surv_times_ = surv_times_test[idx_sel] - pt
        surv_inds_ = surv_inds_test[idx_sel]
        surv_labels_ = np.array([surv_times_, surv_inds_]).T
        surv_preds_ = cond_surv[idx_sel]

        bs_matrix[j] = score("bs", surv_labels_, surv_labels_, surv_preds_, eval_times)
        cindex_matrix[j] = score("c_index", surv_labels_, surv_labels_, surv_preds_, eval_times)

    if args.emit_curves:
        with open(args.output_json, "w") as f:
            json.dump({
                "model": "deeptcsr",
                "surv_curves": all_curves.tolist(),
                "pred_times": np.asarray(pred_times).tolist(),
                "eval_times": np.asarray(eval_times).tolist(),
                "surv_labels_test": np.asarray(surv_labels_test).tolist(),
            }, f)
        return

    # Static t=0
    try:
        exp_times = np.sum(test_surv_curves[:, 0, :], axis=-1) * delta_s
        t0_cindex = float(concordance_index(surv_labels_test[:, 0], exp_times, surv_labels_test[:, 1]))
    except Exception:
        t0_cindex = 0.5

    out = {
        "dynamic_cindex": cindex_matrix.tolist(),
        "dynamic_bs": bs_matrix.tolist(),
        "t0_cindex": t0_cindex,
        "t0_bs": float(np.nanmean(bs_matrix[0]))
    }

    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
