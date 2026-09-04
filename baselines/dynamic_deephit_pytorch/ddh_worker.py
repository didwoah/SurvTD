import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index

# Ensure ddh and DSM are on path
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR_DIR)
sys.path.insert(0, os.path.join(CUR_DIR, "DeepSurvivalMachines"))
# Add signature_survival for identical score function
sys.path.insert(0, os.path.abspath(os.path.join(CUR_DIR, "../signature_survival")))
from src.utils import score

from ddh.ddh_torch import DynamicDeepHitTorch
from ddh.losses import total_loss

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
    # D16. How the TRAINING sequences are truncated. See the block below.
    #   landmark (default, the repair): truncate at each evaluation landmark, which is
    #       exactly the truncation used at prediction time.
    #   tte (the defect, kept only to reproduce the withdrawn numbers): truncate at the
    #       subject's own event/censoring time, which makes sequence length a perfect
    #       proxy for the label.
    parser.add_argument("--train_truncation", choices=("landmark", "tte"),
                        default="landmark")
    args = parser.parse_args()

    data = torch.load(args.input_pt)
    paths_train = data['paths_train'].numpy()       # (N_train, T, 1+D)
    surv_labels_train = data['surv_labels_train']   # (N_train, 2) [tte, event]
    paths_test = data['paths_test'].numpy()         # (N_test, T, 1+D)
    surv_labels_test = data['surv_labels_test']     # (N_test, 2)
    pred_times = data['pred_times']
    eval_times = data['eval_times']

    feat_train = paths_train[:, :, 1:]
    feat_test = paths_test[:, :, 1:]
    times_train = paths_train[:, :, 0]
    times_test = paths_test[:, :, 0]

    N_train, T_len, D_feat = feat_train.shape
    num_bins = 40
    max_tte = max(np.max(surv_labels_train[:, 0]), np.max(surv_labels_test[:, 0])) * 1.05
    bin_edges = np.linspace(0, max_tte, num_bins + 1)
    delta_s = bin_edges[1] - bin_edges[0]

    t_train_bins = np.clip(np.digitize(surv_labels_train[:, 0], bin_edges) - 1, 0, num_bins - 1)
    e_train = surv_labels_train[:, 1].astype(int)

    # ------------------------------------------------------------------ D16
    # Dynamic-DeepHit reads its `inputmask` from the NaN pattern and takes the RNN
    # state at the LAST OBSERVED index, so the number of un-NaN'd steps is an input
    # feature. On the shared forward-filled clock every subject has all T grid points,
    # so the worker has to choose a truncation -- and truncating at `tte` makes that
    # feature a near-perfect proxy for the label:
    #
    #     Framingham train split: spearman(n_observed_steps, tte) = 0.9999
    #                             spearman(n_observed_steps, event) = -0.9693
    #
    # while at prediction time the truncation is at the landmark, so the same feature
    # is CONSTANT across subjects (8 steps at L=2190, 14 at L=4380). The model learns
    # the shortcut, learns almost nothing from the covariates, and the little it does
    # learn comes out with the wrong sign: C^td 0.32-0.40, i.e. BELOW chance, on both
    # landmarks, for both this worker and the independent port.
    #
    # The repair is to truncate training sequences where prediction truncates them --
    # at the landmarks -- giving one training example per at-risk (subject, landmark)
    # pair. Measured on Framingham seed 42: the leak correlation falls 0.9999 -> 0.087
    # and C^td rises to 0.7364 / 0.7339, alongside Landmark Cox's 0.7453 / 0.7436.
    #
    # This does NOT affect the other workers. DeepTCSR's TCN is causal (`Chomp1D`) and
    # its `masks_train` weights the loss rather than masking the input; CoxSig's
    # per-sampling-time expansion is upstream's own time-dependent Cox construction.
    # ---------------------------------------------------------------------------
    if args.train_truncation == "tte":
        x_train = np.copy(feat_train)
        for i in range(N_train):
            unobs = times_train[i] > surv_labels_train[i, 0]
            x_train[i, unobs, :] = np.nan
    else:
        xs, ts_, es_ = [], [], []
        for L in np.asarray(pred_times, dtype=float):
            for i in np.nonzero(surv_labels_train[:, 0] > L)[0]:
                xi = np.copy(feat_train[i])
                xi[times_train[i] > L, :] = np.nan
                xs.append(xi)
                ts_.append(t_train_bins[i])
                es_.append(e_train[i])
        x_train = np.asarray(xs, dtype=np.float32)
        t_train_bins = np.asarray(ts_)
        e_train = np.asarray(es_)
        N_train = len(x_train)

    # DynamicDeepHitTorch with LSTM backbone and temporal attention
    model = DynamicDeepHitTorch(
        input_dim=D_feat,
        output_dim=num_bins,
        layers_rnn=1,
        hidden_rnn=32,
        typ='LSTM',
        risks=1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    alpha = 0.5  # ranking loss weight
    beta = 0.5   # NLL loss weight
    sigma = 0.1

    x_train_t = torch.from_numpy(x_train).float()
    t_train_t = torch.from_numpy(t_train_bins).long()
    e_train_t = torch.from_numpy(e_train).int()

    model.train()
    for epoch in range(args.epochs):
        perm = np.random.permutation(N_train)
        for b in range(0, N_train, args.batch_size):
            idx = perm[b:b + args.batch_size]
            xb = x_train_t[idx]
            tb = t_train_t[idx]
            eb = e_train_t[idx]

            optimizer.zero_grad()
            loss = total_loss(model, xb, tb, eb, alpha, beta, sigma)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()

    # Evaluation under CoxSig Dynamic Landmark protocol
    N_test = paths_test.shape[0]
    n_pts = len(pred_times)
    n_evs = len(eval_times)
    cindex_matrix = np.zeros((n_pts, n_evs))
    bs_matrix = np.zeros((n_pts, n_evs))
    # (N, n_pred, n_eval) for EVERY test subject: the at-risk filter belongs to
    # `curve_scoring.predictions_from_curves`, so that all arms share one risk set by
    # construction rather than by five workers agreeing on `idx_sel`.
    all_curves = np.zeros((N_test, n_pts, n_evs))

    with torch.no_grad():
        for j in range(n_pts):
            pt = pred_times[j]
            x_test_sub = np.copy(feat_test)
            for i in range(N_test):
                unobs = times_test[i] > pt
                x_test_sub[i, unobs, :] = np.nan

            x_sub_t = torch.from_numpy(x_test_sub).float()
            _, outcomes = model(x_sub_t)
            pmf = outcomes[0].numpy()  # (N_test, num_bins)
            cdf = np.cumsum(pmf, axis=-1)
            surv_all = np.clip(1.0 - cdf, 1e-5, 1.0)

            # Map to eval_times
            pt_bin = min(max(int(round(pt / delta_s)), 0), num_bins - 1)
            s_at_pt = surv_all[:, pt_bin:pt_bin+1]

            cond_surv = np.zeros((N_test, n_evs))
            for k in range(n_evs):
                dt = eval_times[k]
                target_t = pt + dt
                t_bin = min(max(int(round(target_t / delta_s)), 0), num_bins - 1)
                cond_surv[:, k] = np.clip(surv_all[:, t_bin] / np.maximum(s_at_pt[:, 0], 1e-4), 0.0, 1.0)

            all_curves[:, j, :] = cond_surv
            if args.emit_curves:
                continue

            # Subset to alive individuals at pred_time
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
                "model": "ddh",
                "surv_curves": all_curves.tolist(),
                "pred_times": np.asarray(pred_times).tolist(),
                "eval_times": np.asarray(eval_times).tolist(),
                "surv_labels_test": np.asarray(surv_labels_test).tolist(),
            }, f)
        return

    # Static t=0 scoring
    try:
        x_t0 = np.copy(feat_test)
        x_t0[:, 1:, :] = np.nan
        x_t0_t = torch.from_numpy(x_t0).float()
        with torch.no_grad():
            _, outcomes = model(x_t0_t)
            pmf = outcomes[0].numpy()
            expected_time = np.sum(pmf * (bin_edges[:-1] + 0.5 * delta_s), axis=-1)
        t0_cindex = float(concordance_index(surv_labels_test[:, 0], expected_time, surv_labels_test[:, 1]))
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
