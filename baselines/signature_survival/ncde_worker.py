import argparse
import json
import torch
import numpy as np
from src.cresnet import ControlledResNet
from _curve_export import curves_on_residual_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emit_curves", action="store_true")
    args = parser.parse_args()

    data = torch.load(args.input_pt)
    paths_train = data['paths_train']
    surv_labels_train = data['surv_labels_train']
    paths_test = data['paths_test']
    surv_labels_test = data['surv_labels_test']
    pred_times = data['pred_times']
    eval_times = data['eval_times']
    sampling_times = data['sampling_times']

    # ControlledResNet architecture from Bleistein et al.
    model = ControlledResNet(
        latent_dim=4,
        hidden_dim=64,
        path_dim=paths_train.shape[-1],
        activation='tanh',
        n_layers=1,
        sampling_times=sampling_times
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train(
        optimizer,
        paths_train,
        surv_labels_train,
        batch_size=32,
        num_epochs=args.epochs,
        verbose=False,
        plot_loss=False
    )

    # 1. Dynamic landmark scoring
    if args.emit_curves:
        curves = curves_on_residual_grid(model.predict_survival, paths_test,
                                         sampling_times, pred_times, eval_times)
        with open(args.output_json, "w") as f:
            json.dump({
                "model": "ncde",
                "surv_curves": curves.tolist(),
                "pred_times": np.asarray(pred_times).tolist(),
                "eval_times": np.asarray(eval_times).tolist(),
                "surv_labels_test": np.asarray(surv_labels_test).tolist(),
            }, f)
        return

    cindex_matrix = model.score(paths_test, surv_labels_test, pred_times, eval_times, 'c_index')
    bs_matrix = model.score(paths_test, surv_labels_test, pred_times, eval_times, 'bs')

    # 2. Static t=0 scoring
    try:
        from lifelines.utils import concordance_index
        # Evaluate model hazard at t=0
        cum_risk = model.predict_risk(paths_test, np.array([0.0]))[:, 0]
        t0_cindex = float(concordance_index(surv_labels_test[:, 0], -cum_risk, surv_labels_test[:, 1]))
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
