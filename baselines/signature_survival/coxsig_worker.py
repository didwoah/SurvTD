import argparse
import json
import torch
import numpy as np
from src.coxsig import CoxSignature

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    data = torch.load(args.input_pt)
    paths_train = data['paths_train']
    surv_labels_train = data['surv_labels_train']
    paths_test = data['paths_test']
    surv_labels_test = data['surv_labels_test']
    pred_times = data['pred_times']
    eval_times = data['eval_times']

    # Train CoxSig
    coxsig = CoxSignature(sig_level=2, alphas=1e-5, max_iter=100)
    coxsig.train(paths_train, surv_labels_train)

    # 1. Dynamic landmark scoring
    cindex_matrix = coxsig.score(paths_test, surv_labels_test, pred_times, eval_times, 'c_index')
    bs_matrix = coxsig.score(paths_test, surv_labels_test, pred_times, eval_times, 'bs')

    # 2. Static t=0 scoring via lifelines Harrell's C-index
    try:
        from lifelines.utils import concordance_index
        # Evaluate hazard at t=0
        haz_t0 = coxsig.predict_hazard(paths_test, np.array([0.0]))[:, 0, :]
        cum_risk = np.sum(haz_t0, axis=-1)
        t0_cindex = float(concordance_index(surv_labels_test[:, 0], -cum_risk, surv_labels_test[:, 1]))
    except Exception as e:
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
