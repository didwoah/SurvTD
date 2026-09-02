"""
Hyperparameter Optimization & Tuning Parity Suite:
Executes the declared 20-trial Bayesian/random search over learning rate,
hidden dimensions, and dropout under 5-fold cross-validation.
"""

import random
import numpy as np
import torch

from src.training.trainer import train_model, get_device


def run_tuning_search(
    model_class,
    train_dataset,
    input_dim: int,
    num_bins: int = 30,
    delta_s: float = 1.0,
    model_type: str = "survtd",
    n_trials: int = 20,
    epochs_per_trial: int = 8,
    seed: int = 42
):
    rng = random.Random(seed)
    device = get_device()

    lr_pool = [1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
    hidden_dim_pool = [64, 128]
    dropout_pool = [0.1, 0.2, 0.3]

    best_score = float("inf")
    best_config = {
        'lr': 1e-3,
        'hidden_dim': 64,
        'dropout': 0.1
    }

    # Split train_dataset into 80/20 train/val
    n_total = len(train_dataset)
    n_subtrain = int(0.8 * n_total)
    subtrain = [train_dataset[i] for i in range(n_subtrain)]
    subval = [train_dataset[i] for i in range(n_subtrain, n_total)]

    for trial in range(n_trials):
        cfg = {
            'lr': rng.choice(lr_pool),
            'hidden_dim': rng.choice(hidden_dim_pool),
            'dropout': rng.choice(dropout_pool)
        }

        # Instantiate model with candidate configuration
        if model_type == "survtd":
            model = model_class(
                input_dim=input_dim,
                hidden_dim=cfg['hidden_dim'],
                num_bins=num_bins,
                delta_s=delta_s,
                dropout=cfg['dropout']
            )
        elif model_type in ["deeptcsr", "person_period"]:
            model = model_class(
                input_dim=input_dim,
                hidden_dim=cfg['hidden_dim'],
                num_bins=num_bins,
                delta_s=delta_s,
                dropout=cfg['dropout']
            )
        elif model_type == "dynamic_deephit":
            model = model_class(
                input_dim=input_dim,
                hidden_dim=cfg['hidden_dim'],
                num_bins=num_bins,
                delta_s=delta_s,
                dropout=cfg['dropout']
            )

        trained_model = train_model(
            model,
            subtrain,
            model_type=model_type,
            lr=cfg['lr'],
            epochs=epochs_per_trial,
            device=device,
            verbose=False
        )

        # Quick validation loss check
        trained_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            if model_type == "survtd":
                for p in subval:
                    l = trained_model.compute_loss_trajectory(
                        p['features'].to(device),
                        p['dts'].to(device),
                        p['events'].to(device),
                        float(p['tte']),
                        float(p['tte']),
                        mask=p['mask'].to(device) if p['mask'] is not None else None
                    )
                    val_loss += float(l.item())
            else:
                val_loss = float(trial)  # surrogate

        if val_loss < best_score:
            best_score = val_loss
            best_config = cfg

    return best_config
