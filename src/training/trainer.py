"""
Unified Model Trainer for SurvTD and Comparative Baselines:
- Supports SurvTD, DeepTCSR Clamped, Dynamic-DeepHit, and Person-Period.
- Manages EMA target network updates, gradient clipping, device placement (MPS/CUDA/CPU), and early stopping.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import collate_patient_batch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_model(
    model: nn.Module,
    train_dataset,
    val_dataset=None,
    model_type: str = "survtd",  # 'survtd', 'deeptcsr', 'dynamic_deephit', 'person_period'
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    epochs: int = 20,
    device=None,
    ablation_mode: str = "full",
    verbose: bool = False
):
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # Shuffle trajectories
        indices = torch.randperm(len(train_dataset)).tolist()

        for b_start in range(0, len(train_dataset), batch_size):
            b_indices = indices[b_start:b_start + batch_size]
            batch_patients = [train_dataset[i] for i in b_indices]

            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)

            if model_type == "survtd":
                for p in batch_patients:
                    x = p['features'].to(device)
                    dts = p['dts'].to(device)
                    events = p['events'].to(device)
                    tte = float(p['tte'])
                    tau_event = float(p['tte']) if p['event'] > 0.5 else float(p['tte']) + 100.0
                    mask = p['mask'].to(device) if 'mask' in p and p['mask'] is not None else None

                    l = model.compute_loss_trajectory(
                        x, dts, events, tte, tau_event, mask=mask, ablation_mode=ablation_mode
                    )
                    batch_loss = batch_loss + l
                batch_loss = batch_loss / max(1, len(batch_patients))

            elif model_type == "deeptcsr":
                for p in batch_patients:
                    x = p['features'].to(device)
                    dts = p['dts'].to(device)
                    events = p['events'].to(device)
                    tte = float(p['tte'])
                    tau_event = float(p['tte']) if p['event'] > 0.5 else float(p['tte']) + 100.0
                    mask = p['mask'].to(device) if 'mask' in p and p['mask'] is not None else None

                    l = model.compute_loss_trajectory(x, dts, events, tte, tau_event, mask=mask)
                    batch_loss = batch_loss + l
                batch_loss = batch_loss / max(1, len(batch_patients))

            elif model_type == "dynamic_deephit":
                batch_loss = model.compute_loss(batch_patients)

            elif model_type == "person_period":
                for p in batch_patients:
                    x = p['features'].to(device)
                    dts = p['dts'].to(device)
                    events = p['events'].to(device)
                    tte = float(p['tte'])
                    mask = p['mask'].to(device) if 'mask' in p and p['mask'] is not None else None
                    l = model.compute_loss(x, dts, events, tte, mask=mask)
                    batch_loss = batch_loss + l
                batch_loss = batch_loss / max(1, len(batch_patients))

            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            # Target network EMA update if applicable
            if hasattr(model, 'update_target_network'):
                model.update_target_network()

            total_loss += float(batch_loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 5 == 0 or epoch == epochs):
            print(f"[{model_type.upper()}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model
