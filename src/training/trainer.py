"""
Unified Model Trainer for SurvTD and Comparative Baselines:
- Supports SurvTD, DeepTCSR Clamped, Dynamic-DeepHit, and Person-Period.
- Manages EMA target network updates, gradient clipping, device placement (MPS/CUDA/CPU),
  validation monitoring (Landmarked C^td / validation loss), and early stopping.

Amended per deviation log:
- Selects best model weights on VALIDATION metric (C^td or validation loss), NOT training loss.
- Supports early stopping with patience.
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import collate_patient_batch
from src.evaluation.landmark import evaluate_landmarked, LandmarkSpec


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def evaluate_val_score(
    model,
    val_dataset,
    val_spec,
    delta_s: float,
    model_type: str,
    ablation_mode: str,
    alpha_anchor: float,
    device: torch.device,
    censoring_est=None,
) -> float:
    """
    Evaluates validation score using landmarked Antolini C^td if spec available,
    or negative validation loss as fallback. Higher is always better.
    """
    model.eval()
    if val_spec is not None and len(val_dataset) > 0:
        try:
            metrics = evaluate_landmarked(model, val_dataset, val_dataset, val_spec, delta_s, device, strict=False)
            valid_c = [m["c_td"] for m in metrics.values() if not np.isnan(m["c_td"])]
            if valid_c:
                return float(np.mean(valid_c))
        except Exception:
            pass

    # Fallback to validation loss (negative, so higher is better)
    total_val_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for p in val_dataset:
            x = p['features'].to(device)
            dts = p['dts'].to(device)
            events = p['events'].to(device)
            tte = float(p['tte'])
            tau_event = float(p['tte']) if p['event'] > 0.5 else float(p['tte']) + 100.0
            mask = p['mask'].to(device) if 'mask' in p and p['mask'] is not None else None
            ipcw_w = float(censoring_est.ipcw(tte, left_limit=True)) if censoring_est is not None else 1.0

            if model_type == "survtd":
                l = model.compute_loss_trajectory(
                    x, dts, events, tte, tau_event, mask=mask,
                    ablation_mode=ablation_mode, alpha_anchor=alpha_anchor,
                    ipcw_weight=ipcw_w
                )
            elif model_type == "deeptcsr":
                l = model.compute_loss_trajectory(
                    x, dts, events, tte, tau_event, mask=mask,
                    alpha_anchor=alpha_anchor,
                    ipcw_weight=ipcw_w
                )
            elif model_type == "dynamic_deephit":
                l = model.compute_loss([p])
            elif model_type == "person_period":
                has_event = bool(p['event'] > 0.5) or bool(torch.any(events > 0.5).item())
                l = model.compute_loss(x, dts, has_event, tte, mask=mask)
            else:
                l = torch.tensor(0.0, device=device)

            total_val_loss += float(l.item())
            n_val += 1

    avg_loss = total_val_loss / max(1, n_val)
    return -float(avg_loss)


def train_model(
    model: nn.Module,
    train_dataset,
    val_dataset=None,
    val_spec: LandmarkSpec | None = None,
    delta_s: float = 1.0,
    model_type: str = "survtd",  # 'survtd', 'deeptcsr', 'dynamic_deephit', 'person_period'
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    epochs: int = 20,
    patience: int = 6,
    es_warmup: int = 5,
    device=None,
    ablation_mode: str = "full",
    alpha_anchor: float | None = None,
    verbose: bool = False
) -> nn.Module:
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = -float("inf")
    best_weights = None
    no_improve_epochs = 0

    history = {
        "train_loss": [],
        "val_score": [],
    }

    # Fit train-split censoring estimator for IPCW scalar loss weighting (A-04, D11)
    censoring_est = None
    try:
        from src.evaluation.censoring import fit_censoring_estimator
        censoring_est = fit_censoring_estimator(train_dataset)
    except Exception:
        censoring_est = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

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
                    ipcw_w = float(censoring_est.ipcw(tte, left_limit=True)) if censoring_est is not None else 1.0

                    l = model.compute_loss_trajectory(
                        x, dts, events, tte, tau_event, mask=mask,
                        ablation_mode=ablation_mode, alpha_anchor=alpha_anchor,
                        ipcw_weight=ipcw_w
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
                    ipcw_w = float(censoring_est.ipcw(tte, left_limit=True)) if censoring_est is not None else 1.0

                    l = model.compute_loss_trajectory(
                        x, dts, events, tte, tau_event, mask=mask,
                        alpha_anchor=alpha_anchor,
                        ipcw_weight=ipcw_w
                    )
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
                    has_event = bool(p['event'] > 0.5) or bool(torch.any(events > 0.5).item())
                    l = model.compute_loss(x, dts, has_event, tte, mask=mask)
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
        history["train_loss"].append(avg_loss)

        # Validation-based checkpointing
        if val_dataset is not None:
            val_score = evaluate_val_score(
                model=model,
                val_dataset=val_dataset,
                val_spec=val_spec,
                delta_s=delta_s,
                model_type=model_type,
                ablation_mode=ablation_mode,
                alpha_anchor=alpha_anchor,
                device=device,
                censoring_est=censoring_est,
            )
            history["val_score"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                best_weights = copy.deepcopy(model.state_dict())
                no_improve_epochs = 0
            elif epoch > es_warmup:
                # Patience does not start accruing until the warm-up grace has
                # elapsed. Measured reason: on synthetic_icu seed 456 SurvTD stopped
                # at ~epoch 7 (335.9s against 675-952s on the other seeds) with a
                # collapsed survival curve (IBS 0.468, AUC 0.474). During cold start
                # the validation C^td sits at chance and does not improve, so a bare
                # patience counter terminates training before the model can leave the
                # degenerate region -- the initialization defect and the stopping rule
                # compound. Declared in amendment A-16.
                no_improve_epochs += 1

            if verbose and (epoch % 5 == 0 or epoch == epochs):
                print(f"[{model_type.upper()}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Val Score: {val_score:.4f}")

            # Early stopping check
            if no_improve_epochs >= patience:
                if verbose:
                    print(f"[{model_type.upper()}] Early stopping triggered at epoch {epoch}")
                break
        else:
            # No validation split: fallback to training loss
            if -avg_loss > best_score:
                best_score = -avg_loss
                best_weights = copy.deepcopy(model.state_dict())
            if verbose and (epoch % 5 == 0 or epoch == epochs):
                print(f"[{model_type.upper()}] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.history = history
    return model
