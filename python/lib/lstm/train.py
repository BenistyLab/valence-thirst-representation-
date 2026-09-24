"""
Training loop for the shared LSTM encoder + phase decoders.

Design notes (adapted from the paper's Section 4.5):
  - The encoder and ALL phase decoder heads are trained jointly in a single
    pass.  For each batch we know which phase it came from, so we route it
    to the matching decoder.
  - Loss: CrossEntropyLoss (averaged over all phases in the batch).
    Optional inverse-frequency class weights address label imbalance.
  - Optimiser: Adam, lr=1e-3, weight_decay=1e-4.
  - Dropout: 0.1 (set at model-construction time).
  - Early stopping: patience=50 epochs on joint validation loss.

Unlike the original paper there is NO two-stage training schedule because
our phases are not repeated (room A is visited twice in the paper – hence
the need to freeze the encoder for the third decoder head).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import PhaseDecoderModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model: PhaseDecoderModel,
    loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    criterion: nn.CrossEntropyLoss,
) -> tuple[float, float]:
    """
    Run one full epoch over all phases.

    Returns average (loss, accuracy) across all samples.
    """
    model.train(train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Zip the iterators so we interleave batches from all phases.
    # If phases have different numbers of batches, we repeat the shorter ones.
    # For simplicity we iterate each phase DataLoader independently and
    # accumulate – the parameter updates will be mixed across phases.
    for phase, loader in loaders.items():
        for batch in loader:
            inputs  = batch["inputs"].to(device)    # (B, T, n_neurons)
            labels  = batch["label"].to(device)     # (B,)
            lengths = batch["lengths"].to(device)   # (B,) – real frame count per trial

            with torch.set_grad_enabled(train):
                _, logits = model(inputs, phase=phase, lengths=lengths)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            B = labels.size(0)
            total_loss += loss.item() * B
            total_correct += int((logits.argmax(-1) == labels).sum())
            total_samples += B

    if total_samples == 0:
        return float("nan"), float("nan")

    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model: PhaseDecoderModel,
    train_loaders: dict[str, DataLoader],
    val_loaders: dict[str, DataLoader],
    *,
    output_dir: str | Path,
    checkpoint_name: str = "lstm_model.pt",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int = 50,
    device: torch.device | None = None,
    class_weights: torch.Tensor | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train the model with early stopping and checkpoint saving.

    Parameters
    ----------
    model:          PhaseDecoderModel (freshly initialised or pre-loaded).
    train_loaders:  {phase: DataLoader} for training data.
    val_loaders:    {phase: DataLoader} for validation data.
    output_dir:     directory where checkpoints will be saved.
    checkpoint_name: filename for the best checkpoint.
    lr:             Adam learning rate.
    weight_decay:   Adam weight decay.
    max_epochs:     maximum number of epochs.
    patience:       early-stopping patience (epochs without improvement).
    device:         torch device (auto-detected if None).
    class_weights:  optional (N_CLASSES,) float32 tensor of inverse-frequency
                    weights passed to CrossEntropyLoss to counter label
                    imbalance.  Compute with dataset.compute_class_weights().
    verbose:        print epoch summaries.

    Returns
    -------
    history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
             as lists of per-epoch values, plus 'best_epoch'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / checkpoint_name

    if device is None:
        device = _get_device()
    model = model.to(device)

    # Build loss criterion – move class weights to the correct device
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    history: dict[str, list] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loaders, optimizer, device, train=True,
            criterion=criterion,
        )
        val_loss, val_acc = _run_epoch(
            model, val_loaders, None, device, train=False,
            criterion=criterion,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "phases": model.phases,
                },
                checkpoint_path,
            )
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1 or improved):
            elapsed = time.time() - t0
            marker = " ✓" if improved else ""
            print(
                f"[{epoch:4d}/{max_epochs}]  "
                f"train loss={train_loss:.4f}  acc={train_acc:.3f}  |  "
                f"val loss={val_loss:.4f}  acc={val_acc:.3f}"
                f"  ({elapsed:.0f}s){marker}"
            )

        if no_improve >= patience:
            if verbose:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)."
                )
            break

    if verbose:
        print(f"\nBest epoch: {best_epoch}  best val loss: {best_val_loss:.4f}")
        print(f"Checkpoint saved to: {checkpoint_path}")

    history["best_epoch"] = best_epoch
    return history


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    model: PhaseDecoderModel,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> dict:
    """Load a saved checkpoint into `model` (in-place). Returns the state dict."""
    if device is None:
        device = _get_device()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    return state