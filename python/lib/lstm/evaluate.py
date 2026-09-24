"""
Evaluation and cross-phase analysis for the trained model.

Two evaluation modes:

1. Within-phase decoding
   For each phase p, take the test set from phase p, encode it, decode with
   decoder_p, and report accuracy.  This verifies that each decoder correctly
   identifies trial types within its own phase.

2. Cross-phase decoding
   Encode trials from a *source* phase and decode them with a *target* phase's
   decoder.  The scientific interest (per Hadas' guidance) is to see how the
   SLM reshapes the internal representation: e.g. pass pre-phase trials through
   the airpuff-SLM decoder and observe whether the predicted trial-type
   structure changes (points that formed a certain cluster now form a different
   cluster).

Helper: extract latent embeddings (z_T) from a DataLoader for dimensionality
        reduction / visualisation (PCA, UMAP, …).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import PhaseDecoderModel
from .dataset import LABEL_NAMES, N_CLASSES


# ---------------------------------------------------------------------------
# Prediction utilities
# ---------------------------------------------------------------------------

def _get_device(model: PhaseDecoderModel) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def predict_phase(
    model: PhaseDecoderModel,
    loader: DataLoader,
    source_phase: str,
    decode_phase: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Run inference on all batches in `loader`.

    Parameters
    ----------
    model:        trained PhaseDecoderModel.
    loader:       DataLoader for the source phase's (test / val) data.
    source_phase: the phase the trials come from (used only for naming).
    decode_phase: the decoder to use; defaults to source_phase (within-phase).

    Returns
    -------
    dict with keys:
        'latents'     : (N, latent_dim) final LSTM latent vectors z_T.
        'logits'      : (N, n_classes)
        'probs'       : (N, n_classes) softmax probabilities.
        'preds'       : (N,)  argmax predicted label.
        'labels'      : (N,)  ground-truth labels.
        'trial_indices': (N,) original trial indices.
    """
    if decode_phase is None:
        decode_phase = source_phase

    device = _get_device(model)
    model.eval()

    all_latents, all_logits, all_labels, all_tidx = [], [], [], []

    for batch in loader:
        inputs = batch["inputs"].to(device)
        labels = batch["label"]
        tidx = batch["trial_idx"]

        z_T, logits = model(inputs, phase=source_phase, decode_phase=decode_phase)

        all_latents.append(z_T.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
        all_tidx.append(tidx.numpy())

    latents = np.concatenate(all_latents, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    tidx_arr = np.concatenate(all_tidx, axis=0)

    probs = torch.softmax(torch.from_numpy(logits_arr), dim=-1).numpy()
    preds = logits_arr.argmax(axis=-1)

    return {
        "latents": latents,
        "logits": logits_arr,
        "probs": probs,
        "preds": preds,
        "labels": labels_arr,
        "trial_indices": tidx_arr,
    }


# ---------------------------------------------------------------------------
# Within-phase accuracy
# ---------------------------------------------------------------------------

def evaluate_within_phase(
    model: PhaseDecoderModel,
    loaders: dict[str, DataLoader],
) -> dict[str, dict]:
    """
    Evaluate within-phase decoding accuracy for every phase in `loaders`.

    Returns
    -------
    results: {phase: {'accuracy': float, 'per_class_accuracy': dict, 'predictions': dict}}
    """
    results: dict[str, dict] = {}

    for phase, loader in loaders.items():
        out = predict_phase(model, loader, source_phase=phase)
        labels = out["labels"]
        preds = out["preds"]

        acc = float((preds == labels).mean())

        per_class: dict[str, float] = {}
        for cls_idx, cls_name in LABEL_NAMES.items():
            mask = labels == cls_idx
            if mask.sum() == 0:
                per_class[cls_name] = float("nan")
            else:
                per_class[cls_name] = float((preds[mask] == cls_idx).mean())

        results[phase] = {
            "accuracy": acc,
            "per_class_accuracy": per_class,
            "n_trials": len(labels),
            "predictions": out,
        }

        print(
            f"  [{phase:8s}] accuracy = {acc:.3f}  "
            + "  ".join(f"{k}={v:.3f}" for k, v in per_class.items())
        )

    return results

def print_interesting_evaluation(results):
    for phase in results.keys():
        print(f"\nPhase {phase}:")
        print(f"\nProbs:")
        print(results[phase]['predictions']['probs'])
        print(f"\nPreds:")
        print(results[phase]['predictions']['preds'])
        print(f"\nLabels:")
        print(results[phase]['predictions']['labels'])
        print(f"\nAccuracy:")
        print(results[phase]['accuracy'])

# ---------------------------------------------------------------------------
# Cross-phase decoding
# ---------------------------------------------------------------------------

def evaluate_cross_phase(
    model: PhaseDecoderModel,
    loaders: dict[str, DataLoader],
    source_phases: list[str] | None = None,
    target_phases: list[str] | None = None,
) -> dict[tuple[str, str], dict]:
    """
    Decode each source phase's trials with every target phase's decoder.

    This is the key cross-phase analysis: by passing pre-phase activity
    through the SLM decoder we can observe how the SLM reshapes the
    internal representation of trial types.

    Parameters
    ----------
    source_phases: which phases to use as input (default: all in loaders).
    target_phases: which decoders to use (default: all model phases).

    Returns
    -------
    cross_results: {(source_phase, target_phase): {'accuracy': float, 'predictions': dict}}
    """
    if source_phases is None:
        source_phases = list(loaders.keys())
    if target_phases is None:
        target_phases = model.phases

    cross_results: dict[tuple[str, str], dict] = {}

    print("\nCross-phase decoding:")
    for src in source_phases:
        loader = loaders[src]
        for tgt in target_phases:
            out = predict_phase(
                model, loader, source_phase=src, decode_phase=tgt
            )
            labels = out["labels"]
            preds = out["preds"]
            acc = float((preds == labels).mean())

            cross_results[(src, tgt)] = {
                "accuracy": acc,
                "n_trials": len(labels),
                "predictions": out,
            }

            marker = " <-- own decoder" if src == tgt else ""
            print(f"  source={src:8s}  decoder={tgt:8s}  acc={acc:.3f}{marker}")

    return cross_results


# ---------------------------------------------------------------------------
# Latent-space extraction (for PCA / UMAP visualisation)
# ---------------------------------------------------------------------------

def extract_latents(
    model: PhaseDecoderModel,
    loaders: dict[str, DataLoader],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Extract the final LSTM latent embeddings z_T for each phase.

    Returns
    -------
    {phase: {'latents': (N, L), 'labels': (N,), 'trial_indices': (N,)}}
    """
    device = _get_device(model)
    model.eval()
    results: dict[str, dict[str, np.ndarray]] = {}

    for phase, loader in loaders.items():
        all_z, all_labels, all_tidx = [], [], []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["inputs"].to(device)
                z_T, _ = model(inputs, phase=phase)
                all_z.append(z_T.cpu().numpy())
                all_labels.append(batch["label"].numpy())
                all_tidx.append(batch["trial_idx"].numpy())

        results[phase] = {
            "latents": np.concatenate(all_z, axis=0),
            "labels": np.concatenate(all_labels, axis=0),
            "trial_indices": np.concatenate(all_tidx, axis=0),
        }

    return results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    results: dict,
    output_dir: str | Path,
    filename: str = "lstm_results.npz",
) -> Path:
    """
    Save evaluation results to an NPZ file.

    The cross-phase results and within-phase results are flattened into
    named arrays for easy inspection in NumPy / MATLAB.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename

    arrays: dict[str, np.ndarray] = {}

    within = results.get("within_phase", {})
    for phase, r in within.items():
        arrays[f"within_{phase}_accuracy"] = np.array([r["accuracy"]])
        for cls, acc in r["per_class_accuracy"].items():
            arrays[f"within_{phase}_{cls}_accuracy"] = np.array([acc])
        pred_dict = r.get("predictions", {})
        for key, val in pred_dict.items():
            arrays[f"within_{phase}_{key}"] = val

    cross = results.get("cross_phase", {})
    for (src, tgt), r in cross.items():
        arrays[f"cross_{src}_to_{tgt}_accuracy"] = np.array([r["accuracy"]])
        pred_dict = r.get("predictions", {})
        for key, val in pred_dict.items():
            arrays[f"cross_{src}_to_{tgt}_{key}"] = val

    np.savez_compressed(save_path, **arrays)
    print(f"\nResults saved to: {save_path}")
    return save_path
