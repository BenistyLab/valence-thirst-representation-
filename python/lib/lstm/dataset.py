"""
Dataset utilities for phase-based trial decoding.

Loads the NPZ trial packs produced by prepare_data.py and wraps them in
PyTorch Dataset / DataLoader objects.

Each trial is a variable-length calcium sequence  (n_neurons, T).
We pad shorter trials with zeros and stack them into fixed-length tensors
of shape  (T_max, n_neurons)  – ready for the LSTM encoder.

Trial types are re-mapped to a compact 0-based integer label so that
CrossEntropyLoss works out of the box.

Typical trial-type condition codes (from the existing codebase):
    water trial   → condition 1  (re-mapped to label 0)
    airpuff trial → condition 2  (re-mapped to label 1)
    NaCl trial    → condition 3  (re-mapped to label 2)

If the actual codes in a recording differ, pass `condition_to_label` to
`load_phase_datasets` to override the mapping.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from lib.io.npz import load_npz_pack
from lib import config


# ---------------------------------------------------------------------------
# Default condition → label mapping
# ---------------------------------------------------------------------------

DEFAULT_CONDITION_TO_LABEL: dict[int, int] = {
    3: 0,   # water
    18: 1,  # airpuff
    4: 2,   # salt (NaCl)
    # condition 26 = SLM trials – intentionally excluded
}

LABEL_NAMES: dict[int, str] = {
    0: "water",
    1: "airpuff",
    2: "salt",
}

N_CLASSES = 3


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrialDataset(Dataset):
    """
    One dataset per phase (pre / airpuff / water).

    Each item is a dict:
        'inputs'   : (T_max, n_neurons) float32 tensor  – padded calcium trace
        'label'    : int64 scalar – 0=water, 1=airpuff, 2=salt
        'lengths'  : int64 scalar – actual (unpadded) sequence length
        'trial_idx': int64 scalar – original trial index (for bookkeeping)

    Parameters
    ----------
    segments:          list of (n_neurons, T_i) arrays.
    trial_types:       raw condition codes, parallel to segments.
    trial_indices:     original trial indices, parallel to segments.
    condition_to_label: mapping from condition code to 0-based label.
    """

    def __init__(
        self,
        segments: list[np.ndarray],
        trial_types: np.ndarray,
        trial_indices: np.ndarray,
        condition_to_label: dict[int, int] | None = None,
    ) -> None:
        cond_map = condition_to_label or DEFAULT_CONDITION_TO_LABEL

        # Filter out any trial whose condition is not in the map
        keep = [
            i for i, tt in enumerate(trial_types) if int(tt) in cond_map
        ]
        if not keep:
            raise ValueError(
                f"No trials with recognised condition codes "
                f"{list(cond_map.keys())} found. "
                f"Conditions present: {set(trial_types.tolist())}."
            )

        filtered_segs = [segments[i] for i in keep]
        filtered_types = trial_types[keep]
        filtered_idx = trial_indices[keep]

        # Pad all trials to the same length along the time axis
        lengths = [s.shape[1] for s in filtered_segs]
        t_max = max(lengths)

        # Stack → (N, n_neurons, T_max) → transpose → (N, T_max, n_neurons)
        n_neurons = filtered_segs[0].shape[0]
        padded = np.zeros((len(filtered_segs), n_neurons, t_max), dtype=np.float32)
        for j, seg in enumerate(filtered_segs):
            padded[j, :, : seg.shape[1]] = seg
        padded = padded.transpose(0, 2, 1)   # (N, T_max, n_neurons)

        labels = np.array([cond_map[int(tt)] for tt in filtered_types], dtype=np.int64)

        self.inputs = torch.from_numpy(padded)
        self.labels = torch.from_numpy(labels)
        self.lengths = torch.tensor(lengths, dtype=torch.int64)
        self.trial_indices = torch.tensor(filtered_idx, dtype=torch.int64)
        self.n_neurons = n_neurons
        self.t_max = t_max

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[idx],
            "label": self.labels[idx],
            "lengths": self.lengths[idx],
            "trial_idx": self.trial_indices[idx],
        }


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _phase_npz_key(phase: str) -> str:
    """Return the NPZ filename stem for a given phase."""
    # prepare_data.py saves files named by phase (e.g. "pre", "airpuff", "water")
    return phase


def load_phase_dataset(
    phase: str,
    condition_to_label: dict[int, int] | None = None,
) -> TrialDataset:
    """
    Load the NPZ pack for `phase` and return a TrialDataset.

    Requires LIVNEH_MOUSE_ID and LIVNEH_DATA_ROOT to be set (or defaults from
    lib/config.py) so that NORMALIZED_DIR points to the right folder.
    """
    key = _phase_npz_key(phase)
    pack = load_npz_pack(key)
    return TrialDataset(
        segments=pack["segments"],
        trial_types=pack["trial_type"],
        trial_indices=pack["trial_idx"],
        condition_to_label=condition_to_label,
    )


def load_phase_datasets(
    phases: list[str] = ("pre", "airpuff", "water"),
    condition_to_label: dict[int, int] | None = None,
) -> dict[str, TrialDataset]:
    """Load a TrialDataset for each phase and return as a dict."""
    return {
        phase: load_phase_dataset(phase, condition_to_label)
        for phase in phases
    }


# ---------------------------------------------------------------------------
# Stratified split helper
# ---------------------------------------------------------------------------

def _stratified_split(
    dataset: TrialDataset,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """
    Split dataset indices so that each class contributes `val_fraction` of
    its trials to the validation set (stratified by label).

    Returns
    -------
    train_indices, val_indices : lists of integer indices into `dataset`.
    """
    labels = dataset.labels.numpy()
    train_idx: list[int] = []
    val_idx: list[int] = []

    for cls in range(N_CLASSES):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) == 0:
            continue
        cls_indices = cls_indices.copy()
        rng.shuffle(cls_indices)
        n_val = max(1, int(len(cls_indices) * val_fraction))
        val_idx.extend(cls_indices[:n_val].tolist())
        train_idx.extend(cls_indices[n_val:].tolist())

    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Class-weight helper  (for weighted CrossEntropyLoss)
# ---------------------------------------------------------------------------

def compute_class_weights(
    datasets: dict[str, TrialDataset],
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from all phase datasets combined.

    The weights are normalised so their mean equals 1, keeping the overall
    gradient scale comparable to an unweighted loss.

    Returns
    -------
    weights : (N_CLASSES,) float32 tensor  – pass directly to
              nn.CrossEntropyLoss(weight=...).
    """
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for ds in datasets.values():
        lbl = ds.labels.numpy()
        for c in range(N_CLASSES):
            counts[c] += int((lbl == c).sum())

    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.mean()   # normalise: mean weight = 1
    return torch.from_numpy(weights.astype(np.float32))


# ---------------------------------------------------------------------------
# DataLoader factory  (stratified train / val split)
# ---------------------------------------------------------------------------

def make_dataloaders(
    datasets: dict[str, TrialDataset],
    val_fraction: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[
    dict[str, DataLoader],
    dict[str, DataLoader],
]:
    """
    Split each phase dataset into train / val using a *stratified* split
    (each class contributes `val_fraction` of its trials to val) and return
    DataLoaders.

    The stratified split ensures that the validation set contains trials from
    every class even when the dataset is heavily imbalanced.  No trial appears
    in both the train and val sets.

    Returns
    -------
    train_loaders : {phase: DataLoader}
    val_loaders   : {phase: DataLoader}
    """
    train_loaders: dict[str, DataLoader] = {}
    val_loaders: dict[str, DataLoader] = {}

    rng = np.random.default_rng(seed)

    for phase, ds in datasets.items():
        train_idx, val_idx = _stratified_split(ds, val_fraction, rng)

        train_loaders[phase] = DataLoader(
            Subset(ds, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loaders[phase] = DataLoader(
            Subset(ds, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loaders, val_loaders