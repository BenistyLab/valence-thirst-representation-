"""
data_loader.py – bridge between the NPZ packs produced by prepare_data.py
and the interface expected by train_mean_ff_multhead.py.

The original data_loader module (by Avigail) was lost; this file re-implements
the one function that train_mean_ff_multhead.py calls:

    runs, condition_names = load_mouse(mouse_id="20Hz/AL45", min_duration=300)

Each run in `runs` exposes:
    run.name          – phase key ("pre", "slm_water", or "slm_airpuff")
    run.X             – np.ndarray  (n_cells, n_frames, n_trials), float32
    run.condition     – np.ndarray  (n_trials,), int64 — raw condition codes
    run.trial_index   – np.ndarray  (n_trials,), int64 — original trial indices

`condition_names` maps each condition code to a human-readable string that
matches what train_mean_ff_multhead.py looks up:
    3  → "Water-Op"
    4  → "NaCl"
    18 → "Pav-Airpuff"

Phase → NPZ file mapping
    "pre"         → pre.npz
    "slm_water"   → water.npz
    "slm_airpuff" → airpuff.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np


# ---------------------------------------------------------------------------
# Phase name used by train_mean_ff_multhead  →  NPZ file key from prepare_data
# ---------------------------------------------------------------------------
_PHASE_TO_NPZ: dict[str, str] = {
    "pre":         "pre",
    "slm_water":   "water",
    "slm_airpuff": "airpuff",
}

# Condition code → name string (must match what train_mean_ff_multhead expects)
CONDITION_NAMES: dict[int, str] = {
    3:  "Water-Op",
    4:  "NaCl",
    18: "Pav-Airpuff",
}


# ---------------------------------------------------------------------------
# Run container
# ---------------------------------------------------------------------------

class Run:
    """Lightweight struct matching the interface the training script expects."""

    def __init__(self, name: str, X: np.ndarray,
                 condition: np.ndarray, trial_index: np.ndarray) -> None:
        self.name        = name          # phase key
        self.X           = X             # (n_cells, n_frames, n_trials) float32
        self.condition   = condition     # (n_trials,) int64
        self.trial_index = trial_index  # (n_trials,) int64


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mouse(session_dir=None, *, min_duration: int = 300,
               to_zscore: bool = False,
               mouse_id: str | None = None) -> tuple[list[Run], dict[int, str]]:
    """
    Load per-phase NPZ packs and return (runs, condition_names).

    Parameters
    ----------
    session_dir
        Unused — kept for API compatibility with the original module.
    mouse_id
        Session id such as "20Hz/AL45". When given, the NPZ files are read
        from the matching normalized_data/ folder even if LIVNEH_MOUSE_ID
        points elsewhere. When None, falls back to LIVNEH_MOUSE_ID / the
        module-level config.NORMALIZED_DIR.
    min_duration
        Trials with fewer than this many frames are dropped entirely.
    to_zscore
        Unused — z-scoring is already applied by prepare_data.py.
    """
    from lib import config as _cfg

    if mouse_id is not None:
        if mouse_id not in _cfg.SESSION_PROTOCOL:
            known = ", ".join(_cfg.KNOWN_MICE)
            raise ValueError(
                f"Unknown session id {mouse_id!r}. Known: {known}"
            )
        protocol   = _cfg.SESSION_PROTOCOL[mouse_id]
        mouse_name = _cfg.SESSION_MOUSE[mouse_id]
        normalized_dir = _cfg.REPO_ROOT / "normalized_data" / protocol / mouse_name
    else:
        normalized_dir = _cfg.NORMALIZED_DIR   # resolved from LIVNEH_MOUSE_ID

    runs: list[Run] = []

    for phase_name, npz_key in _PHASE_TO_NPZ.items():
        npz_path = normalized_dir / f"{npz_key}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Missing {npz_path}.\n"
                f"Run prepare_data.py (or prepare_data_no_cap.py) first for "
                f"mouse {mouse_id or 'current'}."
            )

        d = np.load(npz_path, allow_pickle=True)
        raw      = d["segments"]
        segments = [np.asarray(raw[i], dtype=np.float32) for i in range(len(raw))]
        trial_type = np.asarray(d["trial_type"], dtype=np.int64)
        trial_idx  = np.asarray(d["trial_idx"],  dtype=np.int64)

        if not segments:
            raise ValueError(f"NPZ pack '{npz_path}' contains no trials.")
        n_neurons = segments[0].shape[0]

        # Drop trials shorter than min_duration frames (no zero-padding)
        keep_mask = np.array([s.shape[1] >= min_duration for s in segments])
        n_dropped = int(np.sum(~keep_mask))
        if n_dropped:
            print(
                f"[data_loader] {phase_name}/{npz_key}: dropping {n_dropped} "
                f"trial(s) shorter than {min_duration} frames."
            )
        segments   = [s for s, k in zip(segments, keep_mask) if k]
        trial_type = trial_type[keep_mask]
        trial_idx  = trial_idx[keep_mask]

        if not segments:
            raise ValueError(
                f"NPZ pack '{npz_path}' contains no trials with "
                f">={min_duration} frames."
            )

        # Trim every trial to exactly min_duration frames and stack
        # into (n_cells, n_frames, n_trials) — no zero-padding needed.
        X = np.stack([seg[:, :min_duration] for seg in segments], axis=2)

        runs.append(Run(phase_name, X, trial_type, trial_idx))

    return runs, CONDITION_NAMES
