"""Shared IO helpers for normalized NPZ packs."""
from __future__ import annotations

import re

import numpy as np

from lib import config


def load_npz_pack(key: str) -> dict:
    p = config.NORMALIZED_DIR / f"{key}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run prepare_data.py first.")
    d = np.load(p, allow_pickle=True)
    raw = d["segments"]
    segments = [np.asarray(raw[i], dtype=np.float32) for i in range(len(raw))]
    return {
        "segments": segments,
        "trial_type": np.asarray(d["trial_type"], dtype=np.int64),
        "trial_idx": np.asarray(d["trial_idx"], dtype=np.int64),
        "trial_run": np.asarray(d["trial_run"], dtype=np.int64),
        "cap": int(np.asarray(d["cap"]).ravel()[0]),
        "n_neurons": segments[0].shape[0] if segments else 0,
    }


def safe_filename(label: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(label).strip()) or "cond"
