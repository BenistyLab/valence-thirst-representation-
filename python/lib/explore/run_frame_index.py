"""
Map each trial in a prepare_data_score_per_run.py NPZ pack (train_pool / pre /
water / airpuff) back to the frame indices it came from, in two index spaces:

  - concat_start / concat_end: position within *that pack's own* `concat`
    array (the same local index space lib.explore.extract_trials.py's
    create_trial_boundaries already builds from segment_lengths).
  - frame_start / frame_end: position in the *session's* absolute frame axis
    -- the same axis as motion_corrected_dfFvalid.mat's columns and, from
    lib.explore.lick_rate, lick_rate_df['frame']. This is the piece the packs
    don't store directly: it's reconstructed from exp_data.mat's
    frame_cond_trial (cue_start) via each trial's original trial_idx, using
    prepare_data_score_per_run.py's own _parse_trials (imported, not
    reimplemented) so it stays in lockstep with however the packs were built.

Nothing here recomputes z-scoring or touches prepare_data_score_per_run.py --
it only reads the .npz files it already wrote plus exp_data.mat (via
lib.io.raw.load_exp_data, unchanged).

Once you have frame_start/frame_end for a trial, slice the lick rate frame
axis directly, e.g.:

    seg = lick_rate_df[(lick_rate_df["frame"] >= frame_start)
                        & (lick_rate_df["frame"] <= frame_end)]

Usage:
    from lib.explore.run_frame_index import pack_trial_frame_index, all_packs_frame_index

    df = pack_trial_frame_index("pre")          # one pack
    df_all = all_packs_frame_index()             # train_pool + pre + water + airpuff, concatenated
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.config import NORMALIZED_DIR, default_session_dir, resolve_mouse_id
from lib.io.raw import load_exp_data
from lib.prep.prepare_data_score_per_run import _parse_trials

PACK_KEYS: tuple[str, ...] = ("train_pool", "pre", "water", "airpuff")


def pack_trial_frame_index(
    pack_key: str,
    mouse_id: str | None = None,
    session_dir=None,
) -> pd.DataFrame:
    """Per-trial frame index for one pack (train_pool / pre / water / airpuff).

    Columns:
      pack         -- the pack key, e.g. "pre"
      trial        -- 0-based position of this trial within the pack
                       (same order as the pack's `segments` / concat blocks)
      trial_idx    -- original trial index into the session's full trial list
                       (matches the pack's own `trial_idx` array)
      raw_run      -- literal trialRunMap run number for this trial
      condition_id -- trial type / condition id
      n_frames     -- length of this trial's segment (<= DEFAULT_CAP)
      concat_start, concat_end -- inclusive index range of this trial within
                       the pack's own `concat` array (local to this .npz)
      frame_start, frame_end   -- inclusive index range in the *session's*
                       absolute frame axis -- use these to slice lick_rate_df
                       or dfFvalid's columns.
    """
    mouse_id = resolve_mouse_id(mouse_id)
    if session_dir is None:
        session_dir = default_session_dir(mouse_id)

    npz_path = NORMALIZED_DIR / f"{pack_key}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing {npz_path}. Run prepare_data_zscore_per_run.py for {mouse_id} first."
        )
    with np.load(npz_path, allow_pickle=True) as d:
        trial_idx = np.asarray(d["trial_idx"], dtype=np.int64)
        trial_run = np.asarray(d["trial_run"], dtype=np.int64)
        trial_type = np.asarray(d["trial_type"], dtype=np.int64)
        seg_lens = np.asarray(d["segment_lengths"], dtype=np.int64)

    # Same cue_start values prepare_data_score_per_run.py used to cut each
    # trial's segment out of dfF -- reused as-is, not recomputed by hand.
    exp = load_exp_data(session_dir)
    _, _, cue_start_all, _ = _parse_trials(exp)

    n = len(trial_idx)
    concat_end = np.cumsum(seg_lens)
    concat_start = concat_end - seg_lens

    frame_start = cue_start_all[trial_idx]
    frame_end = frame_start + seg_lens - 1

    return pd.DataFrame({
        "pack": pack_key,
        "trial": np.arange(n),
        "trial_idx": trial_idx,
        "raw_run": trial_run,
        "condition_id": trial_type,
        "n_frames": seg_lens,
        "concat_start": concat_start,
        "concat_end": concat_end - 1,
        "frame_start": frame_start,
        "frame_end": frame_end,
    })


def all_packs_frame_index(
    mouse_id: str | None = None,
    session_dir=None,
    pack_keys: tuple[str, ...] = PACK_KEYS,
) -> pd.DataFrame:
    """pack_trial_frame_index for every pack that exists, concatenated."""
    mouse_id = resolve_mouse_id(mouse_id)
    if session_dir is None:
        session_dir = default_session_dir(mouse_id)

    frames = []
    for key in pack_keys:
        try:
            frames.append(pack_trial_frame_index(key, mouse_id, session_dir))
        except FileNotFoundError as e:
            print(f"{key}: {e}")
    if not frames:
        return pd.DataFrame(columns=[
            "pack", "trial", "trial_idx", "raw_run", "condition_id",
            "n_frames", "concat_start", "concat_end", "frame_start", "frame_end",
        ])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = all_packs_frame_index()
    print(df.groupby("pack")["raw_run"].agg(["count", "min", "max"]))
    print(df.head())
