"""
Extract lick rate ('lickFr') for a mouse as a pandas DataFrame, reusing
Avigail's existing raw-data loaders as-is (no edits to her code, nothing
recomputed).

lickFr lives in exp_data.mat as a single row whose length equals the total
number of imaging frames in the session -- i.e. it is parallel, frame for
frame, to the columns of motion_corrected_dfFvalid.mat (the neuron x frame
array her lib.io.raw.load_dfFvalid loads). This module reads it via
lib.io.raw.load_exp_data (her loader) and lib.config.resolve_mouse_id /
default_session_dir (her mouse/session resolution), the same LIVNEH_MOUSE_ID /
LIVNEH_DATA_ROOT env vars used everywhere else in the repo.

Run from repo root: python extract_lick_rate.py
(set LIVNEH_MOUSE_ID first if you don't want the default mouse)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.config import EXPLORATION_OUT, default_session_dir, resolve_mouse_id
from lib.io.raw import load_exp_data
from lib.explore.run_frame_index import all_packs_frame_index
from lib.explore.extract_trials import create_trial_labelled_df

OUT_DIR = EXPLORATION_OUT / "lick_rate"


def load_lick_rate_df(mouse_id: str | None = None, session_dir=None) -> pd.DataFrame:
    """One row per imaging frame: frame index + lick rate at that frame.

    Loads exp_data.mat via lib.io.raw.load_exp_data (her loader, untouched) and
    pulls out the 'lickFr' field. In the .mat file lickFr is a single row of
    length = number of frames in the session, matching the frame axis of
    dfFvalid's columns (lib.io.raw.load_dfFvalid) -- so this DataFrame's
    'frame' index lines up 1:1 with the neuron data's frame axis, and can be
    sliced with the same cue_start / trial_end frame indices used elsewhere in
    the repo (see lib/prep/prepare_data.py:_parse_trials).
    """
    mouse_id = resolve_mouse_id(mouse_id)
    if session_dir is None:
        session_dir = default_session_dir(mouse_id)

    exp = load_exp_data(session_dir)
    if "lickFr" not in exp:
        raise KeyError(f"'lickFr' not found in exp_data.mat. Keys: {sorted(exp.keys())}")

    lick = np.asarray(exp["lickFr"], dtype=float).reshape(-1)

    return pd.DataFrame({
        "frame": np.arange(lick.size),
        "lick_rate": lick,
    })


def main(mouse_id: str | None = None) -> pd.DataFrame:
    mouse_id = resolve_mouse_id(mouse_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_lick_rate_df(mouse_id)

    csv_path = OUT_DIR / "lick_rate.csv"
    df.to_csv(csv_path, index=False)
    print(f"{mouse_id}: lickFr -> {len(df)} frames -> {csv_path}")

    # Sanity check against the neuron data's frame axis, when it's available too.
    try:
        from lib.io.raw import load_dfFvalid
        dfF = load_dfFvalid(default_session_dir(mouse_id))
        if dfF.shape[1] != len(df):
            print(
                f"WARNING: lickFr has {len(df)} frames but dfFvalid has "
                f"{dfF.shape[1]} frames for {mouse_id} -- they are not the same length."
            )
        else:
            print(f"OK: lickFr length matches dfFvalid frame count ({dfF.shape[1]} frames).")
    except FileNotFoundError:
        pass

    return df

def lick_rate_phase(path, phase):
    """
    Get the lick rate df that is analogous to the labelled df based on the npz file.
    :param path: the npz file corresponding to the mouse
           phase(str): train_pool, pre, water, or airpuff
    :return:
    """
    # get the full lick rate df, includes all runs and all trials
    lr_df = main()
    lr_df = lr_df.transpose()
    lr_df = lr_df.drop('frame')

    # create a map of the absolute values of the ranges (corresponding to the raw data) of each trial in the npz file
    indices_map = all_packs_frame_index()

    # create a multiindex labelled df of the npz file we want, including frame, trial, condition
    trial_boundaries, neuron_df = create_trial_labelled_df(path)

    # create a mask of the absolute values of the phase that we want and access it in the lick rate df
    train_ranges = indices_map.loc[
        indices_map['pack'] == phase,
        ['frame_start', 'frame_end']
    ]

    mask = np.zeros(len(lr_df.columns), dtype=bool)

    for start, end in train_ranges.itertuples(index=False):
        mask |= (lr_df.columns >= start) & (lr_df.columns <= end)

    lr_df = lr_df.loc[:, mask]

    # create the same labels in the lick rate df
    if len(lr_df.columns) != len(neuron_df.columns):
        print("WARNING: different amount of columns in lick rate and corresponding npz file")
    lr_df.columns = neuron_df.columns

    return lr_df, neuron_df


if __name__ == "__main__":
    main()
