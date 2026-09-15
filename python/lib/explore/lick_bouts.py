"""
Find the last "real" licking activity in each trial, so you can pick a frame
cap for lick-rate averaging that doesn't truncate genuine licking bouts but
also isn't dragged out by a single stray/incidental lick near the end of a
trial.

Built entirely on top of lib.explore.lick_rate.lick_rate_phase (unchanged) --
this module only adds "how do we define a significant licking bout, and
where does the last one end" on top of what that already returns.

--- Definition of a "significant lick bout" ---
A sliding window of `window` consecutive frames counts as part of a real
bout if it contains at least `min_licks` lick events. This is a density
threshold rather than a strict "N licks with zero gaps": mice tend to lick
in short, closely-spaced bursts rather than on every single frame, so
requiring a handful of licks within a short window (e.g. 3 licks within 5
frames) catches real bouts while still ignoring one isolated lick with
nothing around it.

Setting window == min_licks makes this exactly "N licks in a row" (the
strict version you were considering) -- e.g. min_licks=3, window=3.
The looser default (min_licks=3, window=5) tolerates a single missed frame
inside a real bout, which is usually what you want with lick data.

--- Usage ---
    from lib.explore.lick_bouts import lick_bout_summary, suggest_cap

    summary = lick_bout_summary(TRAIN_POOL_NPZ_PATH, "train_pool")
    print(summary.sort_values("last_significant_lick_frame", ascending=False).head())

    report = suggest_cap(summary)   # -> dict with a suggested cap + diagnostics
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.explore.lick_rate import lick_rate_phase


def last_significant_lick_frame(
    lick_events: pd.Series,
    min_licks: int = 3,
    window: int = 5,
) -> float:
    """Last frame in `lick_events` that is part of a "significant" lick bout.

    A bout = a run of `window` consecutive frames (by position in
    `lick_events`, which must already be sorted by frame) containing at
    least `min_licks` lick events. Returns the largest frame LABEL (i.e.
    `lick_events.index` value, not position) for which this holds, or
    np.nan if no such bout exists anywhere in the trial.

    `lick_events` should be indexed by frame (any int index is fine -- pass
    trial-relative frames if you want the result relative to trial start)
    with values that are nonzero when a lick happened that frame. Works
    whether lickFr is a strict 0/1 indicator or a small count/rate -- this
    thresholds at > 0.
    """
    licked = (lick_events.to_numpy() > 0).astype(int)
    if licked.sum() == 0:
        return np.nan

    # rolling_sum[i] = number of lick events in the `window` frames ending at
    # position i (i.e. frames [i-window+1, i]).
    rolling_sum = pd.Series(licked).rolling(window=window, min_periods=1).sum()
    in_bout = (rolling_sum >= min_licks).to_numpy()
    if not in_bout.any():
        return np.nan

    last_pos = np.flatnonzero(in_bout)[-1]
    return float(lick_events.index[last_pos])


def max_licks_in_window(lick_events: pd.Series, window: int = 5) -> int:
    """The densest cluster of licks anywhere in this trial: the largest
    number of lick events found in any `window`-frame span.

    Diagnostic only -- use this when `last_significant_lick_frame` keeps
    coming back NaN, to see what's actually achievable. If most trials'
    max is, say, 2, then min_licks=3 can never be met at that window size
    no matter how real the licking is -- you need a smaller min_licks or a
    wider window, not "more significant" licking.
    """
    licked = (lick_events.to_numpy() > 0).astype(int)
    if licked.sum() == 0:
        return 0
    rolling_sum = pd.Series(licked).rolling(window=window, min_periods=1).sum()
    return int(rolling_sum.max())


def lick_bout_summary(
    path: str,
    phase: str,
    min_licks: int = 3,
    window: int = 5,
) -> pd.DataFrame:
    """Per-trial last-significant-lick frame for one npz pack/phase.

    Reuses lick_rate_phase(path, phase) (unchanged) for the trial-labelled
    lick data, then applies last_significant_lick_frame() to each
    (trial_type, trial) group, with frame reset to be relative to that
    trial's own start (0 = first frame of the trial's segment) so the
    result is directly comparable to a frame cap.

    Returns one row per trial with columns:
      trial_type, trial,
      n_frames                     -- frames available in this trial
      n_licks                      -- total lick events in this trial
      max_licks_in_window          -- densest window x this trial reaches
                                       (diagnostic; see max_licks_in_window)
      last_significant_lick_frame  -- 0-based, relative to trial start;
                                       NaN if no bout found at this
                                       min_licks/window
    """
    lr_df, _neuron_df = lick_rate_phase(path, phase)
    # lr_df: 1 row ("lick_rate"), columns = MultiIndex(trial_type, trial, frame)
    licks = lr_df.T["lick_rate"]

    rows = []
    for (trial_type, trial), sub in licks.groupby(level=["trial_type", "trial"]):
        frame_vals = sub.index.get_level_values("frame")
        rel_index = frame_vals - frame_vals.min()
        sub_rel = pd.Series(sub.to_numpy(), index=rel_index).sort_index()

        last_sig = last_significant_lick_frame(sub_rel, min_licks=min_licks, window=window)
        rows.append({
            "trial_type": trial_type,
            "trial": trial,
            "n_frames": int(sub_rel.index.max()) + 1,
            "n_licks": int((sub_rel.to_numpy() > 0).sum()),
            "max_licks_in_window": max_licks_in_window(sub_rel, window=window),
            "last_significant_lick_frame": last_sig,
        })

    return pd.DataFrame(rows)


def suggest_cap(summary: pd.DataFrame, percentile: float = 100.0) -> dict:
    """Suggest a frame cap from a lick_bout_summary() table.

    percentile=100 (default): the smallest cap that still covers every
    trial's last significant lick bout (the max across trials). Trials with
    no significant bout (NaN) don't constrain the cap and are excluded from
    this calculation, but ARE counted in the diagnostics below.

    Pass a lower percentile (e.g. 95) to trade "a handful of long-tailed
    trials get their tail truncated" for a smaller, more typical cap.

    Also reports the shortest trial's available frame count: a cap can't
    exceed that without some trials simply running out of real frames, no
    matter how their licking looks.
    """
    valid = summary["last_significant_lick_frame"].dropna()
    if valid.empty:
        hint = ""
        if "max_licks_in_window" in summary.columns and not summary.empty:
            achievable = int(summary["max_licks_in_window"].max())
            hint = (
                f" The densest cluster found in any trial was {achievable} "
                f"licks in one window -- min_licks must be <= that to ever "
                f"match anything at this window size."
            )
        raise ValueError(
            "No trial had a significant lick bout at this min_licks/window -- "
            "try lowering min_licks or widening window." + hint
        )

    needed_cap = int(np.ceil(np.percentile(valid, percentile)))
    shortest_trial = int(summary["n_frames"].min())

    return {
        "needed_cap": needed_cap,
        "shortest_available_trial_frames": shortest_trial,
        "cap_exceeds_shortest_trial": needed_cap > shortest_trial,
        "n_trials_total": int(len(summary)),
        "n_trials_with_bout": int(valid.shape[0]),
        "n_trials_with_no_bout": int(len(summary) - valid.shape[0]),
        "percentile_used": percentile,
    }
