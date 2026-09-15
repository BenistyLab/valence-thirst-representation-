"""
Summarize trials and cues from Ca_mice_Livne exp_data.mat.

frame_cond_trial [trials x 4] = (trial_type, cue_start_frame, cue_end_frame, trial_end_frame)
exp_cond [conditions x 2] = condition ID and name (when loadable)

Run from repo root: python inspect_trials.py
"""

from pathlib import Path

import numpy as np

from lib.config import default_session_dir
from lib.io.raw import load_exp_data


def _try_load_exp_cond_names(session_dir):
    """Try to load condition names from exp_cond (may be cell array of strings in v7.3)."""
    try:
        import h5py
    except ImportError:
        return None
    mat_path = Path(session_dir) / "exp_data.mat"
    if not mat_path.exists():
        return None
    names = {}
    try:
        with h5py.File(str(mat_path), "r") as f:
            if "exp_cond" not in f:
                return None
            node = f["exp_cond"]
            # MATLAB: exp_cond can be [conditions x 2], col 0 = ID, col 1 = name (cell array)
            if isinstance(node, h5py.Dataset):
                arr = np.array(node)
                if arr.ndim != 2:
                    return None
                for i in range(arr.shape[0]):
                    cond_id = int(arr[i, 0].flat[0]) if arr[i, 0].size else i
                    ref = arr[i, 1]
                    if hasattr(ref, "__len__") and len(ref) > 0:
                        ref = ref.flat[0]
                    try:
                        s = f[ref]
                        if isinstance(s, h5py.Dataset):
                            chars = np.array(s)
                            if chars.dtype.kind == "U" or chars.size == 0:
                                name = "".join(chars.ravel().astype(str)) if chars.size else ""
                            else:
                                name = "".join(chr(c) for c in chars.ravel())
                            names[cond_id] = name.strip()
                    except Exception:
                        pass
            elif isinstance(node, h5py.Group) and "value" in node:
                arr = np.array(node["value"])
                for i in range(arr.shape[0]):
                    cond_id = int(arr[i, 0].flat[0]) if arr[i, 0].size else i
                    ref = arr[i, 1]
                    if hasattr(ref, "__len__") and len(ref) > 0:
                        ref = ref.flat[0]
                    try:
                        s = f[ref]
                        if isinstance(s, h5py.Dataset):
                            chars = np.array(s)
                            if chars.dtype.kind == "U" or chars.size == 0:
                                name = "".join(chars.ravel().astype(str)) if chars.size else ""
                            else:
                                name = "".join(chr(c) for c in chars.ravel())
                            names[cond_id] = name.strip()
                    except Exception:
                        pass
    except Exception:
        return None
    return names if names else None


def main():
    session_dir = default_session_dir()
    if not session_dir.exists():
        print(f"Dataset not found: {session_dir}")
        return

    exp = load_exp_data(session_dir)
    fs = float(np.asarray(exp["fs"]).reshape(-1).flat[0]) if "fs" in exp else None
    fct = np.asarray(exp["frame_cond_trial"])
    if fct.ndim == 1:
        fct = fct.reshape(-1, 4)
    n_trials, n_cols = fct.shape
    if n_cols < 4:
        print("frame_cond_trial has fewer than 4 columns, cannot parse.")
        return

    # Columns: trial_type, cue_start, cue_end, trial_end (all frame indices)
    trial_type = fct[:, 0].astype(int)
    cue_start = fct[:, 1].astype(int)
    cue_end = fct[:, 2].astype(int)
    trial_end = fct[:, 3].astype(int)

    print(f"Total trials: {n_trials}")
    print(f"Frame rate: {fs} Hz\n" if fs else "Frame rate: unknown")

    # Condition ID → name (from raw exp_cond or v7.3 parsed names)
    cond_names = _try_load_exp_cond_names(session_dir)
    if not cond_names and "exp_cond" in exp:
        ec = np.asarray(exp["exp_cond"])
        if ec.ndim == 2:
            cond_names = {}
            for r in range(ec.shape[0]):
                c0 = ec[r, 0]
                c1 = ec[r, 1]
                c0_val = int(c0) if np.isscalar(c0) else int(np.asarray(c0).flat[0])
                if not np.issubdtype(np.asarray(c1).dtype, np.number):
                    cond_names[c0_val] = str(c1).strip()
    if cond_names:
        print("Condition ID → name")
        for cid in sorted(cond_names.keys()):
            print(f"  {cid}: {cond_names[cid]}")

    # Per-condition summary
    unique_types, counts = np.unique(trial_type, return_counts=True)
    print("\nTrials per condition")
    for ut, cnt in zip(unique_types, counts):
        label = cond_names.get(int(ut), f"cond_{ut}") if cond_names else f"cond_{ut}"
        print(f"  {label}: {cnt} trials")

    # Durations (in frames)
    cue_frames = np.maximum(0, cue_end - cue_start)
    trial_frames = np.maximum(0, trial_end - cue_start)
    print("\nCue duration (cue_end - cue_start)")
    print(f"  Frames: min={cue_frames.min()}, max={cue_frames.max()}, mean={cue_frames.mean():.1f}")
    print("\nTrial duration (trial_end - cue_start)")
    print(f"  Frames: min={trial_frames.min()}, max={trial_frames.max()}, mean={trial_frames.mean():.1f}")

    # Runs (trialRunMap)
    if "trialRunMap" in exp:
        trm = np.asarray(exp["trialRunMap"]).reshape(-1)
        if trm.size >= n_trials:
            runs = trm[:n_trials].astype(int)
            unique_runs, run_counts = np.unique(runs, return_counts=True)
            print("\nRuns (trialRunMap)")
            print(f"  Unique runs: {len(unique_runs)} (IDs {unique_runs.min()}–{unique_runs.max()})")
            print(f"  Trials per run: min={run_counts.min()}, max={run_counts.max()}, mean={run_counts.mean():.1f}")

            # Per-run summary (trials, duration, conditions)
            for run_id in unique_runs:
                run_mask = runs == run_id
                n_trials_run = run_mask.sum()
                start_frame = cue_start[run_mask].min()
                end_frame = trial_end[run_mask].max()
                duration_frames = end_frame - start_frame + 1
                run_types = np.unique(trial_type[run_mask])
                if fs:
                    duration_sec = duration_frames / fs
                    duration_min = duration_sec / 60
                    print("\nRun {}".format(run_id))
                    print(f"  Trials: {n_trials_run}")
                    print(f"  Duration: {duration_frames} frames ({duration_sec:.1f} s, {duration_min:.1f} min)")
                else:
                    print("\nRun {}".format(run_id))
                    print(f"  Trials: {n_trials_run}")
                    print(f"  Duration: {duration_frames} frames (frame {start_frame} to {end_frame})")
                print("  Conditions present:")
                for ct in run_types:
                    cnt = (trial_type[run_mask] == ct).sum()
                    label = cond_names.get(int(ct), f"cond_{ct}") if cond_names else f"cond_{ct}"
                    print(f"    {label}: {cnt} trials")

    # Sample of first few trials
    print("\nFirst 10 trials (trial_type, cue_start, cue_end, trial_end)")
    for i in range(min(10, n_trials)):
        tt = trial_type[i]
        label = cond_names.get(tt, str(tt)) if cond_names else str(tt)
        print(f"  Trial {i+1}: {label} ({cue_start[i]}, {cue_end[i]}, {trial_end[i]})")


if __name__ == "__main__":
    main()
