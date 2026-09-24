"""
Plot average lick rate (water + water-free + water(s) trials, including the
free-consumption phase) for every mouse found under `normalized_data`,
reusing Avigail's own pipeline as-is
(lick_rate_phase / all_packs_frame_index / create_trial_labelled_df) --
nothing in lib/ is modified.

One figure per mouse, overlaying every protocol that mouse has data for
(Mock=blue, 20Hz=red, 1Hz=green), each with its own per-phase run
boundary lines in matching color, so the runs can be compared directly.
A mouse missing one or two protocols just gets fewer curves.

Condition ids included
----------------------
  3  – water          (pre / airpuff / water phases)
  7  – water-free     (free-consumption phase)
  8  – water(s)       (free-consumption phase)

If the train_pool was built by prepare_data_free_consumption.py (which
includes the free-consumption phase), all three condition ids appear and
the free-consumption phase boundary is drawn automatically.  If the
train_pool was built by the older prepare_data_score_per_run.py (no
free-consumption), only condition 3 is present and the plot degrades
gracefully to the water-only view.

Why each protocol's data is computed in its own subprocess
------------------------------------------------------------
lib.config resolves LIVNEH_MOUSE_ID into path constants (NORMALIZED_DIR,
OUTPUTS_DIR, ...) exactly once, at import time.  lib.explore.run_frame_index
then reads NORMALIZED_DIR back out as a plain module-level constant when it
locates each mouse's train_pool/pre/water/airpuff .npz packs.  So setting
os.environ["LIVNEH_MOUSE_ID"] later in the *same* process (e.g. mid-loop)
does nothing -- lib.config has already computed NORMALIZED_DIR once and
nothing re-derives it.

So each protocol run is computed in its own subprocess, with LIVNEH_MOUSE_ID
set correctly before that subprocess's Python (and therefore lib.config)
even starts up.  The subprocess writes its trial numbers / values / phase
boundaries out as JSON; this parent process reads that back and does all of
the actual plotting (overlaying protocols), since plotting has to happen
after every protocol for a given mouse has been computed.
"""
import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

# Repo root (the "python" folder) -- needed so `from lib...` imports resolve
# both here and in each per-mouse subprocess.
REPO_ROOT = Path(r"C:\Users\neely.heller\PycharmProjects\valence-thirst-representation-\python")
sys.path.insert(0, str(REPO_ROOT))

PATH = REPO_ROOT / "normalized_data"

PROTOCOL_COLORS = {"Mock": "blue", "20Hz": "red", "1Hz": "green"}

# Condition ids whose trials are included in the lick-rate plot.
# 3 = water (pre/airpuff/water phases)
# 7 = water-free, 8 = water(s)  (free-consumption phase)
WATER_COND_IDS = {3, 7, 8}
WATER_TRIAL_TYPES = {"water", "water-free", "water(s)"}

# Trial types that belong to the free-consumption phase.  These have shorter
# trials, so a lower minimum-frame threshold is used for them.
FREE_CONSUMPTION_TRIAL_TYPES = {"water-free", "water(s)"}

# Minimum frame count to include a trial.
# Standard water trials: 600 frames (average taken over exactly 600 frames).
# Free-consumption trials: much shorter -- set LOW while the data is explored;
# raise once typical trial lengths are known.
MIN_FRAMES = 600
FREE_CONSUMPTION_MIN_FRAMES = 50  # <-- adjust as needed


def mouse_id_for(npz_path: Path, root: Path) -> str:
    """'normalized_data/20Hz/AL45/.../train_pool.npz' -> '20Hz/AL45'.

    Matches the "{protocol}/{mouse}" ids lib.config.KNOWN_MICE expects
    (protocol and mouse are the first two path components under normalized_data).
    """
    rel = npz_path.relative_to(root)
    protocol, mouse = rel.parts[0], rel.parts[1]
    return f"{protocol}/{mouse}"


def _all_water_trial_phases(mouse_id: str):
    """Phase label for every water / water-free / water(s) trial in the
    train_pool, in temporal order, indexed 1..N.

    Uses all_packs_frame_index() for run membership and
    run_to_phase(phase_run_map(...)) to map run -> phase (pre / airpuff /
    water / free-consumption).  When the train_pool contains free-consumption
    trials (conditions 7/8), their phase is 'free-consumption'; when it does
    not, the index naturally contains only condition-3 trials (water-only
    behaviour identical to the old script).

    No lib/ files are changed.
    """
    import pandas as pd
    from lib.explore.run_frame_index import all_packs_frame_index
    from lib.prep.prepare_data_score_per_run import run_to_phase
    from lib.config import phase_run_map

    idx = all_packs_frame_index(mouse_id=mouse_id, pack_keys=("train_pool",))
    water_rows = idx.loc[idx["condition_id"].isin(WATER_COND_IDS)].sort_values("trial")

    run_phase = run_to_phase(phase_run_map(mouse_id))
    phases = water_rows["raw_run"].map(run_phase)
    phases.index = pd.RangeIndex(1, len(phases) + 1)
    return phases


def _phase_boundaries(phases) -> list[float]:
    """x-positions (midpoint between two trials) where the run phase changes."""
    import pandas as pd

    boundaries = []
    prev_trial, prev_phase = None, None
    for trial, phase in phases.items():
        if pd.isna(phase):
            prev_trial, prev_phase = None, None
            continue
        if prev_phase is not None and phase != prev_phase:
            boundaries.append((prev_trial + trial) / 2)
        prev_trial, prev_phase = trial, phase
    return boundaries


def _water_col_pairs(lr_df):
    """Return (trial_type, trial_number) pairs for all water-type columns in
    lr_df, in temporal (column) order, deduplicated.

    Uses dict.fromkeys to preserve first-occurrence order while removing
    the duplicate entries that arise because each trial spans many frames.
    """
    mask = lr_df.columns.get_level_values("trial_type").isin(WATER_TRIAL_TYPES)
    water = lr_df.loc[:, mask]
    pairs = list(dict.fromkeys(
        zip(
            water.columns.get_level_values("trial_type"),
            water.columns.get_level_values("trial"),
        )
    ))
    return water, pairs


def _compute_one(npz_path: str, mouse_id: str) -> dict | None:
    """Average lick rate per trial (first 600 frames), across all water-type
    conditions, in temporal order.

    Runs inside a subprocess where LIVNEH_MOUSE_ID is already set.  Uses
    Avigail's lick_rate_phase unmodified; only the aggregation logic here
    has been extended to cover conditions 7 and 8 alongside condition 3.
    """
    import pandas as pd
    from lib.explore.lick_rate import lick_rate_phase

    lr_df, neuron_df = lick_rate_phase(Path(npz_path), "train_pool")

    # Get all water-type (trial_type, trial_number) pairs in temporal order.
    water, col_pairs = _water_col_pairs(lr_df)

    # For each trial in temporal order: check frame count, compute 600-frame mean.
    valid_global_idxs = []  # 1-based position in col_pairs
    valid_avgs = []
    for global_i, (tt, tnum) in enumerate(col_pairs, start=1):
        trial_data = water.loc[:, (tt, tnum, slice(None))]
        n_frames = trial_data.shape[1]
        is_fc = tt in FREE_CONSUMPTION_TRIAL_TYPES
        threshold = FREE_CONSUMPTION_MIN_FRAMES if is_fc else MIN_FRAMES
        if n_frames < threshold:
            print(f"Dropped {tt} trial {tnum}: only {n_frames} frames "
                  f"(threshold={threshold})")
            continue
        valid_global_idxs.append(global_i)
        # Free-consumption trials: use all available frames (no 600-frame cap).
        # Standard water trials: average over the first MIN_FRAMES frames only.
        frames = trial_data if is_fc else trial_data.iloc[:, :MIN_FRAMES]
        valid_avgs.append(float(frames.values.mean()))

    if not valid_avgs:
        print(f"{mouse_id}: no water trials with >= 600 frames, skipping.")
        return None

    # Phase for each valid trial.  _all_water_trial_phases returns one phase
    # per trial in the same temporal order (indexed 1..N_all); reindex to the
    # valid subset, then re-number 1..N_valid for the x-axis.
    phases_all = _all_water_trial_phases(mouse_id)
    phases = phases_all.reindex(valid_global_idxs)
    phases.index = pd.RangeIndex(1, len(phases) + 1)
    boundaries = _phase_boundaries(phases)

    return {
        "trials": list(range(1, len(valid_avgs) + 1)),
        "values": valid_avgs,
        "boundaries": boundaries,
    }


def _compute_one_sliding_window(
    npz_path: str,
    mouse_id: str,
    window_size: int,
    step: int = 1,
) -> dict | None:
    """Sliding-window lick rate, across all water-type conditions, in temporal
    order.

    Same structure as _compute_one but instead of averaging each trial's first
    600 frames down to one number, a window of `window_size` frames slides
    across the trial's full length in steps of `step`, producing a short trace
    per trial.  A trial is dropped if it has fewer than `window_size` frames.

    Returns:
      - "trials": 1-based sequential indices of valid trials
      - "window_values": {trial -> [window averages along the trial]}
      - "window_size", "step": echoed back
      - "boundaries": same run-phase boundaries as _compute_one
    """
    from lib.explore.lick_rate import lick_rate_phase
    import pandas as pd

    lr_df, neuron_df = lick_rate_phase(Path(npz_path), "train_pool")

    water, col_pairs = _water_col_pairs(lr_df)

    valid_global_idxs = []
    window_values: dict[int, list[float]] = {}  # global_i -> window averages

    for global_i, (tt, tnum) in enumerate(col_pairs, start=1):
        trial_data = water.loc[:, (tt, tnum, slice(None))]
        n_frames = trial_data.shape[1]
        if n_frames < window_size:
            print(f"Dropped {tt} trial {tnum}: only {n_frames} frames "
                  f"(< window_size={window_size})")
            continue
        values = trial_data.values[0]  # single row -> 1-D array of lick rates
        window_values[global_i] = [
            float(values[start : start + window_size].mean())
            for start in range(0, n_frames - window_size + 1, step)
        ]
        valid_global_idxs.append(global_i)

    if not valid_global_idxs:
        print(f"{mouse_id}: no water trials with >= {window_size} frames, skipping.")
        return None

    # Renumber valid trials 1..N for the x-axis.
    sequential = list(range(1, len(valid_global_idxs) + 1))
    seq_window_values = {seq: window_values[gi]
                         for seq, gi in zip(sequential, valid_global_idxs)}

    phases_all = _all_water_trial_phases(mouse_id)
    phases = phases_all.reindex(valid_global_idxs)
    phases.index = pd.RangeIndex(1, len(phases) + 1)
    boundaries = _phase_boundaries(phases)

    return {
        "trials": sequential,
        "window_values": {float(t): seq_window_values[t] for t in sequential},
        "window_size": window_size,
        "step": step,
        "boundaries": boundaries,
    }


def _compute_via_subprocess(
    file: Path,
    mouse_id: str,
    window_size: int | None = None,
    step: int = 1,
) -> dict | None:
    """window_size=None -> the original 600-frames-per-trial average
    (_compute_one); window_size=<n> -> the sliding-window version
    (_compute_one_sliding_window) with that window size and step."""
    env = os.environ.copy()
    env["LIVNEH_MOUSE_ID"] = mouse_id
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    options = {"window_size": window_size, "step": step}

    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        result = subprocess.run(
            [sys.executable, __file__, "--worker", str(file), mouse_id, out_path, json.dumps(options)],
            cwd=str(REPO_ROOT),
            env=env,
        )
        if result.returncode != 0:
            print(f"WARNING: {mouse_id} failed (exit {result.returncode}), skipping.")
            return None
        text = Path(out_path).read_text().strip()
        return json.loads(text) if text else None
    finally:
        Path(out_path).unlink(missing_ok=True)


def _worker_main(npz_path: str, mouse_id: str, out_path: str, options_json: str = "{}") -> None:
    options = json.loads(options_json) if options_json else {}
    window_size = options.get("window_size")
    step = options.get("step", 1)

    if window_size is None:
        data = _compute_one(npz_path, mouse_id)
    else:
        data = _compute_one_sliding_window(npz_path, mouse_id, window_size=window_size, step=step)

    Path(out_path).write_text(json.dumps(data) if data is not None else "")


def main() -> None:
    import matplotlib.pyplot as plt

    files = list(Path(PATH).rglob("train_pool.npz"))

    # Group by mouse only (e.g. "AL45"), across all of its protocols.
    by_mouse: dict[str, dict[str, Path]] = {}
    for file in files:
        protocol, mouse = mouse_id_for(file, Path(PATH)).split("/")
        by_mouse.setdefault(mouse, {})[protocol] = file

    for mouse in sorted(by_mouse):
        plt.figure()
        plotted = False

        for protocol, file in sorted(by_mouse[mouse].items()):
            mouse_id = f"{protocol}/{mouse}"
            print(f"--- {mouse_id}: {file}")

            data = _compute_via_subprocess(file, mouse_id)
            if data is None:
                continue

            color = PROTOCOL_COLORS.get(protocol, "black")
            plt.plot(data["trials"], data["values"], color=color, label=protocol)
            for x in data["boundaries"]:
                plt.axvline(x, color=color, linestyle="--", linewidth=1, alpha=0.6)
            plotted = True

        if not plotted:
            plt.close()
            continue

        plt.xlabel("Trial")
        plt.ylabel("Average lick rate")
        plt.title(f"Average Lick Rate for {mouse}")
        plt.legend()
        plt.show()

    print("end")


def main_sliding_window(window_size: int = 10, step: int = 1) -> None:
    """Same one-figure-per-mouse / overlaid-protocols layout as `main()`, but
    each protocol's curve is the sliding-window lick rate (window_size frames,
    moving by `step` frames each point) concatenated trial-by-trial, instead
    of one point per trial.  A vertical dashed line is drawn at each trial
    boundary so the trial structure stays visible even though the x-axis is
    now "window index" rather than "trial".
    """
    import matplotlib.pyplot as plt

    files = list(Path(PATH).rglob("train_pool.npz"))

    by_mouse: dict[str, dict[str, Path]] = {}
    for file in files:
        protocol, mouse = mouse_id_for(file, Path(PATH)).split("/")
        by_mouse.setdefault(mouse, {})[protocol] = file

    for mouse in sorted(by_mouse):
        plt.figure()
        plotted = False

        for protocol, file in sorted(by_mouse[mouse].items()):
            mouse_id = f"{protocol}/{mouse}"
            print(f"--- {mouse_id}: {file}")

            data = _compute_via_subprocess(file, mouse_id, window_size=window_size, step=step)
            if data is None:
                continue

            # Concatenate each trial's window-average trace in trial order.
            # A NaN is inserted between trials so matplotlib breaks the line
            # there instead of drawing a false connector between non-adjacent
            # windows (which produces steep near-vertical spikes).
            values: list[float] = []
            trial_start_x: dict[float, int] = {}
            for i, trial in enumerate(data["trials"]):
                trial_start_x[trial] = len(values)
                values.extend(data["window_values"][str(trial)])
                if i != len(data["trials"]) - 1:
                    values.append(float("nan"))

            x = list(range(len(values)))
            color = PROTOCOL_COLORS.get(protocol, "black")
            plt.plot(x, values, color=color, label=protocol, linewidth=1)

            # Map run-phase boundaries from trial-space to window-index space.
            trials_sorted = data["trials"]
            for b in data["boundaries"]:
                lower = max(t for t in trials_sorted if t <= b)
                upper = min(t for t in trials_sorted if t >= b)
                if lower == upper:
                    continue
                frac = (b - lower) / (upper - lower)
                x_lower = trial_start_x[lower]
                x_upper = trial_start_x[upper]
                plt.axvline(
                    x_lower + frac * (x_upper - x_lower),
                    color=color, linestyle="--", linewidth=1, alpha=0.6,
                )

            plotted = True

        if not plotted:
            plt.close()
            continue

        plt.xlabel(f"Window index (window={window_size} frames, step={step})")
        plt.ylabel("Average lick rate (sliding window)")
        plt.title(f"Sliding-Window Average Lick Rate for {mouse}")
        plt.legend()
        plt.show()

    print("end")


if __name__ == "__main__":
    if len(sys.argv) in (5, 6) and sys.argv[1] == "--worker":
        options_json = sys.argv[5] if len(sys.argv) == 6 else "{}"
        _worker_main(sys.argv[2], sys.argv[3], sys.argv[4], options_json)
    else:
        main()