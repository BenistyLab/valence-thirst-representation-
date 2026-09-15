"""
Plot average lick rate (during the water phase) for every mouse found under
`normalized_data`, reusing Avigail's own pipeline as-is
(lick_rate_phase / all_packs_frame_index / create_trial_labelled_df) --
nothing in lib/ is modified.

One figure per mouse, overlaying every protocol that mouse has data for
(Mock=blue, 20Hz=red, 1Hz=green), each with its own pre/airpuff/water
run-phase boundary lines in matching color, so the runs can be compared
directly. A mouse missing one or two protocols just gets fewer curves.

Why each protocol's data is computed in its own subprocess
------------------------------------------------------------
lib.config resolves LIVNEH_MOUSE_ID into path constants (NORMALIZED_DIR,
OUTPUTS_DIR, ...) exactly once, at import time. lib.explore.run_frame_index
then reads NORMALIZED_DIR back out as a plain module-level constant when it
locates each mouse's train_pool/pre/water/airpuff .npz packs. So setting
os.environ["LIVNEH_MOUSE_ID"] later in the *same* process (e.g. mid-loop)
does nothing -- lib.config has already computed NORMALIZED_DIR once and
nothing re-derives it. That's exactly why this used to only ever work for
whichever single mouse/protocol LIVNEH_MOUSE_ID happened to be set to
*before* the interpreter started.

So each protocol run is computed in its own subprocess, with LIVNEH_MOUSE_ID
set correctly before that subprocess's Python (and therefore lib.config)
even starts up. The subprocess writes its trial numbers / values / phase
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
# both here and in each per-mouse subprocess, since this script itself lives
# outside the repo (PyCharm scratch file).
REPO_ROOT = Path(r"C:\Users\neely.heller\PycharmProjects\valence-thirst-representation-\python")
sys.path.insert(0, str(REPO_ROOT))

PATH = REPO_ROOT / "normalized_data"

PROTOCOL_COLORS = {"Mock": "blue", "20Hz": "red", "1Hz": "green"}


def mouse_id_for(npz_path: Path, root: Path) -> str:
    """'normalized_data/20Hz/AL45/.../train_pool.npz' -> '20Hz/AL45'.

    Matches the "{protocol}/{mouse}" ids lib.config.KNOWN_MICE expects
    (protocol and mouse are the first two path components under normalized_data).
    """
    rel = npz_path.relative_to(root)
    protocol, mouse = rel.parts[0], rel.parts[1]
    return f"{protocol}/{mouse}"


def _water_trial_phases(mouse_id: str):
    """1-based chronological trial number -> 'pre' / 'airpuff' / 'water'
    (the *run phase*, not the trial condition), for every water-condition
    trial in the train_pool pack -- numbered the same way
    lib.explore.extract_trials.create_trial_labelled_df numbers the 'water'
    level of lr_df/neuron_df's columns, so it lines up 1:1 with the trial
    numbers plotted below.

    Built entirely from her own code: all_packs_frame_index() for each
    trial's raw_run, NUMBER_TRIAL_TYPES for the water condition id, and
    run_to_phase(phase_run_map(...)) -- the same run->phase mapping
    prepare_data_score_per_run.main() uses to fit mu/sigma per phase.
    """
    import pandas as pd
    from lib.explore.run_frame_index import all_packs_frame_index
    from lib.explore.extract_trials import NUMBER_TRIAL_TYPES
    from lib.prep.prepare_data_score_per_run import run_to_phase
    from lib.config import phase_run_map

    water_cond_id = {name: cid for cid, name in NUMBER_TRIAL_TYPES.items()}["water"]
    idx = all_packs_frame_index(mouse_id=mouse_id, pack_keys=("train_pool",))
    water_rows = idx.loc[idx["condition_id"] == water_cond_id].sort_values("trial")

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


def _compute_one(npz_path: str, mouse_id: str) -> dict | None:
    """Runs inside a subprocess where LIVNEH_MOUSE_ID is already set, so
    lib.config's NORMALIZED_DIR (and everything derived from it) matches
    this mouse before lib.explore.lick_rate / run_frame_index ever import.
    This is Avigail's own code, called unmodified -- only the result
    (trials/values/boundaries) is returned, nothing is plotted here.
    """
    import pandas as pd
    from lib.explore.lick_rate import lick_rate_phase

    lr_df, neuron_df = lick_rate_phase(Path(npz_path), "train_pool")
    water = lr_df.loc[:, "water"]

    valid_trials = []
    for trial in water.columns.get_level_values("trial").unique():
        trial_data = water.loc[:, (trial, slice(None))]
        n_frames = trial_data.shape[1]
        if n_frames < 600:
            print(f"Dropped trial {trial}: only {n_frames} frames")
        else:
            valid_trials.append(trial)

    if not valid_trials:
        print(f"{mouse_id}: no water trials with >= 600 frames, skipping.")
        return None

    # Keep only valid trials and first 600 frames of each
    water_600 = pd.concat(
        [water.loc[:, (trial, slice(None))].iloc[:, :600] for trial in valid_trials],
        axis=1,
    )

    # Average across the first 600 frames of each trial
    avg_lick_rate = water_600.groupby(level="trial", axis=1).mean()

    phases = _water_trial_phases(mouse_id).reindex(avg_lick_rate.columns)
    boundaries = _phase_boundaries(phases)

    return {
        "trials": [float(t) for t in avg_lick_rate.columns],
        "values": [float(v) for v in avg_lick_rate.iloc[0]],
        "boundaries": boundaries,
    }


def _compute_via_subprocess(file: Path, mouse_id: str) -> dict | None:
    env = os.environ.copy()
    env["LIVNEH_MOUSE_ID"] = mouse_id
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        result = subprocess.run(
            [sys.executable, __file__, "--worker", str(file), mouse_id, out_path],
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


def _worker_main(npz_path: str, mouse_id: str, out_path: str) -> None:
    data = _compute_one(npz_path, mouse_id)
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


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
