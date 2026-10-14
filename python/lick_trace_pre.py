"""
lick_trace_pre.py – Sliding-window lick trace for the Pre phase, by trial type.

For each mouse/protocol session that has a pre.npz, produces one figure with
three traces (frames 0–300):
  • water   – mean ± SEM sliding-window lick rate across all water trials in Pre
  • nacl    – mean ± SEM across all nacl trials in Pre
  • airpuff – mean ± SEM across all airpuff trials in Pre

Figures are saved under:
  python/outputs/<protocol>/<mouse>/exploration/lick_rate/
  pre_lick_trace_w<window_size>_s<step>.png

Usage
-----
  python lick_trace_pre.py                        # defaults: window=10, step=1
  python lick_trace_pre.py --window-size 20 --step 5
  python lick_trace_pre.py --window-size 30 --step 10 --no-show  # save only

The script reuses the existing lib.explore.lick_rate.lick_rate_phase pipeline
without modifying any library files, and uses the same subprocess trick as
average_lick_rate.py so that lib.config is initialised fresh for each session.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ── repo root ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

PATH = REPO_ROOT / "normalized_data"

# ── constants ──────────────────────────────────────────────────────────────
MAX_FRAMES = 300          # analyse only the first 300 frames of each trial
TRIAL_TYPES = ["water", "nacl", "airpuff"]
TRIAL_COLORS = {
    "water":   "#2196F3",   # blue
    "nacl":    "#FF9800",   # orange
    "airpuff": "#F44336",   # red
}


# ══════════════════════════════════════════════════════════════════════════
#  Worker – runs in a subprocess with LIVNEH_MOUSE_ID already set
# ══════════════════════════════════════════════════════════════════════════

def _compute_pre_lick_trace(mouse_id: str, window_size: int, step: int) -> dict | None:
    """
    For a single session (mouse_id), compute per-trial-type sliding-window
    lick traces over the first MAX_FRAMES frames of each Pre trial.

    Returns
    -------
    {
      "x_values": [frame positions of window centres, length = n_windows],
      "trial_types": {
          "water":   {"mean": [...], "sem": [...], "n_trials": int} | None,
          "nacl":    {...} | None,
          "airpuff": {...} | None,
      }
    }
    or None if no data could be extracted.
    """
    import numpy as np
    from lib.config import REPO_ROOT as LIB_REPO_ROOT, SESSION_PROTOCOL, SESSION_MOUSE
    from lib.explore.lick_rate import lick_rate_phase

    protocol   = SESSION_PROTOCOL[mouse_id]
    mouse_name = SESSION_MOUSE[mouse_id]
    pre_npz    = LIB_REPO_ROOT / "normalized_data" / protocol / mouse_name / "pre.npz"

    if not pre_npz.exists():
        print(f"[worker] {mouse_id}: pre.npz not found at {pre_npz}, skipping.")
        return None

    try:
        lr_df, neuron_df = lick_rate_phase(pre_npz, "pre")
    except Exception as exc:
        print(f"[worker] {mouse_id}: lick_rate_phase failed: {exc}")
        return None

    # lr_df  → shape (1, total_pre_frames), columns = MultiIndex(trial_type, trial, frame)
    # The 'frame' level is the concatenated-array position (not relative within trial).
    # All we need is to iterate by (trial_type, trial_number) and take values in order.

    result: dict = {}
    x_values: list[int] | None = None

    for tt in TRIAL_TYPES:
        # Select all columns for this trial type
        type_mask = lr_df.columns.get_level_values("trial_type") == tt
        if not type_mask.any():
            result[tt] = None
            continue

        tt_cols = lr_df.loc[:, type_mask]
        trial_nums = tt_cols.columns.get_level_values("trial").unique().tolist()

        sliding_traces: list[list[float]] = []

        for tnum in sorted(trial_nums):
            trial_mask = tt_cols.columns.get_level_values("trial") == tnum
            trial_data = tt_cols.loc[:, trial_mask]
            frames = trial_data.values[0]                  # 1-D lick-rate array
            frames = frames[:MAX_FRAMES].astype(float)     # cap at 300 frames
            n_frames = len(frames)

            if n_frames < window_size:
                print(f"[worker] {mouse_id}: dropped {tt} trial {tnum} "
                      f"({n_frames} frames < window_size={window_size})")
                continue

            windows = [
                float(frames[s : s + window_size].mean())
                for s in range(0, n_frames - window_size + 1, step)
            ]
            sliding_traces.append(windows)

        if not sliding_traces:
            result[tt] = None
            continue

        # Align traces to the shortest (in case trials differ slightly)
        min_len = min(len(t) for t in sliding_traces)
        arr     = np.array([t[:min_len] for t in sliding_traces])  # (n_trials, n_windows)

        mean = arr.mean(axis=0)
        sem  = arr.std(axis=0) / max(np.sqrt(len(sliding_traces)), 1.0)

        # Window-centre frame positions for the x-axis
        centres = [s + window_size // 2
                   for s in range(0, MAX_FRAMES - window_size + 1, step)][:min_len]

        if x_values is None:
            x_values = centres

        result[tt] = {
            "mean":     mean.tolist(),
            "sem":      sem.tolist(),
            "n_trials": len(sliding_traces),
        }

    if x_values is None:
        print(f"[worker] {mouse_id}: no usable trials found.")
        return None

    return {"x_values": x_values, "trial_types": result}


def _worker_main(mouse_id: str, out_path: str, window_size: int, step: int) -> None:
    data = _compute_pre_lick_trace(mouse_id, window_size, step)
    Path(out_path).write_text(json.dumps(data) if data is not None else "")


# ══════════════════════════════════════════════════════════════════════════
#  Subprocess launcher
# ══════════════════════════════════════════════════════════════════════════

def _compute_via_subprocess(mouse_id: str, window_size: int, step: int) -> dict | None:
    """Runs _worker_main in a fresh subprocess so lib.config is re-initialised
    with LIVNEH_MOUSE_ID set correctly for this session."""
    env = os.environ.copy()
    env["LIVNEH_MOUSE_ID"] = mouse_id
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        result = subprocess.run(
            [sys.executable, __file__,
             "--worker", mouse_id, out_path, str(window_size), str(step)],
            cwd=str(REPO_ROOT),
            env=env,
        )
        if result.returncode != 0:
            print(f"WARNING: {mouse_id} subprocess failed (exit {result.returncode}), skipping.")
            return None
        text = Path(out_path).read_text().strip()
        return json.loads(text) if text else None
    finally:
        Path(out_path).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  Main – collect + plot
# ══════════════════════════════════════════════════════════════════════════

def main(window_size: int = 10, step: int = 1, show: bool = True) -> None:
    import matplotlib.pyplot as plt

    pre_files = sorted(PATH.rglob("pre.npz"))
    if not pre_files:
        print(f"No pre.npz files found under {PATH}")
        return

    # Build session list from the directory tree
    sessions: list[tuple[str, Path]] = []
    for f in pre_files:
        rel = f.relative_to(PATH)
        if len(rel.parts) < 2:
            continue
        protocol, mouse_name = rel.parts[0], rel.parts[1]
        mouse_id = f"{protocol}/{mouse_name}"
        sessions.append((mouse_id, f))

    print(f"Found {len(sessions)} session(s) with pre.npz.\n")

    for mouse_id, _ in sessions:
        print(f"─── {mouse_id}")
        data = _compute_via_subprocess(mouse_id, window_size=window_size, step=step)

        if data is None:
            print(f"    No data returned, skipping.\n")
            continue

        x_values = data["x_values"]
        protocol, mouse_name = mouse_id.split("/")

        fig, ax = plt.subplots(figsize=(10, 4))

        any_plotted = False
        for tt in TRIAL_TYPES:
            tt_data = data["trial_types"].get(tt)
            if tt_data is None:
                print(f"    {tt}: no trials")
                continue

            n      = tt_data["n_trials"]
            mean   = tt_data["mean"]
            sem    = tt_data["sem"]
            color  = TRIAL_COLORS[tt]
            x      = x_values[: len(mean)]

            ax.plot(x, mean, color=color, label=f"{tt}  (n={n})", linewidth=1.8)
            ax.fill_between(
                x,
                [m - s for m, s in zip(mean, sem)],
                [m + s for m, s in zip(mean, sem)],
                color=color, alpha=0.18,
            )
            print(f"    {tt}: {n} trial(s)")
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            print(f"    Nothing to plot.\n")
            continue

        ax.set_xlim(0, MAX_FRAMES)
        ax.set_xlabel(f"Frame (0–{MAX_FRAMES})", fontsize=11)
        ax.set_ylabel("Lick rate (sliding-window mean ± SEM)", fontsize=11)
        ax.set_title(
            f"Pre-phase lick trace  –  {mouse_id}\n"
            f"window = {window_size} frames,  step = {step} frame(s)",
            fontsize=12,
        )
        ax.legend(framealpha=0.85)
        fig.tight_layout()

        # Save
        out_dir = (REPO_ROOT / "outputs" / protocol / mouse_name
                   / "exploration" / "lick_rate")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"pre_lick_trace_w{window_size}_s{step}.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"    Saved → {out_file}\n")

        if show:
            plt.show()
        else:
            plt.close(fig)

    print("Done.")


# ══════════════════════════════════════════════════════════════════════════
#  Entry-point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Internal worker mode (called by _compute_via_subprocess)
    if sys.argv[1:2] == ["--worker"]:
        # argv: script --worker mouse_id out_path window_size step
        _, _, _mouse_id, _out_path, _ws, _step = sys.argv
        _worker_main(_mouse_id, _out_path, int(_ws), int(_step))
        sys.exit(0)

    # Normal user-facing mode
    parser = argparse.ArgumentParser(
        description=(
            "Plot sliding-window lick rate in the Pre phase, split by trial type "
            "(water / nacl / airpuff), one figure per mouse/session."
        )
    )
    parser.add_argument(
        "--window-size", "-w", type=int, default=10,
        help="Sliding-window size in frames (default: 10)",
    )
    parser.add_argument(
        "--step", "-s", type=int, default=1,
        help="Step between consecutive windows in frames (default: 1)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save figures to disk without calling plt.show() (useful on headless machines)",
    )
    args = parser.parse_args()
    main(window_size=args.window_size, step=args.step, show=not args.no_show)
