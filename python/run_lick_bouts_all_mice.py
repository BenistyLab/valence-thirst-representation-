"""
Run the water-trial lick-bout analysis (run_lick_bouts_one_mouse.py) for
every mouse that has a train_pool.npz under normalized_data/, then collect
all the per-mouse cap suggestions into one table.

Why this runs each mouse as its OWN subprocess rather than looping over
mice in one process and calling into lib.explore.lick_bouts directly: see
run_lick_bouts_one_mouse.py's docstring, and the create_all_npz_files fix
from before -- lib/config.py freezes several path/phase constants at import
time from whatever LIVNEH_MOUSE_ID happens to be set to right then, so an
in-process loop that just reassigns os.environ["LIVNEH_MOUSE_ID"] silently
keeps reusing the first mouse's config for every mouse after it. A fresh
subprocess per mouse (matching how run_mouse.py already handles this same
class of problem) sidesteps it completely: each subprocess's environment is
set BEFORE that process's first `import lib...`, so lib.config computes
everything correctly for that one mouse, every time.

Run from the repo root (or anywhere -- REPO_ROOT is derived from this file):
    python run_lick_bouts_all_mice.py

Writes one aggregate CSV:
    outputs/lick_bout_cap_summary_all_mice.csv
with one row per mouse: mouse_id, needed_cap, shortest_available_trial_frames,
cap_exceeds_shortest_trial, n_trials_total, n_trials_with_bout,
n_trials_with_no_bout -- plus per-mouse detail CSVs/plots under each mouse's
own outputs/<protocol>/<mouse>/exploration/lick_bouts/ (see the worker script).

Each subprocess also saves its own histogram PNG, but doesn't display it --
plt.show() would block that subprocess until you close the window, one mouse
at a time, before the next one even starts. Instead, once every mouse is
done, THIS process (which never touched matplotlib.use("Agg"), unlike the
worker) rebuilds every mouse's histogram from its saved per-trial CSV into
one grid figure and shows it all at once, interactively, at the end.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
WORKER = REPO_ROOT / "run_lick_bouts_one_mouse.py"
NORMALIZED_DIR = REPO_ROOT / "normalized_data"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def mouse_id_from_npz_path(npz_path: Path) -> str:
    """normalized_data/<protocol>/<mouse>/train_pool.npz -> "<protocol>/<mouse>",
    matching the LIVNEH_MOUSE_ID format lib.config.resolve_mouse_id expects."""
    protocol = npz_path.parent.parent.name
    mouse = npz_path.parent.name
    return f"{protocol}/{mouse}"


def show_all_histograms(files: list[Path], all_reports: pd.DataFrame | None) -> None:
    """One grid figure, one histogram subplot per mouse, shown interactively.

    Rebuilds each histogram from that mouse's own saved
    train_pool_water_bout_summary.csv (written by the worker) rather than
    re-running the analysis -- this process just plots, cheaply, after every
    mouse is already done.
    """
    cap_by_mouse = {}
    if all_reports is not None:
        cap_by_mouse = dict(zip(all_reports["mouse_id"], all_reports["needed_cap"]))

    panels = []  # (mouse_id, last_significant_lick_frame values, cap or None)
    for npz_path in files:
        protocol = npz_path.parent.parent.name
        mouse = npz_path.parent.name
        mouse_id = f"{protocol}/{mouse}"
        summary_csv = OUTPUTS_DIR / protocol / mouse / "exploration" / "lick_bouts" / "train_pool_water_bout_summary.csv"
        if not summary_csv.exists():
            continue
        vals = pd.read_csv(summary_csv)["last_significant_lick_frame"].dropna()
        panels.append((mouse_id, vals, cap_by_mouse.get(mouse_id)))

    if not panels:
        print("No per-mouse histogram data found -- nothing to show.")
        return

    n = len(panels)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, (mouse_id, vals, cap) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        if vals.empty:
            ax.text(0.5, 0.5, "no significant bouts", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(vals, bins=30)
            if cap is not None and not pd.isna(cap):
                ax.axvline(cap, color="red", linestyle="--", label=f"cap={int(cap)}")
                ax.legend(fontsize=8)
        ax.set_title(mouse_id)
        ax.set_xlabel("Last significant lick frame")
        ax.set_ylabel("Number of trials")

    # Hide any unused grid cells (e.g. 5 mice in a 2x3 grid).
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Last significant lick bout per mouse (train_pool, trial_type=water)")
    fig.tight_layout()
    plt.show()


def main() -> None:
    files = sorted(NORMALIZED_DIR.rglob("train_pool.npz"))
    if not files:
        raise SystemExit(f"No train_pool.npz found under {NORMALIZED_DIR}")

    print(f"Found {len(files)} train_pool.npz file(s) under {NORMALIZED_DIR}")

    failures: list[str] = []
    for npz_path in files:
        mouse_id = mouse_id_from_npz_path(npz_path)

        env = os.environ.copy()
        env["LIVNEH_MOUSE_ID"] = mouse_id
        env.setdefault("OMP_NUM_THREADS", "2")

        result = subprocess.run(
            [sys.executable, str(WORKER), str(npz_path)],
            cwd=REPO_ROOT,
            env=env,
        )
        if result.returncode != 0:
            failures.append(mouse_id)
            print(f"  FAILED: {mouse_id} (exit code {result.returncode})")
        print()

    # Collect each mouse's cap report (written by the worker) into one table.
    reports = []
    for npz_path in files:
        protocol = npz_path.parent.parent.name
        mouse = npz_path.parent.name
        report_csv = OUTPUTS_DIR / protocol / mouse / "exploration" / "lick_bouts" / "train_pool_water_cap_report.csv"
        if report_csv.exists():
            reports.append(pd.read_csv(report_csv))
        else:
            print(f"  (no report found for {protocol}/{mouse} -- worker likely failed)")

    if reports:
        all_reports = pd.concat(reports, ignore_index=True)
        out_path = OUTPUTS_DIR / "lick_bout_cap_summary_all_mice.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        all_reports.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")
        print(all_reports.to_string())

    show_all_histograms(files, all_reports if reports else None)

    if failures:
        print()
        print("Failed mice:", ", ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
