"""
Run the train_pool / water-trial lick-bout analysis for ONE mouse.

This is meant to be launched as its own subprocess (one per mouse) by
run_lick_bouts_all_mice.py -- see that script's docstring for why: lib/config.py
computes several module-level constants (NORMALIZED_DIR, PHASE_RUN_MAP,
EXPLORATION_OUT, ...) once, at import time, from whatever LIVNEH_MOUSE_ID is
set when it's FIRST imported in the process. Looping over mice in one process
by just reassigning os.environ["LIVNEH_MOUSE_ID"] leaves those frozen (see the
create_all_npz_files.py fix from before) -- a fresh process per mouse sidesteps
that entirely, since LIVNEH_MOUSE_ID is already set before this script's very
first `import lib...` line runs.

Usage:
    python run_lick_bouts_one_mouse.py <path-to-train_pool.npz>

LIVNEH_MOUSE_ID (and LIVNEH_DATA_ROOT) must already be set correctly in this
process's environment -- the driver script sets them before launching this.

Writes, under this mouse's own outputs/<protocol>/<mouse>/exploration/lick_bouts/:
  train_pool_water_bout_summary.csv  -- one row per water trial
  train_pool_water_cap_report.csv    -- one row: this mouse's cap suggestion
  train_pool_water_bout_hist.png     -- histogram of last significant lick frame
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")  # no display available when run as a subprocess in a loop
import matplotlib.pyplot as plt
import pandas as pd

from lib.config import EXPLORATION_OUT, resolve_mouse_id
from lib.explore.lick_bouts import lick_bout_summary, suggest_cap

TRIAL_TYPE = "water"
MIN_LICKS = 3
WINDOW = 10


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python run_lick_bouts_one_mouse.py <path-to-train_pool.npz>")
    npz_path = Path(sys.argv[1])

    mouse_id = resolve_mouse_id()  # reads LIVNEH_MOUSE_ID from this process's env
    print(f"=== {mouse_id} ({npz_path}) ===")

    full_summary = lick_bout_summary(str(npz_path), "train_pool", min_licks=MIN_LICKS, window=WINDOW)
    summary = full_summary[full_summary["trial_type"] == TRIAL_TYPE].reset_index(drop=True)
    summary.insert(0, "mouse_id", mouse_id)

    out_dir = EXPLORATION_OUT / "lick_bouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "train_pool_water_bout_summary.csv", index=False)

    n_no_bout = int(summary["last_significant_lick_frame"].isna().sum())
    print(f"  {len(summary)} '{TRIAL_TYPE}' trials, {n_no_bout} with no significant bout")

    try:
        report = suggest_cap(summary, percentile=100)
    except ValueError as e:
        print(f"  no cap suggestion: {e}")
        report = {
            "mouse_id": mouse_id,
            "needed_cap": None,
            "shortest_available_trial_frames": int(summary["n_frames"].min()) if len(summary) else None,
            "cap_exceeds_shortest_trial": None,
            "n_trials_total": len(summary),
            "n_trials_with_bout": 0,
            "n_trials_with_no_bout": n_no_bout,
            "percentile_used": 100,
        }
    else:
        report = {"mouse_id": mouse_id, **report}
        print(f"  {report}")

        plt.figure()
        plt.hist(summary["last_significant_lick_frame"].dropna(), bins=30)
        plt.axvline(report["needed_cap"], color="red", linestyle="--", label=f"cap={report['needed_cap']}")
        plt.xlabel("Last significant lick frame (relative to trial start)")
        plt.ylabel("Number of trials")
        plt.title(f"{mouse_id} train_pool, trial_type={TRIAL_TYPE} (min_licks={MIN_LICKS}, window={WINDOW})")
        plt.legend()
        plt.savefig(out_dir / "train_pool_water_bout_hist.png", dpi=150, bbox_inches="tight")
        plt.close()

    pd.DataFrame([report]).to_csv(out_dir / "train_pool_water_cap_report.csv", index=False)


if __name__ == "__main__":
    main()
