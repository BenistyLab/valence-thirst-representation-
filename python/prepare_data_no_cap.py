"""
Run Avigail's prepare_data pipeline with the per-trial frame cap effectively
disabled, WITHOUT editing lib/prep/prepare_data.py.

Why this works without touching her file: lib/prep/prepare_data.py defines a
module-level constant `DEFAULT_CAP = 500` and its main() reads that name as a
global at call time (not as a bound default argument), e.g.:
    L = frames_from_cue(cue_start[i], trial_end[i], n_frames_total, DEFAULT_CAP)
    pooled = _build_pack(..., cap=DEFAULT_CAP, ...)
    eval_specs = {"pre": (DEFAULT_CAP, ...), ...}
So overriding `lib.prep.prepare_data.DEFAULT_CAP` on the module object BEFORE
calling main() changes what every one of those call sites sees, exactly as if
you'd edited the constant yourself -- but the file on disk is untouched.

frames_from_cue() does `min(cap, trial_end - cue_start + 1)`, so setting cap to
a number far larger than any real trial (in frames) means every trial keeps
its FULL length from cue onset to trial end, bounded only by the recording
itself. That's "no cap" in effect.

Usage (same env vars as her normal pipeline):
    $env:LIVNEH_DATA_ROOT = "C:\path\to\Ca_mice_Livne"
    $env:LIVNEH_MOUSE_ID = "20Hz/AL45"      # or whichever session
    python prepare_data_no_cap.py

This writes the same train_pool.npz / pre.npz / airpuff.npz / water.npz files
prepare_data.py normally would, just with full-length trials instead of
500-frame-capped ones. The z-scoring (mean/std pooled across the training
runs, excluding SLM) is unchanged -- only the trial length cap is affected.

Note: each saved pack's `cap` field will just record whatever large number we
used here (metadata only) -- it's not a meaningful frame count, just a flag
that no real cap was applied.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib.prep.prepare_data_score_per_run as prepare_data

# Any recording length works: uncapped trials are still bounded by however
# many frames the session actually recorded, so a huge sentinel is safe.
NO_CAP_SENTINEL = 10_000_000
prepare_data.DEFAULT_CAP = NO_CAP_SENTINEL

if __name__ == "__main__":
    prepare_data.main()
