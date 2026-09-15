"""
Run prepare_data_no_cap.py (the uncapped, per-phase z-scored NPZ builder) for
every mouse in ALL_MICE, writing each mouse's train_pool/pre/water/airpuff.npz
into ITS OWN normalized_data/<protocol>/<mouse>/ folder.

Why the original create_all_npz_files.py only ever wrote one mouse's data:
lib/config.py computes several module-level constants ONCE, at import time,
by calling resolve_mouse_id() right then -- e.g.:

    _mouse_id = resolve_mouse_id()
    PHASE_RUN_MAP = phase_run_map(_mouse_id)
    NORMALIZED_DIR = REPO_ROOT / "normalized_data" / _protocol / _mouse
    OUTPUTS_DIR = ...

lib/prep/prepare_data_score_per_run.py then does
`from lib.config import NORMALIZED_DIR, PHASE_RUN_MAP, ...` at its own
top level, which copies those values into ITS OWN module namespace once.
`importlib.reload(prepare_data)` re-runs that import line, but Python's
import system serves lib.config back from sys.modules (its cache) -- since
lib.config itself was never reloaded, that "from" import just re-fetches the
exact same frozen values every time, no matter what LIVNEH_MOUSE_ID is set to
in os.environ afterwards. So every mouse's dfF/exp_data loaded correctly
(default_session_dir() DOES read the env var live, since it calls
resolve_mouse_id() at call time, not at import time) -- but every mouse's
output got written to the SAME first-imported mouse's NORMALIZED_DIR folder,
overwriting the same four .npz files each time, and every mouse's trial/run
selection used that first mouse's PHASE_RUN_MAP too. Reloading just
`prepare_data` (as the original script tried) can't fix this: it needs
lib.config reloaded FIRST, in the same process, for prepare_data's re-import
of those names to pick up anything different.

Rather than chase every module-level constant that would need this treatment
(and re-broken again the next time a new frozen constant is added upstream),
this uses the same pattern lib/run_mouse.py already uses for exactly this
multi-mouse situation: run each mouse in its OWN fresh subprocess, with
LIVNEH_MOUSE_ID set in that subprocess's environment before Python even
starts. A fresh process means lib.config computes its frozen constants
correctly for that one mouse, every time -- no reload bookkeeping needed.

Nothing in lib/ is changed; this only calls prepare_data_no_cap.py (already a
separate, non-Avigail entry-point script) once per mouse.
"""
import os
import subprocess
import sys
from pathlib import Path

# REPO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = r"C:\Users\neely.heller\PycharmProjects\valence-thirst-representation-\python"

# SCRIPT = REPO_ROOT / "prepare_data_no_cap.py"
# this calls prepare_data_no_cap but you can change it or change the cap
SCRIPT = r"C:\Users\neely.heller\PycharmProjects\valence-thirst-representation-\python\prepare_data_no_cap.py"

ALL_MICE_MOCK = ['Mock/AL42', 'Mock/AL48', 'Mock/AL49_1', 'Mock/AL49_2']
ALL_MICE_20HZ = ['20Hz/AL41', '20Hz/AL42', '20Hz/AL45', '20Hz/AL48', '20Hz/AL49_1', '20Hz/AL49_2']
ALL_MICE_1HZ = ['1Hz/AL41', '1Hz/AL42', '1Hz/AL45', '1Hz/AL48', '1Hz/AL49_1']
ALL_MICE = ALL_MICE_20HZ + ALL_MICE_1HZ + ALL_MICE_MOCK

failures = []

for mouse in ALL_MICE:
    env = os.environ.copy()
    env["LIVNEH_MOUSE_ID"] = mouse
    env.setdefault("OMP_NUM_THREADS", "2")

    print(f"=== {mouse} ===")
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
    )
    if result.returncode != 0:
        failures.append(mouse)
        print(f"  FAILED: {mouse} (exit code {result.returncode})")
    print()

print("end")
if failures:
    print("Failed mice:", ", ".join(failures))
