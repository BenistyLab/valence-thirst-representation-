"""Run the full livneh-mice pipeline for one or more mice."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (
    Path.home()
    / "Benisty Lab Dropbox"
    / "Benisty Lab Team Folder"
    / "datasets"
    / "Ca_mice_Livne"
)

ALL_STEPS = (
    "prep",
    "dpca",
    "dpca_downstream",
    "svm_axes",
    "svm_no_dpca",
    "axes_trajectories",
    "lda",
    "trajectories",
)
STEP_SCRIPTS: dict[str, str] = {
    "prep": "prepare_data.py",
    "dpca": "run_dpca.py",
    "dpca_downstream": "run_dpca_downstream.py",
    "svm_axes": "run_svm_axes.py",
    "svm_no_dpca": "run_svm_no_dpca.py",
    "axes_trajectories": "run_axes_trajectories.py",
    "lda": "run_lda.py",
    "trajectories": "run_trajectories.py",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the livneh-mice analysis pipeline per mouse."
    )
    p.add_argument(
        "--mouse",
        nargs="+",
        metavar="ID",
        help="Session id(s) to run, as protocol/mouse (e.g. 20Hz/AL45).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all catalogued sessions under 20Hz, 1Hz, and Mock.",
    )
    p.add_argument(
        "--protocol",
        nargs="+",
        metavar="NAME",
        help="Run every session in these protocols (20Hz, 1Hz, Mock).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override LIVNEH_DATA_ROOT (default: env var or Dropbox path).",
    )
    p.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=f"Comma-separated steps to run (default: all). Choices: {', '.join(ALL_STEPS)}",
    )
    p.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip prepare_data.py (alias for omitting prep from --steps).",
    )
    return p.parse_args()


def _resolve_mice(args: argparse.Namespace) -> list[str]:
    from lib.config import KNOWN_MICE, PROTOCOLS, SESSION_PROTOCOL, resolve_mouse_id

    n_selectors = sum(bool(x) for x in (args.all, args.mouse, args.protocol))
    if n_selectors > 1:
        raise SystemExit("Use only one of --all, --mouse, or --protocol.")
    if args.all:
        mice = list(KNOWN_MICE)
    elif args.protocol:
        unknown = [p for p in args.protocol if p not in PROTOCOLS]
        if unknown:
            raise SystemExit(
                f"Unknown protocol(s): {', '.join(unknown)}. "
                f"Choices: {', '.join(PROTOCOLS)}"
            )
        wanted = set(args.protocol)
        mice = [sid for sid in KNOWN_MICE if SESSION_PROTOCOL[sid] in wanted]
        if not mice:
            raise SystemExit(f"No sessions found for protocol(s): {', '.join(args.protocol)}")
    elif args.mouse:
        mice = [resolve_mouse_id(m) for m in args.mouse]
    else:
        raise SystemExit("Specify --mouse ID [ID ...], --protocol NAME [...], or --all.")
    return mice


def _resolve_steps(args: argparse.Namespace) -> list[str]:
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise SystemExit(f"Unknown step(s): {', '.join(sorted(unknown))}")
    if args.skip_prep and "prep" in steps:
        steps = [s for s in steps if s != "prep"]
    return steps


def _resolve_data_root(args: argparse.Namespace) -> Path:
    if args.data_root is not None:
        return args.data_root.expanduser()
    env = os.environ.get("LIVNEH_DATA_ROOT")
    if env:
        return Path(env)
    if DEFAULT_DATA_ROOT.exists():
        return DEFAULT_DATA_ROOT
    raise SystemExit(
        "Set LIVNEH_DATA_ROOT or pass --data-root to your Ca_mice_Livne folder."
    )


def _run_step(script: str, env: dict[str, str]) -> None:
    cmd = [sys.executable, str(REPO_ROOT / script)]
    print(f"  > {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    args = _parse_args()
    mice = _resolve_mice(args)
    steps = _resolve_steps(args)
    data_root = _resolve_data_root(args)

    from lib.config import MOUSE_SESSION_REL, SESSION_MOUSE, SESSION_PROTOCOL, resolve_mouse_id

    print(f"Data root: {data_root}")
    print(f"Mice: {', '.join(mice)}")
    print(f"Steps: {', '.join(steps)}")
    print()

    summaries: list[tuple[str, Path, Path]] = []
    failures: list[tuple[str, str]] = []

    for mouse_id in mice:
        resolve_mouse_id(mouse_id)
        protocol = SESSION_PROTOCOL[mouse_id]
        mouse = SESSION_MOUSE[mouse_id]
        session_dir = data_root / MOUSE_SESSION_REL[mouse_id]
        if not session_dir.exists():
            raise SystemExit(f"Session data not found for {mouse_id}: {session_dir}")

        env = os.environ.copy()
        env["LIVNEH_DATA_ROOT"] = str(data_root)
        env["LIVNEH_MOUSE_ID"] = mouse_id
        env.setdefault("OMP_NUM_THREADS", "2")

        print(f"=== {mouse_id} ===")
        print(f"  session: {session_dir}")

        failed_step: str | None = None
        for step in steps:
            try:
                _run_step(STEP_SCRIPTS[step], env)
            except subprocess.CalledProcessError:
                failed_step = step
                failures.append((mouse_id, step))
                print(f"  FAILED {mouse_id} at step {step}")
                if len(mice) == 1:
                    raise
                break

        if failed_step is None:
            norm_dir = REPO_ROOT / "normalized_data" / protocol / mouse
            out_dir = REPO_ROOT / "outputs" / protocol / mouse
            summaries.append((mouse_id, norm_dir, out_dir))
        print()

    print("Done. Output locations:")
    for mouse_id, norm_dir, out_dir in summaries:
        print(f"  {mouse_id}:")
        print(f"    normalized_data: {norm_dir}")
        print(f"    outputs:         {out_dir}")
    if failures:
        print()
        print("Failed sessions:")
        for mouse_id, step in failures:
            print(f"  {mouse_id}: stopped at {step}")
        raise SystemExit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
