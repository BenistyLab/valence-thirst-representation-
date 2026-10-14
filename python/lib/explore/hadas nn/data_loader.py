"""Load calcium-imaging trials and group runs using the MATLAB loadMouse rules.

MATLAB input files are expected to be v7.3 MAT files (HDF5). Arrays returned
by this module use Python conventions: activity is cells x time x trials,
trial metadata are one-dimensional, and sample indices are zero-based only
inside the loader. The extracted window is equivalent to MATLAB's
``dfFvalid(:, st:st+minDuration-1)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import h5py
import numpy as np


@dataclass
class Run:
    """Trials from one raw acquisition run or a MATLAB-style united group."""

    name: str
    type: int | tuple[int, ...]
    duration: int
    X: np.ndarray                         # cells x time x trials
    condition: np.ndarray
    cue_offset: np.ndarray
    lick_fr: np.ndarray                   # time x trials
    trial_index: np.ndarray               # zero-based row in frame_cond_trial
    slm_target_intent: str = "none"
    water_op: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    salt: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    water_free: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    waters: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    airpuff: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    slm_high: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))


def _require_hdf5(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not h5py.is_hdf5(path):
        raise ValueError(f"{path} is not a MATLAB v7.3/HDF5 file")


def _matlab_char(h5: h5py.File, ref) -> str:
    if not ref:
        return ""
    values = np.asarray(h5[ref]).reshape(-1)
    return "".join(chr(int(v)) for v in values if int(v) != 0)


def _matlab_ref_value(h5: h5py.File, ref) -> np.ndarray:
    if not ref:
        return np.asarray([])
    return np.asarray(h5[ref])


def _load_condition_table(h5: h5py.File) -> dict[int, str]:
    refs = h5["exp_cond"]
    # MATLAB stores cell arrays column-major; HDF5 exposes the dimensions reversed.
    refs = np.asarray(refs).T
    result: dict[int, str] = {}
    for condition_ref, name_ref in refs:
        condition_id = int(np.asarray(_matlab_ref_value(h5, condition_ref)).reshape(-1)[0])
        result[condition_id] = _matlab_char(h5, name_ref)
    return result


def _load_slm_runs(h5: h5py.File) -> dict[int, tuple[str, str]]:
    """Return raw run -> (intent, XML source); intent comes from MATLAB's XML name."""
    runs = h5["slm_info/runs"]
    run_refs = np.asarray(runs["run_number"]).reshape(-1)
    xml_refs = np.asarray(runs["xml_source"]).reshape(-1)
    found: dict[int, tuple[str, str]] = {}
    for run_ref, xml_ref in zip(run_refs, xml_refs):
        run_number = int(np.asarray(_matlab_ref_value(h5, run_ref)).reshape(-1)[0])
        xml_source = _matlab_char(h5, xml_ref)
        match = re.search(r"SLM_cell_(airpuff|water)_", Path(xml_source).name, re.IGNORECASE)
        if match is None:
            raise ValueError(
                f"Cannot infer SLM target intent for run {run_number} from XML path: {xml_source}"
            )
        found[run_number] = (match.group(1).lower(), xml_source)
    if not found:
        raise ValueError("exp_data.mat has no slm_info.runs entries")
    return found


def _empty_run(name: str, raw_run_numbers: tuple[int, ...], duration: int,
               n_cells: int, dtype) -> Run:
    return Run(name=name, type=raw_run_numbers, duration=duration,
               X=np.empty((n_cells, duration, 0), dtype=dtype),
               condition=np.empty(0, dtype=np.int64),
               cue_offset=np.empty(0, dtype=np.int64),
               lick_fr=np.empty((duration, 0), dtype=np.float32),
               trial_index=np.empty(0, dtype=np.int64))


def _unite(runs: list[Run], name: str, duration: int, n_cells: int, dtype) -> Run:
    if not runs:
        return _empty_run(name, (), duration, n_cells, dtype)
    duration = min(duration, *(run.duration for run in runs))
    raw_types = tuple(dict.fromkeys(
        raw for run in runs for raw in (run.type if isinstance(run.type, tuple) else (run.type,))
    ))
    def cat(attr: str, axis: int = 0):
        arrays = [getattr(run, attr) for run in runs]
        return np.concatenate(arrays, axis=axis) if arrays else np.empty(0)
    return Run(
        name=name,
        type=raw_types,
        duration=duration,
        X=np.concatenate([run.X[:, :duration, :] for run in runs], axis=2),
        condition=cat("condition"),
        cue_offset=cat("cue_offset"),
        lick_fr=np.concatenate([run.lick_fr[:duration, :] for run in runs], axis=1),
        trial_index=cat("trial_index"),
        slm_target_intent=runs[-1].slm_target_intent,
        water_op=cat("water_op"), salt=cat("salt"),
        water_free=cat("water_free"), waters=cat("waters"),
        airpuff=cat("airpuff"), slm_high=cat("slm_high"),
    )


def load_mouse(session_dir: str | Path, min_duration: int = 300,
               to_zscore: bool = False) -> tuple[list[Run], dict[int, str]]:
    """Load a mouse session and form the same groups as MATLAB ``loadMouse``.

    Returns runs in MATLAB order: combined pre-SLM, combined airpuff-SLM,
    combined water-SLM, each raw run after the last water-SLM run, and all
    retained raw runs combined. As in MATLAB, the first raw run is removed
    when it has fewer than 10 duration-qualified trials.
    """
    if min_duration <= 0:
        raise ValueError("min_duration must be positive")
    session_dir = Path(session_dir)
    exp_path = session_dir / "exp_data.mat"
    activity_path = session_dir / "motion_corrected_dfFvalid.mat"
    _require_hdf5(exp_path)
    _require_hdf5(activity_path)

    with h5py.File(exp_path, "r") as exp:
        for key in ("frame_cond_trial", "trialRunMap", "exp_cond", "slm_info/runs"):
            if key not in exp:
                raise KeyError(f"{exp_path} is missing {key}")
        frame_cond_trial = np.asarray(exp["frame_cond_trial"]).T
        trial_run = np.asarray(exp["trialRunMap"]).reshape(-1).astype(np.int64)
        condition_names = _load_condition_table(exp)
        slm_runs = _load_slm_runs(exp)
        lick_fr = np.asarray(exp["lickFr"]).reshape(-1)

    with h5py.File(activity_path, "r") as activity:
        if "dfFvalid" not in activity:
            raise KeyError(f"{activity_path} is missing dfFvalid")
        df_node = activity["dfFvalid"]
        # MATLAB/HDF5 reverses dimensions relative to the MATLAB cells x time array.
        n_frames, n_cells = df_node.shape
        df_time_by_cell = df_node

        n_trials = frame_cond_trial.shape[0]
        if frame_cond_trial.shape[1] < 4:
            raise ValueError("frame_cond_trial must have at least four columns")
        if trial_run.size != n_trials:
            raise ValueError(
                f"trialRunMap has {trial_run.size} rows but frame_cond_trial has {n_trials}"
            )
        if lick_fr.size < n_frames:
            raise ValueError(f"lickFr has {lick_fr.size} samples but dfFvalid has {n_frames}")

        cond = frame_cond_trial[:, 0].astype(np.int64)
        starts = frame_cond_trial[:, 1].astype(np.int64)  # MATLAB one-based start samples
        cue_offsets = frame_cond_trial[:, 2].astype(np.int64)
        ends = frame_cond_trial[:, 3].astype(np.int64)
        raw_run_numbers = np.unique(trial_run)
        raw_runs: list[Run] = []

        cond_id_by_name = {name: cid for cid, name in condition_names.items()}
        indicators = {
            "water_op": cond_id_by_name.get("Water-Op"),
            "salt": cond_id_by_name.get("NaCl"),
            "water_free": cond_id_by_name.get("water-free"),
            "waters": cond_id_by_name.get("water(s)"),
            "airpuff": cond_id_by_name.get("Pav-Airpuff"),
            "slm_high": cond_id_by_name.get("pre-cue-water-SLMhigh(pav)"),
        }

        for raw_run in raw_run_numbers:
            candidates = np.flatnonzero(trial_run == raw_run)
            durations = ends[candidates] - starts[candidates] - 1
            selected = candidates[durations >= min_duration]
            # MATLAB indexing: st:st+minDuration-1 -> Python [st-1:st-1+minDuration].
            selected = selected[(starts[selected] >= 1) &
                                (starts[selected] - 1 + min_duration <= n_frames)]
            x = np.empty((n_cells, min_duration, selected.size), dtype=np.float32)
            lick = np.empty((min_duration, selected.size), dtype=np.float32)
            for j, trial_index in enumerate(selected):
                first = int(starts[trial_index]) - 1
                last = first + min_duration
                x[:, :, j] = np.asarray(df_time_by_cell[first:last, :], dtype=np.float32).T
                lick[:, j] = lick_fr[first:last]
            run = Run(
                name=f"run_{int(raw_run)}", type=int(raw_run), duration=min_duration,
                X=x, condition=cond[selected], cue_offset=cue_offsets[selected],
                lick_fr=lick, trial_index=selected.astype(np.int64),
                slm_target_intent=slm_runs.get(int(raw_run), ("none", ""))[0],
            )
            for field_name, condition_id in indicators.items():
                setattr(run, field_name,
                        np.zeros(selected.size, dtype=bool) if condition_id is None
                        else run.condition == condition_id)
            raw_runs.append(run)

    if raw_runs and raw_runs[0].X.shape[2] < 10:
        raw_runs = raw_runs[1:]
    if not raw_runs:
        return [], condition_names

    first_slm = next((i for i, run in enumerate(raw_runs)
                      if run.slm_target_intent != "none"), len(raw_runs))
    pre = _unite(raw_runs[:first_slm], "pre", min_duration, n_cells, np.float32)
    airpuff = _unite([r for r in raw_runs if r.slm_target_intent == "airpuff"],
                     "slm_airpuff", min_duration, n_cells, np.float32)
    water = _unite([r for r in raw_runs if r.slm_target_intent == "water"],
                   "slm_water", min_duration, n_cells, np.float32)

    slm_water_indices = [i for i, r in enumerate(raw_runs)
                         if r.slm_target_intent == "water"]
    tail = raw_runs[slm_water_indices[-1] + 1:] if slm_water_indices else []
    groups = [pre, airpuff, water]
    groups.extend(tail)
    groups.append(_unite(raw_runs, "all", min_duration, n_cells, np.float32))

    # Keep only nonempty groups; MATLAB's intended ``uniteRuns`` cannot combine
    # an empty index set, so omitting absent groups makes the Python result usable.
    groups = [run for run in groups if run.X.shape[2] > 0]
    if to_zscore:
        for run in groups:
            mean = run.X.mean(axis=(1, 2), keepdims=True)
            std = run.X.std(axis=(1, 2), keepdims=True, ddof=1)
            std[std == 0] = 1
            run.X = (run.X - mean) / std
    return groups, condition_names
