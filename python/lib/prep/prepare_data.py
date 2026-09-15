"""
Prepare normalized NPZ packs for analysis.

Training pool:
  pre + airpuff + water phases, excluding SLM trials.

Eval packs:
  pre, airpuff, water.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.config import NORMALIZED_DIR, PHASE_RUN_MAP, SLM_COND_ID, default_session_dir
from lib.io.raw import load_dfFvalid, load_exp_data

DEFAULT_CAP = 500
EPS = 1e-8
OUT_DIR = NORMALIZED_DIR
REPORT_PATH = NORMALIZED_DIR / "prep_report.txt"


@dataclass
class Pack:
    key: str
    cap: int
    concat: np.ndarray
    segments: list[np.ndarray]
    segment_lengths: np.ndarray
    trial_idx: np.ndarray
    trial_type: np.ndarray
    trial_run: np.ndarray
    avail_from_cue: np.ndarray


def _parse_trials(exp: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fct = np.asarray(exp["frame_cond_trial"])
    if fct.ndim == 1:
        fct = fct.reshape(-1, 4)
    trm = np.asarray(exp["trialRunMap"]).reshape(-1)
    n_trials = fct.shape[0]
    if trm.size < n_trials:
        raise ValueError("trialRunMap shorter than trials")
    trial_run = trm[:n_trials].astype(int)
    cue_start = fct[:, 1].astype(int)
    trial_end = fct[:, 3].astype(int)
    trial_type = fct[:, 0].astype(int)
    return trial_type, trial_run, cue_start, trial_end


def frames_from_cue(cue_start: int, trial_end: int, n_frames_total: int, cap: int) -> int:
    cs = int(cue_start)
    te = int(trial_end)
    if cs < 0 or cs >= n_frames_total:
        return 0
    te2 = min(te, n_frames_total - 1)
    if te2 < cs:
        return 0
    return min(cap, te2 - cs + 1)


def avail_full_from_cue(cue_start: int, trial_end: int, n_frames_total: int) -> int:
    cs = int(cue_start)
    te = int(trial_end)
    if cs < 0 or cs >= n_frames_total:
        return 0
    te2 = min(te, n_frames_total - 1)
    if te2 < cs:
        return 0
    return te2 - cs + 1


def _is_train_trial(run_i: int, cond_i: int, phase_runs: dict[str, tuple[int, ...]]) -> bool:
    train_runs = {
        int(r)
        for phase in ("pre", "airpuff", "water")
        for r in phase_runs.get(phase, ())
    }
    if int(run_i) not in train_runs:
        return False
    return int(cond_i) != SLM_COND_ID


def _fit_mu_sigma(dfF: np.ndarray, trial_sel: list[tuple[int, int]], cue_start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not trial_sel:
        raise ValueError("No trials selected for z-score fit.")
    chunks = [dfF[:, int(cue_start[i]) : int(cue_start[i]) + L] for i, L in trial_sel]
    big = np.concatenate(chunks, axis=1)
    mu = big.mean(axis=1).astype(np.float64)
    sig = np.maximum(big.std(axis=1), EPS).astype(np.float64)
    return mu, sig


def _build_pack(
    key: str,
    cap: int,
    dfF: np.ndarray,
    trial_type: np.ndarray,
    trial_run: np.ndarray,
    cue_start: np.ndarray,
    trial_end: np.ndarray,
    n_frames_total: int,
    keep_pred,
    mu: np.ndarray,
    sig: np.ndarray,
) -> Pack | None:
    picked: list[tuple[int, int]] = []
    for i in range(len(trial_run)):
        if not keep_pred(int(trial_run[i]), int(trial_type[i])):
            continue
        L = frames_from_cue(cue_start[i], trial_end[i], n_frames_total, cap)
        if L >= 1:
            picked.append((i, L))
    if not picked:
        return None

    segments = []
    trial_idx = []
    seg_lens = []
    avails = []
    for i, L in picked:
        cs = int(cue_start[i])
        raw = dfF[:, cs : cs + L]
        segments.append(((raw - mu[:, None]) / sig[:, None]).astype(np.float32))
        trial_idx.append(i)
        seg_lens.append(L)
        avails.append(avail_full_from_cue(cue_start[i], trial_end[i], n_frames_total))

    concat = np.concatenate([s.astype(np.float64) for s in segments], axis=1)
    trial_idx_arr = np.array(trial_idx, dtype=np.int64)
    return Pack(
        key=key,
        cap=cap,
        concat=concat,
        segments=segments,
        segment_lengths=np.array(seg_lens, dtype=np.int32),
        trial_idx=trial_idx_arr,
        trial_type=trial_type[trial_idx_arr].astype(np.int64),
        trial_run=trial_run[trial_idx_arr].astype(np.int64),
        avail_from_cue=np.array(avails, dtype=np.int32),
    )


def _save_pack(pack: Pack) -> None:
    seg_obj = np.empty(len(pack.segments), dtype=object)
    for j, s in enumerate(pack.segments):
        seg_obj[j] = s
    out_npz = OUT_DIR / f"{pack.key}.npz"
    np.savez_compressed(
        out_npz,
        concat=pack.concat.astype(np.float32),
        trial_idx=pack.trial_idx,
        trial_type=pack.trial_type,
        trial_run=pack.trial_run,
        segment_lengths=pack.segment_lengths,
        avail_from_cue=pack.avail_from_cue,
        cap=np.array([pack.cap]),
        segments=seg_obj,
    )
    print(f"Saved {out_npz} ({len(pack.segments)} trials, cap={pack.cap})")


def _count_trials_for_runs(
    trial_run: np.ndarray,
    trial_type: np.ndarray,
    cue_start: np.ndarray,
    trial_end: np.ndarray,
    n_frames_total: int,
    runs: tuple[int, ...],
    cap: int,
    exclude_slm: bool,
) -> int:
    count = 0
    for i in range(len(trial_run)):
        if int(trial_run[i]) not in runs:
            continue
        if exclude_slm and int(trial_type[i]) == SLM_COND_ID:
            continue
        if frames_from_cue(cue_start[i], trial_end[i], n_frames_total, cap) >= 1:
            count += 1
    return count


def main(session_dir: Path | None = None) -> None:
    session_dir = Path(session_dir or default_session_dir())
    phase_runs = PHASE_RUN_MAP
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dfF = load_dfFvalid(session_dir)
    exp = load_exp_data(session_dir)
    trial_type, trial_run, cue_start, trial_end = _parse_trials(exp)
    n_frames_total = dfF.shape[1]

    train_sel = []
    for i in range(len(trial_run)):
        rr = int(trial_run[i])
        tt = int(trial_type[i])
        if not _is_train_trial(rr, tt, phase_runs):
            continue
        L = frames_from_cue(cue_start[i], trial_end[i], n_frames_total, DEFAULT_CAP)
        if L >= 1:
            train_sel.append((i, L))
    mu, sig = _fit_mu_sigma(dfF, train_sel, cue_start)

    pooled = _build_pack(
        key="train_pool",
        cap=DEFAULT_CAP,
        dfF=dfF,
        trial_type=trial_type,
        trial_run=trial_run,
        cue_start=cue_start,
        trial_end=trial_end,
        n_frames_total=n_frames_total,
        keep_pred=lambda r, t: _is_train_trial(r, t, phase_runs),
        mu=mu,
        sig=sig,
    )
    if pooled is None:
        raise RuntimeError("No trials in training pool.")
    _save_pack(pooled)

    pre_runs = set(int(x) for x in phase_runs.get("pre", ()))
    airpuff_runs = set(int(x) for x in phase_runs.get("airpuff", ()))
    water_runs = set(int(x) for x in phase_runs.get("water", ()))
    eval_specs = {
        "pre": (DEFAULT_CAP, lambda r, t: int(r) in pre_runs),
        "airpuff": (DEFAULT_CAP, lambda r, t: int(r) in airpuff_runs),
        "water": (DEFAULT_CAP, lambda r, t: int(r) in water_runs),
    }

    report = []
    report.append("livneh-mice data prep report")
    report.append(f"SLM exclusion condition id: {SLM_COND_ID}")
    report.append("")
    report.append(f"train_pool trials: {len(pooled.segments)}")
    report.append(f"train_pool run counts: {dict(zip(*np.unique(pooled.trial_run, return_counts=True)))}")
    report.append("")
    report.append("training-vs-overall trial counts (cap=500, >=1 frame from cue):")
    for phase, runs in (("pre", pre_runs), ("airpuff", airpuff_runs), ("water", water_runs)):
        if not runs:
            continue
        used = _count_trials_for_runs(
            trial_run, trial_type, cue_start, trial_end, n_frames_total, tuple(sorted(runs)), DEFAULT_CAP, exclude_slm=True
        )
        total = _count_trials_for_runs(
            trial_run, trial_type, cue_start, trial_end, n_frames_total, tuple(sorted(runs)), DEFAULT_CAP, exclude_slm=False
        )
        report.append(
            f"  {phase}: runs={tuple(sorted(runs))}, used_for_training_no_slm={used}, overall_including_slm={total}"
        )
    report.append("")

    for key, (cap, pred) in eval_specs.items():
        pack = _build_pack(
            key=key,
            cap=cap,
            dfF=dfF,
            trial_type=trial_type,
            trial_run=trial_run,
            cue_start=cue_start,
            trial_end=trial_end,
            n_frames_total=n_frames_total,
            keep_pred=pred,
            mu=mu,
            sig=sig,
        )
        if pack is None:
            report.append(f"{key}: no trials found, skipped")
            continue
        _save_pack(pack)
        report.append(f"{key}: {len(pack.segments)} trials, cap={cap}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report))
    print(f"Wrote {REPORT_PATH}")
