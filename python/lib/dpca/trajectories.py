"""
Representation trajectories and stability.

Single-trial path (concatenated cue-aligned frames, no padding):
  NPZ segments -> X_flat (sum_i T_i, n_neurons) + row metadata ->
  StandardScaler fit on X_flat -> global PCA and projection onto pooled
  four-class LDA loadings (CSV).

Trial-average path (parallel):
  Mean tensor (n_conditions, T_common, n_neurons) with T_common = min segment
  length over kept trials -> z-score across (condition, time) ->
  demixed PCA (dPCA; machenslab / PyPI `dpca`) on shape (n_neurons, n_conditions, T).
  With ``regularizer='auto'``, trial-by-trial ``trialX`` is passed (required by dPCA for CV);
  ``protect=['t']`` is set so shuffling respects time within trials. Use ``--dpca-regularizer none``
  for a faster unregularized fit without trial tensors.
  Figures under ``dpca/<task>/``: ``marginal_t_*.png``, ``marginal_s_*.png``, ``traj_st_3d.png``,
  ``traj_st_2d_panels.png``.

  Joint valence × consumption (``dpca/valence_x_consumption/``): one shared dPCA
  with four mean trajectories — Aversive, Appetitive, NonConsumption, Consumption
  (binary class means from each task, plotted together).

Install (not run by this repo automatically):
  pip install dpca numexpr

The estimator class lives in submodule ``dPCA.dPCA``; use
``from dPCA.dPCA import dPCA`` (``from dPCA import dPCA`` imports the module, not the class).

Tensor axes for dPCA (machenslab implementation):
  X.shape = (n_neurons, n_features_1, n_features_2, ...) matching labels string
  order; here labels='st' -> (n_neurons, n_stimulus, n_time).

LDA loadings CSVs may only expose LD1 and LD2; the third trajectory coordinate
uses cue-relative frame index (scaled) so 3D plots remain readable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lib.lda.decode import build_binary_labels
from lib.io.npz import load_npz_pack
from lib.tasks import TASK_SPECS, cond_name_map as _cond_name_map
from lib.config import EVAL_RUN_KEYS, LDA_FOUR_CLASS_DIR, SLM_COND_ID, TRAJECTORIES_OUT

# Four binary-class means in one dPCA: valence pair + consumption pair.
# (name, task_key, class_id, color, linestyle)
JOINT_BINARY_LEVELS: tuple[tuple[str, str, int, str, str], ...] = (
    ("Aversive", "hedonic_valence", 0, "#b22222", "-"),
    ("Appetitive", "hedonic_valence", 1, "#0b3d91", "-"),
    ("NonConsumption", "consumption_mode", 0, "#2e8b57", "--"),
    ("Consumption", "consumption_mode", 1, "#9467bd", "--"),
)
JOINT_TASK_KEY = "valence_x_consumption"


def _is_train_trial(run_i: int, cond_i: int) -> bool:
    """Match prepare_data: exclude SLM trials from semantic phase training pool."""
    return int(cond_i) != SLM_COND_ID


def trial_keep_train_pool(trial_run: int, trial_type: int) -> bool:
    if int(trial_run) in (2, 3, 4, 5):
        return _is_train_trial(int(trial_run), int(trial_type))
    return True


def _load_lda_weight_matrix(loadings_path: Path, n_neurons: int) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(loadings_path)
    cols = [c for c in df.columns if c.startswith("lda_ld") and c.endswith("_loading")]
    cols = sorted(cols, key=lambda c: int(c.replace("lda_ld", "").replace("_loading", "")))
    if not cols:
        raise ValueError(f"No lda_ld*_loading columns in {loadings_path}")
    idx = df["neuron_index"].to_numpy(dtype=np.int64)
    W = np.zeros((n_neurons, len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        v = df[c].to_numpy(dtype=np.float64)
        for i in range(len(idx)):
            ii = int(idx[i])
            if 0 <= ii < n_neurons:
                W[ii, j] = v[i]
    return W, cols


def _build_flat_and_meta(
    run_keys: list[str],
    task_key: str,
    exclude_slm_train_pool: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """Return X_flat, run_id, global_trial_ix, frame_t, y_bin, trial_run, trial_type, n_neurons."""
    rows_x: list[np.ndarray] = []
    rows_run: list[int] = []
    rows_trial: list[int] = []
    rows_t: list[int] = []
    rows_y: list[int] = []
    rows_trun: list[int] = []
    rows_ttp: list[int] = []
    global_trial = 0
    n_neurons = 0

    for rk in run_keys:
        pack = load_npz_pack(rk)
        n_neurons = int(pack["n_neurons"])
        y_cond = pack["trial_type"]
        tr = pack["trial_run"]
        cmap = _cond_name_map(rk)
        yb, keep = build_binary_labels(y_cond, cmap, task_key)

        for local_i, seg in enumerate(pack["segments"]):
            if exclude_slm_train_pool and not trial_keep_train_pool(int(tr[local_i]), int(y_cond[local_i])):
                continue
            if not bool(keep[local_i]):
                continue
            segf = np.asarray(seg, dtype=np.float64)
            if segf.ndim != 2 or segf.shape[0] != n_neurons:
                continue
            T = segf.shape[1]
            if T < 1:
                continue
            for t in range(T):
                rows_x.append(segf[:, t])
                rows_run.append(run_keys.index(rk))
                rows_trial.append(global_trial)
                rows_t.append(t)
                rows_y.append(int(yb[local_i]))
                rows_trun.append(int(tr[local_i]))
                rows_ttp.append(int(y_cond[local_i]))
            global_trial += 1

    if not rows_x:
        raise RuntimeError(f"No rows assembled for task={task_key}, runs={run_keys}.")

    X_flat = np.vstack(rows_x)
    return (
        X_flat,
        np.asarray(rows_run, dtype=np.int64),
        np.asarray(rows_trial, dtype=np.int64),
        np.asarray(rows_t, dtype=np.int64),
        np.asarray(rows_y, dtype=np.int64),
        np.asarray(rows_trun, dtype=np.int64),
        np.asarray(rows_ttp, dtype=np.int64),
        n_neurons,
    )


def _trial_lengths_from_meta(trial_ix: np.ndarray, frame_t: np.ndarray) -> np.ndarray:
    """Per global trial index, number of frames (max t + 1)."""
    if trial_ix.size == 0:
        return np.array([], dtype=np.int64)
    n_tr = int(trial_ix.max()) + 1
    lens = np.zeros(n_tr, dtype=np.int64)
    for i in range(trial_ix.size):
        ti = int(trial_ix[i])
        lens[ti] = max(lens[ti], int(frame_t[i]) + 1)
    return lens


def _collect_segs_by_class(
    run_keys: list[str],
    task_key: str,
    exclude_slm_train_pool: bool,
) -> tuple[dict[int, list[np.ndarray]], int, int]:
    """Binary task segments per class; T_common = min segment length."""
    segs_by_class: dict[int, list[np.ndarray]] = {0: [], 1: []}
    min_T: int | None = None
    n_neurons: int | None = None

    for rk in run_keys:
        pack = load_npz_pack(rk)
        nn = int(pack["n_neurons"])
        if n_neurons is None:
            n_neurons = nn
        elif nn != n_neurons:
            raise ValueError(f"Inconsistent n_neurons: {n_neurons} vs {nn} in {rk}")
        y_cond = pack["trial_type"]
        tr = pack["trial_run"]
        cmap = _cond_name_map(rk)
        yb, keep = build_binary_labels(y_cond, cmap, task_key)

        for local_i, seg in enumerate(pack["segments"]):
            if exclude_slm_train_pool and not trial_keep_train_pool(int(tr[local_i]), int(y_cond[local_i])):
                continue
            if not bool(keep[local_i]):
                continue
            segf = np.asarray(seg, dtype=np.float64)
            if segf.shape[0] != n_neurons:
                continue
            T = segf.shape[1]
            if T < 1:
                continue
            cls = int(yb[local_i])
            segs_by_class[cls].append(segf)
            min_T = T if min_T is None else min(min_T, T)

    if min_T is None or min_T < 2 or n_neurons is None:
        raise RuntimeError("dPCA: insufficient time samples or no trials after filtering.")
    for cls in (0, 1):
        if not segs_by_class[cls]:
            raise RuntimeError(f"dPCA: no trials for class {cls} ({task_key}).")

    return segs_by_class, min_T, n_neurons


def _from_segs_build_mean_trialx(
    segs_by_class: dict[int, list[np.ndarray]],
    min_T: int,
    neuron_ix: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]] | None:
    """trialX (n_trials_max, n_sub, 2, T), X_mean (n_sub, 2, T). neuron_ix=None -> all neurons."""
    segs0 = segs_by_class[0]
    segs1 = segs_by_class[1]
    if not segs0 or not segs1:
        return None
    n_full = int(segs0[0].shape[0])
    if neuron_ix is None:
        ix = np.arange(n_full, dtype=np.int64)
    else:
        ix = np.unique(np.asarray(neuron_ix, dtype=np.int64))
        ix = ix[(ix >= 0) & (ix < n_full)]
    if ix.size < 3:
        return None
    n_sub = int(ix.size)
    n0, n1 = len(segs0), len(segs1)
    nmax = max(n0, n1)
    trialX = np.full((nmax, n_sub, 2, min_T), np.nan, dtype=np.float64)
    for i, s in enumerate(segs0):
        trialX[i, :, 0, :] = s[ix, :min_T]
    for i, s in enumerate(segs1):
        trialX[i, :, 1, :] = s[ix, :min_T]
    X_mean = np.nanmean(trialX, axis=0)
    counts = {0: n0, 1: n1}
    return trialX, X_mean, counts


def _build_dpca_mean_and_trialx(
    run_keys: list[str],
    task_key: str,
    exclude_slm_train_pool: bool,
) -> tuple[np.ndarray, np.ndarray, int, dict[int, int]]:
    """trialX (n_trials_max, n_neurons, 2, T); X_mean (n_neurons, 2, T)."""
    segs, min_T, _n_neurons = _collect_segs_by_class(run_keys, task_key, exclude_slm_train_pool)
    out = _from_segs_build_mean_trialx(segs, min_T, None)
    if out is None:
        raise RuntimeError("dPCA: could not build mean/trial tensors.")
    trialX, X_mean, counts = out
    return trialX, X_mean, min_T, counts


def _collect_segs_by_joint_binary_levels(
    run_keys: list[str],
    levels: tuple[tuple[str, str, int, str, str], ...],
    exclude_slm_train_pool: bool,
) -> tuple[dict[int, list[np.ndarray]], int, int]:
    """Segments per binary label (valence + consumption); a trial may appear in two levels."""
    segs_by_s: dict[int, list[np.ndarray]] = {si: [] for si in range(len(levels))}
    min_T: int | None = None
    n_neurons: int | None = None

    for rk in run_keys:
        pack = load_npz_pack(rk)
        nn = int(pack["n_neurons"])
        if n_neurons is None:
            n_neurons = nn
        elif nn != n_neurons:
            raise ValueError(f"Inconsistent n_neurons: {n_neurons} vs {nn} in {rk}")
        y_cond = pack["trial_type"]
        tr = pack["trial_run"]
        cmap = _cond_name_map(rk)
        y_by_task = {
            tk: build_binary_labels(y_cond, cmap, tk)
            for tk in ("hedonic_valence", "consumption_mode")
        }

        for local_i, seg in enumerate(pack["segments"]):
            ci = int(y_cond[local_i])
            if exclude_slm_train_pool and not trial_keep_train_pool(int(tr[local_i]), ci):
                continue
            segf = np.asarray(seg, dtype=np.float64)
            if segf.ndim != 2 or segf.shape[0] != n_neurons:
                continue
            T = segf.shape[1]
            if T < 1:
                continue
            placed = False
            for si, (_name, task_key, cls_id, _col, _ls) in enumerate(levels):
                yb, keep = y_by_task[task_key]
                if not bool(keep[local_i]):
                    continue
                if int(yb[local_i]) != int(cls_id):
                    continue
                segs_by_s[si].append(segf)
                placed = True
            if placed:
                min_T = T if min_T is None else min(min_T, T)

    if min_T is None or min_T < 2 or n_neurons is None:
        raise RuntimeError("dPCA joint: insufficient time samples or no trials after filtering.")
    for si, (name, *_rest) in enumerate(levels):
        if not segs_by_s[si]:
            raise RuntimeError(f"dPCA joint: no trials for label '{name}'.")
    return segs_by_s, min_T, n_neurons


def _from_segs_build_mean_trialx_n(
    segs_by_s: dict[int, list[np.ndarray]],
    n_stim: int,
    min_T: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """trialX (n_trials_max, n_neurons, S, T), X_mean (n_neurons, S, T)."""
    for si in range(n_stim):
        if not segs_by_s.get(si):
            raise RuntimeError(f"dPCA joint: empty stimulus index {si}.")
    n_neurons = int(segs_by_s[0][0].shape[0])
    counts = {si: len(segs_by_s[si]) for si in range(n_stim)}
    nmax = max(counts.values())
    trialX = np.full((nmax, n_neurons, n_stim, min_T), np.nan, dtype=np.float64)
    for si in range(n_stim):
        for i, s in enumerate(segs_by_s[si]):
            trialX[i, :, si, :] = s[:, :min_T]
    X_mean = np.nanmean(trialX, axis=0)
    return trialX, X_mean, counts


def _build_joint_dpca_mean_and_trialx(
    run_keys: list[str],
    exclude_slm_train_pool: bool,
) -> tuple[np.ndarray, np.ndarray, int, dict[int, int], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Build tensors for 4 binary-label dPCA; return labels, colors, linestyles."""
    levels = JOINT_BINARY_LEVELS
    segs, min_T, _nn = _collect_segs_by_joint_binary_levels(run_keys, levels, exclude_slm_train_pool)
    trialX, X_mean, counts = _from_segs_build_mean_trialx_n(segs, len(levels), min_T)
    labels = tuple(name for name, *_rest in levels)
    colors = tuple(col for _n, _tk, _c, col, _ls in levels)
    linestyles = tuple(ls for *_rest, ls in levels)
    return trialX, X_mean, min_T, counts, labels, colors, linestyles


def _zscore_tensor(tensor: np.ndarray) -> np.ndarray:
    """Z-score each neuron across (condition, time). tensor (C, T, N)."""
    c, t, n = tensor.shape
    flat = tensor.reshape(-1, n)
    mu = flat.mean(axis=0)
    sig = np.maximum(flat.std(axis=0), 1e-8)
    return (tensor - mu) / sig


def _zscore_mean_and_trialx(X_mean: np.ndarray, trialX: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-neuron z-score from all non-NaN trialX values; apply same affine to X_mean."""
    N = X_mean.shape[0]
    Xm = X_mean.astype(np.float64, copy=True)
    Xt = trialX.astype(np.float64, copy=True)
    for n in range(N):
        v = Xt[:, n, :, :].ravel()
        v = v[~np.isnan(v)]
        if v.size < 2:
            mu, sig = 0.0, 1.0
        else:
            mu = float(np.mean(v))
            sig = max(float(np.std(v)), 1e-8)
        Xm[n, :, :] = (X_mean[n, :, :].astype(np.float64) - mu) / sig
        Xt[:, n, :, :] = (trialX[:, n, :, :].astype(np.float64) - mu) / sig
    return Xm, Xt


def _split_coords_by_trial(
    coords: np.ndarray,
    trial_ix: np.ndarray,
    frame_t: np.ndarray,
    lens: np.ndarray,
) -> list[np.ndarray]:
    """coords (S, k) aligned with rows; return list of (T_i, k) per trial."""
    out: list[np.ndarray] = [np.full((int(lens[j]), coords.shape[1]), np.nan) for j in range(len(lens))]
    for r in range(trial_ix.size):
        j = int(trial_ix[r])
        t = int(frame_t[r])
        out[j][t, :] = coords[r, :]
    return out


def _speed_per_trial(traj_list: list[np.ndarray], k3: int = 3) -> np.ndarray:
    k = min(k3, traj_list[0].shape[1]) if traj_list else 0
    speeds = np.zeros(len(traj_list), dtype=np.float64)
    for j, tr in enumerate(traj_list):
        if tr.shape[0] < 2:
            continue
        d = np.diff(tr[:, :k], axis=0)
        speeds[j] = float(np.mean(np.linalg.norm(d, axis=1)))
    return speeds


def _mean_sem_timecourse(
    traj_list: list[np.ndarray], y_bin: np.ndarray, k3: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per class, per time: mean and SEM over trials that extend to t. traj_list aligned with unique trials in order."""
    k = min(k3, traj_list[0].shape[1]) if traj_list else 0
    max_t = max((tr.shape[0] for tr in traj_list), default=0)
    mean0 = np.full((max_t, k), np.nan)
    mean1 = np.full((max_t, k), np.nan)
    sem0 = np.full((max_t, k), np.nan)
    sem1 = np.full((max_t, k), np.nan)
    for t in range(max_t):
        v0 = [tr[t, :k] for tr, y in zip(traj_list, y_bin) if y == 0 and tr.shape[0] > t]
        v1 = [tr[t, :k] for tr, y in zip(traj_list, y_bin) if y == 1 and tr.shape[0] > t]
        if v0:
            a = np.stack(v0, axis=0)
            mean0[t] = np.nanmean(a, axis=0)
            sem0[t] = np.nanstd(a, axis=0) / np.sqrt(max(1, a.shape[0]))
        if v1:
            a = np.stack(v1, axis=0)
            mean1[t] = np.nanmean(a, axis=0)
            sem1[t] = np.nanstd(a, axis=0) / np.sqrt(max(1, a.shape[0]))
    return mean0, sem0, mean1, sem1


def _plot_spaghetti_3d(
    traj_list: list[np.ndarray],
    y_bin: np.ndarray,
    run_ix: np.ndarray,
    run_names: list[str],
    class_names: tuple[str, str],
    title: str,
    out_path: Path,
    k3: int = 3,
    max_trials: int = 40,
    seed: int = 42,
    line_alpha: float = 0.18,
    linewidth: float = 0.55,
) -> None:
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(traj_list))
    if len(idxs) > max_trials:
        idxs = rng.choice(idxs, size=max_trials, replace=False)
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    colors = {0: "#b22222", 1: "#0b3d91"}
    styles = {i: ("-", 0.35 + 0.4 * (i / max(1, len(run_names) - 1))) for i in range(len(run_names))}
    for j in idxs:
        tr = traj_list[int(j)]
        if tr.shape[0] < 2:
            continue
        k = min(k3, tr.shape[1])
        r = int(run_ix[j]) if j < run_ix.size else 0
        ls, _al = styles.get(r, ("-", 0.5))
        ax.plot(
            tr[:, 0],
            tr[:, 1],
            tr[:, 2] if k > 2 else np.zeros(tr.shape[0]),
            color=colors[int(y_bin[j])],
            alpha=line_alpha,
            linewidth=linewidth,
            linestyle=ls,
        )
    ax.set_title(title + " (subsample of single-trial trajectories)", fontsize=10)
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")
    ax.set_zlabel("Axis 3")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_trajectory_3d(
    mean0: np.ndarray,
    mean1: np.ndarray,
    class_names: tuple[str, str],
    title: str,
    out_path: Path,
    k3: int = 3,
) -> None:
    """Mean path in the first three axes (cue time order along the line)."""
    k = min(k3, mean0.shape[1], mean1.shape[1])
    if k < 2:
        return
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    z0 = mean0[:, 2] if k > 2 else np.zeros(mean0.shape[0])
    z1 = mean1[:, 2] if k > 2 else np.zeros(mean1.shape[0])
    ax.plot(mean0[:, 0], mean0[:, 1], z0, color="#b22222", linewidth=2.0, label=class_names[0])
    ax.plot(mean1[:, 0], mean1[:, 1], z1, color="#0b3d91", linewidth=2.0, label=class_names[1])
    ax.scatter(mean0[0, 0], mean0[0, 1], z0[0], color="#b22222", s=36, marker="o", zorder=5)
    ax.scatter(mean1[0, 0], mean1[0, 1], z1[0], color="#0b3d91", s=36, marker="o", zorder=5)
    ax.set_title(title + " (mean trajectories)", fontsize=10)
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")
    ax.set_zlabel("Axis 3")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_trajectory_2d_panels(
    mean0: np.ndarray,
    mean1: np.ndarray,
    class_names: tuple[str, str],
    title: str,
    out_path: Path,
    k3: int = 3,
) -> None:
    """2×2 or 1×3 panels: parametric mean trajectory in (1,2), (1,3), (2,3) planes."""
    k = min(k3, mean0.shape[1], mean1.shape[1])
    if k < 2:
        return
    pairs = [(0, 1)] if k < 3 else [(0, 1), (0, 2), (1, 2)]
    n_p = len(pairs)
    fig, axes = plt.subplots(1, n_p, figsize=(4.2 * n_p, 4.0), squeeze=False)
    for ax_i, (i0, i1) in enumerate(pairs):
        ax = axes[0, ax_i]
        ax.plot(mean0[:, i0], mean0[:, i1], color="#b22222", linewidth=1.8, label=class_names[0])
        ax.plot(mean1[:, i0], mean1[:, i1], color="#0b3d91", linewidth=1.8, label=class_names[1])
        ax.scatter(mean0[0, i0], mean0[0, i1], color="#b22222", s=28, zorder=5, marker="o")
        ax.scatter(mean1[0, i0], mean1[0, i1], color="#0b3d91", s=28, zorder=5, marker="o")
        ax.set_xlabel(f"Axis {i0 + 1}")
        ax.set_ylabel(f"Axis {i1 + 1}")
        ax.set_title(f"Mean traj axes {i0 + 1} vs {i1 + 1}")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("auto")
        if ax_i == n_p - 1:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(title + " (mean trajectories, 2D projections)", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mean2d(mean0: np.ndarray, sem0: np.ndarray, mean1: np.ndarray, sem1: np.ndarray, class_names: tuple[str, str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    t0 = np.arange(mean0.shape[0])
    ax.plot(t0, mean0[:, 0], color="#b22222", label=class_names[0])
    ax.fill_between(t0, mean0[:, 0] - sem0[:, 0], mean0[:, 0] + sem0[:, 0], color="#b22222", alpha=0.2)
    ax.plot(t0, mean1[:, 0], color="#0b3d91", label=class_names[1])
    ax.fill_between(t0, mean1[:, 0] - sem1[:, 0], mean1[:, 0] + sem1[:, 0], color="#0b3d91", alpha=0.2)
    ax.set_xlabel("Cue-relative frame")
    ax.set_ylabel("Axis 1 (mean ± SEM)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dpca_condition_traj(
    Z_st: np.ndarray,
    class_names: Sequence[str],
    title: str,
    out_path: Path,
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
) -> None:
    """Z_st shape (n_comp, n_cond, n_time)."""
    if Z_st.ndim != 3 or Z_st.shape[1] < 1:
        return
    default_colors = ("#b22222", "#0b3d91", "#2e8b57", "#9467bd")
    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    T = Z_st.shape[2]
    for c in range(Z_st.shape[1]):
        x = Z_st[0, c, :]
        y = Z_st[1, c, :] if Z_st.shape[0] > 1 else np.zeros(T)
        z = Z_st[2, c, :] if Z_st.shape[0] > 2 else np.arange(T, dtype=np.float64) * 0.01
        if colors is not None and c < len(colors):
            col = colors[c]
        else:
            col = default_colors[c % len(default_colors)]
        ls = linestyles[c] if linestyles is not None and c < len(linestyles) else "-"
        lbl = class_names[c] if c < len(class_names) else f"cond{c}"
        ax.plot(x, y, z, color=col, label=lbl, linewidth=1.4, linestyle=ls)
        ax.scatter(x[0], y[0], z[0], color=col, s=28, zorder=5, marker="o")
    ax.set_title(title)
    ax.set_xlabel("dPC1 (st)")
    ax.set_ylabel("dPC2 (st)")
    ax.set_zlabel("dPC3 (st)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dpca_st_2d_panels(
    Z_st: np.ndarray,
    class_names: Sequence[str],
    title: str,
    out_path: Path,
    colors: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
) -> None:
    """2D projections of the (st) mean trajectories in dPC space."""
    n_comp, n_cond, _T = Z_st.shape
    if n_comp < 2 or n_cond < 1:
        return
    default_colors = ("#b22222", "#0b3d91", "#2e8b57", "#9467bd")
    pairs = [(0, 1), (0, 2), (1, 2)] if n_comp >= 3 else [(0, 1)]
    n_p = len(pairs)
    fig, axes = plt.subplots(1, n_p, figsize=(4.0 * n_p, 3.8), squeeze=False)
    for ax_i, (i0, i1) in enumerate(pairs):
        ax = axes[0, ax_i]
        for c in range(n_cond):
            if colors is not None and c < len(colors):
                col = colors[c]
            else:
                col = default_colors[c % len(default_colors)]
            ls = linestyles[c] if linestyles is not None and c < len(linestyles) else "-"
            lbl = class_names[c] if c < len(class_names) else f"c{c}"
            ax.plot(
                Z_st[i0, c, :],
                Z_st[i1, c, :],
                color=col,
                linewidth=1.6,
                label=lbl,
                linestyle=ls,
            )
            ax.scatter(Z_st[i0, c, 0], Z_st[i1, c, 0], color=col, s=26, zorder=5, marker="o")
        ax.set_xlabel(f"dPC{i0 + 1} (st)")
        ax.set_ylabel(f"dPC{i1 + 1} (st)")
        ax.set_title(f"st: axes {i0 + 1} vs {i1 + 1}")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("auto")
        if ax_i == n_p - 1:
            ax.legend(fontsize=6.5)
    fig.suptitle(title + " (st, 2D)", fontsize=10, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dpca_t_marginal(
    Z_t: np.ndarray,
    class_names: tuple[str, str],
    title: str,
    out_path: Path,
    n_lines: int = 2,
) -> None:
    """Time marginal: dPC k vs cue frame (both classes). Z_t shape (n_comp, n_cond, n_time)."""
    if Z_t.ndim != 3 or Z_t.shape[2] < 2:
        return
    n_show = min(n_lines, int(Z_t.shape[0]))
    fig, axes = plt.subplots(n_show, 1, figsize=(7.0, 2.8 * n_show), squeeze=False)
    t_ax = np.arange(Z_t.shape[2])
    colors = ["#b22222", "#0b3d91"]
    for row in range(n_show):
        ax = axes[row, 0]
        for c in range(min(2, Z_t.shape[1])):
            ax.plot(t_ax, Z_t[row, c, :], color=colors[c], linewidth=1.3, label=class_names[c] if c < 2 else f"c{c}")
        ax.set_ylabel(f"dPC{row + 1} (t)")
        ax.grid(True, alpha=0.25)
        if row == n_show - 1:
            ax.set_xlabel("Cue-relative frame")
        if row == 0:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(title + " — marginal t (shared / time-like)", fontsize=10, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dpca_s_marginal(
    Z_s: np.ndarray,
    class_names: tuple[str, str],
    title: str,
    out_path: Path,
) -> None:
    """Stimulus marginal: dPC vs time (expect weak time dependence) + time-mean bars."""
    if Z_s.ndim != 3:
        return
    n_comp, n_cond, T = Z_s.shape
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    colors = ["#b22222", "#0b3d91"]
    t_ax = np.arange(T)
    for c in range(min(2, n_cond)):
        ax0.plot(t_ax, Z_s[0, c, :], color=colors[c], linewidth=1.3, label=class_names[c] if c < 2 else f"c{c}")
    ax0.axhline(0.0, color="0.5", linewidth=0.6, linestyle="--", alpha=0.6)
    ax0.set_xlabel("Cue-relative frame")
    ax0.set_ylabel("dPC1 (s)")
    ax0.set_title("s: 1st dPC vs time")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=7)

    n_b = min(4, n_comp)
    x = np.arange(n_b)
    w = 0.35
    m0 = [float(np.nanmean(Z_s[k, 0, :])) for k in range(n_b)]
    m1 = [float(np.nanmean(Z_s[k, 1, :])) for k in range(n_b)]
    ax1.bar(x - w / 2, m0, width=w, color=colors[0], label=class_names[0])
    ax1.bar(x + w / 2, m1, width=w, color=colors[1], label=class_names[1])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"dPC{k + 1}" for k in range(n_b)])
    ax1.axhline(0.0, color="0.5", linewidth=0.6, linestyle="--", alpha=0.6)
    ax1.set_ylabel("Time-mean score")
    ax1.set_title("s: time-mean of each dPC by class")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(fontsize=7)
    fig.suptitle(title + " — marginal s (stimulus-like)", fontsize=10, y=1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dpca_explained_variance(ev: dict, out_path: Path, title: str) -> None:
    """Bar chart: variance explained by dPC1 in each marginalization (per dPCA API)."""
    if not ev:
        return
    keys = sorted(ev.keys(), key=lambda k: (len(k), k))
    vals = []
    for k in keys:
        arr = np.asarray(ev[k], dtype=np.float64).ravel()
        vals.append(float(arr[0]) if arr.size else 0.0)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(range(len(keys)), vals, color="steelblue", edgecolor="0.25", linewidth=0.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=25, ha="right")
    ax.set_ylabel("Expl. var. ratio (1st dPC)")
    ax.set_title(title + " — dPCA explained variance (1st component per marginalization)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_dpca_manifest(Z: dict, ev: dict | None, out_path: Path) -> None:
    lines = ["dPCA output manifest", "Marginalization keys (Z arrays): " + ", ".join(sorted(Z.keys())), ""]
    if ev and isinstance(ev, dict):
        lines.append("explained_variance_ratio_ keys: " + ", ".join(sorted(ev.keys())))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _traj_list_to_flat_offsets(traj_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offs = np.zeros(len(traj_list) + 1, dtype=np.int64)
    for i, tr in enumerate(traj_list):
        offs[i + 1] = offs[i] + int(tr.shape[0])
    if not traj_list:
        return np.zeros((0, 0), dtype=np.float32), offs
    flat = np.vstack([t.astype(np.float32) for t in traj_list])
    return flat, offs


def _dispersion_per_time(traj_list: list[np.ndarray], y_trial: np.ndarray, k3: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Per cue frame, mean trace of cov of first k coords, split by class (naive)."""
    if not traj_list or traj_list[0].shape[1] < 1:
        return np.array([]), np.array([])
    max_t = max((tr.shape[0] for tr in traj_list), default=0)
    k = min(k3, traj_list[0].shape[1])
    d0 = np.full(max_t, np.nan)
    d1 = np.full(max_t, np.nan)
    for t in range(max_t):
        for cls, arr in ((0, d0), (1, d1)):
            pts = [tr[t, :k] for tr, y in zip(traj_list, y_trial) if y == cls and tr.shape[0] > t]
            if len(pts) >= 2:
                a = np.stack(pts, axis=0)
                cov = np.cov(a.T)
                arr[t] = float(np.trace(cov))
    return d0, d1


def run_one_task(
    task_key: str,
    run_keys: list[str],
    out_root: Path,
    n_components: int,
    dpca_regularizer: float | str | None,
    exclude_slm_train_pool: bool,
    seed: int,
    plot_spaghetti: bool = True,
    spaghetti_max_trials: int = 40,
    spaghetti_alpha: float = 0.18,
) -> None:
    spec = TASK_SPECS[task_key]
    class_names = (spec["class0_name"], spec["class1_name"])

    X_flat, row_run, trial_ix, frame_t, y_row, trial_run_row, trial_type_row, n_neurons = _build_flat_and_meta(
        run_keys, task_key, exclude_slm_train_pool
    )
    lens = _trial_lengths_from_meta(trial_ix, frame_t)
    n_trials = len(lens)

    # One y_bin / run index per trial (first row of each trial)
    y_trial = np.empty(n_trials, dtype=np.int64)
    run_of_trial = np.empty(n_trials, dtype=np.int64)
    for j in range(n_trials):
        ix = int(np.flatnonzero(trial_ix == j)[0])
        y_trial[j] = int(y_row[ix])
        run_of_trial[j] = int(row_run[ix])

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_flat)

    # --- Global PCA ---
    gdir = out_root / "global_pca" / task_key
    gdir.mkdir(parents=True, exist_ok=True)
    pca_g = PCA(n_components=min(n_components, Xz.shape[1], Xz.shape[0]), random_state=seed)
    Zg = pca_g.fit_transform(Xz)
    traj_g = _split_coords_by_trial(Zg, trial_ix, frame_t, lens)
    tr_first = np.zeros(n_trials, dtype=np.int64)
    tt_first = np.zeros(n_trials, dtype=np.int64)
    for j in range(n_trials):
        m = trial_ix == j
        i0 = int(np.flatnonzero(m)[0])
        tr_first[j] = trial_run_row[i0]
        tt_first[j] = trial_type_row[i0]
    flat_g, off_g = _traj_list_to_flat_offsets(traj_g)
    np.savez_compressed(
        gdir / "coords_flat.npz",
        coords_flat=flat_g,
        trial_offsets=off_g,
        y_trial=y_trial,
        run_of_trial=run_of_trial,
        trial_run=tr_first,
        trial_type=tt_first,
        explained_variance_ratio=pca_g.explained_variance_ratio_.astype(np.float32),
    )
    np.savez_compressed(
        gdir / "pca_model.npz",
        components=pca_g.components_.astype(np.float32),
        mean_=scaler.mean_.astype(np.float32),
        scale_=scaler.scale_.astype(np.float32),
    )
    m0, s0, m1, s1 = _mean_sem_timecourse(traj_g, y_trial)
    _plot_mean2d(m0, s0, m1, s1, class_names, f"Global PCA axis1 mean±SEM ({task_key})", gdir / "mean_sem_axis1.png")
    _plot_mean_trajectory_3d(m0, m1, class_names, f"Global PCA ({task_key})", gdir / "mean_traj_3d.png")
    _plot_mean_trajectory_2d_panels(m0, m1, class_names, f"Global PCA ({task_key})", gdir / "mean_traj_2d_panels.png")
    if plot_spaghetti:
        _plot_spaghetti_3d(
            traj_g,
            y_trial,
            run_of_trial,
            run_keys,
            class_names,
            f"Global PCA ({task_key})",
            gdir / "spaghetti_3d.png",
            seed=seed,
            max_trials=spaghetti_max_trials,
            line_alpha=spaghetti_alpha,
        )
    spd = _speed_per_trial(traj_g)
    disp0, disp1 = _dispersion_per_time(traj_g, y_trial)
    np.savez_compressed(
        gdir / "metrics.npz",
        per_trial_speed=spd.astype(np.float32),
        y_trial=y_trial,
        trace_cov_t_class0=disp0.astype(np.float32),
        trace_cov_t_class1=disp1.astype(np.float32),
    )

    # --- LDA projection ---
    load_path = LDA_FOUR_CLASS_DIR / f"pooled_{task_key}_four_class_lda_loadings.csv"
    if not load_path.exists():
        raise FileNotFoundError(f"Missing LDA loadings: {load_path} (run run_lda.py first).")
    W, wcols = _load_lda_weight_matrix(load_path, n_neurons)
    ld12 = Xz @ W
    tnorm = frame_t.astype(np.float64)
    z3 = (tnorm / max(1.0, float(frame_t.max()))) * (np.std(ld12[:, 0]) + 1e-8) * 0.25
    if ld12.shape[1] >= 3:
        Zlda = np.column_stack([ld12[:, 0], ld12[:, 1], ld12[:, 2]])
    else:
        Zlda = np.column_stack([ld12[:, 0], ld12[:, 1], z3])
    traj_lda = _split_coords_by_trial(Zlda, trial_ix, frame_t, lens)
    ldir = out_root / "lda_proj" / task_key
    ldir.mkdir(parents=True, exist_ok=True)
    flat_l, off_l = _traj_list_to_flat_offsets(traj_lda)
    (ldir / "lda_columns.json").write_text(json.dumps(wcols), encoding="utf-8")
    np.savez_compressed(
        ldir / "coords_flat.npz",
        coords_flat=flat_l,
        trial_offsets=off_l,
        y_trial=y_trial,
        run_of_trial=run_of_trial,
    )
    m0l, s0l, m1l, s1l = _mean_sem_timecourse(traj_lda, y_trial)
    _plot_mean2d(m0l, s0l, m1l, s1l, class_names, f"LDA axis1 mean±SEM ({task_key})", ldir / "mean_sem_axis1.png")
    _plot_mean_trajectory_3d(m0l, m1l, class_names, f"LDA projection ({task_key})", ldir / "mean_traj_3d.png")
    _plot_mean_trajectory_2d_panels(m0l, m1l, class_names, f"LDA projection ({task_key})", ldir / "mean_traj_2d_panels.png")
    if plot_spaghetti:
        _plot_spaghetti_3d(
            traj_lda,
            y_trial,
            run_of_trial,
            run_keys,
            class_names,
            f"LDA projection ({task_key})",
            ldir / "spaghetti_3d.png",
            seed=seed,
            max_trials=spaghetti_max_trials,
            line_alpha=spaghetti_alpha,
        )
    spd_l = _speed_per_trial(traj_lda)
    d0l, d1l = _dispersion_per_time(traj_lda, y_trial)
    np.savez_compressed(
        ldir / "metrics.npz",
        per_trial_speed=spd_l.astype(np.float32),
        y_trial=y_trial,
        trace_cov_t_class0=d0l.astype(np.float32),
        trace_cov_t_class1=d1l.astype(np.float32),
    )

    # --- dPCA ---
    ddir = out_root / "dpca" / task_key
    ddir.mkdir(parents=True, exist_ok=True)
    trialX, X_mean, T_common, counts = _build_dpca_mean_and_trialx(run_keys, task_key, exclude_slm_train_pool)
    info = {
        "task": task_key,
        "T_common": T_common,
        "n_trials_class0": counts[0],
        "n_trials_class1": counts[1],
        "n_trials_max_padded": int(trialX.shape[0]),
    }
    (ddir / "trial_counts.json").write_text(json.dumps(info, indent=2))

    try:
        from dPCA.dPCA import dPCA as DPCAClass  # type: ignore[import-not-found]
    except ImportError:
        try:
            from dpca.dPCA import dPCA as DPCAClass  # type: ignore[import-not-found]
        except ImportError:
            DPCAClass = None  # type: ignore[misc, assignment]

    if DPCAClass is None:
        (ddir / "SKIPPED_no_dpca_module.txt").write_text(
            "Install: pip install dpca numexpr\n"
            "Estimator: from dPCA.dPCA import dPCA\n"
            "Then re-run this script.\n"
        )
    else:
        reg_in = dpca_regularizer
        use_auto = isinstance(reg_in, str) and reg_in.strip().lower() == "auto"
        reg = reg_in
        if isinstance(reg, str) and reg.lower() == "none":
            reg = None

        if use_auto:
            X_fit, trialX_fit = _zscore_mean_and_trialx(X_mean, trialX)
        else:
            tensor_2tn = np.transpose(X_mean, (1, 2, 0))
            tensor_z = _zscore_tensor(tensor_2tn)
            X_fit = np.transpose(tensor_z, (2, 0, 1))
            trialX_fit = None

        ncomp = min(max(3, n_components), 10, X_fit.shape[1] * X_fit.shape[2])
        dpca = DPCAClass(labels="st", n_components=ncomp, regularizer=reg)
        dpca.debug = 0
        if use_auto:
            dpca.protect = ["t"]
            Z = dpca.fit_transform(X_fit, trialX=trialX_fit)
        else:
            Z = dpca.fit_transform(X_fit)
        z_kw = {f"Z_{k}": np.asarray(v, dtype=np.float32) for k, v in Z.items()}
        np.savez_compressed(ddir / "dpca_result.npz", **z_kw)
        ev_obj = getattr(dpca, "explained_variance_ratio_", None)
        if isinstance(ev_obj, dict):
            (ddir / "explained_variance_ratio.json").write_text(
                json.dumps({str(k): np.asarray(v, dtype=float).tolist() for k, v in ev_obj.items()}, indent=2)
            )
            _plot_dpca_explained_variance(ev_obj, ddir / "explained_variance_pc1_bars.png", f"dPCA ({task_key})")
        _write_dpca_manifest(Z, ev_obj if isinstance(ev_obj, dict) else None, ddir / "dpca_manifest.txt")

        base = f"dPCA ({task_key})"
        if "t" in Z and isinstance(Z["t"], np.ndarray) and Z["t"].ndim == 3:
            _plot_dpca_t_marginal(np.asarray(Z["t"], dtype=np.float64), class_names, base, ddir / "marginal_t_dpc_vs_time.png")
        if "s" in Z and isinstance(Z["s"], np.ndarray) and Z["s"].ndim == 3:
            _plot_dpca_s_marginal(np.asarray(Z["s"], dtype=np.float64), class_names, base, ddir / "marginal_s_dpc_vs_time_and_bars.png")
        if "st" in Z and isinstance(Z["st"], np.ndarray) and Z["st"].ndim == 3:
            Zst = np.asarray(Z["st"], dtype=np.float64)
            _plot_dpca_condition_traj(Zst, class_names, f"{base} — interaction (st)", ddir / "traj_st_3d.png")
            _plot_dpca_st_2d_panels(Zst, class_names, base, ddir / "traj_st_2d_panels.png")


def _import_dpca_class():
    try:
        from dPCA.dPCA import dPCA as DPCAClass  # type: ignore[import-not-found]
        return DPCAClass
    except ImportError:
        try:
            from dpca.dPCA import dPCA as DPCAClass  # type: ignore[import-not-found]
            return DPCAClass
        except ImportError:
            return None


def run_valence_x_consumption(
    run_keys: list[str],
    out_root: Path,
    n_components: int,
    dpca_regularizer: float | str | None,
    exclude_slm_train_pool: bool,
) -> None:
    """Shared dPCA with four paths: Aversive, Appetitive, NonConsumption, Consumption."""
    ddir = out_root / "dpca" / JOINT_TASK_KEY
    ddir.mkdir(parents=True, exist_ok=True)
    trialX, X_mean, T_common, counts, labels, colors, linestyles = _build_joint_dpca_mean_and_trialx(
        run_keys, exclude_slm_train_pool
    )
    info = {
        "task": JOINT_TASK_KEY,
        "T_common": T_common,
        "labels": list(labels),
        "n_trials": {labels[si]: int(counts[si]) for si in range(len(labels))},
        "n_trials_max_padded": int(trialX.shape[0]),
        "note": "Overlapping trial sets: valence and consumption partitions of the same pool.",
    }
    (ddir / "trial_counts.json").write_text(json.dumps(info, indent=2))

    DPCAClass = _import_dpca_class()
    if DPCAClass is None:
        (ddir / "SKIPPED_no_dpca_module.txt").write_text(
            "Install: pip install dpca numexpr\n"
            "Estimator: from dPCA.dPCA import dPCA\n"
            "Then re-run this script.\n"
        )
        return

    reg_in = dpca_regularizer
    use_auto = isinstance(reg_in, str) and reg_in.strip().lower() == "auto"
    reg = reg_in
    if isinstance(reg, str) and reg.lower() == "none":
        reg = None

    if use_auto:
        X_fit, trialX_fit = _zscore_mean_and_trialx(X_mean, trialX)
    else:
        tensor_2tn = np.transpose(X_mean, (1, 2, 0))
        tensor_z = _zscore_tensor(tensor_2tn)
        X_fit = np.transpose(tensor_z, (2, 0, 1))
        trialX_fit = None

    ncomp = min(max(3, n_components), 10, X_fit.shape[1] * X_fit.shape[2])
    dpca = DPCAClass(labels="st", n_components=ncomp, regularizer=reg)
    dpca.debug = 0
    if use_auto:
        dpca.protect = ["t"]
        Z = dpca.fit_transform(X_fit, trialX=trialX_fit)
    else:
        Z = dpca.fit_transform(X_fit)

    z_kw = {f"Z_{k}": np.asarray(v, dtype=np.float32) for k, v in Z.items()}
    np.savez_compressed(ddir / "dpca_result.npz", **z_kw)
    ev_obj = getattr(dpca, "explained_variance_ratio_", None)
    if isinstance(ev_obj, dict):
        (ddir / "explained_variance_ratio.json").write_text(
            json.dumps({str(k): np.asarray(v, dtype=float).tolist() for k, v in ev_obj.items()}, indent=2)
        )
        _plot_dpca_explained_variance(ev_obj, ddir / "explained_variance_pc1_bars.png", f"dPCA ({JOINT_TASK_KEY})")
    _write_dpca_manifest(Z, ev_obj if isinstance(ev_obj, dict) else None, ddir / "dpca_manifest.txt")

    base = "dPCA (valence + consumption labels)"
    if "st" in Z and isinstance(Z["st"], np.ndarray) and Z["st"].ndim == 3:
        Zst = np.asarray(Z["st"], dtype=np.float64)
        _plot_dpca_condition_traj(
            Zst,
            labels,
            f"{base} — interaction (st)",
            ddir / "traj_st_3d.png",
            colors=colors,
            linestyles=linestyles,
        )
        _plot_dpca_st_2d_panels(
            Zst,
            labels,
            base,
            ddir / "traj_st_2d_panels.png",
            colors=colors,
            linestyles=linestyles,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Representation trajectories (Livneh).")
    p.add_argument("--runs", nargs="+", default=list(EVAL_RUN_KEYS), help="NPZ keys under normalized_data/")
    p.add_argument("--out-dir", type=Path, default=TRAJECTORIES_OUT, help="Output directory.")
    p.add_argument(
        "--task",
        choices=["both", "hedonic_valence", "consumption_mode", "valence_x_consumption"],
        default="both",
        help="'both' runs binary tasks plus valence×consumption joint dPCA.",
    )
    p.add_argument("--n-components", type=int, default=20)
    p.add_argument(
        "--dpca-regularizer",
        default="auto",
        help="dPCA: 'auto' (CV; needs trial-by-trial data, slower), 'none', or a float regularizer.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-spaghetti-plots",
        action="store_true",
        help="Skip 3D subsampled single-trial spaghetti; mean trajectory figures are still written.",
    )
    p.add_argument(
        "--spaghetti-max-trials",
        type=int,
        default=40,
        help="Max trials drawn in spaghetti 3D (random subset, stratified by trial list order).",
    )
    p.add_argument(
        "--spaghetti-alpha",
        type=float,
        default=0.18,
        help="Line alpha for spaghetti 3D (lower = less cloud-like).",
    )
    p.add_argument(
        "--exclude-slm-train-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude SLM like prepare_data train pool.",
    )
    args = p.parse_args()
    run_keys = list(args.runs)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    reg: float | str | None
    if isinstance(args.dpca_regularizer, str):
        s = args.dpca_regularizer.strip().lower()
        if s in ("auto", "none"):
            reg = "auto" if s == "auto" else None
        else:
            reg = float(s)
    else:
        reg = args.dpca_regularizer

    run_binary = args.task in ("both", "hedonic_valence", "consumption_mode")
    run_joint = args.task in ("both", "valence_x_consumption")
    if run_binary:
        tasks = list(TASK_SPECS.keys()) if args.task == "both" else [args.task]
        for tk in tasks:
            run_one_task(
                tk,
                run_keys,
                out_root,
                int(args.n_components),
                reg,
                bool(args.exclude_slm_train_pool),
                int(args.seed),
                plot_spaghetti=not bool(args.no_spaghetti_plots),
                spaghetti_max_trials=int(args.spaghetti_max_trials),
                spaghetti_alpha=float(args.spaghetti_alpha),
            )
    if run_joint:
        run_valence_x_consumption(
            run_keys,
            out_root,
            int(args.n_components),
            reg,
            bool(args.exclude_slm_train_pool),
        )
    print(f"Done. Outputs under: {out_root.resolve()}")


if __name__ == "__main__":
    main()
