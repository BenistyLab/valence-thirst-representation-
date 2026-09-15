"""
Demixed PCA (dPCA) on semantic phase packs (pre, airpuff, water).

Each analysis pools trials into stimulus levels on axis s (labels='st');
marginalizations: t, s, st. Outputs under outputs/dpca/<run_key>/<name>/.

Install: pip install dpca numexpr
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dPCA.dPCA import dPCA
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

from lib.config import DPCA_OUT as DPCA_OUT_ROOT, PHASE_KEYS, REPO_ROOT

N_COMPONENTS = 5
ANALYSIS_T_FRAMES = 300
REGULARIZER = "auto"
LOADING_MARGINALS = ("t", "s", "st")
LOADING_N_PCS = 2
LOADING_TOP_NEURONS = 30
LOADING_SIM_ANALYSES = ("3conditions", "hedonic_valence", "consumption_mode")
LOADING_SIM_ANALYSIS_PAIRS = (
    ("3conditions", "3conditions"),
    ("3conditions", "hedonic_valence"),
    ("3conditions", "consumption_mode"),
    ("hedonic_valence", "hedonic_valence"),
    ("hedonic_valence", "consumption_mode"),
    ("consumption_mode", "consumption_mode"),
)
LOADING_SIM_PCS = (0, 1)
LOADING_VECTOR_REFERENCE_ANALYSIS = "3conditions"
LOADING_VECTOR_ANALYSES = ("3conditions", "hedonic_valence", "consumption_mode")
LOADING_VECTOR_COLORS = {
    "3conditions": "#0b3d91",
    "hedonic_valence": "#b22222",
    "consumption_mode": "#2e8b57",
}
SCATTER_MEAN_FRAME_WINDOW = (60, 200)
TRAJ_SEM_ALPHA = 0.28
STIM_ENCODER_MARGINALS = ("s", "st")
STIM_SUBGROUP_TOP_FRAC = 0.10

SLM_COND_ID = 26

COLORS_3 = ("#0b3d91", "#b22222", "#2e8b57")
COLORS_4 = ("#0b3d91", "#b22222", "#2e8b57", "#9467bd")  # water, nacl, airpuff, slm
COLORS_2 = ("#b22222", "#0b3d91")  # class0, class1 (aversive / non-consumption first)

_STIMULUS_3 = (
    ("water", frozenset({3})),
    ("nacl", frozenset({4})),
    ("airpuff", frozenset({18})),
)
_STIMULUS_4 = _STIMULUS_3 + (("slm", frozenset({SLM_COND_ID})),)


@dataclass(frozen=True)
class AnalysisSpec:
    subdir: str
    title: str
    stimulus_levels: tuple[tuple[str, frozenset[int]], ...]
    colors: tuple[str, ...]


def _hedonic_valence_spec(run_key: str) -> AnalysisSpec:
    return AnalysisSpec(
        "hedonic_valence",
        f"{run_key} — hedonic valence (appetitive vs aversive)",
        (
            ("aversive", frozenset({4, 18})),
            ("appetitive", frozenset({3})),
        ),
        COLORS_2,
    )


def _consumption_mode_spec(run_key: str) -> AnalysisSpec:
    return AnalysisSpec(
        "consumption_mode",
        f"{run_key} — consumption mode (consumption vs non-consumption)",
        (
            ("non_consumption", frozenset({18})),
            ("consumption", frozenset({3, 4})),
        ),
        COLORS_2,
    )


def _pack_cond_ids(pack: dict) -> frozenset[int]:
    return frozenset(int(x) for x in pack["trial_type"])


def _standard_stimulus_analyses(run_key: str, *, include_slm: bool) -> tuple[AnalysisSpec, ...]:
    hedonic = _hedonic_valence_spec(run_key)
    consumption = _consumption_mode_spec(run_key)
    if include_slm:
        return (
            AnalysisSpec(
                "4conditions",
                f"{run_key} — water / nacl / airpuff / slm",
                _STIMULUS_4,
                COLORS_4,
            ),
            AnalysisSpec(
                "3conditions",
                f"{run_key} — water / nacl / airpuff (SLM excluded)",
                _STIMULUS_3,
                COLORS_3,
            ),
            hedonic,
            consumption,
            AnalysisSpec(
                "slm_vs_not",
                f"{run_key} — SLM vs non-SLM (stimulus separation)",
                (
                    ("not_slm", frozenset({3, 4, 18})),
                    ("slm", frozenset({SLM_COND_ID})),
                ),
                COLORS_2,
            ),
        )
    return (
        AnalysisSpec(
            "3conditions",
            f"{run_key} — water / nacl / airpuff",
            _STIMULUS_3,
            COLORS_3,
        ),
        hedonic,
        consumption,
    )


def _waters_waterfree_analyses(run_key: str) -> tuple[AnalysisSpec, ...]:
    return (
        AnalysisSpec(
            "consumption_mode",
            f"{run_key} — waters vs waterfree (consumption mode)",
            (
                ("non_consumption", frozenset({8})),
                ("consumption", frozenset({7})),
            ),
            COLORS_2,
        ),
    )


def analyses_for_run(run_key: str, pack: dict | None = None) -> tuple[AnalysisSpec, ...]:
    if pack is None:
        return _standard_stimulus_analyses(run_key, include_slm=False)

    cond_ids = _pack_cond_ids(pack)
    has_stim = bool(cond_ids & {3, 4, 18})
    has_slm = SLM_COND_ID in cond_ids
    has_waters = bool(cond_ids & {7, 8})

    if has_stim:
        return _standard_stimulus_analyses(run_key, include_slm=has_slm)
    if has_waters:
        return _waters_waterfree_analyses(run_key)
    return ()


RUN_KEYS: tuple[str, ...] = tuple(PHASE_KEYS)


def loading_sim_run_pairs() -> tuple[tuple[str, str], ...]:
    return tuple((a, b) for i, a in enumerate(RUN_KEYS) for b in RUN_KEYS[i:])


def _cond_to_stim_index(stimulus_levels: tuple[tuple[str, frozenset[int]], ...]) -> dict[int, int]:
    out: dict[int, int] = {}
    for si, (_name, cids) in enumerate(stimulus_levels):
        for cid in cids:
            if cid in out:
                raise ValueError(f"Condition id {cid} appears in more than one stimulus level.")
            out[cid] = si
    return out


def _collect_segments(
    pack: dict,
    stimulus_levels: tuple[tuple[str, frozenset[int]], ...],
) -> tuple[dict[int, list[np.ndarray]], int, int]:
    cond_to_s = _cond_to_stim_index(stimulus_levels)
    S = len(stimulus_levels)
    segs_by_s: dict[int, list[np.ndarray]] = {si: [] for si in range(S)}
    min_T: int | None = None
    n_neurons: int | None = None

    y_cond = pack["trial_type"]
    for local_i, seg in enumerate(pack["segments"]):
        ci = int(y_cond[local_i])
        if ci not in cond_to_s:
            continue
        segf = np.asarray(seg, dtype=np.float64)
        if segf.ndim != 2:
            continue
        nn = int(segf.shape[0])
        if n_neurons is None:
            n_neurons = nn
        elif nn != n_neurons:
            raise ValueError(f"Inconsistent n_neurons: {n_neurons} vs {nn}")
        T = int(segf.shape[1])
        if T < 2:
            continue
        segs_by_s[cond_to_s[ci]].append(segf)
        min_T = T if min_T is None else min(min_T, T)

    if min_T is None or n_neurons is None:
        names = ", ".join(n for n, _ in stimulus_levels)
        raise RuntimeError(f"No trials matched stimulus levels: {names}.")
    for si, (name, _) in enumerate(stimulus_levels):
        if not segs_by_s[si]:
            raise RuntimeError(f"No trials for stimulus level '{name}'.")

    return segs_by_s, min_T, n_neurons


def _build_mean_and_trialx(
    segs_by_s: dict[int, list[np.ndarray]],
    stimulus_levels: tuple[tuple[str, frozenset[int]], ...],
    n_frames: int,
    n_neurons: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int], dict[str, int]]:
    S = len(stimulus_levels)
    candidate_counts = {stimulus_levels[si][0]: len(segs_by_s[si]) for si in range(S)}
    kept_by_s: dict[int, list[np.ndarray]] = {}
    ignored_short_counts: dict[str, int] = {}
    counts: dict[str, int] = {}
    for si, (name, _) in enumerate(stimulus_levels):
        kept = [seg for seg in segs_by_s[si] if int(seg.shape[1]) >= n_frames]
        kept_by_s[si] = kept
        ignored_short_counts[name] = len(segs_by_s[si]) - len(kept)
        counts[name] = len(kept)
        if not kept:
            raise RuntimeError(f"No trials for stimulus level '{name}' have at least {n_frames} frames.")
    nmax = max(counts.values())
    trialX = np.full((nmax, n_neurons, S, n_frames), np.nan, dtype=np.float64)
    for si in range(S):
        for i, seg in enumerate(kept_by_s[si]):
            trialX[i, :, si, :] = seg[:, :n_frames]
    X_mean = np.nanmean(trialX, axis=0)
    return trialX, X_mean, counts, candidate_counts, ignored_short_counts


def _zscore_mean_and_trialx(X_mean: np.ndarray, trialX: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _plot_explained_variance(ev: dict, out_path: Path, title: str) -> None:
    keys = [k for k in ("t", "s", "st") if k in ev]
    if not keys:
        keys = sorted(ev.keys(), key=lambda k: (len(k), k))
    vals = [float(np.asarray(ev[k], dtype=np.float64).ravel()[0]) if np.asarray(ev[k]).size else 0.0 for k in keys]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(range(len(keys)), vals, color="steelblue", edgecolor="0.25", linewidth=0.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)
    ax.set_ylabel("Expl. var. ratio (1st dPC)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_marginal_over_time(
    Z_marg: np.ndarray,
    marg: str,
    title: str,
    stim_names: tuple[str, ...],
    colors: tuple[str, ...],
    out_path: Path,
    trial_Z: list[np.ndarray] | None = None,
) -> None:
    if Z_marg.ndim != 3 or Z_marg.shape[2] < 2:
        return
    n_show = min(2, int(Z_marg.shape[0]))
    fig, axes = plt.subplots(n_show, 1, figsize=(7.5, 2.6 * n_show), squeeze=False)
    t_ax = np.arange(Z_marg.shape[2])
    for row in range(n_show):
        ax = axes[row, 0]
        for c in range(Z_marg.shape[1]):
            col = colors[c % len(colors)]
            y = Z_marg[row, c, :]
            if trial_Z is not None and c < len(trial_Z) and trial_Z[c].shape[0] > 1:
                _, sem = _mean_sem_trials(trial_Z[c])
                if row < sem.shape[0]:
                    band = sem[row, :]
                    valid = ~np.isnan(y) & ~np.isnan(band)
                    if np.any(valid):
                        ax.fill_between(
                            t_ax[valid],
                            y[valid] - band[valid],
                            y[valid] + band[valid],
                            color=col,
                            alpha=TRAJ_SEM_ALPHA,
                            linewidth=0,
                            zorder=1,
                        )
            ax.plot(
                t_ax,
                y,
                color=col,
                linewidth=1.3,
                label=stim_names[c] if c < len(stim_names) else f"s{c}",
                zorder=3,
            )
        ax.set_ylabel(f"dPC{row + 1} ({marg})")
        ax.grid(True, alpha=0.25)
        if row == 0:
            ax.legend(fontsize=7, loc="best")
        if row == n_show - 1:
            ax.set_xlabel("Cue-relative frame")
    sem_note = " (mean ± SEM)" if trial_Z is not None else ""
    fig.suptitle(f"{title}{sem_note}", fontsize=10, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_marginal_t(
    Z_t: np.ndarray,
    stim_names: tuple[str, ...],
    colors: tuple[str, ...],
    out_path: Path,
    trial_Z: list[np.ndarray] | None = None,
) -> None:
    _plot_marginal_over_time(
        Z_t,
        "t",
        "dPCA marginal t (time-like, shared across stimuli)",
        stim_names,
        colors,
        out_path,
        trial_Z,
    )


def _plot_marginal_s(
    Z_s: np.ndarray,
    stim_names: tuple[str, ...],
    colors: tuple[str, ...],
    out_path: Path,
    trial_Z: list[np.ndarray] | None = None,
) -> None:
    _plot_marginal_over_time(
        Z_s,
        "s",
        "dPCA marginal s (stimulus-like)",
        stim_names,
        colors,
        out_path,
        trial_Z,
    )


def _plot_marginal_st(
    Z_st: np.ndarray,
    stim_names: tuple[str, ...],
    colors: tuple[str, ...],
    out_path: Path,
    trial_Z: list[np.ndarray] | None = None,
) -> None:
    _plot_marginal_over_time(
        Z_st,
        "st",
        "dPCA marginal st (stimulus x time interaction)",
        stim_names,
        colors,
        out_path,
        trial_Z,
    )


def _project_trials_marg(model: dPCA, trialX: np.ndarray, marginalization: str) -> list[np.ndarray]:
    """Per stimulus: single-trial paths in a dPCA marginalization, shape (n_trials, n_comp, T)."""
    S = trialX.shape[2]
    out: list[np.ndarray] = []
    for s in range(S):
        trajs: list[np.ndarray] = []
        for i in range(trialX.shape[0]):
            Xi = trialX[i, :, s, :]
            if np.any(np.isnan(Xi)):
                continue
            X3 = np.asarray(Xi, dtype=np.float64).reshape(Xi.shape[0], 1, Xi.shape[1])
            Z = model.transform(X3, marginalization=marginalization)
            trajs.append(np.asarray(Z[:, 0, :], dtype=np.float64))
        out.append(np.stack(trajs, axis=0) if trajs else np.zeros((0, 1, trialX.shape[3])))
    return out


def _mean_sem_trials(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM across trials; trials shape (n_trials, n_comp, T)."""
    if trials.size == 0:
        return trials, trials
    mean = np.nanmean(trials, axis=0)
    n_eff = np.sum(~np.isnan(trials), axis=0).astype(np.float64)
    centered = trials - mean[None, :, :]
    sum_sq = np.nansum(centered * centered, axis=0)
    std = np.sqrt(sum_sq / np.maximum(n_eff - 1.0, 1.0))
    sem = std / np.sqrt(np.maximum(n_eff, 1.0))
    sem[n_eff <= 1.0] = 0.0
    return mean, sem


def _fill_sem_band_2d(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    x_sem: np.ndarray,
    y_sem: np.ndarray,
    color: str,
    alpha: float,
) -> None:
    """Axis-aligned mean ± SEM envelope around a 2D parametric curve."""
    xu, xl = x + x_sem, x - x_sem
    yu, yl = y + y_sem, y - y_sem
    px = np.concatenate([xu, xl[::-1]])
    py = np.concatenate([yu, yl[::-1]])
    ax.fill(px, py, color=color, alpha=alpha, linewidth=0, zorder=1)


def _mark_cue_onset_2d(ax: plt.Axes, xs: list[float], ys: list[float]) -> None:
    if not xs:
        return
    ax.scatter(
        xs,
        ys,
        s=52,
        c="k",
        edgecolors="white",
        linewidths=1.4,
        zorder=100,
    )


def _mark_cue_onset_3d(ax: Axes3D, points: list[tuple[float, float, float]]) -> None:
    if not points:
        return
    xs, ys, zs = zip(*points)
    ax.scatter(
        xs,
        ys,
        zs,
        s=56,
        c="k",
        edgecolors="white",
        linewidths=1.4,
        depthshade=False,
        zorder=100,
    )


def _plot_traj_marg_2d(
    Z: np.ndarray,
    marg: str,
    stim_names: tuple[str, ...],
    colors: tuple[str, ...],
    out_path: Path,
    trial_Z: list[np.ndarray] | None = None,
) -> None:
    """Paths in dPC space (parametric in cue time): 2D projections with SEM shading."""
    n_comp, n_stim, _T = Z.shape
    if n_comp < 2 or n_stim < 1 or Z.shape[2] < 2:
        return
    pairs = [(0, 1), (0, 2), (1, 2)] if n_comp >= 3 else [(0, 1)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.0 * len(pairs), 3.8), squeeze=False)
    for ax_i, (i0, i1) in enumerate(pairs):
        ax = axes[0, ax_i]
        starts_x: list[float] = []
        starts_y: list[float] = []
        for c in range(n_stim):
            col = colors[c % len(colors)]
            lbl = stim_names[c] if c < len(stim_names) else f"s{c}"
            mx, my = Z[i0, c, :], Z[i1, c, :]
            if trial_Z is not None and c < len(trial_Z) and trial_Z[c].shape[0] > 1:
                _, sem = _mean_sem_trials(trial_Z[c])
                _fill_sem_band_2d(ax, mx, my, sem[i0, :], sem[i1, :], col, TRAJ_SEM_ALPHA)
            ax.plot(mx, my, color=col, linewidth=1.6, label=lbl, zorder=3)
            starts_x.append(float(mx[0]))
            starts_y.append(float(my[0]))
        _mark_cue_onset_2d(ax, starts_x, starts_y)
        ax.set_xlabel(f"dPC{i0 + 1} ({marg})")
        ax.set_ylabel(f"dPC{i1 + 1} ({marg})")
        ax.set_title(f"axes {i0 + 1} vs {i1 + 1}")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("auto")
        if ax_i == len(pairs) - 1:
            ax.legend(fontsize=7)
    fig.suptitle(f"dPCA {marg} trajectories over cue time (mean ± SEM, 2D)", fontsize=10, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_traj_marg_3d(
    Z: np.ndarray,
    marg: str,
    stim_names: tuple[str, ...],
    colors: tuple[str, ...],
    out_path: Path,
) -> None:
    """Paths in dPC space over cue time: 3D mean trajectories only (SEM shading in 2D only)."""
    if Z.ndim != 3 or Z.shape[1] < 1 or Z.shape[2] < 2:
        return
    T = Z.shape[2]
    use_dpc3 = Z.shape[0] > 2
    fig = plt.figure(figsize=(7.0, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    onset_pts: list[tuple[float, float, float]] = []
    for c in range(Z.shape[1]):
        col = colors[c % len(colors)]
        lbl = stim_names[c] if c < len(stim_names) else f"s{c}"
        x = Z[0, c, :]
        y = Z[1, c, :] if Z.shape[0] > 1 else np.zeros(T)
        z = Z[2, c, :] if use_dpc3 else np.arange(T, dtype=np.float64) * 0.01
        ax.plot(x, y, z, color=col, label=lbl, linewidth=1.6, zorder=3)
        onset_pts.append((float(x[0]), float(y[0]), float(z[0])))
    _mark_cue_onset_3d(ax, onset_pts)
    ax.set_xlabel(f"dPC1 ({marg})")
    ax.set_ylabel(f"dPC2 ({marg})")
    ax.set_zlabel(f"dPC3 ({marg})" if use_dpc3 else "cue frame (scaled)")
    ax.set_title(f"dPCA {marg} trajectories over cue time (3D)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pc12_trial_scatter(
    trial_Z: list[np.ndarray],
    marg: str,
    stim_names: tuple[str, ...],
    colors: tuple[str, ...],
    out_path: Path,
) -> None:
    """Single-trial window-mean dPC1 vs dPC2 points, colored by stimulus label."""
    if not trial_Z:
        return
    has_points = any(trials.ndim == 3 and trials.shape[0] > 0 and trials.shape[1] >= 2 for trials in trial_Z)
    if not has_points:
        return

    win_start, win_stop = SCATTER_MEAN_FRAME_WINDOW
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for c, trials in enumerate(trial_Z):
        if trials.ndim != 3 or trials.shape[0] < 1 or trials.shape[1] < 2:
            continue
        frame_stop = min(win_stop, trials.shape[2])
        if frame_stop <= win_start:
            continue
        x = np.nanmean(trials[:, 0, win_start:frame_stop], axis=1)
        y = np.nanmean(trials[:, 1, win_start:frame_stop], axis=1)
        valid = ~np.isnan(x) & ~np.isnan(y)
        if not np.any(valid):
            continue
        xv = x[valid]
        yv = y[valid]
        col = colors[c % len(colors)]
        lbl = stim_names[c] if c < len(stim_names) else f"s{c}"
        all_x.append(xv)
        all_y.append(yv)
        ax.scatter(
            xv,
            yv,
            s=30,
            alpha=0.78,
            color=col,
            edgecolors="none",
            label=f"{lbl} (n={xv.size})",
        )

    if all_x and all_y:
        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        max_abs = float(max(np.max(np.abs(x_all)), np.max(np.abs(y_all))))
        if max_abs < 1e-12:
            ax.text(
                0.02,
                0.02,
                "Trial means are ~0 for this marginal",
                transform=ax.transAxes,
                fontsize=8,
                color="0.35",
                ha="left",
                va="bottom",
            )

    ax.axhline(0.0, color="0.55", linewidth=0.7, linestyle="--", zorder=0)
    ax.axvline(0.0, color="0.55", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_xlabel(f"Mean dPC1 ({marg}), frames {win_start}:{win_stop}")
    ax.set_ylabel(f"Mean dPC2 ({marg}), frames {win_start}:{win_stop}")
    ax.set_title(
        f"dPCA {marg}: trial-window mean PC1 vs PC2\n"
        f"frames {win_start}:{win_stop}; one point per trial"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _per_neuron_stimulus_separation(X_mean: np.ndarray) -> np.ndarray:
    """Mean over time of (max - min) response across stimulus conditions. Shape (N,)."""
    return np.mean(np.ptp(X_mean, axis=1), axis=1)


def _concentration_top_fraction(abs_w: np.ndarray, top_frac: float) -> float:
    abs_w = np.asarray(abs_w, dtype=np.float64).ravel()
    s = float(abs_w.sum())
    if s <= 0:
        return float("nan")
    k = max(1, int(np.ceil(len(abs_w) * top_frac)))
    top_ix = np.argsort(abs_w)[-k:]
    return float(abs_w[top_ix].sum() / s)


def _gmm_unimodal_test(weights: np.ndarray) -> dict:
    """BIC comparison: 1 vs 2 Gaussian components on encoder weights (paper Fig. 6 / S7)."""
    from sklearn.mixture import GaussianMixture

    x = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    g1 = GaussianMixture(n_components=1, random_state=0, n_init=3).fit(x)
    g2 = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(x)
    bic1, bic2 = float(g1.bic(x)), float(g2.bic(x))
    return {
        "bic_1": bic1,
        "bic_2": bic2,
        "delta_bic_2_minus_1": bic2 - bic1,
        "prefers_two_clusters": bic2 < bic1,
    }


def _plot_encoder_weight_distributions(
    P: np.ndarray, marg: str, out_path: Path, n_pcs: int, bin_width: float = 0.005
) -> None:
    """Paper Figure 6 style: KDE of encoder weights per dPC, centred near zero."""
    n_show = min(n_pcs, P.shape[1])
    if n_show < 1:
        return
    wmin = float(np.min(P[:, :n_show]))
    wmax = float(np.max(P[:, :n_show]))
    pad = max(0.02, 0.05 * (wmax - wmin + 1e-8))
    bins = np.arange(wmin - pad, wmax + pad + bin_width, bin_width)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for k in range(n_show):
        col = P[:, k]
        ax.hist(col, bins=bins, density=True, histtype="step", linewidth=1.2, label=f"dPC{k + 1}")
    ax.axvline(0.0, color="0.45", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Encoder weight")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Encoder weight distributions ({marg})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _analyze_stimulus_neuron_subgroups(
    P_dict: dict,
    X_mean: np.ndarray,
    load_dir: Path,
    n_pcs: int,
) -> dict:
    """
    Test whether stimulus-related encoder weights concentrate in a neuron subgroup
    (vs paper's unimodal, centre-at-zero population distribution).
    """
    load_dir.mkdir(parents=True, exist_ok=True)
    sep = _per_neuron_stimulus_separation(X_mean)
    report: dict = {"stimulus_separation_index": {"description": "mean_t(max_s - min_s) fluorescence per neuron"}}

    for marg in STIM_ENCODER_MARGINALS:
        if marg not in P_dict:
            continue
        P = np.asarray(P_dict[marg], dtype=np.float64)
        n_show = min(n_pcs, P.shape[1])
        _plot_encoder_weight_distributions(P, marg, load_dir / f"encoder_weight_distributions_{marg}.png", n_show)

        marg_out: dict = {"per_dpc": []}
        top_sets: list[set[int]] = []
        n_neurons = P.shape[0]
        k_top = max(1, int(np.ceil(n_neurons * STIM_SUBGROUP_TOP_FRAC)))

        for ki in range(n_show):
            w = P[:, ki]
            aw = np.abs(w)
            gmm = _gmm_unimodal_test(w)
            top_ix = np.argsort(aw)[-k_top:]
            top_sets.append(set(int(i) for i in top_ix))
            r = float(np.corrcoef(sep, aw)[0, 1]) if np.std(aw) > 0 and np.std(sep) > 0 else float("nan")
            marg_out["per_dpc"].append(
                {
                    "dpc_index": ki,
                    "concentration_top_fraction": _concentration_top_fraction(aw, STIM_SUBGROUP_TOP_FRAC),
                    "gmm": gmm,
                    "corr_abs_loading_vs_stimulus_separation": r,
                    "top_neuron_indices": [int(i) for i in top_ix[::-1][:15]],
                }
            )

        if len(top_sets) >= 2:
            inter = set.intersection(*top_sets)
            union = set.union(*top_sets)
            marg_out["top_fraction_overlap_jaccard"] = (
                float(len(inter) / len(union)) if union else float("nan")
            )
        marg_out["interpretation"] = (
            "Paper (Fig. 6): weights unimodal at zero, no subpopulations. "
            "prefers_two_clusters=True or high concentration_top_fraction suggests a dedicated subgroup; "
            "high corr_abs_loading_vs_stimulus_separation links dPCA weights to raw condition tuning."
        )
        report[marg] = marg_out

        if "s" in marg and n_show >= 1:
            w0 = P[:, 0]
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.5, 3.8))
            ax0.hist(w0, bins=80, density=True, color="0.75", edgecolor="none")
            ax0.axvline(0.0, color="k", linewidth=0.8, linestyle="--")
            if marg_out["per_dpc"][0]["gmm"]["prefers_two_clusters"]:
                from sklearn.mixture import GaussianMixture

                g2 = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(w0.reshape(-1, 1))
                for mu in g2.means_.ravel():
                    ax0.axvline(float(mu), color="#b22222", linewidth=1.2, linestyle=":")
            ax0.set_xlabel("Encoder weight dPC1 (s)")
            ax0.set_ylabel("Density")
            ax0.set_title("dPC1 (s) weight distribution")

            ax1.scatter(sep, np.abs(w0), s=8, alpha=0.35, c="0.35", edgecolors="none")
            top_ix = np.argsort(np.abs(w0))[-k_top:]
            ax1.scatter(sep[top_ix], np.abs(w0[top_ix]), s=14, c="#b22222", edgecolors="k", linewidths=0.3)
            ax1.set_xlabel("Stimulus separation (data)")
            ax1.set_ylabel("|Encoder weight| dPC1 (s)")
            ax1.set_title(f"Top {int(STIM_SUBGROUP_TOP_FRAC * 100)}% |loading| highlighted")
            ax1.grid(True, alpha=0.25)
            fig.suptitle("Stimulus subpopulation check (encoder s)", fontsize=10, y=1.02)
            fig.tight_layout()
            fig.savefig(load_dir / "stimulus_subgroup_dpc1_s.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    (load_dir / "stimulus_subgroup_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _save_and_plot_loadings(
    dpca_fit: dPCA,
    out_dir: Path,
    marginals: tuple[str, ...],
    n_pcs: int,
    top_k: int,
    X_mean: np.ndarray | None = None,
) -> None:
    P_dict = getattr(dpca_fit, "P", None)
    D_dict = getattr(dpca_fit, "D", None)
    if not isinstance(P_dict, dict) or not isinstance(D_dict, dict):
        return

    load_dir = out_dir / "loadings"
    load_dir.mkdir(parents=True, exist_ok=True)
    npz_kw: dict[str, np.ndarray] = {}
    for marg in marginals:
        if marg in P_dict:
            npz_kw[f"encoder_P_{marg}"] = np.asarray(P_dict[marg], dtype=np.float32)
        if marg in D_dict:
            npz_kw[f"decoder_D_{marg}"] = np.asarray(D_dict[marg], dtype=np.float32)
    if npz_kw:
        np.savez_compressed(load_dir / "encoder_decoder_matrices.npz", **npz_kw)

    csv_path = load_dir / "encoder_top_neurons.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["marginalization", "dpc_index", "neuron_index", "encoder_loading"])
        for marg in marginals:
            if marg not in P_dict:
                continue
            P = np.asarray(P_dict[marg], dtype=np.float64)
            n_show = min(n_pcs, P.shape[1])
            for k in range(n_show):
                col = P[:, k]
                for ni in np.argsort(np.abs(col))[-top_k:][::-1]:
                    w.writerow([marg, k, int(ni), float(col[int(ni)])])

    for marg in marginals:
        if marg not in P_dict:
            continue
        P = np.asarray(P_dict[marg], dtype=np.float64)
        n_show = min(n_pcs, P.shape[1])
        fig, axes = plt.subplots(1, n_show, figsize=(3.4 * n_show, 5.8), squeeze=False)
        for k in range(n_show):
            col = P[:, k]
            ix = np.argsort(np.abs(col))[-top_k:][::-1]
            ax = axes[0, k]
            vals = col[ix]
            bar_colors = ["#b22222" if v < 0 else "#0b3d91" for v in vals]
            ax.barh(np.arange(top_k), vals, color=bar_colors, edgecolor="none")
            ax.invert_yaxis()
            ax.set_yticks(np.arange(top_k))
            ax.set_yticklabels([str(int(i)) for i in ix], fontsize=6)
            ax.axvline(0.0, color="0.5", linewidth=0.6)
            ax.set_xlabel("encoder weight")
            ax.set_title(f"dPC{k + 1}")
            ax.grid(True, axis="x", alpha=0.25)
        fig.suptitle(
            f"Encoder loadings ({marg}) — top {top_k} neurons by |weight|",
            fontsize=10,
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(load_dir / f"encoder_loadings_{marg}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if X_mean is not None:
        _analyze_stimulus_neuron_subgroups(P_dict, X_mean, load_dir / "population_encoder", n_pcs)


def _load_encoder_matrices(load_path: Path) -> dict[str, np.ndarray] | None:
    if not load_path.exists():
        return None
    with np.load(load_path) as data:
        return {str(k): np.asarray(data[k], dtype=np.float64) for k in data.files}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    if av.shape != bv.shape:
        return None
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0:
        return None
    return float(np.dot(av, bv) / denom)


def _encoder_loadings_path(dpca_root: Path, run_key: str, analysis: str) -> Path:
    return dpca_root / run_key / analysis / "loadings" / "encoder_decoder_matrices.npz"


def _loading_similarity_rows_for_pair(
    dpca_root: Path,
    run_a: str,
    run_b: str,
    warnings: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], dict[str, np.ndarray] | None] = {}

    def matrices_for(run_key: str, analysis: str) -> dict[str, np.ndarray] | None:
        cache_key = (run_key, analysis)
        if cache_key not in cache:
            load_path = _encoder_loadings_path(dpca_root, run_key, analysis)
            cache[cache_key] = _load_encoder_matrices(load_path)
            if cache[cache_key] is None:
                warnings.append(f"Missing encoder matrices: {load_path}")
        return cache[cache_key]

    for analysis_a, analysis_b in LOADING_SIM_ANALYSIS_PAIRS:
        mats_a = matrices_for(run_a, analysis_a)
        mats_b = matrices_for(run_b, analysis_b)
        if mats_a is None or mats_b is None:
            continue

        for marg in LOADING_MARGINALS:
            key = f"encoder_P_{marg}"
            if key not in mats_a:
                warnings.append(f"Missing {key}: {run_a}/{analysis_a}")
                continue
            if key not in mats_b:
                warnings.append(f"Missing {key}: {run_b}/{analysis_b}")
                continue
            Pa = np.asarray(mats_a[key], dtype=np.float64)
            Pb = np.asarray(mats_b[key], dtype=np.float64)
            if Pa.shape[0] != Pb.shape[0]:
                warnings.append(
                    f"Neuron-count mismatch for {key}: "
                    f"{run_a}/{analysis_a} has {Pa.shape[0]}, {run_b}/{analysis_b} has {Pb.shape[0]}"
                )
                continue

            for pc_ix in LOADING_SIM_PCS:
                if pc_ix >= Pa.shape[1] or pc_ix >= Pb.shape[1]:
                    warnings.append(
                        f"Missing PC{pc_ix + 1} in {key}: "
                        f"{run_a}/{analysis_a} shape={Pa.shape}, {run_b}/{analysis_b} shape={Pb.shape}"
                    )
                    continue
                cos = _cosine_similarity(Pa[:, pc_ix], Pb[:, pc_ix])
                if cos is None:
                    warnings.append(
                        f"Could not compute cosine for {key} PC{pc_ix + 1}: "
                        f"{run_a}/{analysis_a} vs {run_b}/{analysis_b}"
                    )
                    continue
                rows.append(
                    {
                        "run_a": run_a,
                        "run_b": run_b,
                        "analysis_a": analysis_a,
                        "analysis_b": analysis_b,
                        "marginalization": marg,
                        "pc": pc_ix + 1,
                        "cosine_similarity": cos,
                    }
                )

    return rows


def _plot_loading_similarity_heatmaps(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    run_pair: str,
    row_order: tuple[tuple[str, str], ...] = LOADING_SIM_ANALYSIS_PAIRS,
    row_group_break: int | None = None,
    title_suffix: str = "",
) -> None:
    if not rows:
        return

    pair_labels = [f"{analysis_a}\nvs\n{analysis_b}" for analysis_a, analysis_b in row_order]
    pair_to_ix = {pair: ix for ix, pair in enumerate(row_order)}
    columns = [(marg, pc_ix + 1) for marg in LOADING_MARGINALS for pc_ix in LOADING_SIM_PCS]
    col_to_ix = {col: ix for ix, col in enumerate(columns)}
    data = np.full((len(pair_labels), len(columns)), np.nan, dtype=np.float64)

    for row in rows:
        pair = (str(row["analysis_a"]), str(row["analysis_b"]))
        col = (str(row["marginalization"]), int(row["pc"]))
        if pair not in pair_to_ix or col not in col_to_ix:
            continue
        data[pair_to_ix[pair], col_to_ix[col]] = float(row["cosine_similarity"])

    fig_w = max(13.0, 2.2 * len(columns) + 3.0)
    fig_h = max(3.5, 0.55 * len(pair_labels) + 1.6)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)
    title = f"{run_pair} loading cosine similarity"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    fig.suptitle(title, y=1.02)

    plot_configs = (
        ("Signed cosine", data, "RdBu_r", -1.0, 1.0, "Cosine similarity"),
        ("Absolute cosine", np.abs(data), "viridis", 0.0, 1.0, "Absolute cosine similarity"),
    )
    for ax, (title, values, cmap_name, vmin, vmax, cbar_label) in zip(axes, plot_configs):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#f0f0f0")
        im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels([f"{marg} PC{pc}" for marg, pc in columns], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pair_labels)))
        ax.set_yticklabels(pair_labels)
        if row_group_break is not None and 0 < row_group_break < len(pair_labels):
            ax.axhline(row_group_break - 0.5, color="0.15", linewidth=1.1)

        for row_ix in range(values.shape[0]):
            for col_ix in range(values.shape[1]):
                if np.isfinite(values[row_ix, col_ix]):
                    value = values[row_ix, col_ix]
                    text_color = "white" if abs(value) > 0.65 else "black"
                    ax.text(col_ix, row_ix, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_loading_similarity_heatmap_signed_marg(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    run_pair: str,
    marginalization: str,
    row_order: tuple[tuple[str, str], ...] = LOADING_SIM_ANALYSIS_PAIRS,
    row_group_break: int | None = None,
    title_suffix: str = "",
) -> None:
    """Signed cosine heatmap for one marginalization (PC1 and PC2 only)."""
    if not rows:
        return

    pair_labels = [f"{analysis_a}\nvs\n{analysis_b}" for analysis_a, analysis_b in row_order]
    pair_to_ix = {pair: ix for ix, pair in enumerate(row_order)}
    columns = [(marginalization, pc_ix + 1) for pc_ix in LOADING_SIM_PCS]
    col_to_ix = {col: ix for ix, col in enumerate(columns)}
    data = np.full((len(pair_labels), len(columns)), np.nan, dtype=np.float64)

    for row in rows:
        if str(row["marginalization"]) != marginalization:
            continue
        pair = (str(row["analysis_a"]), str(row["analysis_b"]))
        col = (marginalization, int(row["pc"]))
        if pair not in pair_to_ix or col not in col_to_ix:
            continue
        data[pair_to_ix[pair], col_to_ix[col]] = float(row["cosine_similarity"])

    fig_w = max(5.5, 2.2 * len(columns) + 2.5)
    fig_h = max(3.5, 0.55 * len(pair_labels) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    title = f"{run_pair} loading cosine similarity ({marginalization})"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    fig.suptitle(title, y=1.02)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f0f0f0")
    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([f"{marg} PC{pc}" for marg, pc in columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pair_labels)))
    ax.set_yticklabels(pair_labels)
    if row_group_break is not None and 0 < row_group_break < len(pair_labels):
        ax.axhline(row_group_break - 0.5, color="0.15", linewidth=1.1)

    for row_ix in range(data.shape[0]):
        for col_ix in range(data.shape[1]):
            if np.isfinite(data[row_ix, col_ix]):
                value = data[row_ix, col_ix]
                text_color = "white" if abs(value) > 0.65 else "black"
                ax.text(col_ix, row_ix, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _loading_similarity_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, str, int], float]:
    lookup: dict[tuple[str, str, str, int], float] = {}
    for row in rows:
        key = (
            str(row["analysis_a"]),
            str(row["analysis_b"]),
            str(row["marginalization"]),
            int(row["pc"]),
        )
        lookup[key] = float(row["cosine_similarity"])
    return lookup


def _angle_from_cosine(cosine_similarity: float, reference_angle: float) -> float:
    cos = float(np.clip(cosine_similarity, -1.0, 1.0))
    return reference_angle + float(np.arccos(cos))


def _vector_for_similarity(
    lookup: dict[tuple[str, str, str, int], float],
    *,
    analysis: str,
    marginalization: str,
    pc: int,
    use_absolute: bool = False,
) -> tuple[float, float] | None:
    reference_angle = 0.0 if pc == 1 else np.pi / 2.0
    key = (analysis, analysis, marginalization, pc)
    if key not in lookup:
        return None
    cos = abs(lookup[key]) if use_absolute else lookup[key]
    angle = _angle_from_cosine(cos, reference_angle)
    return float(np.cos(angle)), float(np.sin(angle))


def _pairwise_cosine_value(
    lookup: dict[tuple[str, str, str, int], float],
    analysis_a: str,
    analysis_b: str,
    marginalization: str,
    pc: int,
    use_absolute: bool = False,
) -> float | None:
    key = (analysis_a, analysis_b, marginalization, pc)
    if key in lookup:
        return abs(lookup[key]) if use_absolute else lookup[key]
    key = (analysis_b, analysis_a, marginalization, pc)
    if key in lookup:
        return abs(lookup[key]) if use_absolute else lookup[key]
    if analysis_a == analysis_b:
        return 1.0
    return None


def _pairwise_loading_embedding(
    lookup: dict[tuple[str, str, str, int], float],
    *,
    marginalization: str,
    pc: int,
    use_absolute: bool = False,
) -> dict[str, tuple[float, float]]:
    analyses = LOADING_VECTOR_ANALYSES
    gram = np.eye(len(analyses), dtype=np.float64)
    for i, analysis_i in enumerate(analyses):
        for j, analysis_j in enumerate(analyses):
            if i >= j:
                continue
            value = _pairwise_cosine_value(
                lookup,
                analysis_i,
                analysis_j,
                marginalization,
                pc,
                use_absolute=use_absolute,
            )
            if value is None:
                return {}
            gram[i, j] = gram[j, i] = float(np.clip(value, -1.0, 1.0))

    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:2]
    vals = np.maximum(eigvals[order], 0.0)
    coords = eigvecs[:, order] * np.sqrt(vals)[None, :]

    ref_ix = analyses.index(LOADING_VECTOR_REFERENCE_ANALYSIS)
    ref = coords[ref_ix]
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm > 1e-12:
        angle = float(np.arctan2(ref[1], ref[0]))
        target_angle = 0.0 if pc == 1 else np.pi / 2.0
        rot = target_angle - angle
        rot_mat = np.asarray(
            [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]],
            dtype=np.float64,
        )
        coords = coords @ rot_mat.T

    return {
        analysis: (float(coords[ix, 0]), float(coords[ix, 1]))
        for ix, analysis in enumerate(analyses)
    }


def _draw_loading_similarity_arrow(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    color: str,
    linestyle: str,
    alpha: float,
) -> None:
    ax.annotate(
        "",
        xy=(x, y),
        xytext=(0.0, 0.0),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linestyle": linestyle,
            "linewidth": 1.8,
            "alpha": alpha,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
    )


def _plot_loading_similarity_vectors(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    run_pair: str,
    use_absolute: bool = False,
) -> None:
    """Schematic 2D arrows from pairwise loading cosines; not an exact embedding."""
    if not rows:
        return

    lookup = _loading_similarity_lookup(rows)
    row_configs = (
        "Same-label stability\nread each color separately",
        "Between-label geometry\ncompare colors",
    )
    fig, axes = plt.subplots(
        len(row_configs),
        len(LOADING_MARGINALS),
        figsize=(9.4, 7.0),
        squeeze=False,
    )
    fig.suptitle(
        f"{run_pair}: schematic PC1/PC2 loading-axis rotation\n"
        f"{'absolute' if use_absolute else 'signed'} cosine; "
        "top row = heatmap rows 1, 4, 6 only; bottom row = heatmap rows 1-6 together",
        fontsize=10,
        y=0.965,
    )

    for row_ix, row_label in enumerate(row_configs):
        for col_ix, marg in enumerate(LOADING_MARGINALS):
            ax = axes[row_ix, col_ix]
            circle = plt.Circle((0.0, 0.0), 1.0, edgecolor="0.82", facecolor="none", linewidth=0.8)
            ax.add_patch(circle)
            ax.axhline(0.0, color="0.82", linewidth=0.7, zorder=0)
            ax.axvline(0.0, color="0.82", linewidth=0.7, zorder=0)

            for analysis in LOADING_VECTOR_ANALYSES:
                color = LOADING_VECTOR_COLORS[analysis]
                for pc in (1, 2):
                    if row_ix == 0:
                        vec = _vector_for_similarity(
                            lookup,
                            analysis=analysis,
                            marginalization=marg,
                            pc=pc,
                            use_absolute=use_absolute,
                        )
                    else:
                        embedded = _pairwise_loading_embedding(
                            lookup,
                            marginalization=marg,
                            pc=pc,
                            use_absolute=use_absolute,
                        )
                        vec = embedded.get(analysis)
                    if vec is None:
                        continue
                    x, y = vec
                    linestyle = "-" if pc == 1 else "--"
                    alpha = 0.95 if pc == 1 else 0.65
                    _draw_loading_similarity_arrow(ax, x, y, color=color, linestyle=linestyle, alpha=alpha)

            ax.set_title(marg if row_ix == 0 else "")
            if col_ix == 0:
                ax.set_ylabel(row_label)
            ax.set_xlim(-1.35, 1.35)
            ax.set_ylim(-1.35, 1.35)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks((-1.0, 0.0, 1.0))
            ax.set_yticks((-1.0, 0.0, 1.0))
            ax.grid(True, alpha=0.18)

    color_handles = [
        Line2D([0], [0], color=LOADING_VECTOR_COLORS[analysis], lw=2, label=analysis)
        for analysis in LOADING_VECTOR_ANALYSES
    ]
    style_handles = [
        Line2D([0], [0], color="0.25", lw=2, linestyle="-", label="PC1"),
        Line2D([0], [0], color="0.25", lw=2, linestyle="--", label="PC2"),
    ]
    fig.legend(
        handles=color_handles,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=len(color_handles),
        frameon=False,
    )
    fig.legend(
        handles=style_handles,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        ncol=len(style_handles),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84), w_pad=0.25, h_pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _safe_plot_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _plot_loading_vector_scatter(
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    *,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    xv = np.asarray(x, dtype=np.float64).ravel()
    yv = np.asarray(y, dtype=np.float64).ravel()
    valid = ~np.isnan(xv) & ~np.isnan(yv)
    if not np.any(valid):
        return
    xv = xv[valid]
    yv = yv[valid]

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.scatter(xv, yv, s=12, alpha=0.45, color="0.25", edgecolors="none")
    ax.axhline(0.0, color="0.55", linewidth=0.7, linestyle="--")
    ax.axvline(0.0, color="0.55", linewidth=0.7, linestyle="--")

    lo = float(min(np.min(xv), np.min(yv)))
    hi = float(max(np.max(xv), np.max(yv)))
    if np.isfinite(lo) and np.isfinite(hi):
        pad = max(1e-8, 0.08 * (hi - lo if hi > lo else max(abs(hi), 1e-8)))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.65", linewidth=0.8, zorder=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_loading_vector_scatters_for_pair(
    dpca_root: Path,
    out_dir: Path,
    run_a: str,
    run_b: str,
    warnings: list[str],
) -> int:
    scatter_dir = out_dir / "loading_scatter"
    n_written = 0
    cache: dict[tuple[str, str], dict[str, np.ndarray] | None] = {}

    def matrices_for(run_key: str, analysis: str) -> dict[str, np.ndarray] | None:
        cache_key = (run_key, analysis)
        if cache_key not in cache:
            load_path = _encoder_loadings_path(dpca_root, run_key, analysis)
            cache[cache_key] = _load_encoder_matrices(load_path)
            if cache[cache_key] is None:
                warnings.append(f"Missing encoder matrices for scatter: {load_path}")
        return cache[cache_key]

    for analysis_a, analysis_b in LOADING_SIM_ANALYSIS_PAIRS:
        if run_a == run_b and analysis_a == analysis_b:
            continue
        mats_a = matrices_for(run_a, analysis_a)
        mats_b = matrices_for(run_b, analysis_b)
        if mats_a is None or mats_b is None:
            continue

        for marg in LOADING_MARGINALS:
            key = f"encoder_P_{marg}"
            if key not in mats_a or key not in mats_b:
                continue
            Pa = np.asarray(mats_a[key], dtype=np.float64)
            Pb = np.asarray(mats_b[key], dtype=np.float64)
            if Pa.shape[0] != Pb.shape[0]:
                continue

            for pc_ix in LOADING_SIM_PCS:
                if pc_ix >= Pa.shape[1] or pc_ix >= Pb.shape[1]:
                    continue
                x = Pa[:, pc_ix]
                y = Pb[:, pc_ix]
                file_name = (
                    f"{_safe_plot_token(analysis_a)}_vs_{_safe_plot_token(analysis_b)}_"
                    f"{marg}_pc{pc_ix + 1}.png"
                )
                _plot_loading_vector_scatter(
                    x,
                    y,
                    scatter_dir / file_name,
                    x_label=f"{run_a} {analysis_a} {marg} PC{pc_ix + 1} loading",
                    y_label=f"{run_b} {analysis_b} {marg} PC{pc_ix + 1} loading",
                    title=f"{run_a}/{analysis_a} vs {run_b}/{analysis_b} ({marg} PC{pc_ix + 1})",
                )
                n_written += 1

    return n_written


def _write_loading_similarity_outputs(dpca_root: Path) -> None:
    out_root = dpca_root / "loading_similarity"
    fieldnames = [
        "run_a",
        "run_b",
        "analysis_a",
        "analysis_b",
        "marginalization",
        "pc",
        "cosine_similarity",
    ]

    for run_a, run_b in loading_sim_run_pairs():
        pair_name = f"{run_a}_vs_{run_b}"
        out_dir = out_root / pair_name
        out_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        rows = _loading_similarity_rows_for_pair(dpca_root, run_a, run_b, warnings)

        csv_path = out_dir / "loading_similarity.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

        payload = {
            "run_pair": pair_name,
            "run_a": run_a,
            "run_b": run_b,
            "analyses": list(LOADING_SIM_ANALYSES),
            "analysis_pairs": [list(pair) for pair in LOADING_SIM_ANALYSIS_PAIRS],
            "marginalizations": list(LOADING_MARGINALS),
            "pcs": [pc + 1 for pc in LOADING_SIM_PCS],
            "rows": rows,
            "warnings": warnings,
        }
        (out_dir / "loading_similarity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _plot_loading_similarity_heatmaps(
            rows,
            out_dir / "loading_similarity_heatmaps.png",
            run_pair=pair_name,
        )
        same_label_pairs = tuple(pair for pair in LOADING_SIM_ANALYSIS_PAIRS if pair[0] == pair[1])
        between_label_pairs = tuple(pair for pair in LOADING_SIM_ANALYSIS_PAIRS if pair[0] != pair[1])
        if run_a == run_b:
            signed_row_order = between_label_pairs
            signed_row_group_break = None
            signed_title_suffix = "between-label only"
        else:
            signed_row_order = (*same_label_pairs, *between_label_pairs)
            signed_row_group_break = len(same_label_pairs)
            signed_title_suffix = "same-label first"
        for marg in ("t", "s"):
            _plot_loading_similarity_heatmap_signed_marg(
                rows,
                out_dir / f"loading_similarity_heatmap_{marg}_signed.png",
                run_pair=pair_name,
                marginalization=marg,
                row_order=signed_row_order,
                row_group_break=signed_row_group_break,
                title_suffix=signed_title_suffix,
            )
        _plot_loading_similarity_heatmaps(
            rows,
            out_dir / "loading_similarity_heatmaps_grouped.png",
            run_pair=pair_name,
            row_order=(*same_label_pairs, *between_label_pairs),
            row_group_break=len(same_label_pairs),
            title_suffix="same-label first",
        )
        _plot_loading_similarity_vectors(
            rows,
            out_dir / "loading_similarity_vectors.png",
            run_pair=pair_name,
        )
        _plot_loading_similarity_vectors(
            rows,
            out_dir / "loading_similarity_vectors_abs.png",
            run_pair=pair_name,
            use_absolute=True,
        )
        n_scatter = _plot_loading_vector_scatters_for_pair(dpca_root, out_dir, run_a, run_b, warnings)

        print(f"[loading_similarity/{pair_name}] wrote {len(rows)} rows to {out_dir}")
        print(f"  loading scatter plots: {n_scatter}")
        for warning in warnings:
            print(f"  warning: {warning}")


def run_analysis(spec: AnalysisSpec, pack: dict, *, run_key: str, out_root: Path) -> None:
    out_dir = out_root / spec.subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    segs_by_s, min_T, n_neurons = _collect_segments(pack, spec.stimulus_levels)
    trialX, X_mean, counts, candidate_counts, ignored_short_counts = _build_mean_and_trialx(
        segs_by_s,
        spec.stimulus_levels,
        ANALYSIS_T_FRAMES,
        n_neurons,
    )
    stim_names = tuple(n for n, _ in spec.stimulus_levels)

    meta = {
        "run_key": run_key,
        "analysis": spec.subdir,
        "title": spec.title,
        "stimulus_levels": [
            {"name": n, "condition_ids": sorted(cids)} for n, cids in spec.stimulus_levels
        ],
        "trial_counts": counts,
        "candidate_trial_counts": candidate_counts,
        "ignored_short_trial_counts": ignored_short_counts,
        "ignored_short_trial_total": int(sum(ignored_short_counts.values())),
        "frame_window": ANALYSIS_T_FRAMES,
        "min_candidate_trial_frames": min_T,
        "n_neurons": n_neurons,
        "n_trials_max_per_stimulus": int(trialX.shape[0]),
        "tensor_shape_mean": list(X_mean.shape),
        "labels": "st",
        "regularizer": REGULARIZER,
        "n_components": N_COMPONENTS,
    }

    X_fit, trialX_fit = _zscore_mean_and_trialx(X_mean, trialX)
    S, T = X_fit.shape[1], X_fit.shape[2]
    ncomp = min(max(3, N_COMPONENTS), 10, S * T)
    model = dPCA(labels="st", n_components=ncomp, regularizer=REGULARIZER)
    model.debug = 0
    model.protect = ["t"]
    Z = model.fit_transform(X_fit, trialX=trialX_fit)

    lam = getattr(model, "regularizer", None)
    if lam is not None:
        meta["optimal_regularizer"] = float(lam) if np.isscalar(lam) else lam
    meta["marginalization_keys"] = sorted(Z.keys())
    (out_dir / "trial_counts.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    z_kw = {f"Z_{k}": np.asarray(v, dtype=np.float32) for k, v in Z.items()}
    np.savez_compressed(out_dir / "dpca_result.npz", **z_kw)

    ev_obj = getattr(model, "explained_variance_ratio_", None)
    if isinstance(ev_obj, dict):
        (out_dir / "explained_variance_ratio.json").write_text(
            json.dumps({str(k): np.asarray(v, dtype=float).tolist() for k, v in ev_obj.items()}, indent=2),
            encoding="utf-8",
        )
        _plot_explained_variance(
            ev_obj,
            out_dir / "explained_variance_pc1_bars.png",
            f"{spec.title} — dPCA explained variance",
        )

    _save_and_plot_loadings(
        model, out_dir, LOADING_MARGINALS, LOADING_N_PCS, LOADING_TOP_NEURONS, X_mean=X_mean
    )

    manifest = [
        spec.title,
        f"Output: {(DPCA_OUT_ROOT / run_key / spec.subdir).relative_to(REPO_ROOT)}/",
        "Marginalizations: " + ", ".join(sorted(Z.keys())),
        "Stimulus levels: " + ", ".join(stim_names),
        f"Frame window: first {ANALYSIS_T_FRAMES} frames; shorter matched trials are ignored",
        "PCs over time: marginal_{t,s,st}.png (mean±SEM from projected single trials)",
        f"PC1 vs PC2 scatter: scatter_{{t,s,st}}_pc1_pc2.png "
        f"(plain scatter; one trial-mean point per trial over frames "
        f"{SCATTER_MEAN_FRAME_WINDOW[0]}:{SCATTER_MEAN_FRAME_WINDOW[1]}, colored by label)",
        "Trajectories: traj_{s,st}_2d.png (mean±SEM), traj_{s,st}_3d.png (mean only); black dot = cue onset",
        "Population encoder: loadings/population_encoder/ (Fig. 6 distributions, stimulus_subgroup_*.png/json)",
    ]
    (out_dir / "dpca_manifest.txt").write_text("\n".join(manifest), encoding="utf-8")

    if "t" in Z and isinstance(Z["t"], np.ndarray) and Z["t"].ndim == 3:
        Zt = np.asarray(Z["t"], dtype=np.float64)
        trial_Z_t = _project_trials_marg(model, trialX_fit, "t")
        _plot_marginal_t(Zt, stim_names, spec.colors, out_dir / "marginal_t.png", trial_Z_t)
        _plot_pc12_trial_scatter(trial_Z_t, "t", stim_names, spec.colors, out_dir / "scatter_t_pc1_pc2.png")
    if "s" in Z and isinstance(Z["s"], np.ndarray) and Z["s"].ndim == 3:
        Zs = np.asarray(Z["s"], dtype=np.float64)
        trial_Z_s = _project_trials_marg(model, trialX_fit, "s")
        _plot_marginal_s(Zs, stim_names, spec.colors, out_dir / "marginal_s.png", trial_Z_s)
        _plot_pc12_trial_scatter(trial_Z_s, "s", stim_names, spec.colors, out_dir / "scatter_s_pc1_pc2.png")
        _plot_traj_marg_2d(Zs, "s", stim_names, spec.colors, out_dir / "traj_s_2d.png", trial_Z_s)
        _plot_traj_marg_3d(Zs, "s", stim_names, spec.colors, out_dir / "traj_s_3d.png")
    if "st" in Z and isinstance(Z["st"], np.ndarray) and Z["st"].ndim == 3:
        Zst = np.asarray(Z["st"], dtype=np.float64)
        trial_Z_st = _project_trials_marg(model, trialX_fit, "st")
        _plot_marginal_st(Zst, stim_names, spec.colors, out_dir / "marginal_st.png", trial_Z_st)
        _plot_pc12_trial_scatter(trial_Z_st, "st", stim_names, spec.colors, out_dir / "scatter_st_pc1_pc2.png")
        _plot_traj_marg_2d(Zst, "st", stim_names, spec.colors, out_dir / "traj_st_2d.png", trial_Z_st)
        _plot_traj_marg_3d(Zst, "st", stim_names, spec.colors, out_dir / "traj_st_3d.png")

    print(f"[{spec.subdir}] saved to {out_dir}")
    print(f"  candidate_trials: {candidate_counts}")
    print(
        f"  ignored_short_trials(<{ANALYSIS_T_FRAMES} frames): "
        f"{ignored_short_counts}, total={sum(ignored_short_counts.values())}"
    )
    print(f"  used_trials: {counts}, frame_window={ANALYSIS_T_FRAMES}, n_components={ncomp}")


def main() -> None:
    import argparse

    from lib.io.npz import load_npz_pack

    p = argparse.ArgumentParser(description="dPCA on phase packs (pre, airpuff, water).")
    p.add_argument("--run", choices=RUN_KEYS, help="Run one eval pack only (default: all in RUN_KEYS).")
    args = p.parse_args()
    run_keys = (args.run,) if args.run else RUN_KEYS

    for run_key in run_keys:
        pack = load_npz_pack(run_key)
        out_root = DPCA_OUT_ROOT / run_key
        for spec in analyses_for_run(run_key, pack):
            run_analysis(spec, pack, run_key=run_key, out_root=out_root)

    _write_loading_similarity_outputs(DPCA_OUT_ROOT)


if __name__ == "__main__":
    main()
