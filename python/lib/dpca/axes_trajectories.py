"""
Phase PCA trajectories in bins of chronological trials.

Fits simple PCA (3 components) on a normalized phase pack (per condition and
across water/nacl/airpuff), then plots mean within-trial trajectories for
consecutive bins of trials (default 5). One PNG per condition:

  top row    — PC1 vs PC2 (2D)
  bottom row — PC1 / PC2 / PC3 (3D)

Left column = condition-only PCA; right = all-conditions PCA.

Writes under ``outputs/<protocol>/<mouse>/axes_trajectories/{pre,airpuff,water}_pca/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from lib.config import (
    AXES_TRAJECTORIES_AIRPUFF_PCA_OUT,
    AXES_TRAJECTORIES_OUT,
    AXES_TRAJECTORIES_PRE_PCA_OUT,
    AXES_TRAJECTORIES_WATER_PCA_OUT,
    resolve_mouse_id,
)
from lib.io.npz import load_npz_pack

# Same stimulus levels as dPCA 3-condition analyses (no SLM).
_STIMULUS_3: tuple[tuple[str, frozenset[int]], ...] = (
    ("water", frozenset({3})),
    ("nacl", frozenset({4})),
    ("airpuff", frozenset({18})),
)
_COND_NAMES = tuple(name for name, _ in _STIMULUS_3)
_COND_IDS = {name: ids for name, ids in _STIMULUS_3}
_N_PCA = 3

_PHASE_OUT: dict[str, Path] = {
    "pre": AXES_TRAJECTORIES_PRE_PCA_OUT,
    "airpuff": AXES_TRAJECTORIES_AIRPUFF_PCA_OUT,
    "water": AXES_TRAJECTORIES_WATER_PCA_OUT,
}
_DEFAULT_PHASES = ("pre", "airpuff", "water")


def _stack_frames(segments: list[np.ndarray]) -> np.ndarray:
    """Stack cue-aligned frames as rows (sum_i T_i, N)."""
    rows: list[np.ndarray] = []
    for seg in segments:
        segf = np.asarray(seg, dtype=np.float64)
        if segf.ndim != 2 or segf.shape[1] < 1:
            continue
        for t in range(segf.shape[1]):
            rows.append(segf[:, t])
    if not rows:
        raise RuntimeError("No frames to stack for PCA.")
    return np.vstack(rows)


def _fit_scaler_pca(
    segments: list[np.ndarray],
    n_components: int = _N_PCA,
    seed: int = 42,
) -> tuple[StandardScaler, PCA]:
    X = _stack_frames(segments)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    n_comp = min(n_components, Xz.shape[1], Xz.shape[0])
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(Xz)
    return scaler, pca


def _project_mean_traj(
    mean_nt: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    """Project (N, T) mean trajectory → (T, n_comp) PC coords."""
    X = mean_nt.T  # (T, N)
    Xz = scaler.transform(X)
    return pca.transform(Xz)


def _trials_by_condition(pack: dict) -> dict[str, list[np.ndarray]]:
    """Group cue-aligned segments by condition, preserving pack (chronological) order."""
    out: dict[str, list[np.ndarray]] = {name: [] for name in _COND_NAMES}
    trial_type = pack["trial_type"]
    for i, seg in enumerate(pack["segments"]):
        cid = int(trial_type[i])
        for name, ids in _COND_IDS.items():
            if cid in ids:
                out[name].append(np.asarray(seg, dtype=np.float64))
                break
    return out


def _bin_indices(n_trials: int, bin_size: int) -> list[tuple[int, int]]:
    """0-based [start, stop) pairs for consecutive bins (remainder kept)."""
    if n_trials < 1 or bin_size < 1:
        return []
    bins: list[tuple[int, int]] = []
    start = 0
    while start < n_trials:
        stop = min(start + bin_size, n_trials)
        bins.append((start, stop))
        start = stop
    return bins


def _mean_traj_for_bin(segments: list[np.ndarray], start: int, stop: int) -> np.ndarray:
    """Average truncated trials in [start, stop) → (N, T) with T = min length in bin."""
    segs = segments[start:stop]
    if not segs:
        raise ValueError("Empty bin.")
    T = min(int(s.shape[1]) for s in segs)
    if T < 1:
        raise ValueError("Bin has no frames.")
    stacked = np.stack([s[:, :T] for s in segs], axis=0)  # (n_trials, N, T)
    return np.mean(stacked, axis=0)


def _bin_color(cmap, bi: int, n_bins: int):
    return cmap(bi / max(n_bins - 1, 1)) if n_bins > 1 else cmap(0.5)


def _plot_2d_panel(ax, trajs: list[np.ndarray], bin_labels: list[str], panel_title: str, cmap) -> None:
    n_bins = len(bin_labels)
    for bi, (traj, label) in enumerate(zip(trajs, bin_labels)):
        color = _bin_color(cmap, bi, n_bins)
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.8, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], color="black", s=28, zorder=5, marker="o")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(panel_title, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best", title="Trial bins")


def _plot_3d_panel(ax, trajs: list[np.ndarray], bin_labels: list[str], panel_title: str, cmap) -> None:
    n_bins = len(bin_labels)
    for bi, (traj, label) in enumerate(zip(trajs, bin_labels)):
        color = _bin_color(cmap, bi, n_bins)
        if traj.shape[1] < 3:
            z = np.zeros(traj.shape[0])
        else:
            z = traj[:, 2]
        ax.plot(traj[:, 0], traj[:, 1], z, color=color, linewidth=1.8, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], z[0], color="black", s=28, zorder=5, marker="o")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(panel_title, fontsize=11)
    ax.legend(fontsize=6, loc="best", title="Trial bins")


def _plot_condition_figure(
    phase_key: str,
    cond_name: str,
    mouse_id: str,
    bin_labels: list[str],
    trajs_within: list[np.ndarray],
    trajs_all: list[np.ndarray],
    out_path: Path,
) -> None:
    cmap = plt.colormaps["viridis"]
    fig = plt.figure(figsize=(11.5, 9.5))

    ax_2d_w = fig.add_subplot(2, 2, 1)
    ax_2d_a = fig.add_subplot(2, 2, 2)
    ax_3d_w = fig.add_subplot(2, 2, 3, projection="3d")
    ax_3d_a = fig.add_subplot(2, 2, 4, projection="3d")

    _plot_2d_panel(ax_2d_w, trajs_within, bin_labels, f"PCA within {cond_name} (PC1–PC2)", cmap)
    _plot_2d_panel(ax_2d_a, trajs_all, bin_labels, "PCA all conditions (PC1–PC2)", cmap)
    _plot_3d_panel(ax_3d_w, trajs_within, bin_labels, f"PCA within {cond_name} (PC1–PC3)", cmap)
    _plot_3d_panel(ax_3d_a, trajs_all, bin_labels, "PCA all conditions (PC1–PC3)", cmap)

    fig.suptitle(
        f"{mouse_id} · {phase_key} · {cond_name} — mean trajectories (bins of chronological trials)",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_axes_trajectories_phase(
    phase_key: str,
    out_dir: Path | None = None,
    bin_size: int = 5,
    seed: int = 42,
) -> Path:
    if phase_key not in _PHASE_OUT:
        raise ValueError(f"Unknown phase {phase_key!r}; expected one of {tuple(_PHASE_OUT)}")

    mouse_id = resolve_mouse_id()
    out_root = Path(out_dir) if out_dir is not None else _PHASE_OUT[phase_key]
    out_root.mkdir(parents=True, exist_ok=True)

    pack = load_npz_pack(phase_key)
    by_cond = _trials_by_condition(pack)

    all_segs: list[np.ndarray] = []
    for name in _COND_NAMES:
        all_segs.extend(by_cond[name])
    if not all_segs:
        raise RuntimeError(f"No water/nacl/airpuff trials found in {phase_key} pack.")

    scaler_all, pca_all = _fit_scaler_pca(all_segs, n_components=_N_PCA, seed=seed)

    for cond_name in _COND_NAMES:
        segs = by_cond[cond_name]
        if not segs:
            print(f"[axes_trajectories] skip {phase_key}/{cond_name}: no trials")
            continue

        scaler_cond, pca_cond = _fit_scaler_pca(segs, n_components=_N_PCA, seed=seed)
        bins = _bin_indices(len(segs), bin_size)
        bin_labels: list[str] = []
        trajs_within: list[np.ndarray] = []
        trajs_all: list[np.ndarray] = []

        for start, stop in bins:
            mean_nt = _mean_traj_for_bin(segs, start, stop)
            trajs_within.append(_project_mean_traj(mean_nt, scaler_cond, pca_cond))
            trajs_all.append(_project_mean_traj(mean_nt, scaler_all, pca_all))
            bin_labels.append(f"trials {start + 1}–{stop}")

        out_path = out_root / f"{cond_name}_pc1_pc2.png"
        _plot_condition_figure(
            phase_key,
            cond_name,
            mouse_id,
            bin_labels,
            trajs_within,
            trajs_all,
            out_path,
        )
        print(f"[axes_trajectories] wrote {out_path} ({len(segs)} trials, {len(bins)} bins)")

    print(f"[axes_trajectories] done {phase_key} -> {out_root}")
    return out_root


def run_axes_trajectories(
    phases: tuple[str, ...] | list[str] | None = None,
    out_dir: Path | None = None,
    bin_size: int = 5,
    seed: int = 42,
) -> list[Path]:
    """Run for one or more phases. ``out_dir`` only applies when a single phase is requested."""
    phase_list = list(phases) if phases is not None else list(_DEFAULT_PHASES)
    unknown = [p for p in phase_list if p not in _PHASE_OUT]
    if unknown:
        raise ValueError(f"Unknown phase(s) {unknown}; expected one of {tuple(_PHASE_OUT)}")

    written: list[Path] = []
    for phase_key in phase_list:
        phase_out = out_dir if (out_dir is not None and len(phase_list) == 1) else None
        if out_dir is not None and len(phase_list) > 1:
            phase_out = Path(out_dir) / f"{phase_key}_pca"
        written.append(
            run_axes_trajectories_phase(
                phase_key,
                out_dir=phase_out,
                bin_size=bin_size,
                seed=seed,
            )
        )
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        nargs="+",
        default=list(_DEFAULT_PHASES),
        choices=list(_PHASE_OUT.keys()),
        help="Phase pack(s) to analyze (default: pre airpuff water).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory override. For a single --phase, writes directly there; "
            f"for multiple phases, writes <out-dir>/<phase>_pca/. "
            f"Default: under {AXES_TRAJECTORIES_OUT}/{{phase}}_pca/."
        ),
    )
    p.add_argument("--bin-size", type=int, default=5, help="Trials per chronological bin.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_axes_trajectories(
        phases=tuple(args.phase),
        out_dir=args.out_dir,
        bin_size=int(args.bin_size),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
