"""
SVM separating axes in pre 5D s-dPCA space.

Fits linear SVM hyperplanes on pre window-mean features:
  - valence: water vs nacl+airpuff
  - thirst: water vs nacl

Also writes water_vs_airpuff_valence/ with the same three cross-phase
figures as svm_no_dpca/water_vs_airpuff_valence/:
  - valence: water vs airpuff (nacl held out, then projected)
  - thirst: water vs nacl (airpuff held out, then projected)

Projects all phases onto fixed pre decoder/center + SVM axes; writes trial-bin
summaries, within-trial 1D trajectories, and cross-phase comparisons under
axes_trajectories/svm_axes/.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import colorsys
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from sklearn.svm import LinearSVC

from lib.config import (
    AXES_TRAJECTORIES_SVM_AXES_OUT,
    AXES_TRAJECTORIES_SVM_AXES_WATER_AIRPUFF_VALENCE_OUT,
    resolve_mouse_id,
)
from lib.dpca import analysis as base
from lib.dpca.condition_distances import _prepare_trial_tensor, _trial_features_s
from lib.dpca.thirst_valence_axes import _load_decoder_s, _spec_for_phase
from lib.io.npz import load_npz_pack

REFERENCE_RUN = "pre"
ANALYSIS_3 = "3conditions"
DEFAULT_N_PCS = 5
DEFAULT_PHASES = ("pre", "airpuff", "water")
PLOT_CONDS = ("water", "nacl", "airpuff")


@dataclass
class SvmAxis:
    name: str
    w: np.ndarray
    b: float
    training: dict[str, Any]


@dataclass
class TrialRecord:
    condition: str
    trial_score: float
    traj_scores: np.ndarray
    features_5d: np.ndarray
    order_in_phase: int


@dataclass
class JointTrialRecord:
    condition: str
    valence_score: float
    thirst_score: float
    valence_traj: np.ndarray
    thirst_traj: np.ndarray
    features_5d: np.ndarray
    order_in_phase: int


@dataclass
class BinSummary:
    bin_index: int
    label: str
    mean_score: float
    sem_score: float
    mean_traj: np.ndarray


_PHASE_COLORS = {
    "pre": "#0b3d91",
    "airpuff": "#b22222",
    "water": "#2e8b57",
}
_COND_COLORS = {
    "water": base.COLORS_3[0],
    "nacl": base.COLORS_3[1],
    "airpuff": base.COLORS_3[2],
    "slm": base.COLORS_4[3],
}
PLOT_CONDS_WITH_SLM = PLOT_CONDS + ("slm",)


def _pre_3cond_spec() -> base.AnalysisSpec:
    return base.AnalysisSpec(
        ANALYSIS_3,
        f"{REFERENCE_RUN} — water / nacl / airpuff",
        base._STIMULUS_3,  # noqa: SLF001
        base.COLORS_3,
    )


def _stim_names_3() -> tuple[str, ...]:
    return tuple(name for name, _ in base._STIMULUS_3)  # noqa: SLF001


def _fit_svm_axis(
    features: np.ndarray,
    y: np.ndarray,
    *,
    axis_name: str,
    positive_description: str,
) -> SvmAxis:
    if features.shape[0] < 2 or len(np.unique(y)) < 2:
        raise RuntimeError(f"Not enough classes to fit SVM for {axis_name}.")

    clf = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=20_000)
    clf.fit(features, y)
    w = np.asarray(clf.coef_[0], dtype=np.float64)
    b = float(clf.intercept_[0])

    pos = y == 1
    neg = y == 0
    if float(np.mean(clf.decision_function(features[pos]))) < float(np.mean(clf.decision_function(features[neg]))):
        w = -w
        b = -b

    w_norm = float(np.linalg.norm(w))
    meta = {
        "axis": axis_name,
        "positive_direction": positive_description,
        "n_positive": int(np.sum(pos)),
        "n_negative": int(np.sum(neg)),
        "w_norm": w_norm,
        "w_unit": (w / w_norm).tolist() if w_norm > 1e-12 else w.tolist(),
        "b": b,
    }
    return SvmAxis(name=axis_name, w=w, b=b, training=meta)


def _fit_axes_on_pre(
    decoder: np.ndarray,
    center: np.ndarray,
    frame_window: tuple[int, int],
    n_pcs: int,
    *,
    valence_mode: str = "nacl_airpuff",
) -> tuple[SvmAxis, SvmAxis]:
    spec = _pre_3cond_spec()
    trial_x, _, _ = _prepare_trial_tensor(REFERENCE_RUN, spec)
    features, labels = _trial_features_s(
        trial_x,
        center,
        decoder,
        n_pcs=n_pcs,
        frame_window=frame_window,
    )
    stim_names = _stim_names_3()
    name_by_ix = {i: stim_names[i] for i in range(len(stim_names))}
    cond_names = np.array([name_by_ix[int(ix)] for ix in labels])

    if valence_mode == "water_vs_airpuff":
        valence_mask = np.isin(cond_names, ("water", "airpuff"))
        y_val = (cond_names[valence_mask] == "airpuff").astype(np.int64)
        valence = _fit_svm_axis(
            features[valence_mask],
            y_val,
            axis_name="valence",
            positive_description="toward airpuff (aversive)",
        )
    elif valence_mode == "nacl_airpuff":
        y_val = np.isin(cond_names, ("nacl", "airpuff")).astype(np.int64)
        valence = _fit_svm_axis(
            features,
            y_val,
            axis_name="valence",
            positive_description="toward nacl+airpuff (aversive)",
        )
    else:
        raise ValueError(f"Unknown valence_mode {valence_mode!r}")

    thirst_mask = np.isin(cond_names, ("water", "nacl"))
    y_thirst = (cond_names[thirst_mask] == "nacl").astype(np.int64)
    thirst = _fit_svm_axis(
        features[thirst_mask],
        y_thirst,
        axis_name="thirst",
        positive_description="toward nacl",
    )
    return valence, thirst


def _chronological_order_map(
    phase: str,
    spec: base.AnalysisSpec,
) -> dict[tuple[int, int], int]:
    """Map (stim_ix, trial_ix) in the trial tensor to chronological order in the phase pack."""
    pack = load_npz_pack(phase)
    cond_to_s = base._cond_to_stim_index(spec.stimulus_levels)  # noqa: SLF001
    stim_names = tuple(name for name, _ in spec.stimulus_levels)
    counts: dict[int, int] = {si: 0 for si in range(len(stim_names))}
    order_map: dict[tuple[int, int], int] = {}
    chrono = 0
    y_cond = pack["trial_type"]
    for local_i, seg in enumerate(pack["segments"]):
        ci = int(y_cond[local_i])
        if ci not in cond_to_s:
            continue
        si = cond_to_s[ci]
        if stim_names[si] not in PLOT_CONDS_WITH_SLM:
            continue
        segf = np.asarray(seg, dtype=np.float64)
        if segf.ndim != 2 or int(segf.shape[1]) < base.ANALYSIS_T_FRAMES:
            continue
        trial_ix = counts[si]
        counts[si] += 1
        order_map[(si, trial_ix)] = chrono
        chrono += 1
    return order_map


def _enumerate_trials(
    trial_x: np.ndarray,
    center: np.ndarray,
    decoder: np.ndarray,
    stim_names: tuple[str, ...],
    frame_window: tuple[int, int],
    axis: SvmAxis,
    *,
    n_pcs: int,
    order_map: dict[tuple[int, int], int] | None = None,
) -> list[TrialRecord]:
    start, stop = frame_window
    stop = min(stop, int(trial_x.shape[3]))
    n_use = min(n_pcs, int(decoder.shape[1]))
    dec = np.asarray(decoder[:, :n_use], dtype=np.float64)
    w = axis.w[:n_use]
    b = axis.b

    records: list[TrialRecord] = []
    for stim_ix, cond_name in enumerate(stim_names):
        if cond_name not in PLOT_CONDS:
            continue
        for trial_ix in range(trial_x.shape[0]):
            xi = np.asarray(trial_x[trial_ix, :, stim_ix, :], dtype=np.float64)
            if np.any(np.isnan(xi)):
                continue
            projected = dec.T @ (xi - center[:, None])  # (n_pcs, T)
            feat = np.mean(projected[:, start:stop], axis=1)
            trial_score = float(w @ feat + b)
            traj_scores = (w @ projected + b).astype(np.float64)
            order_in_phase = int(order_map.get((stim_ix, trial_ix), len(records))) if order_map else len(records)
            records.append(
                TrialRecord(
                    condition=cond_name,
                    trial_score=trial_score,
                    traj_scores=traj_scores,
                    features_5d=feat.astype(np.float64),
                    order_in_phase=order_in_phase,
                )
            )
    return records


def _enumerate_joint_trials(
    trial_x: np.ndarray,
    center: np.ndarray,
    decoder: np.ndarray,
    stim_names: tuple[str, ...],
    frame_window: tuple[int, int],
    valence_axis: SvmAxis,
    thirst_axis: SvmAxis,
    *,
    n_pcs: int,
    order_map: dict[tuple[int, int], int] | None = None,
) -> list[JointTrialRecord]:
    start, stop = frame_window
    stop = min(stop, int(trial_x.shape[3]))
    n_use = min(n_pcs, int(decoder.shape[1]))
    dec = np.asarray(decoder[:, :n_use], dtype=np.float64)
    w_v = valence_axis.w[:n_use]
    b_v = valence_axis.b
    w_t = thirst_axis.w[:n_use]
    b_t = thirst_axis.b

    records: list[JointTrialRecord] = []
    for stim_ix, cond_name in enumerate(stim_names):
        if cond_name not in PLOT_CONDS_WITH_SLM:
            continue
        for trial_ix in range(trial_x.shape[0]):
            xi = np.asarray(trial_x[trial_ix, :, stim_ix, :], dtype=np.float64)
            if np.any(np.isnan(xi)):
                continue
            projected = dec.T @ (xi - center[:, None])
            feat = np.mean(projected[:, start:stop], axis=1)
            order_in_phase = int(order_map.get((stim_ix, trial_ix), len(records))) if order_map else len(records)
            records.append(
                JointTrialRecord(
                    condition=cond_name,
                    valence_score=float(w_v @ feat + b_v),
                    thirst_score=float(w_t @ feat + b_t),
                    valence_traj=(w_v @ projected + b_v).astype(np.float64),
                    thirst_traj=(w_t @ projected + b_t).astype(np.float64),
                    features_5d=feat.astype(np.float64),
                    order_in_phase=order_in_phase,
                )
            )
    return records


def _axis_records_from_joint(joint: list[JointTrialRecord], axis_name: str) -> list[TrialRecord]:
    if axis_name == "valence":
        return [
            TrialRecord(
                r.condition,
                r.valence_score,
                r.valence_traj,
                r.features_5d,
                r.order_in_phase,
            )
            for r in joint
        ]
    return [
        TrialRecord(
            r.condition,
            r.thirst_score,
            r.thirst_traj,
            r.features_5d,
            r.order_in_phase,
        )
        for r in joint
    ]


def _orthogonal_unit(w: np.ndarray) -> np.ndarray:
    """Unit vector in feature space orthogonal to w."""
    w_unit = w / max(float(np.linalg.norm(w)), 1e-12)
    e = np.zeros_like(w_unit)
    e[int(np.argmin(np.abs(w_unit)))] = 1.0
    v = e - w_unit * float(np.dot(e, w_unit))
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        e = np.roll(e, 1)
        v = e - w_unit * float(np.dot(e, w_unit))
        n = float(np.linalg.norm(v))
    return v / max(n, 1e-12)


def _shade_condition_color(base_hex: str, t: float) -> tuple[float, float, float]:
    """Same hue as base; t=0 darker, t=1 lighter (HLS lightness sweep)."""
    r, g, b = to_rgb(base_hex)
    h, lightness, sat = colorsys.rgb_to_hls(r, g, b)
    lightness_out = 0.28 + 0.48 * float(t)
    sat_out = min(1.0, sat * (0.9 + 0.1 * float(t)))
    return colorsys.hls_to_rgb(h, lightness_out, sat_out)


def _order_t_values(orders: np.ndarray) -> np.ndarray:
    if orders.size == 0:
        return orders
    lo, hi = float(np.min(orders)), float(np.max(orders))
    if hi <= lo:
        return np.full(orders.shape, 0.5, dtype=np.float64)
    return (orders - lo) / (hi - lo)


def _plot_valence_thirst_plane(
    records: list[JointTrialRecord],
    phase: str,
    mouse_id: str,
    out_path: Path,
    *,
    show_centroids: bool = True,
    shade_by_order: bool = False,
) -> None:
    if not records:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    valence_scores = np.array([r.valence_score for r in records], dtype=np.float64)
    thirst_scores = np.array([r.thirst_score for r in records], dtype=np.float64)
    orders = np.array([r.order_in_phase for r in records], dtype=np.float64)
    order_t = _order_t_values(orders)

    for cond_name in PLOT_CONDS:
        mask = np.array([r.condition == cond_name for r in records])
        if not np.any(mask):
            continue
        if shade_by_order:
            colors = np.array(
                [_shade_condition_color(_COND_COLORS[cond_name], float(t)) for t in order_t[mask]],
                dtype=np.float64,
            )
            ax.scatter(
                valence_scores[mask],
                thirst_scores[mask],
                c=colors,
                label=cond_name,
                alpha=1.0,
                s=28,
                edgecolors="0.25",
                linewidths=0.35,
                zorder=2,
            )
        else:
            ax.scatter(
                valence_scores[mask],
                thirst_scores[mask],
                c=_COND_COLORS[cond_name],
                label=cond_name,
                alpha=0.55,
                s=28,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
        if show_centroids:
            cx = float(np.mean(valence_scores[mask]))
            cy = float(np.mean(thirst_scores[mask]))
            ax.scatter(
                cx,
                cy,
                c=_COND_COLORS[cond_name],
                s=160,
                marker="D",
                edgecolors="black",
                linewidths=1.0,
                zorder=4,
            )
            ax.annotate(
                cond_name,
                (cx, cy),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                fontweight="bold",
            )

    ax.axvline(0.0, color="#333333", linewidth=2.0, linestyle="-", label="Valence boundary", zorder=1)
    ax.axhline(0.0, color="#666666", linewidth=2.0, linestyle="--", label="Thirst boundary", zorder=1)
    ax.set_xlabel("Valence SVM signed score")
    ax.set_ylabel("Thirst SVM signed score")
    if shade_by_order:
        ax.set_title(
            f"{mouse_id} · {phase} — valence × thirst SVM plane\n"
            "Point shade = trial order in phase (dark → light)",
            fontsize=11,
        )
    else:
        ax.set_title(f"{mouse_id} · {phase} — valence × thirst SVM plane")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _shared_axis_limits(arrays: list[np.ndarray], pad_frac: float = 0.05) -> tuple[float, float]:
    parts = [a for a in arrays if a.size > 0]
    if not parts:
        return -1.0, 1.0
    vals = np.concatenate(parts)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        pad = max(abs(lo), 1.0) * pad_frac
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _draw_joint_plane_ax(
    ax,
    records: list[JointTrialRecord],
    *,
    plot_conds: tuple[str, ...] = PLOT_CONDS,
    show_centroids: bool = True,
    shade_by_order: bool = False,
    point_size: float = 24,
    centroid_size: float = 120,
    annotate_centroids: bool = False,
    legend: bool = False,
    legend_conds: tuple[str, ...] | None = None,
) -> None:
    valence_scores = np.array([r.valence_score for r in records], dtype=np.float64)
    thirst_scores = np.array([r.thirst_score for r in records], dtype=np.float64)
    orders = np.array([r.order_in_phase for r in records], dtype=np.float64)
    order_t = _order_t_values(orders)

    for cond_name in plot_conds:
        mask = np.array([r.condition == cond_name for r in records])
        if not np.any(mask):
            continue
        if shade_by_order:
            colors = np.array(
                [_shade_condition_color(_COND_COLORS[cond_name], float(t)) for t in order_t[mask]],
                dtype=np.float64,
            )
            ax.scatter(
                valence_scores[mask],
                thirst_scores[mask],
                c=colors,
                label=cond_name,
                alpha=1.0,
                s=point_size,
                edgecolors="0.25",
                linewidths=0.3,
                zorder=2,
            )
        else:
            ax.scatter(
                valence_scores[mask],
                thirst_scores[mask],
                c=_COND_COLORS[cond_name],
                label=cond_name,
                alpha=0.55,
                s=point_size,
                edgecolors="white",
                linewidths=0.3,
                zorder=2,
            )
        if show_centroids:
            cx = float(np.mean(valence_scores[mask]))
            cy = float(np.mean(thirst_scores[mask]))
            ax.scatter(
                cx,
                cy,
                c=_COND_COLORS[cond_name],
                s=centroid_size,
                marker="D",
                edgecolors="black",
                linewidths=0.9,
                zorder=4,
            )
            if annotate_centroids:
                ax.annotate(
                    cond_name,
                    (cx, cy),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=9,
                    fontweight="bold",
                )

    ax.axvline(0.0, color="#333333", linewidth=1.8, linestyle="-", zorder=1)
    ax.axhline(0.0, color="#666666", linewidth=1.8, linestyle="--", zorder=1)
    ax.grid(True, alpha=0.25)
    if legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=_COND_COLORS[c],
                markeredgecolor="white",
                markersize=7,
                label=c,
            )
            for c in (legend_conds or plot_conds)
        ]
        ax.legend(handles=handles, fontsize=7, loc="best")


def _plot_valence_thirst_plane_cross_phase(
    phases: tuple[str, ...],
    phase_joint: dict[str, list[JointTrialRecord]],
    mouse_id: str,
    out_path: Path,
    *,
    show_centroids: bool = True,
    shade_by_order: bool = False,
    plot_conds: tuple[str, ...] = PLOT_CONDS,
    title_suffix: str = "",
) -> None:
    def _kept(records: list[JointTrialRecord]) -> list[JointTrialRecord]:
        return [r for r in records if r.condition in plot_conds]

    active = [p for p in phases if _kept(phase_joint.get(p, []))]
    if not active:
        return

    all_valence: list[np.ndarray] = []
    all_thirst: list[np.ndarray] = []
    present: list[str] = []
    for phase in active:
        records = _kept(phase_joint[phase])
        all_valence.append(np.array([r.valence_score for r in records], dtype=np.float64))
        all_thirst.append(np.array([r.thirst_score for r in records], dtype=np.float64))
        for cond_name in plot_conds:
            if cond_name not in present and any(r.condition == cond_name for r in records):
                present.append(cond_name)
    xlim = _shared_axis_limits(all_valence)
    ylim = _shared_axis_limits(all_thirst)
    legend_conds = tuple(present) if present else plot_conds

    fig, axes = plt.subplots(1, len(active), figsize=(6.0 * len(active), 5.8), squeeze=False)
    for ax_i, phase in enumerate(active):
        records = _kept(phase_joint[phase])
        ax = axes[0, ax_i]
        _draw_joint_plane_ax(
            ax,
            records,
            plot_conds=plot_conds,
            show_centroids=show_centroids,
            shade_by_order=shade_by_order,
            legend=ax_i == 0,
            legend_conds=legend_conds,
        )
        ax.set_xlabel("Valence SVM score")
        ax.set_ylabel("Thirst SVM score")
        ax.set_title(phase)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    phase_note = "across phases" if not title_suffix else f"across phases; {title_suffix}"
    if shade_by_order:
        fig.suptitle(
            f"{mouse_id} — valence × thirst SVM plane ({phase_note})\n"
            "Point shade = trial order in phase (dark → light)",
            fontsize=12,
            y=1.04,
        )
    else:
        fig.suptitle(
            f"{mouse_id} — valence × thirst SVM plane ({phase_note})",
            fontsize=12,
            y=1.02,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_separation_cloud(
    axis: SvmAxis,
    phase: str,
    records: list[TrialRecord],
    mouse_id: str,
    out_path: Path,
) -> None:
    if not records:
        return

    feats = np.stack([r.features_5d for r in records], axis=0)
    w = axis.w[: feats.shape[1]]
    v_orth = _orthogonal_unit(w)
    orth_coords = feats @ v_orth

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    ax0 = axes[0]
    scores = np.array([r.trial_score for r in records], dtype=np.float64)
    for cond_name in PLOT_CONDS:
        mask = np.array([r.condition == cond_name for r in records])
        if not np.any(mask):
            continue
        ax0.scatter(
            scores[mask],
            orth_coords[mask],
            c=_COND_COLORS[cond_name],
            label=cond_name,
            alpha=0.7,
            s=32,
            edgecolors="white",
            linewidths=0.4,
        )
    ax0.axvline(0.0, color="black", linewidth=2.0, linestyle="-", label="SVM boundary")
    ax0.set_xlabel("SVM signed score")
    ax0.set_ylabel("Orthogonal s-space coordinate")
    ax0.set_title("Aligned with SVM normal")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8, loc="best")

    ax1 = axes[1]
    orders = np.array([r.order_in_phase for r in records], dtype=np.float64)
    order_t = _order_t_values(orders)
    for cond_name in PLOT_CONDS:
        mask = np.array([r.condition == cond_name for r in records])
        if not np.any(mask):
            continue
        colors = np.array(
            [_shade_condition_color(_COND_COLORS[cond_name], float(t)) for t in order_t[mask]],
            dtype=np.float64,
        )
        ax1.scatter(
            scores[mask],
            orth_coords[mask],
            c=colors,
            label=cond_name,
            alpha=1.0,
            s=36,
            edgecolors="0.25",
            linewidths=0.35,
        )
    ax1.axvline(0.0, color="black", linewidth=2.0, linestyle="-", label="SVM boundary")
    ax1.set_xlabel("SVM signed score")
    ax1.set_ylabel("Orthogonal s-space coordinate")
    ax1.set_title("Aligned with SVM normal (shade = trial order)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8, loc="best")
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap="gray"),
        ax=ax1,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label("Trial order (dark → light)")

    fig.suptitle(
        f"{mouse_id} · {phase} · {axis.name} axis — trial clouds vs SVM boundary",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_separation_cloud_cross_phase(
    axis: SvmAxis,
    mouse_id: str,
    phases: tuple[str, ...],
    phase_records: dict[str, list[TrialRecord]],
    out_path: Path,
) -> None:
    n_phases = sum(1 for p in phases if phase_records.get(p))
    if n_phases < 1:
        return

    fig, axes = plt.subplots(1, n_phases, figsize=(5.5 * n_phases, 5.0), squeeze=False)
    ax_i = 0
    w = axis.w

    for phase in phases:
        records = phase_records.get(phase, [])
        if not records:
            continue
        feats = np.stack([r.features_5d for r in records], axis=0)
        w_use = w[: feats.shape[1]]
        v_orth = _orthogonal_unit(w_use)
        scores = np.array([r.trial_score for r in records], dtype=np.float64)
        orth_coords = feats @ v_orth

        ax = axes[0, ax_i]
        for cond_name in PLOT_CONDS:
            mask = np.array([r.condition == cond_name for r in records])
            if not np.any(mask):
                continue
            ax.scatter(
                scores[mask],
                orth_coords[mask],
                c=_COND_COLORS[cond_name],
                label=cond_name,
                alpha=0.7,
                s=30,
                edgecolors="white",
                linewidths=0.4,
            )
        ax.axvline(0.0, color="black", linewidth=2.0, linestyle="-")
        ax.set_xlabel("SVM signed score")
        ax.set_ylabel("Orthogonal coordinate")
        ax.set_title(phase)
        ax.grid(True, alpha=0.25)
        if ax_i == 0:
            ax.legend(fontsize=7, loc="best")
        ax_i += 1

    for j in range(ax_i, axes.shape[1]):
        axes[0, j].set_visible(False)

    fig.suptitle(
        f"{mouse_id} · {axis.name} axis — trial clouds vs SVM boundary (across phases)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_separation_cloud_ax(
    ax,
    axis: SvmAxis,
    records: list[TrialRecord],
    *,
    legend: bool,
) -> None:
    feats = np.stack([r.features_5d for r in records], axis=0)
    w_use = axis.w[: feats.shape[1]]
    v_orth = _orthogonal_unit(w_use)
    scores = np.array([r.trial_score for r in records], dtype=np.float64)
    orth_coords = feats @ v_orth
    for cond_name in PLOT_CONDS:
        mask = np.array([r.condition == cond_name for r in records])
        if not np.any(mask):
            continue
        ax.scatter(
            scores[mask],
            orth_coords[mask],
            c=_COND_COLORS[cond_name],
            label=cond_name,
            alpha=0.7,
            s=30,
            edgecolors="white",
            linewidths=0.4,
        )
    ax.axvline(0.0, color="black", linewidth=2.0, linestyle="-")
    ax.set_xlabel("SVM signed score")
    ax.set_ylabel("Orthogonal coordinate")
    ax.grid(True, alpha=0.25)
    if legend:
        ax.legend(fontsize=7, loc="best")


def _plot_separation_cloud_both_axes(
    mouse_id: str,
    phases: tuple[str, ...],
    thirst_axis: SvmAxis,
    valence_axis: SvmAxis,
    phase_thirst: dict[str, list[TrialRecord]],
    phase_valence: dict[str, list[TrialRecord]],
    out_path: Path,
    *,
    title_suffix: str = "",
) -> None:
    active = [p for p in phases if phase_thirst.get(p) or phase_valence.get(p)]
    if not active:
        return

    n_phases = len(active)
    fig, axes = plt.subplots(2, n_phases, figsize=(5.5 * n_phases, 9.6), squeeze=False)
    row_axes = (
        (thirst_axis, phase_thirst),
        (valence_axis, phase_valence),
    )
    for row, (axis, phase_records) in enumerate(row_axes):
        for col, phase in enumerate(active):
            ax = axes[row, col]
            records = phase_records.get(phase, [])
            if not records:
                ax.set_visible(False)
                continue
            _draw_separation_cloud_ax(ax, axis, records, legend=row == 0 and col == 0)
            if row == 0:
                ax.set_title(phase)
            if col == 0:
                ax.set_ylabel(f"{axis.name} axis\nOrthogonal coordinate")

    title = f"{mouse_id} — trial clouds vs SVM boundary (across phases)"
    if title_suffix:
        title = f"{title}; {title_suffix}"
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bin_indices(n_trials: int, bin_size: int) -> list[tuple[int, int]]:
    if n_trials < 1 or bin_size < 1:
        return []
    bins: list[tuple[int, int]] = []
    start = 0
    while start < n_trials:
        stop = min(start + bin_size, n_trials)
        bins.append((start, stop))
        start = stop
    return bins


def _sem_1d(values: np.ndarray) -> float:
    n = int(values.size)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def _mean_traj_for_bin(trajs: list[np.ndarray], start: int, stop: int) -> np.ndarray:
    segs = trajs[start:stop]
    T = min(int(t.shape[0]) for t in segs)
    stacked = np.stack([t[:T] for t in segs], axis=0)
    return np.mean(stacked, axis=0)


def _plot_trial_bins(
    cond_name: str,
    phase: str,
    axis: SvmAxis,
    mouse_id: str,
    bin_labels: list[str],
    means: np.ndarray,
    sems: np.ndarray,
    out_path: Path,
) -> None:
    x = np.arange(len(bin_labels))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.errorbar(x, means, yerr=sems, fmt="o-", color="#0b3d91", capsize=4, linewidth=1.8)
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("SVM signed score")
    ax.set_xlabel("Chronological trial bin")
    ax.set_title(f"{mouse_id} · {phase} · {cond_name} · {axis.name} axis")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_traj_bins(
    cond_name: str,
    phase: str,
    axis: SvmAxis,
    mouse_id: str,
    bin_labels: list[str],
    mean_trajs: list[np.ndarray],
    out_path: Path,
) -> None:
    cmap = plt.colormaps["viridis"]
    n_bins = len(bin_labels)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for bi, (traj, label) in enumerate(zip(mean_trajs, bin_labels)):
        color = cmap(bi / max(n_bins - 1, 1)) if n_bins > 1 else cmap(0.5)
        t = np.arange(traj.shape[0])
        ax.plot(t, traj, color=color, linewidth=1.8, label=label)
        ax.scatter(t[0], traj[0], color="black", s=28, zorder=5, marker="o")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Cue-relative frame")
    ax.set_ylabel("SVM signed score")
    ax.set_title(f"{mouse_id} · {phase} · {cond_name} · {axis.name} axis (mean traj by bin)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best", title="Trial bins")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_condition_axis(
    records: list[TrialRecord],
    cond_name: str,
    phase: str,
    axis: SvmAxis,
    mouse_id: str,
    out_dir: Path,
    bin_size: int,
) -> None:
    cond_records = [r for r in records if r.condition == cond_name]
    if not cond_records:
        return

    bins = _bin_indices(len(cond_records), bin_size)
    bin_labels: list[str] = []
    trial_means: list[float] = []
    trial_sems: list[float] = []
    mean_trajs: list[np.ndarray] = []

    for start, stop in bins:
        chunk = cond_records[start:stop]
        scores = np.array([r.trial_score for r in chunk], dtype=np.float64)
        trajs = [r.traj_scores for r in chunk]
        bin_labels.append(f"trials {start + 1}–{stop}")
        trial_means.append(float(np.mean(scores)))
        trial_sems.append(_sem_1d(scores))
        mean_trajs.append(_mean_traj_for_bin(trajs, 0, len(trajs)))

    _plot_trial_bins(
        cond_name,
        phase,
        axis,
        mouse_id,
        bin_labels,
        np.asarray(trial_means),
        np.asarray(trial_sems),
        out_dir / f"{cond_name}_trial_bins.png",
    )
    _plot_traj_bins(
        cond_name,
        phase,
        axis,
        mouse_id,
        bin_labels,
        mean_trajs,
        out_dir / f"{cond_name}_traj_bins.png",
    )


def _summarize_bins(
    records: list[TrialRecord],
    cond_name: str,
    bin_size: int,
) -> list[BinSummary]:
    cond_records = [r for r in records if r.condition == cond_name]
    if not cond_records:
        return []

    summaries: list[BinSummary] = []
    for bi, (start, stop) in enumerate(_bin_indices(len(cond_records), bin_size)):
        chunk = cond_records[start:stop]
        scores = np.array([r.trial_score for r in chunk], dtype=np.float64)
        trajs = [r.traj_scores for r in chunk]
        summaries.append(
            BinSummary(
                bin_index=bi,
                label=f"trials {start + 1}–{stop}",
                mean_score=float(np.mean(scores)),
                sem_score=_sem_1d(scores),
                mean_traj=_mean_traj_for_bin(trajs, 0, len(trajs)),
            )
        )
    return summaries


def _plot_phase_means(
    cond_name: str,
    axis: SvmAxis,
    mouse_id: str,
    phases: tuple[str, ...],
    phase_records: dict[str, list[TrialRecord]],
    out_path: Path,
) -> None:
    means: list[float] = []
    sems: list[float] = []
    labels: list[str] = []
    for phase in phases:
        cond_scores = np.array(
            [r.trial_score for r in phase_records.get(phase, []) if r.condition == cond_name],
            dtype=np.float64,
        )
        if cond_scores.size == 0:
            continue
        labels.append(phase)
        means.append(float(np.mean(cond_scores)))
        sems.append(_sem_1d(cond_scores))

    if not labels:
        return

    x = np.arange(len(labels))
    colors = [_PHASE_COLORS.get(p, "#333333") for p in labels]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.85, capsize=4, edgecolor="0.2")
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean SVM signed score")
    ax.set_xlabel("Phase")
    ax.set_title(f"{mouse_id} · {cond_name} · {axis.name} axis (across phases)")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_trial_bins_across_phases(
    cond_name: str,
    axis: SvmAxis,
    mouse_id: str,
    phases: tuple[str, ...],
    phase_bins: dict[str, list[BinSummary]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    any_plotted = False
    for phase in phases:
        bins = phase_bins.get(phase, [])
        if not bins:
            continue
        x = np.array([b.bin_index for b in bins], dtype=np.float64)
        means = np.array([b.mean_score for b in bins], dtype=np.float64)
        sems = np.array([b.sem_score for b in bins], dtype=np.float64)
        color = _PHASE_COLORS.get(phase, "#333333")
        ax.errorbar(x, means, yerr=sems, fmt="o-", color=color, capsize=4, linewidth=1.8, label=phase)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Chronological trial bin index (within phase)")
    ax.set_ylabel("SVM signed score")
    ax.set_title(f"{mouse_id} · {cond_name} · {axis.name} axis (trial bins across phases)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, title="Phase")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_traj_grand_mean_across_phases(
    cond_name: str,
    axis: SvmAxis,
    mouse_id: str,
    phases: tuple[str, ...],
    phase_records: dict[str, list[TrialRecord]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    any_plotted = False
    for phase in phases:
        trajs = [r.traj_scores for r in phase_records.get(phase, []) if r.condition == cond_name]
        if not trajs:
            continue
        T = min(int(t.shape[0]) for t in trajs)
        mean_traj = np.mean(np.stack([t[:T] for t in trajs], axis=0), axis=0)
        t = np.arange(T)
        color = _PHASE_COLORS.get(phase, "#333333")
        ax.plot(t, mean_traj, color=color, linewidth=1.8, label=phase)
        ax.scatter(t[0], mean_traj[0], color="black", s=24, zorder=5, marker="o")
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Cue-relative frame")
    ax.set_ylabel("SVM signed score")
    ax.set_title(f"{mouse_id} · {cond_name} · {axis.name} axis (grand-mean trajectory across phases)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, title="Phase")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_traj_bins_across_phases(
    cond_name: str,
    axis: SvmAxis,
    mouse_id: str,
    phases: tuple[str, ...],
    phase_bins: dict[str, list[BinSummary]],
    out_path: Path,
) -> None:
    max_bins = max((len(b) for b in phase_bins.values()), default=0)
    if max_bins < 1:
        return

    n_cols = min(3, max_bins)
    n_rows = int(np.ceil(max_bins / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False)

    for bi in range(max_bins):
        ax = axes[bi // n_cols, bi % n_cols]
        any_line = False
        for phase in phases:
            bins = phase_bins.get(phase, [])
            if bi >= len(bins):
                continue
            traj = bins[bi].mean_traj
            t = np.arange(traj.shape[0])
            color = _PHASE_COLORS.get(phase, "#333333")
            ax.plot(t, traj, color=color, linewidth=1.6, label=phase)
            ax.scatter(t[0], traj[0], color="black", s=20, zorder=5, marker="o")
            any_line = True
        if not any_line:
            ax.set_visible(False)
            continue
        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.7)
        ax.set_title(f"Bin {bi + 1}", fontsize=9)
        ax.grid(True, alpha=0.25)
        if bi % n_cols == n_cols - 1:
            ax.legend(fontsize=6, loc="best")

    for j in range(max_bins, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.supxlabel("Cue-relative frame", fontsize=10)
    fig.supylabel("SVM signed score", fontsize=10)
    fig.suptitle(
        f"{mouse_id} · {cond_name} · {axis.name} axis (trajectory bins across phases)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cross_phase(
    axis: SvmAxis,
    mouse_id: str,
    phases: tuple[str, ...],
    phase_records: dict[str, list[TrialRecord]],
    out_dir: Path,
    bin_size: int,
) -> None:
    cross_out = out_dir / "cross_phase"
    phase_bins = {
        phase: {
            cond: _summarize_bins(recs, cond, bin_size)
            for cond in PLOT_CONDS
        }
        for phase, recs in phase_records.items()
    }

    for cond_name in PLOT_CONDS:
        cond_phase_bins = {phase: phase_bins[phase][cond_name] for phase in phases if phase in phase_bins}
        if not any(cond_phase_bins.values()):
            continue

        _plot_phase_means(
            cond_name,
            axis,
            mouse_id,
            phases,
            phase_records,
            cross_out / f"{cond_name}_phase_mean.png",
        )
        _plot_trial_bins_across_phases(
            cond_name,
            axis,
            mouse_id,
            phases,
            cond_phase_bins,
            cross_out / f"{cond_name}_trial_bins_across_phases.png",
        )
        _plot_traj_grand_mean_across_phases(
            cond_name,
            axis,
            mouse_id,
            phases,
            phase_records,
            cross_out / f"{cond_name}_traj_grand_mean_across_phases.png",
        )
        _plot_traj_bins_across_phases(
            cond_name,
            axis,
            mouse_id,
            phases,
            cond_phase_bins,
            cross_out / f"{cond_name}_traj_bins_across_phases.png",
        )

    for phase, recs in phase_records.items():
        if recs:
            _plot_separation_cloud(
                axis,
                phase,
                recs,
                mouse_id,
                cross_out / f"{phase}_separation_cloud.png",
            )
    _plot_separation_cloud_cross_phase(
        axis,
        mouse_id,
        phases,
        phase_records,
        cross_out / "separation_cloud_across_phases.png",
    )


def _write_axes_definition(
    path: Path,
    *,
    frame_window: tuple[int, int],
    n_pcs: int,
    valence: SvmAxis,
    thirst: SvmAxis,
    valence_sign: str = "positive = toward nacl+airpuff (aversive)",
    thirst_sign: str = "positive = toward nacl",
) -> None:
    payload = {
        "reference_run": REFERENCE_RUN,
        "analysis": ANALYSIS_3,
        "n_pcs": n_pcs,
        "projection_frame_window": list(frame_window),
        "sign_conventions": {
            "valence": valence_sign,
            "thirst": thirst_sign,
        },
        "valence": valence.training,
        "thirst": thirst.training,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_svm_axes(
    out_dir: Path | None = None,
    phases: tuple[str, ...] | list[str] | None = None,
    bin_size: int = 5,
    n_pcs: int = DEFAULT_N_PCS,
    frame_window: tuple[int, int] | None = None,
) -> Path:
    mouse_id = resolve_mouse_id()
    out_root = Path(out_dir) if out_dir is not None else AXES_TRAJECTORIES_SVM_AXES_OUT
    out_root.mkdir(parents=True, exist_ok=True)

    if frame_window is None:
        frame_window = base.SCATTER_MEAN_FRAME_WINDOW

    decoder = _load_decoder_s(REFERENCE_RUN, ANALYSIS_3)
    pre_spec = _pre_3cond_spec()
    _, pre_center, _ = _prepare_trial_tensor(REFERENCE_RUN, pre_spec)

    valence_axis, thirst_axis = _fit_axes_on_pre(
        decoder,
        pre_center,
        frame_window,
        n_pcs,
    )
    _write_axes_definition(
        out_root / "axes_definition.json",
        frame_window=frame_window,
        n_pcs=n_pcs,
        valence=valence_axis,
        thirst=thirst_axis,
    )

    phase_list = tuple(phases) if phases is not None else DEFAULT_PHASES
    phase_valence_records: dict[str, list[TrialRecord]] = {}
    phase_thirst_records: dict[str, list[TrialRecord]] = {}
    phase_joint_records: dict[str, list[JointTrialRecord]] = {}

    for phase in phase_list:
        spec = _spec_for_phase(phase)
        stim_names = tuple(name for name, _ in spec.stimulus_levels)
        trial_x, _, _ = _prepare_trial_tensor(phase, spec)
        order_map = _chronological_order_map(phase, spec)
        joint = _enumerate_joint_trials(
            trial_x,
            pre_center,
            decoder,
            stim_names,
            frame_window,
            valence_axis,
            thirst_axis,
            n_pcs=n_pcs,
            order_map=order_map,
        )
        joint_core = [r for r in joint if r.condition in PLOT_CONDS]
        records = _axis_records_from_joint(joint_core, "valence")
        thirst_records = _axis_records_from_joint(joint_core, "thirst")
        phase_joint_records[phase] = joint
        phase_valence_records[phase] = records
        phase_thirst_records[phase] = thirst_records

        plane_out = out_root / "valence_thirst" / phase
        _plot_valence_thirst_plane(joint_core, phase, mouse_id, plane_out / "plane.png")
        _plot_valence_thirst_plane(
            joint_core,
            phase,
            mouse_id,
            plane_out / "plane_by_order.png",
            show_centroids=False,
            shade_by_order=True,
        )

        for axis, recs in ((valence_axis, records), (thirst_axis, thirst_records)):
            phase_out = out_root / axis.name / phase
            for cond_name in PLOT_CONDS:
                _plot_condition_axis(
                    recs,
                    cond_name,
                    phase,
                    axis,
                    mouse_id,
                    phase_out,
                    bin_size,
                )
            _plot_separation_cloud(
                axis,
                phase,
                recs,
                mouse_id,
                phase_out / "separation_cloud.png",
            )
        print(f"[svm_axes] wrote {phase} plots under {out_root}")

    _plot_cross_phase(
        valence_axis,
        mouse_id,
        phase_list,
        phase_valence_records,
        out_root / valence_axis.name,
        bin_size,
    )
    _plot_cross_phase(
        thirst_axis,
        mouse_id,
        phase_list,
        phase_thirst_records,
        out_root / thirst_axis.name,
        bin_size,
    )
    _plot_valence_thirst_plane_cross_phase(
        phase_list,
        phase_joint_records,
        mouse_id,
        out_root / "valence_thirst" / "cross_phase" / "plane_panels.png",
    )
    _plot_valence_thirst_plane_cross_phase(
        phase_list,
        phase_joint_records,
        mouse_id,
        out_root / "valence_thirst" / "cross_phase" / "plane_panels_by_order.png",
        show_centroids=False,
        shade_by_order=True,
    )
    _plot_valence_thirst_plane_cross_phase(
        phase_list,
        phase_joint_records,
        mouse_id,
        out_root / "valence_thirst" / "cross_phase" / "plane_panels_slm.png",
        plot_conds=PLOT_CONDS_WITH_SLM,
    )
    print(f"[svm_axes] wrote valence x thirst plane plots under {out_root / 'valence_thirst'}")
    print(f"[svm_axes] wrote cross-phase plots under {out_root}")

    print(f"[svm_axes] done -> {out_root}")
    return out_root


def run_svm_axes_water_vs_airpuff_valence(
    out_dir: Path | None = None,
    phases: tuple[str, ...] | list[str] | None = None,
    n_pcs: int = DEFAULT_N_PCS,
    frame_window: tuple[int, int] | None = None,
) -> Path:
    mouse_id = resolve_mouse_id()
    out_root = (
        Path(out_dir)
        if out_dir is not None
        else AXES_TRAJECTORIES_SVM_AXES_WATER_AIRPUFF_VALENCE_OUT
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if frame_window is None:
        frame_window = base.SCATTER_MEAN_FRAME_WINDOW

    decoder = _load_decoder_s(REFERENCE_RUN, ANALYSIS_3)
    pre_spec = _pre_3cond_spec()
    _, pre_center, _ = _prepare_trial_tensor(REFERENCE_RUN, pre_spec)
    valence_axis, thirst_axis = _fit_axes_on_pre(
        decoder,
        pre_center,
        frame_window,
        n_pcs,
        valence_mode="water_vs_airpuff",
    )
    _write_axes_definition(
        out_root / "axes_definition.json",
        frame_window=frame_window,
        n_pcs=n_pcs,
        valence=valence_axis,
        thirst=thirst_axis,
        valence_sign="positive = toward airpuff (aversive)",
    )

    phase_list = tuple(phases) if phases is not None else DEFAULT_PHASES
    phase_thirst_records: dict[str, list[TrialRecord]] = {}
    phase_valence_records: dict[str, list[TrialRecord]] = {}
    phase_joint_records: dict[str, list[JointTrialRecord]] = {}

    for phase in phase_list:
        spec = _spec_for_phase(phase)
        stim_names = tuple(name for name, _ in spec.stimulus_levels)
        trial_x, _, _ = _prepare_trial_tensor(phase, spec)
        order_map = _chronological_order_map(phase, spec)
        joint = _enumerate_joint_trials(
            trial_x,
            pre_center,
            decoder,
            stim_names,
            frame_window,
            valence_axis,
            thirst_axis,
            n_pcs=n_pcs,
            order_map=order_map,
        )
        joint_core = [r for r in joint if r.condition in PLOT_CONDS]
        phase_joint_records[phase] = joint
        phase_thirst_records[phase] = _axis_records_from_joint(joint_core, "thirst")
        phase_valence_records[phase] = _axis_records_from_joint(joint_core, "valence")
        print(f"[svm_axes/water_vs_airpuff_valence] scored {phase} ({len(joint)} trials)")

    title_suffix = "valence = water vs airpuff"
    _plot_separation_cloud_both_axes(
        mouse_id,
        phase_list,
        thirst_axis,
        valence_axis,
        phase_thirst_records,
        phase_valence_records,
        out_root / "separation_cloud_across_phases.png",
        title_suffix=title_suffix,
    )
    _plot_valence_thirst_plane_cross_phase(
        phase_list,
        phase_joint_records,
        mouse_id,
        out_root / "plane_panels.png",
        title_suffix=title_suffix,
    )
    _plot_valence_thirst_plane_cross_phase(
        phase_list,
        phase_joint_records,
        mouse_id,
        out_root / "plane_panels_slm.png",
        plot_conds=PLOT_CONDS_WITH_SLM,
        title_suffix=title_suffix,
    )
    print(f"[svm_axes/water_vs_airpuff_valence] done -> {out_root}")
    return out_root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {AXES_TRAJECTORIES_SVM_AXES_OUT}).",
    )
    p.add_argument(
        "--phases",
        nargs="+",
        default=list(DEFAULT_PHASES),
        choices=list(DEFAULT_PHASES),
        help="Phases to project and plot.",
    )
    p.add_argument("--bin-size", type=int, default=5, help="Trials per chronological bin.")
    p.add_argument("--n-pcs", type=int, default=DEFAULT_N_PCS, help="s-dPCA dimensions.")
    args = p.parse_args()
    run_svm_axes(
        out_dir=args.out_dir,
        phases=tuple(args.phases),
        bin_size=int(args.bin_size),
        n_pcs=int(args.n_pcs),
    )
    variant_root = (
        Path(args.out_dir) / "water_vs_airpuff_valence"
        if args.out_dir is not None
        else AXES_TRAJECTORIES_SVM_AXES_WATER_AIRPUFF_VALENCE_OUT
    )
    run_svm_axes_water_vs_airpuff_valence(
        out_dir=variant_root,
        phases=tuple(args.phases),
        n_pcs=int(args.n_pcs),
    )


if __name__ == "__main__":
    main()
