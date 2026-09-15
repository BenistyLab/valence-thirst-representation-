"""
SVM separating axes on z-scored neuron activity (no dPCA).

Default two pre-trained linear SVMs (same as svm_axes, no dPCA):
  - valence: water vs nacl+airpuff
  - thirst: water vs nacl

Also writes water_vs_airpuff_valence/:
  - valence: water vs airpuff (nacl held out, then projected)
  - thirst: water vs nacl (airpuff held out, then projected)

Fits and projects on window-mean neuron features from the normalized trial
tensors, skipping the 5D s-dPCA subspace. Writes only the requested
cross-phase figures under axes_trajectories/svm_no_dpca/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from lib.config import (
    AXES_TRAJECTORIES_SVM_NO_DPCA_OUT,
    AXES_TRAJECTORIES_SVM_NO_DPCA_WATER_AIRPUFF_VALENCE_OUT,
    resolve_mouse_id,
)
from lib.dpca import analysis as base
from lib.dpca.condition_distances import _prepare_trial_tensor
from lib.dpca.svm_axes import (
    DEFAULT_PHASES,
    PLOT_CONDS,
    PLOT_CONDS_WITH_SLM,
    REFERENCE_RUN,
    JointTrialRecord,
    SvmAxis,
    TrialRecord,
    _axis_records_from_joint,
    _chronological_order_map,
    _fit_svm_axis,
    _PHASE_COLORS,
    _plot_separation_cloud_both_axes,
    _plot_valence_thirst_plane_cross_phase,
    _pre_3cond_spec,
    _stim_names_3,
    _summarize_bins,
)
from lib.dpca.thirst_valence_axes import _spec_for_phase

ANALYSIS_3 = "3conditions"


def _window_mean_features(
    trial_x: np.ndarray,
    center: np.ndarray,
    frame_window: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial window-mean of centered neuron traces. Labels are stim indices."""
    start, stop = frame_window
    stop = min(stop, int(trial_x.shape[3]))
    if stop <= start:
        raise ValueError(f"Frame window {frame_window} is outside trial length {trial_x.shape[3]}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    for stim_ix in range(trial_x.shape[2]):
        for trial_ix in range(trial_x.shape[0]):
            xi = np.asarray(trial_x[trial_ix, :, stim_ix, :], dtype=np.float64)
            if np.any(np.isnan(xi)):
                continue
            centered = xi - center[:, None]
            features.append(np.mean(centered[:, start:stop], axis=1))
            labels.append(int(stim_ix))

    if not features:
        raise RuntimeError("No finite trial features were assembled.")
    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def _fit_axes_on_pre(
    center: np.ndarray,
    frame_window: tuple[int, int],
    *,
    valence_mode: str = "nacl_airpuff",
) -> tuple[SvmAxis, SvmAxis]:
    spec = _pre_3cond_spec()
    trial_x, _, _ = _prepare_trial_tensor(REFERENCE_RUN, spec)
    features, labels = _window_mean_features(trial_x, center, frame_window)
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


def _enumerate_joint_trials(
    trial_x: np.ndarray,
    center: np.ndarray,
    stim_names: tuple[str, ...],
    frame_window: tuple[int, int],
    valence_axis: SvmAxis,
    thirst_axis: SvmAxis,
    *,
    order_map: dict[tuple[int, int], int] | None = None,
) -> list[JointTrialRecord]:
    start, stop = frame_window
    stop = min(stop, int(trial_x.shape[3]))
    w_v = valence_axis.w
    b_v = valence_axis.b
    w_t = thirst_axis.w
    b_t = thirst_axis.b

    records: list[JointTrialRecord] = []
    for stim_ix, cond_name in enumerate(stim_names):
        if cond_name not in PLOT_CONDS_WITH_SLM:
            continue
        for trial_ix in range(trial_x.shape[0]):
            xi = np.asarray(trial_x[trial_ix, :, stim_ix, :], dtype=np.float64)
            if np.any(np.isnan(xi)):
                continue
            centered = xi - center[:, None]
            feat = np.mean(centered[:, start:stop], axis=1)
            order_in_phase = int(order_map.get((stim_ix, trial_ix), len(records))) if order_map else len(records)
            records.append(
                JointTrialRecord(
                    condition=cond_name,
                    valence_score=float(w_v @ feat + b_v),
                    thirst_score=float(w_t @ feat + b_t),
                    valence_traj=(w_v @ centered + b_v).astype(np.float64),
                    thirst_traj=(w_t @ centered + b_t).astype(np.float64),
                    features_5d=feat.astype(np.float64),
                    order_in_phase=order_in_phase,
                )
            )
    return records


def _draw_trial_bins_ax(
    ax,
    phases: tuple[str, ...],
    phase_bins: dict[str, list],
    *,
    legend: bool,
) -> bool:
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
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_ylabel("SVM signed score")
    ax.grid(True, alpha=0.25)
    if legend and any_plotted:
        ax.legend(fontsize=8, title="Phase")
    return any_plotted


def _plot_trial_bins_across_phases_both_axes(
    cond_name: str,
    mouse_id: str,
    phases: tuple[str, ...],
    thirst_axis: SvmAxis,
    valence_axis: SvmAxis,
    thirst_bins: dict[str, list],
    valence_bins: dict[str, list],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)
    plotted = False
    for ax, axis, phase_bins in (
        (axes[0], thirst_axis, thirst_bins),
        (axes[1], valence_axis, valence_bins),
    ):
        if _draw_trial_bins_ax(ax, phases, phase_bins, legend=True):
            plotted = True
        ax.set_title(f"{axis.name} axis")

    if not plotted:
        plt.close(fig)
        return

    axes[1].set_xlabel("Chronological trial bin index (within phase)")
    fig.suptitle(f"{mouse_id} · {cond_name} (trial bins across phases)", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _axis_meta(axis: SvmAxis) -> dict[str, Any]:
    meta = dict(axis.training)
    meta.pop("w_unit", None)
    return meta


def _write_axes_definition(
    path: Path,
    *,
    frame_window: tuple[int, int],
    n_neurons: int,
    valence: SvmAxis,
    thirst: SvmAxis,
    valence_sign: str,
    thirst_sign: str = "positive = toward nacl",
) -> None:
    payload: dict[str, Any] = {
        "reference_run": REFERENCE_RUN,
        "analysis": ANALYSIS_3,
        "feature_space": "normalized neuron window-mean (no dPCA)",
        "n_neurons": n_neurons,
        "projection_frame_window": list(frame_window),
        "sign_conventions": {
            "valence": valence_sign,
            "thirst": thirst_sign,
        },
        "valence": _axis_meta(valence),
        "thirst": _axis_meta(thirst),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_svm_no_dpca(
    out_dir: Path | None = None,
    phases: tuple[str, ...] | list[str] | None = None,
    bin_size: int = 5,
    frame_window: tuple[int, int] | None = None,
    *,
    valence_mode: str = "nacl_airpuff",
    write_trial_bins: bool = True,
    title_suffix: str = "no dPCA",
    valence_sign: str = "positive = toward nacl+airpuff (aversive)",
) -> Path:
    mouse_id = resolve_mouse_id()
    out_root = Path(out_dir) if out_dir is not None else AXES_TRAJECTORIES_SVM_NO_DPCA_OUT
    out_root.mkdir(parents=True, exist_ok=True)

    if frame_window is None:
        frame_window = base.SCATTER_MEAN_FRAME_WINDOW

    pre_spec = _pre_3cond_spec()
    _, pre_center, _ = _prepare_trial_tensor(REFERENCE_RUN, pre_spec)
    valence_axis, thirst_axis = _fit_axes_on_pre(
        pre_center, frame_window, valence_mode=valence_mode
    )
    _write_axes_definition(
        out_root / "axes_definition.json",
        frame_window=frame_window,
        n_neurons=int(pre_center.shape[0]),
        valence=valence_axis,
        thirst=thirst_axis,
        valence_sign=valence_sign,
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
            stim_names,
            frame_window,
            valence_axis,
            thirst_axis,
            order_map=order_map,
        )
        joint_core = [r for r in joint if r.condition in PLOT_CONDS]
        phase_joint_records[phase] = joint
        phase_thirst_records[phase] = _axis_records_from_joint(joint_core, "thirst")
        phase_valence_records[phase] = _axis_records_from_joint(joint_core, "valence")
        print(f"[svm_no_dpca] scored {phase} ({len(joint)} trials)")

    if write_trial_bins:
        for cond_name in PLOT_CONDS:
            thirst_bins = {
                phase: _summarize_bins(recs, cond_name, bin_size)
                for phase, recs in phase_thirst_records.items()
            }
            valence_bins = {
                phase: _summarize_bins(recs, cond_name, bin_size)
                for phase, recs in phase_valence_records.items()
            }
            if not any(thirst_bins.values()) and not any(valence_bins.values()):
                continue
            _plot_trial_bins_across_phases_both_axes(
                cond_name,
                mouse_id,
                phase_list,
                thirst_axis,
                valence_axis,
                thirst_bins,
                valence_bins,
                out_root / f"{cond_name}_trial_bins_across_phases.png",
            )

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

    print(f"[svm_no_dpca] done -> {out_root}")
    return out_root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {AXES_TRAJECTORIES_SVM_NO_DPCA_OUT}).",
    )
    p.add_argument(
        "--phases",
        nargs="+",
        default=list(DEFAULT_PHASES),
        choices=list(DEFAULT_PHASES),
        help="Phases to project and plot.",
    )
    p.add_argument("--bin-size", type=int, default=5, help="Trials per chronological bin.")
    args = p.parse_args()
    run_svm_no_dpca(
        out_dir=args.out_dir,
        phases=tuple(args.phases),
        bin_size=int(args.bin_size),
    )
    variant_root = (
        Path(args.out_dir) / "water_vs_airpuff_valence"
        if args.out_dir is not None
        else AXES_TRAJECTORIES_SVM_NO_DPCA_WATER_AIRPUFF_VALENCE_OUT
    )
    run_svm_no_dpca(
        out_dir=variant_root,
        phases=tuple(args.phases),
        bin_size=int(args.bin_size),
        valence_mode="water_vs_airpuff",
        write_trial_bins=False,
        title_suffix="no dPCA; valence = water vs airpuff",
        valence_sign="positive = toward airpuff (aversive)",
    )


if __name__ == "__main__":
    main()
