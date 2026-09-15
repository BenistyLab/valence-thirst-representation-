"""
Fixed thirst / valence axes from pre-run dPCA s-space.

Defines:
  - thirst axis: unit(centroid(nacl) - centroid(water)) in pre 3conditions 5D s-space
  - valence A:  unit(mean(nacl, airpuff) - water) in the same space
  - valence B:  pre hedonic_valence s-dPC1 (window mean)

Projects all phases onto those fixed axes (including SLM when present) and writes
centroid plots plus SLM-shift / separability summaries.

Two normalization modes control how non-reference phases are prepared before projection:
  - per_phase (``thirst_valence_axes/within_phase``): each phase z-scored + centered on
    its own tensor. Isolates within-phase condition separation; removes cross-phase drift.
  - shared (``thirst_valence_axes/drift``): every phase z-scored + centered with the
    reference (pre) tensor's stats, so cross-phase drift is preserved and across-phase
    centroid movement is interpretable as drift. Adds drift_summary.csv + drift plots.

Run after run_dpca.py (needs pre/3conditions and pre/hedonic_valence decoder matrices).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from lib.dpca import analysis as base
from lib.dpca.condition_distances import (
    _centroids_by_condition,
    _trial_features_s,
)
from lib.io.npz import load_npz_pack

REFERENCE_RUN = base.RUN_KEYS[0]  # "pre"
ANALYSIS_3 = "3conditions"
ANALYSIS_HEDONIC = "hedonic_valence"
DEFAULT_N_PCS = 5
NON_SLM_CONDS = ("water", "nacl", "airpuff")
OUTPUT_ROOT = base.DPCA_OUT_ROOT / "thirst_valence_axes"

_COLOR_BY_COND = {
    "water": base.COLORS_4[0],
    "nacl": base.COLORS_4[1],
    "airpuff": base.COLORS_4[2],
    "slm": base.COLORS_4[3],
}
_PHASE_MARKERS = {
    "pre": "o",
    "airpuff": "s",
    "water": "^",
}


def _load_decoder_s(run_key: str, analysis: str) -> np.ndarray:
    path = base.DPCA_OUT_ROOT / run_key / analysis / "loadings" / "encoder_decoder_matrices.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_dpca.py first so decoder matrices are saved."
        )
    with np.load(path) as data:
        if "decoder_D_s" not in data.files:
            raise KeyError(f"decoder_D_s not found in {path}")
        return np.asarray(data["decoder_D_s"], dtype=np.float64)


def _stimulus_levels_for_phase(phase: str) -> tuple[tuple[str, frozenset[int]], ...]:
    pack = load_npz_pack(phase)
    cond_ids = frozenset(int(x) for x in pack["trial_type"])
    if base.SLM_COND_ID in cond_ids:
        return base._STIMULUS_4  # noqa: SLF001
    return base._STIMULUS_3  # noqa: SLF001


def _spec_for_phase(phase: str) -> base.AnalysisSpec:
    levels = _stimulus_levels_for_phase(phase)
    colors = base.COLORS_4 if len(levels) == 4 else base.COLORS_3
    return base.AnalysisSpec(
        "thirst_valence_projection",
        f"{phase} — thirst/valence projection",
        levels,
        colors,
    )


SharedNorm = tuple[np.ndarray, np.ndarray, np.ndarray]  # (mu, sigma, center) per neuron


def _fit_zscore_stats(trial_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-neuron mean/std over all (trial, stim, frame) samples, ignoring NaNs."""
    n_neurons = int(trial_x.shape[1])
    mu = np.zeros(n_neurons, dtype=np.float64)
    sig = np.ones(n_neurons, dtype=np.float64)
    for n in range(n_neurons):
        v = trial_x[:, n, :, :].ravel()
        v = v[~np.isnan(v)]
        if v.size >= 2:
            mu[n] = float(np.mean(v))
            sig[n] = max(float(np.std(v)), 1e-8)
    return mu, sig


def _apply_zscore(
    x_mean: np.ndarray, trial_x: np.ndarray, mu: np.ndarray, sig: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x_fit = (x_mean.astype(np.float64) - mu[:, None, None]) / sig[:, None, None]
    trial_x_fit = (trial_x.astype(np.float64) - mu[None, :, None, None]) / sig[None, :, None, None]
    return x_fit, trial_x_fit


def _fit_shared_norm(reference_run: str, spec: base.AnalysisSpec) -> SharedNorm:
    """Fit z-score stats + center on the reference phase for reuse across all phases."""
    pack = load_npz_pack(reference_run)
    segs_by_s, _, n_neurons = base._collect_segments(pack, spec.stimulus_levels)  # noqa: SLF001
    trial_x, x_mean, *_ = base._build_mean_and_trialx(  # noqa: SLF001
        segs_by_s,
        spec.stimulus_levels,
        base.ANALYSIS_T_FRAMES,
        n_neurons,
    )
    mu, sig = _fit_zscore_stats(trial_x)
    x_fit, _ = _apply_zscore(x_mean, trial_x, mu, sig)
    center = np.mean(x_fit.reshape((x_fit.shape[0], -1)), axis=1)
    return mu, sig, center


def _prepare_trial_tensor(
    run_key: str,
    spec: base.AnalysisSpec,
    shared_stats: SharedNorm | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pack = load_npz_pack(run_key)
    segs_by_s, min_t, n_neurons = base._collect_segments(pack, spec.stimulus_levels)  # noqa: SLF001
    trial_x, x_mean, counts, candidate_counts, ignored_short_counts = base._build_mean_and_trialx(  # noqa: SLF001
        segs_by_s,
        spec.stimulus_levels,
        base.ANALYSIS_T_FRAMES,
        n_neurons,
    )
    if shared_stats is None:
        x_fit, trial_x_fit = base._zscore_mean_and_trialx(x_mean, trial_x)  # noqa: SLF001
        center = np.mean(x_fit.reshape((x_fit.shape[0], -1)), axis=1)
        norm_mode = "per_phase"
    else:
        mu, sig, shared_center = shared_stats
        _, trial_x_fit = _apply_zscore(x_mean, trial_x, mu, sig)
        center = np.asarray(shared_center, dtype=np.float64)
        norm_mode = "shared"
    meta = {
        "run_key": run_key,
        "normalize_mode": norm_mode,
        "stimulus_levels": [name for name, _ in spec.stimulus_levels],
        "trial_counts": counts,
        "candidate_trial_counts": candidate_counts,
        "ignored_short_trial_counts": ignored_short_counts,
        "frame_window": base.ANALYSIS_T_FRAMES,
        "projection_frame_window": list(base.SCATTER_MEAN_FRAME_WINDOW),
        "min_candidate_trial_frames": min_t,
        "n_neurons": int(n_neurons),
    }
    return trial_x_fit, center, meta


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise RuntimeError("Cannot normalize near-zero axis vector.")
    return np.asarray(v, dtype=np.float64) / n


def _define_thirst_valence_axes(centroids: dict[str, np.ndarray]) -> dict[str, Any]:
    water = centroids["water"]
    nacl = centroids["nacl"]
    airpuff = centroids["airpuff"]
    aversive_mean = 0.5 * (nacl + airpuff)

    thirst = _unit(nacl - water)
    valence_centroid = _unit(aversive_mean - water)
    cos_ang = float(np.clip(np.dot(thirst, valence_centroid), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_ang)))

    return {
        "thirst_axis": thirst,
        "valence_centroid_axis": valence_centroid,
        "angle_deg": angle_deg,
        "pre_centroids": {k: v.tolist() for k, v in centroids.items()},
        "sign_conventions": {
            "thirst": "positive = more nacl-like (thirst/salt)",
            "valence_centroid": "positive = more aversive (toward mean of nacl+airpuff)",
            "valence_hedonic": "positive = aversive pole of pre hedonic_valence s-dPC1",
        },
    }


def _project_onto_axis(features: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return (features - origin[None, :]) @ axis


def _hedonic_s_coords(
    trial_x: np.ndarray,
    center: np.ndarray,
    hedonic_decoder: np.ndarray,
    *,
    frame_window: tuple[int, int],
) -> np.ndarray:
    """1D window-mean projection onto hedonic s-dPC1 (column 0)."""
    start, stop = frame_window
    stop = min(stop, int(trial_x.shape[3]))
    decoder = np.asarray(hedonic_decoder[:, :1], dtype=np.float64)
    values: list[float] = []
    for stim_ix in range(trial_x.shape[2]):
        for trial_ix in range(trial_x.shape[0]):
            xi = np.asarray(trial_x[trial_ix, :, stim_ix, :], dtype=np.float64)
            if np.any(np.isnan(xi)):
                continue
            projected = decoder.T @ (xi - center[:, None])
            values.append(float(np.mean(projected[0, start:stop])))
    return np.asarray(values, dtype=np.float64)


def _flip_hedonic_sign(
    hedonic_coords: np.ndarray,
    labels: np.ndarray,
    stim_names: tuple[str, ...],
) -> tuple[np.ndarray, bool]:
    """
    Ensure positive hedonic = aversive: if pre water mean > aversive mean, flip.
    Returns (coords, flipped).
    """
    name_to_ix = {n: i for i, n in enumerate(stim_names)}
    water_mask = labels == name_to_ix["water"]
    aversive_mask = np.isin(labels, [name_to_ix["nacl"], name_to_ix["airpuff"]])
    water_mean = float(np.mean(hedonic_coords[water_mask]))
    aversive_mean = float(np.mean(hedonic_coords[aversive_mask]))
    if water_mean > aversive_mean:
        return -hedonic_coords, True
    return hedonic_coords, False


def _sem_1d(values: np.ndarray) -> float:
    n = int(values.size)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_phase_data(
    phase: str,
    decoder_3cond: np.ndarray,
    hedonic_decoder: np.ndarray,
    *,
    origin: np.ndarray,
    thirst_axis: np.ndarray,
    valence_centroid_axis: np.ndarray,
    n_pcs: int,
    frame_window: tuple[int, int],
    flip_hedonic: bool | None = None,
    shared_stats: SharedNorm | None = None,
) -> dict[str, Any]:
    spec = _spec_for_phase(phase)
    stim_names = tuple(n for n, _ in spec.stimulus_levels)
    trial_x, center, meta = _prepare_trial_tensor(phase, spec, shared_stats=shared_stats)
    features_5d, labels = _trial_features_s(
        trial_x,
        center,
        decoder_3cond,
        n_pcs=n_pcs,
        frame_window=frame_window,
    )
    thirst = _project_onto_axis(features_5d, origin, thirst_axis)
    valence_c = _project_onto_axis(features_5d, origin, valence_centroid_axis)
    hedonic_raw = _hedonic_s_coords(trial_x, center, hedonic_decoder, frame_window=frame_window)
    if hedonic_raw.shape[0] != features_5d.shape[0]:
        raise RuntimeError(
            f"Hedonic coord count ({hedonic_raw.shape[0]}) != feature count ({features_5d.shape[0]}) "
            f"for phase {phase}."
        )
    if flip_hedonic is None:
        hedonic, flipped = _flip_hedonic_sign(hedonic_raw, labels, stim_names)
    else:
        hedonic = -hedonic_raw if flip_hedonic else hedonic_raw
        flipped = flip_hedonic

    trial_rows: list[dict[str, Any]] = []
    trial_counters: dict[str, int] = {n: 0 for n in stim_names}
    for i in range(features_5d.shape[0]):
        cond = stim_names[int(labels[i])]
        trial_counters[cond] += 1
        trial_rows.append(
            {
                "phase": phase,
                "condition": cond,
                "trial_index_within_condition": trial_counters[cond],
                "thirst": float(thirst[i]),
                "valence_centroid": float(valence_c[i]),
                "valence_hedonic": float(hedonic[i]),
            }
        )

    return {
        "phase": phase,
        "stim_names": stim_names,
        "meta": meta,
        "features_5d": features_5d,
        "labels": labels,
        "thirst": thirst,
        "valence_centroid": valence_c,
        "valence_hedonic": hedonic,
        "trial_rows": trial_rows,
        "hedonic_flipped": flipped,
    }


def _centroid_summary_rows(phase_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase = phase_data["phase"]
    labels = phase_data["labels"]
    stim_names = phase_data["stim_names"]
    for axis_name, values in (
        ("thirst", phase_data["thirst"]),
        ("valence_centroid", phase_data["valence_centroid"]),
        ("valence_hedonic", phase_data["valence_hedonic"]),
    ):
        for stim_ix, name in enumerate(stim_names):
            pts = values[labels == stim_ix]
            rows.append(
                {
                    "phase": phase,
                    "condition": name,
                    "axis": axis_name,
                    "n_trials": int(pts.size),
                    "mean": float(np.mean(pts)),
                    "sem": _sem_1d(pts),
                }
            )
    return rows


def _separability_rows(
    phase_data_by_phase: dict[str, dict[str, Any]],
    reference_phase: str,
) -> list[dict[str, Any]]:
    """Per-phase separations along thirst and both valence axes vs reference."""
    ref = phase_data_by_phase[reference_phase]
    ref_means = _means_by_condition(ref)
    rows: list[dict[str, Any]] = []

    metrics = (
        ("thirst_water_nacl", "thirst", "water", "nacl"),
        ("valence_centroid_water_aversive", "valence_centroid", "water", "aversive"),
        ("valence_hedonic_water_aversive", "valence_hedonic", "water", "aversive"),
    )
    ref_vals: dict[str, float] = {}
    for key, axis, a, b in metrics:
        ref_vals[key] = _sep(ref_means, axis, a, b)

    for phase, pdata in phase_data_by_phase.items():
        means = _means_by_condition(pdata)
        for key, axis, a, b in metrics:
            val = _sep(means, axis, a, b)
            ref_v = ref_vals[key]
            fold = float(val / ref_v) if abs(ref_v) > 1e-12 else float("nan")
            rows.append(
                {
                    "phase": phase,
                    "metric": key,
                    "axis": axis,
                    "separation": val,
                    "reference_phase": reference_phase,
                    "reference_separation": ref_v,
                    "fold_change_vs_reference": fold,
                }
            )
    return rows


def _means_by_condition(pdata: dict[str, Any]) -> dict[str, dict[str, float]]:
    labels = pdata["labels"]
    stim_names = pdata["stim_names"]
    out: dict[str, dict[str, float]] = {}
    for stim_ix, name in enumerate(stim_names):
        mask = labels == stim_ix
        out[name] = {
            "thirst": float(np.mean(pdata["thirst"][mask])),
            "valence_centroid": float(np.mean(pdata["valence_centroid"][mask])),
            "valence_hedonic": float(np.mean(pdata["valence_hedonic"][mask])),
        }
    # synthetic aversive = mean of nacl + airpuff means
    if "nacl" in out and "airpuff" in out:
        out["aversive"] = {
            ax: 0.5 * (out["nacl"][ax] + out["airpuff"][ax])
            for ax in ("thirst", "valence_centroid", "valence_hedonic")
        }
    return out


def _sep(means: dict[str, dict[str, float]], axis: str, a: str, b: str) -> float:
    return float(means[b][axis] - means[a][axis])


def _slm_shift_rows(phase_data_by_phase: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, pdata in phase_data_by_phase.items():
        stim_names = pdata["stim_names"]
        if "slm" not in stim_names:
            continue
        means = _means_by_condition(pdata)
        non_slm = [c for c in NON_SLM_CONDS if c in means]
        if not non_slm:
            continue
        for axis in ("thirst", "valence_centroid", "valence_hedonic"):
            non_slm_mean = float(np.mean([means[c][axis] for c in non_slm]))
            slm_mean = means["slm"][axis]
            # also vs each non-SLM condition
            rows.append(
                {
                    "phase": phase,
                    "axis": axis,
                    "slm_mean": slm_mean,
                    "non_slm_mean": non_slm_mean,
                    "slm_minus_non_slm": float(slm_mean - non_slm_mean),
                    "slm_minus_water": float(slm_mean - means["water"][axis]),
                    "slm_minus_nacl": float(slm_mean - means["nacl"][axis]),
                    "slm_minus_airpuff": float(slm_mean - means["airpuff"][axis]),
                }
            )
    return rows


def _shared_limits(
    phase_data_by_phase: dict[str, dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    pad_frac: float = 0.12,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = np.concatenate([d[x_key] for d in phase_data_by_phase.values()])
    ys = np.concatenate([d[y_key] for d in phase_data_by_phase.values()])
    def _lim(arr: np.ndarray) -> tuple[float, float]:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        span = hi - lo if hi > lo else 1.0
        pad = pad_frac * span
        return lo - pad, hi + pad
    return _lim(xs), _lim(ys)


def _scatter_phase_panel(
    ax: plt.Axes,
    pdata: dict[str, Any],
    *,
    x_key: str,
    y_key: str,
    title: str,
    show_legend: bool,
) -> None:
    labels = pdata["labels"]
    stim_names = pdata["stim_names"]
    for stim_ix, name in enumerate(stim_names):
        mask = labels == stim_ix
        col = _COLOR_BY_COND.get(name, "0.4")
        ax.scatter(
            pdata[x_key][mask],
            pdata[y_key][mask],
            s=28,
            alpha=0.55,
            color=col,
            edgecolors="none",
            label=name,
            zorder=2,
        )
        cx = float(np.mean(pdata[x_key][mask]))
        cy = float(np.mean(pdata[y_key][mask]))
        ax.scatter(
            [cx],
            [cy],
            s=130,
            color=col,
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        ax.annotate(name, (cx, cy), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # connect water-nacl and water-aversive midpoints for thirst/valence context
    means = _means_by_condition(pdata)
    if all(k in means for k in ("water", "nacl", "airpuff")):
        wx, wy = means["water"][x_key], means["water"][y_key]
        nx, ny = means["nacl"][x_key], means["nacl"][y_key]
        ax_mean_x = 0.5 * (means["nacl"][x_key] + means["airpuff"][x_key])
        ax_mean_y = 0.5 * (means["nacl"][y_key] + means["airpuff"][y_key])
        ax.plot([wx, nx], [wy, ny], color="0.45", linewidth=1.0, linestyle="--", zorder=3)
        ax.plot([wx, ax_mean_x], [wy, ax_mean_y], color="0.45", linewidth=1.0, linestyle=":", zorder=3)

    ax.axhline(0.0, color="0.7", linewidth=0.7, zorder=1)
    ax.axvline(0.0, color="0.7", linewidth=0.7, zorder=1)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(fontsize=8, loc="best")


def _plot_faceted(
    phase_data_by_phase: dict[str, dict[str, Any]],
    out_path: Path,
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    phases = [p for p in base.RUN_KEYS if p in phase_data_by_phase]
    xlim, ylim = _shared_limits(phase_data_by_phase, x_key, y_key)
    fig, axes = plt.subplots(1, len(phases), figsize=(5.2 * len(phases), 5.0), sharex=True, sharey=True)
    if len(phases) == 1:
        axes = [axes]
    for ax, phase in zip(axes, phases):
        _scatter_phase_panel(
            ax,
            phase_data_by_phase[phase],
            x_key=x_key,
            y_key=y_key,
            title=phase,
            show_legend=(phase == phases[0]),
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay(
    phase_data_by_phase: dict[str, dict[str, Any]],
    out_path: Path,
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    phases = [p for p in base.RUN_KEYS if p in phase_data_by_phase]
    for phase in phases:
        pdata = phase_data_by_phase[phase]
        marker = _PHASE_MARKERS.get(phase, "o")
        labels = pdata["labels"]
        stim_names = pdata["stim_names"]
        for stim_ix, name in enumerate(stim_names):
            mask = labels == stim_ix
            col = _COLOR_BY_COND.get(name, "0.4")
            ax.scatter(
                pdata[x_key][mask],
                pdata[y_key][mask],
                s=18,
                alpha=0.25,
                color=col,
                marker=marker,
                edgecolors="none",
                zorder=2,
            )
            cx = float(np.mean(pdata[x_key][mask]))
            cy = float(np.mean(pdata[y_key][mask]))
            ax.scatter(
                [cx],
                [cy],
                s=140,
                color=col,
                marker=marker,
                edgecolors="black",
                linewidths=1.3,
                zorder=5,
                label=f"{phase}:{name}",
            )
    ax.axhline(0.0, color="0.7", linewidth=0.7)
    ax.axvline(0.0, color="0.7", linewidth=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_slm_shift_bars(slm_rows: list[dict[str, Any]], out_path: Path) -> None:
    if not slm_rows:
        return
    phases = sorted({r["phase"] for r in slm_rows}, key=lambda p: list(base.RUN_KEYS).index(p) if p in base.RUN_KEYS else 99)
    axes_names = ("thirst", "valence_centroid", "valence_hedonic")
    axis_labels = {
        "thirst": "Thirst",
        "valence_centroid": "Valence (centroid)",
        "valence_hedonic": "Valence (hedonic)",
    }
    x = np.arange(len(axes_names), dtype=np.float64)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    cmap = plt.get_cmap("tab10")
    for i, phase in enumerate(phases):
        vals = []
        for axis in axes_names:
            match = next((r for r in slm_rows if r["phase"] == phase and r["axis"] == axis), None)
            vals.append(float(match["slm_minus_non_slm"]) if match else float("nan"))
        offset = (i - (len(phases) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, color=cmap(i % 10), label=phase)
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([axis_labels[a] for a in axes_names])
    ax.set_ylabel("SLM − mean(non-SLM) centroid")
    ax.set_title("SLM shift on fixed thirst / valence axes")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_separation_across_phases(sep_rows: list[dict[str, Any]], out_path: Path) -> None:
    metrics = (
        "thirst_water_nacl",
        "valence_centroid_water_aversive",
        "valence_hedonic_water_aversive",
    )
    metric_labels = {
        "thirst_water_nacl": "Thirst: water→nacl",
        "valence_centroid_water_aversive": "ValenceA: water→aversive",
        "valence_hedonic_water_aversive": "ValenceB: water→aversive",
    }
    phases = [p for p in base.RUN_KEYS if any(r["phase"] == p for r in sep_rows)]
    x = np.arange(len(phases), dtype=np.float64)
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, metric in enumerate(metrics):
        vals = []
        for phase in phases:
            match = next((r for r in sep_rows if r["phase"] == phase and r["metric"] == metric), None)
            vals.append(float(match["separation"]) if match else float("nan"))
        ax.bar(x + (i - 1) * width, vals, width=width, label=metric_labels[metric])
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Signed separation (b − a)")
    ax.set_title("Condition separation on fixed thirst / valence axes")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


_DRIFT_AXES = ("thirst", "valence_centroid", "valence_hedonic")
_DRIFT_AXIS_LABELS = {
    "thirst": "Thirst",
    "valence_centroid": "Valence (centroid)",
    "valence_hedonic": "Valence (hedonic)",
}


def _drift_rows(
    phase_data_by_phase: dict[str, dict[str, Any]],
    reference_phase: str,
) -> list[dict[str, Any]]:
    """Overall (grand-mean) and water-centroid position per axis, per phase, vs reference.

    Only meaningful under shared normalization, where all phases live in one common
    frame, so a change in these positions reflects real cross-phase drift.
    """
    ref = phase_data_by_phase[reference_phase]
    ref_grand = {ax: float(np.mean(ref[ax])) for ax in _DRIFT_AXES}
    ref_means = _means_by_condition(ref)
    rows: list[dict[str, Any]] = []
    for phase, pdata in phase_data_by_phase.items():
        means = _means_by_condition(pdata)
        for ax in _DRIFT_AXES:
            grand = float(np.mean(pdata[ax]))
            water = float(means["water"][ax]) if "water" in means else float("nan")
            ref_water = float(ref_means["water"][ax]) if "water" in ref_means else float("nan")
            rows.append(
                {
                    "phase": phase,
                    "axis": ax,
                    "grand_mean": grand,
                    "reference_grand_mean": ref_grand[ax],
                    "grand_mean_drift_vs_reference": grand - ref_grand[ax],
                    "water_centroid": water,
                    "water_centroid_drift_vs_reference": water - ref_water,
                }
            )
    return rows


def _plot_drift(
    drift_rows: list[dict[str, Any]],
    out_path: Path,
    *,
    value_key: str,
    title: str,
) -> None:
    if not drift_rows:
        return
    phases = [p for p in base.RUN_KEYS if any(r["phase"] == p for r in drift_rows)]
    x = np.arange(len(phases), dtype=np.float64)
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, axis_name in enumerate(_DRIFT_AXES):
        vals = []
        for phase in phases:
            match = next(
                (r for r in drift_rows if r["phase"] == phase and r["axis"] == axis_name),
                None,
            )
            vals.append(float(match[value_key]) if match else float("nan"))
        ax.bar(x + (i - 1) * width, vals, width=width, label=_DRIFT_AXIS_LABELS[axis_name])
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel(value_key.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_thirst_valence_axes(
    *,
    reference_run: str = REFERENCE_RUN,
    n_pcs: int = DEFAULT_N_PCS,
    frame_window: tuple[int, int] | None = None,
    out_root: Path | None = None,
    normalize_mode: str = "per_phase",
) -> Path:
    if normalize_mode not in ("per_phase", "shared"):
        raise ValueError(f"normalize_mode must be 'per_phase' or 'shared', got {normalize_mode!r}")
    if frame_window is None:
        frame_window = base.SCATTER_MEAN_FRAME_WINDOW
    if out_root is not None:
        out_root = Path(out_root)
    elif normalize_mode == "shared":
        out_root = OUTPUT_ROOT / "drift"
    else:
        out_root = OUTPUT_ROOT / "within_phase"
    out_root.mkdir(parents=True, exist_ok=True)

    decoder_3 = _load_decoder_s(reference_run, ANALYSIS_3)
    decoder_hedonic = _load_decoder_s(reference_run, ANALYSIS_HEDONIC)

    # Define axes from reference-phase 3conditions centroids in decoder_3 space.
    # Axes are ALWAYS defined in the per-phase-normalized reference space so they
    # stay identical across normalize_mode; only how OTHER phases are normalized
    # before projection changes.
    ref_spec = base.AnalysisSpec(
        ANALYSIS_3,
        f"{reference_run} — 3conditions (axis definition)",
        base._STIMULUS_3,  # noqa: SLF001
        base.COLORS_3,
    )
    ref_trial_x, ref_center, ref_meta = _prepare_trial_tensor(reference_run, ref_spec)
    # Under shared normalization every phase is projected using the reference
    # phase's per-neuron mean/std + center, so cross-phase drift is preserved.
    shared_stats: SharedNorm | None = None
    if normalize_mode == "shared":
        shared_stats = _fit_shared_norm(reference_run, ref_spec)
    ref_features, ref_labels = _trial_features_s(
        ref_trial_x,
        ref_center,
        decoder_3,
        n_pcs=n_pcs,
        frame_window=frame_window,
    )
    ref_stim_names = tuple(n for n, _ in ref_spec.stimulus_levels)
    ref_centroids = _centroids_by_condition(ref_features, ref_labels, ref_stim_names)
    axes_def = _define_thirst_valence_axes(ref_centroids)
    thirst_axis = axes_def["thirst_axis"]
    valence_centroid_axis = axes_def["valence_centroid_axis"]
    origin = ref_centroids["water"]  # thirst/valence-A coords relative to pre water

    # Determine hedonic flip from reference phase first
    phase_data_by_phase: dict[str, dict[str, Any]] = {}
    ref_pdata = _collect_phase_data(
        reference_run,
        decoder_3,
        decoder_hedonic,
        origin=origin,
        thirst_axis=thirst_axis,
        valence_centroid_axis=valence_centroid_axis,
        n_pcs=n_pcs,
        frame_window=frame_window,
        flip_hedonic=None,
        shared_stats=shared_stats,
    )
    flip_hedonic = bool(ref_pdata["hedonic_flipped"])
    phase_data_by_phase[reference_run] = ref_pdata

    for phase in base.RUN_KEYS:
        if phase == reference_run:
            continue
        try:
            load_npz_pack(phase)
        except FileNotFoundError:
            print(f"[thirst_valence] skipping missing phase {phase}")
            continue
        phase_data_by_phase[phase] = _collect_phase_data(
            phase,
            decoder_3,
            decoder_hedonic,
            origin=origin,
            thirst_axis=thirst_axis,
            valence_centroid_axis=valence_centroid_axis,
            n_pcs=n_pcs,
            frame_window=frame_window,
            flip_hedonic=flip_hedonic,
            shared_stats=shared_stats,
        )

    # Tables
    trial_rows: list[dict[str, Any]] = []
    centroid_rows: list[dict[str, Any]] = []
    for pdata in phase_data_by_phase.values():
        trial_rows.extend(pdata["trial_rows"])
        centroid_rows.extend(_centroid_summary_rows(pdata))
    sep_rows = _separability_rows(phase_data_by_phase, reference_run)
    slm_rows = _slm_shift_rows(phase_data_by_phase)
    drift_rows = _drift_rows(phase_data_by_phase, reference_run)

    if normalize_mode == "shared":
        norm_note = (
            f"Every phase is z-scored and centered with the {reference_run} tensor's per-neuron "
            "mean/std + center (shared normalization), so cross-phase drift is preserved and "
            "across-phase centroid movement is interpretable as drift."
        )
    else:
        norm_note = (
            "Each phase is z-scored and centered with its own trial tensor before projection "
            "(per-phase normalization); this removes cross-phase drift and isolates within-phase "
            "condition separation."
        )
    axes_payload = {
        "reference_run": reference_run,
        "normalize_mode": normalize_mode,
        "n_pcs": n_pcs,
        "frame_window": list(frame_window),
        "thirst_axis": thirst_axis.tolist(),
        "valence_centroid_axis": valence_centroid_axis.tolist(),
        "angle_between_thirst_and_valence_centroid_deg": axes_def["angle_deg"],
        "origin_condition": "water",
        "origin_5d": origin.tolist(),
        "pre_centroids_5d": axes_def["pre_centroids"],
        "sign_conventions": axes_def["sign_conventions"],
        "hedonic_s_dpc1_flipped_to_aversive_positive": flip_hedonic,
        "reference_tensor": ref_meta,
        "notes": [
            "Thirst and valence-centroid axes are unit vectors in pre 3conditions s-space (5D window means).",
            norm_note,
            "valence_hedonic uses pre/hedonic_valence decoder_D_s column 0; sign flipped if needed so aversive > water.",
            "SLM trials (cond 26) are included when present in the phase pack.",
        ],
    }
    (out_root / "axes_definition.json").write_text(
        json.dumps(axes_payload, indent=2), encoding="utf-8"
    )
    _write_csv(trial_rows, out_root / "trial_coordinates.csv")
    _write_csv(centroid_rows, out_root / "centroid_summary.csv")
    _write_csv(sep_rows, out_root / "separability_summary.csv")
    _write_csv(slm_rows, out_root / "slm_shift_summary.csv")
    _write_csv(drift_rows, out_root / "drift_summary.csv")

    mode_tag = f" [{normalize_mode} norm]"

    # Plots
    _plot_faceted(
        phase_data_by_phase,
        out_root / "centroids_faceted_valence_centroid.png",
        x_key="thirst",
        y_key="valence_centroid",
        x_label="Thirst (nacl − water)",
        y_label="Valence (aversive − water, centroid)",
        title=f"Fixed thirst / valence-centroid axes from {reference_run}{mode_tag}",
    )
    _plot_faceted(
        phase_data_by_phase,
        out_root / "centroids_faceted_valence_hedonic.png",
        x_key="thirst",
        y_key="valence_hedonic",
        x_label="Thirst (nacl − water)",
        y_label="Valence (hedonic s-dPC1)",
        title=f"Fixed thirst / hedonic-valence axes from {reference_run}{mode_tag}",
    )
    _plot_overlay(
        phase_data_by_phase,
        out_root / "centroids_overlay_valence_centroid.png",
        x_key="thirst",
        y_key="valence_centroid",
        x_label="Thirst (nacl − water)",
        y_label="Valence (aversive − water, centroid)",
        title=f"Overlay: thirst × valence-centroid (axes from {reference_run}){mode_tag}",
    )
    _plot_overlay(
        phase_data_by_phase,
        out_root / "centroids_overlay_valence_hedonic.png",
        x_key="thirst",
        y_key="valence_hedonic",
        x_label="Thirst (nacl − water)",
        y_label="Valence (hedonic s-dPC1)",
        title=f"Overlay: thirst × hedonic valence (axes from {reference_run}){mode_tag}",
    )
    _plot_slm_shift_bars(slm_rows, out_root / "slm_shift_by_phase.png")
    _plot_separation_across_phases(sep_rows, out_root / "separation_across_phases.png")
    if normalize_mode == "shared":
        _plot_drift(
            drift_rows,
            out_root / "drift_grand_mean_by_phase.png",
            value_key="grand_mean",
            title=f"Overall drift: grand-mean position on fixed axes{mode_tag}",
        )
        _plot_drift(
            drift_rows,
            out_root / "drift_water_centroid_by_phase.png",
            value_key="water_centroid_drift_vs_reference",
            title=f"Water-centroid drift vs {reference_run} on fixed axes{mode_tag}",
        )

    print(f"[thirst_valence] ({normalize_mode}) wrote outputs to {out_root}")
    return out_root


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("per_phase", "shared", "both"),
        default="both",
        help=(
            "Normalization for projecting non-reference phases. 'per_phase' isolates "
            "within-phase separation; 'shared' preserves cross-phase drift; 'both' runs each."
        ),
    )
    args = parser.parse_args()
    modes = ("per_phase", "shared") if args.mode == "both" else (args.mode,)
    for mode in modes:
        run_thirst_valence_axes(normalize_mode=mode)


if __name__ == "__main__":
    main()
