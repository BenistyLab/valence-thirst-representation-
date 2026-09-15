"""
Pairwise condition distances in each run's native dPCA stimulus (s) space.

For the 3conditions analysis (water / nacl / airpuff), projects trials onto
that phase's own decoder_D_s, computes centroid separations, and compares how
separability changes across configured phases.

Run after run_dpca.py.

Writes Euclidean outputs to condition_distances/ and Mahalanobis (pooled covariance)
outputs to condition_distances/mahalanobis/.
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from lib.dpca import analysis as base
from lib.io.npz import load_npz_pack

ANALYSIS = "3conditions"
DEFAULT_N_PCS = 5
REFERENCE_RUN = base.RUN_KEYS[0]
PC_VARIANTS: tuple[str, ...] = ("all", "pc1", "pc2", "pc3", "pc4", "pc5")
PAIR_NAMES: tuple[str, ...] = ("water_nacl", "water_airpuff", "nacl_airpuff")
OUTPUT_ROOT = base.DPCA_OUT_ROOT / "condition_distances"
MAHALANOBIS_OUTPUT_ROOT = OUTPUT_ROOT / "mahalanobis"
COV_REGULARIZATION = 1e-6


def _load_decoder_matrices(run_key: str) -> dict[str, np.ndarray]:
    path = base.DPCA_OUT_ROOT / run_key / ANALYSIS / "loadings" / "encoder_decoder_matrices.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_dpca.py first so decoder matrices are saved."
        )
    with np.load(path) as data:
        return {str(k): np.asarray(data[k], dtype=np.float64) for k in data.files}


def _three_conditions_spec(run_key: str) -> base.AnalysisSpec | None:
    try:
        pack = load_npz_pack(run_key)
    except FileNotFoundError:
        return None
    for spec in base.analyses_for_run(run_key, pack):
        if spec.subdir == ANALYSIS:
            return spec
    return None


def _prepare_trial_tensor(run_key: str, spec: base.AnalysisSpec) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pack = load_npz_pack(run_key)
    segs_by_s, min_t, n_neurons = base._collect_segments(pack, spec.stimulus_levels)  # noqa: SLF001
    trial_x, x_mean, counts, candidate_counts, ignored_short_counts = base._build_mean_and_trialx(  # noqa: SLF001
        segs_by_s,
        spec.stimulus_levels,
        base.ANALYSIS_T_FRAMES,
        n_neurons,
    )
    x_fit, trial_x_fit = base._zscore_mean_and_trialx(x_mean, trial_x)  # noqa: SLF001
    center = np.mean(x_fit.reshape((x_fit.shape[0], -1)), axis=1)
    meta = {
        "run_key": run_key,
        "analysis": spec.subdir,
        "stimulus_levels": [name for name, _ in spec.stimulus_levels],
        "trial_counts": counts,
        "candidate_trial_counts": candidate_counts,
        "ignored_short_trial_counts": ignored_short_counts,
        "ignored_short_trial_total": int(sum(ignored_short_counts.values())),
        "frame_window": base.ANALYSIS_T_FRAMES,
        "projection_frame_window": list(base.SCATTER_MEAN_FRAME_WINDOW),
        "min_candidate_trial_frames": min_t,
        "n_neurons": n_neurons,
        "tensor_shape_mean": list(x_mean.shape),
    }
    return trial_x_fit, center, meta


def _trial_features_s(
    trial_x: np.ndarray,
    center: np.ndarray,
    decoder_d_s: np.ndarray,
    *,
    n_pcs: int,
    frame_window: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    start, stop = frame_window
    stop = min(stop, int(trial_x.shape[3]))
    if stop <= start:
        raise ValueError(f"Frame window {frame_window} is outside trial length {trial_x.shape[3]}")

    n_use = min(n_pcs, int(decoder_d_s.shape[1]))
    decoder = np.asarray(decoder_d_s[:, :n_use], dtype=np.float64)
    features: list[np.ndarray] = []
    labels: list[int] = []

    for stim_ix in range(trial_x.shape[2]):
        for trial_ix in range(trial_x.shape[0]):
            xi = np.asarray(trial_x[trial_ix, :, stim_ix, :], dtype=np.float64)
            if np.any(np.isnan(xi)):
                continue
            projected = decoder.T @ (xi - center[:, None])
            features.append(np.mean(projected[:, start:stop], axis=1))
            labels.append(int(stim_ix))

    if not features:
        raise RuntimeError("No finite trial features were assembled.")
    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def _centroids_by_condition(
    features: np.ndarray,
    labels: np.ndarray,
    stim_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for stim_ix, name in enumerate(stim_names):
        mask = labels == stim_ix
        if not np.any(mask):
            raise RuntimeError(f"No trials for stimulus level {name!r}.")
        centroids[name] = np.mean(features[mask], axis=0)
    return centroids


def _centroid_sem_by_condition(
    features: np.ndarray,
    labels: np.ndarray,
    stim_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """SEM of the mean feature vector per condition (all trials, no resampling)."""
    sems: dict[str, np.ndarray] = {}
    for stim_ix, name in enumerate(stim_names):
        pts = features[labels == stim_ix]
        n = int(pts.shape[0])
        if n <= 1:
            sems[name] = np.zeros(pts.shape[1], dtype=np.float64)
        else:
            sems[name] = np.std(pts, axis=0, ddof=1) / np.sqrt(n)
    return sems


def _pc_indices(pc_variant: str) -> slice | int:
    if pc_variant == "all":
        return slice(None)
    if pc_variant.startswith("pc") and pc_variant[2:].isdigit():
        return int(pc_variant[2:]) - 1
    raise ValueError(f"Unknown pc_variant: {pc_variant!r}")


def _distance_between(centroid_a: np.ndarray, centroid_b: np.ndarray, pc_variant: str) -> float:
    ix = _pc_indices(pc_variant)
    if pc_variant == "all":
        return float(np.linalg.norm(centroid_a[ix] - centroid_b[ix]))
    return float(abs(centroid_a[ix] - centroid_b[ix]))


def _pairwise_distances(centroids: dict[str, np.ndarray], stim_names: tuple[str, ...], pc_variant: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for (a, b) in combinations(stim_names, 2):
        out[f"{a}_{b}"] = _distance_between(centroids[a], centroids[b], pc_variant)
    return out


def _feature_slice(features: np.ndarray, pc_variant: str) -> np.ndarray:
    ix = _pc_indices(pc_variant)
    if pc_variant == "all":
        return np.asarray(features, dtype=np.float64)
    return np.asarray(features[:, ix], dtype=np.float64).reshape((features.shape[0], 1))


def _pooled_covariance(features: np.ndarray, pc_variant: str, *, reg: float = COV_REGULARIZATION) -> np.ndarray:
    x = _feature_slice(features, pc_variant)
    if x.shape[0] < 2:
        raise RuntimeError("Need at least 2 trials for pooled covariance.")
    cov = np.cov(x, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    cov.flat[:: cov.shape[0] + 1] += reg
    return cov


def _mahalanobis_between(
    centroid_a: np.ndarray,
    centroid_b: np.ndarray,
    cov_inv: np.ndarray,
    pc_variant: str,
) -> float:
    ix = _pc_indices(pc_variant)
    diff = np.asarray(centroid_a[ix] - centroid_b[ix], dtype=np.float64).reshape(-1)
    val = float(diff @ cov_inv @ diff)
    return float(np.sqrt(max(val, 0.0)))


def _pairwise_mahalanobis_distances(
    centroids: dict[str, np.ndarray],
    stim_names: tuple[str, ...],
    cov_inv: np.ndarray,
    pc_variant: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for (a, b) in combinations(stim_names, 2):
        out[f"{a}_{b}"] = _mahalanobis_between(centroids[a], centroids[b], cov_inv, pc_variant)
    return out


def _pairwise_mahalanobis_distance_sem(
    centroids: dict[str, np.ndarray],
    centroid_sems: dict[str, np.ndarray],
    stim_names: tuple[str, ...],
    cov_inv: np.ndarray,
    pc_variant: str,
) -> dict[str, float]:
    """Approximate SEM via delta method with diagonal centroid uncertainty."""
    out: dict[str, float] = {}
    for (a, b) in combinations(stim_names, 2):
        ix = _pc_indices(pc_variant)
        diff = np.asarray(centroids[a][ix] - centroids[b][ix], dtype=np.float64).reshape(-1)
        dist = float(np.sqrt(max(float(diff @ cov_inv @ diff), 0.0)))
        if dist <= 0.0 or not np.isfinite(dist):
            out[f"{a}_{b}"] = float("nan")
            continue
        grad = cov_inv @ diff / dist
        sa = np.asarray(centroid_sems[a][ix], dtype=np.float64).reshape(-1)
        sb = np.asarray(centroid_sems[b][ix], dtype=np.float64).reshape(-1)
        var_d = float(np.sum((grad * sa) ** 2) + np.sum((grad * sb) ** 2))
        out[f"{a}_{b}"] = float(np.sqrt(max(var_d, 0.0)))
    return out


def _pairwise_distance_sem(
    centroids: dict[str, np.ndarray],
    centroid_sems: dict[str, np.ndarray],
    stim_names: tuple[str, ...],
    pc_variant: str,
) -> dict[str, float]:
    """SEM of pairwise centroid distance from trial scatter (delta method for all PCs)."""
    out: dict[str, float] = {}
    for (a, b) in combinations(stim_names, 2):
        ca = centroids[a]
        cb = centroids[b]
        sa = centroid_sems[a]
        sb = centroid_sems[b]
        if pc_variant == "all":
            diff = ca - cb
            dist = float(np.linalg.norm(diff))
            if dist <= 0.0 or not np.isfinite(dist):
                out[f"{a}_{b}"] = float("nan")
                continue
            grad_a = diff / dist
            grad_b = -grad_a
            var_d = float(np.sum((grad_a * sa) ** 2) + np.sum((grad_b * sb) ** 2))
            out[f"{a}_{b}"] = float(np.sqrt(max(var_d, 0.0)))
        else:
            k = _pc_indices(pc_variant)
            assert isinstance(k, int)
            out[f"{a}_{b}"] = float(np.sqrt(sa[k] ** 2 + sb[k] ** 2))
    return out


def _normalize_distance_sem(
    distances: dict[str, float],
    distance_sems: dict[str, float],
) -> dict[str, float]:
    """Approximate SEM of normalized distances via fixed normalization denominator."""
    mean_d = float(np.mean(list(distances.values())))
    if mean_d <= 0.0 or not np.isfinite(mean_d):
        return {k: float("nan") for k in distances}
    return {pair: float(distance_sems[pair] / mean_d) for pair in distances}


def _approx_normal_ci(point: float, sem: float, *, z: float = 1.96) -> tuple[float, float]:
    if not np.isfinite(point) or not np.isfinite(sem):
        return float("nan"), float("nan")
    return float(point - z * sem), float(point + z * sem)


def _fold_change_sem(distance: float, distance_sem: float, ref_distance: float, ref_sem: float) -> float:
    if ref_distance <= 0.0 or not np.isfinite(ref_distance) or not np.isfinite(distance):
        return float("nan")
    rel_run = distance_sem / distance if distance > 0 else float("nan")
    rel_ref = ref_sem / ref_distance
    if not np.isfinite(rel_run) or not np.isfinite(rel_ref):
        return float("nan")
    fold = distance / ref_distance
    return float(fold * np.sqrt(rel_run ** 2 + rel_ref ** 2))


def _normalize_distances(distances: dict[str, float]) -> dict[str, float]:
    mean_d = float(np.mean(list(distances.values())))
    if mean_d <= 0.0 or not np.isfinite(mean_d):
        return {k: float("nan") for k in distances}
    return {k: float(v / mean_d) for k, v in distances.items()}


def _within_condition_spread(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: dict[str, np.ndarray],
    stim_names: tuple[str, ...],
    pc_variant: str,
) -> dict[str, float]:
    ix = _pc_indices(pc_variant)
    spreads: dict[str, float] = {}
    for stim_ix, name in enumerate(stim_names):
        mask = labels == stim_ix
        pts = features[mask]
        if pc_variant == "all":
            dists = np.linalg.norm(pts - centroids[name][None, :], axis=1)
        else:
            dists = np.abs(pts[:, ix] - centroids[name][ix])
        spreads[name] = float(np.mean(dists))
    return spreads


def _direction_label(ci_low: float, ci_high: float) -> str:
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        if ci_low > 1.0:
            return "further"
        if ci_high < 1.0:
            return "closer"
    return "unchanged"


def _analyze_run(
    run_key: str,
    *,
    n_pcs: int,
    frame_window: tuple[int, int],
    metric: str = "euclidean",
) -> dict[str, Any] | None:
    spec = _three_conditions_spec(run_key)
    if spec is None:
        return None
    matrix_path = base.DPCA_OUT_ROOT / run_key / ANALYSIS / "loadings" / "encoder_decoder_matrices.npz"
    if not matrix_path.exists():
        return None
    stim_names = tuple(name for name, _ in spec.stimulus_levels)
    decoder_matrices = _load_decoder_matrices(run_key)
    if "decoder_D_s" not in decoder_matrices:
        raise RuntimeError(f"Missing decoder_D_s for {run_key}/{ANALYSIS}")
    trial_x, center, meta = _prepare_trial_tensor(run_key, spec)
    features, labels = _trial_features_s(
        trial_x,
        center,
        decoder_matrices["decoder_D_s"],
        n_pcs=n_pcs,
        frame_window=frame_window,
    )
    centroids = _centroids_by_condition(features, labels, stim_names)
    centroid_sems = _centroid_sem_by_condition(features, labels, stim_names)

    variant_rows: dict[str, dict[str, Any]] = {}
    cov_by_variant: dict[str, np.ndarray] = {}
    if metric == "mahalanobis":
        for variant in PC_VARIANTS:
            cov_by_variant[variant] = _pooled_covariance(features, variant)
    for variant in PC_VARIANTS:
        if metric == "mahalanobis":
            cov_inv = np.linalg.pinv(cov_by_variant[variant])
            dists = _pairwise_mahalanobis_distances(centroids, stim_names, cov_inv, variant)
            dist_sems = _pairwise_mahalanobis_distance_sem(
                centroids, centroid_sems, stim_names, cov_inv, variant
            )
        else:
            dists = _pairwise_distances(centroids, stim_names, variant)
            dist_sems = _pairwise_distance_sem(centroids, centroid_sems, stim_names, variant)
        norm_dists = _normalize_distances(dists)
        norm_sems = _normalize_distance_sem(dists, dist_sems)
        spreads = _within_condition_spread(features, labels, centroids, stim_names, variant)
        pair_stats: dict[str, Any] = {}
        for pair in PAIR_NAMES:
            ci_low, ci_high = _approx_normal_ci(dists[pair], dist_sems[pair])
            norm_ci_low, norm_ci_high = _approx_normal_ci(norm_dists[pair], norm_sems[pair])
            pair_stats[pair] = {
                "distance": dists[pair],
                "normalized_distance": norm_dists[pair],
                "distance_sem": dist_sems[pair],
                "distance_ci_low": ci_low,
                "distance_ci_high": ci_high,
                "normalized_distance_sem": norm_sems[pair],
                "normalized_distance_ci_low": norm_ci_low,
                "normalized_distance_ci_high": norm_ci_high,
            }
        variant_rows[variant] = {
            "pairwise": pair_stats,
            "mean_pairwise_distance": float(np.mean(list(dists.values()))),
            "mean_normalized_pairwise_distance": float(np.mean(list(norm_dists.values()))),
            "within_condition_spread": spreads,
            "mean_within_condition_spread": float(np.mean(list(spreads.values()))),
            "centroids": {name: centroids[name].tolist() for name in stim_names},
        }
        if metric == "mahalanobis":
            cov = cov_by_variant[variant]
            variant_rows[variant]["pooled_covariance"] = cov.tolist()
            variant_rows[variant]["pooled_covariance_shape"] = list(cov.shape)

    return {
        "run_key": run_key,
        "meta": meta,
        "stim_names": stim_names,
        "colors": list(spec.colors),
        "features": features,
        "labels": labels,
        "metric": metric,
        "variants": variant_rows,
    }


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_distance_rows(
    run_results: dict[str, dict[str, Any]],
    *,
    reference_run: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ref_variants = run_results.get(reference_run, {}).get("variants", {})
    for run_key, result in run_results.items():
        trial_counts = result["meta"]["trial_counts"]
        for variant, variant_data in result["variants"].items():
            ref_pair_stats = ref_variants.get(variant, {}).get("pairwise", {})
            for pair in PAIR_NAMES:
                stats = variant_data["pairwise"][pair]
                row: dict[str, object] = {
                    "run": run_key,
                    "pc_variant": variant,
                    "pair": pair,
                    "distance": stats["distance"],
                    "normalized_distance": stats["normalized_distance"],
                    "distance_sem": stats["distance_sem"],
                    "distance_ci_low": stats["distance_ci_low"],
                    "distance_ci_high": stats["distance_ci_high"],
                    "normalized_distance_sem": stats["normalized_distance_sem"],
                    "normalized_distance_ci_low": stats["normalized_distance_ci_low"],
                    "normalized_distance_ci_high": stats["normalized_distance_ci_high"],
                }
                a_name, b_name = pair.split("_", 1)
                row["trial_count_a"] = trial_counts.get(a_name)
                row["trial_count_b"] = trial_counts.get(b_name)
                if run_key != reference_run and pair in ref_pair_stats:
                    ref_stats = ref_pair_stats[pair]
                    ref_dist = float(ref_stats["distance"])
                    fold = float(stats["distance"] / ref_dist) if ref_dist > 0 else float("nan")
                    row["fold_change_vs_reference"] = fold
                    fc_sem = _fold_change_sem(
                        float(stats["distance"]),
                        float(stats["distance_sem"]),
                        ref_dist,
                        float(ref_stats["distance_sem"]),
                    )
                    fc_low, fc_high = _approx_normal_ci(fold, fc_sem)
                    row["fold_change_ci_low"] = fc_low
                    row["fold_change_ci_high"] = fc_high
                    row["direction_vs_reference"] = _direction_label(fc_low, fc_high)
                else:
                    row["fold_change_vs_reference"] = 1.0 if run_key == reference_run else float("nan")
                    row["fold_change_ci_low"] = 1.0 if run_key == reference_run else float("nan")
                    row["fold_change_ci_high"] = 1.0 if run_key == reference_run else float("nan")
                    row["direction_vs_reference"] = "reference" if run_key == reference_run else "unchanged"
                rows.append(row)
    return rows


def _build_summary_rows(run_results: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_key, result in run_results.items():
        for variant, variant_data in result["variants"].items():
            rows.append(
                {
                    "run": run_key,
                    "pc_variant": variant,
                    "mean_pairwise_distance": variant_data["mean_pairwise_distance"],
                    "mean_normalized_pairwise_distance": variant_data["mean_normalized_pairwise_distance"],
                    "mean_within_condition_spread": variant_data["mean_within_condition_spread"],
                    "within_condition_spread_water": variant_data["within_condition_spread"]["water"],
                    "within_condition_spread_nacl": variant_data["within_condition_spread"]["nacl"],
                    "within_condition_spread_airpuff": variant_data["within_condition_spread"]["airpuff"],
                }
            )
    return rows


def _runs_in_rows(rows: list[dict[str, object]]) -> list[str]:
    present = {str(row["run"]) for row in rows}
    return [rk for rk in base.RUN_KEYS if rk in present]


def _plot_pair_distances_by_run(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    pc_variant: str,
    metric_label: str = "Pairwise centroid distance",
) -> None:
    variant_rows = [row for row in rows if row["pc_variant"] == pc_variant]
    runs = _runs_in_rows(variant_rows)
    pairs = list(PAIR_NAMES)
    x = np.arange(len(pairs), dtype=np.float64)
    width = 0.24
    cmap = plt.get_cmap("tab10")
    colors = {rk: cmap(i % 10) for i, rk in enumerate(runs)}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for idx, run_key in enumerate(runs):
        vals: list[float] = []
        errs: list[float] = []
        for pair in pairs:
            match = next((row for row in variant_rows if row["run"] == run_key and row["pair"] == pair), None)
            vals.append(float(match["distance"]) if match else float("nan"))
            errs.append(float(match["distance_sem"]) if match else float("nan"))
        ax.bar(
            x + (idx - 1) * width,
            vals,
            width=width,
            yerr=errs,
            capsize=3,
            color=colors.get(run_key, "0.4"),
            label=run_key,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "–") for p in pairs])
    ax.set_ylabel(metric_label)
    ax.set_title(f"Condition pair distances in native s-space ({pc_variant})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_normalized_pair_trends(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    pc_variant: str,
    metric_label: str = "Normalized pairwise distance",
) -> None:
    variant_rows = [row for row in rows if row["pc_variant"] == pc_variant]
    pair_colors = {"water_nacl": "#0b3d91", "water_airpuff": "#b22222", "nacl_airpuff": "#2e8b57"}
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    runs = _runs_in_rows(variant_rows)
    x = np.arange(len(runs), dtype=np.float64)
    for pair in PAIR_NAMES:
        ys: list[float] = []
        sems: list[float] = []
        for run in runs:
            match = next((row for row in variant_rows if row["run"] == run and row["pair"] == pair), None)
            if match is None:
                ys.append(float("nan"))
                sems.append(float("nan"))
            else:
                ys.append(float(match["normalized_distance"]))
                sems.append(float(match["normalized_distance_sem"]))
        if not any(np.isfinite(ys)):
            continue
        y_arr = np.asarray(ys, dtype=np.float64)
        sem_arr = np.asarray(sems, dtype=np.float64)
        col = pair_colors[pair]
        ax.fill_between(
            x,
            y_arr - sem_arr,
            y_arr + sem_arr,
            color=col,
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(x, y_arr, marker="o", linewidth=1.8, label=pair.replace("_", "–"), color=col)
    ax.axhline(1.0, color="0.45", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_ylabel(metric_label)
    ax.set_title(
        f"Normalized pair trends across runs ({pc_variant})\n(shaded: mean ± SEM of centroid, all trials)"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_separability_across_runs(
    summary_rows: list[dict[str, object]],
    out_path: Path,
    *,
    title: str = "Mean pairwise separability across runs (native s-space)",
    y_label: str = "Mean pairwise distance",
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), squeeze=False)
    runs = _runs_in_rows(summary_rows)
    x = np.arange(len(runs), dtype=np.float64)
    for ax, variant in zip(axes.ravel(), PC_VARIANTS):
        variant_rows = [row for row in summary_rows if row["pc_variant"] == variant]
        ys = [
            float(next(row["mean_pairwise_distance"] for row in variant_rows if row["run"] == run))
            for run in runs
        ]
        ax.plot(x, ys, marker="o", color="#0b3d91", linewidth=1.8)
        ax.set_xticks(x)
        ax.set_xticklabels(runs, fontsize=8)
        ax.set_title(variant, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(y_label, fontsize=8)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_pair_distance_heatmap(distance_rows: list[dict[str, object]], out_path: Path) -> None:
    runs = _runs_in_rows(distance_rows)
    pairs = list(PAIR_NAMES)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), squeeze=False)
    for ax, variant in zip(axes.ravel(), PC_VARIANTS):
        mat = np.full((len(runs), len(pairs)), np.nan, dtype=np.float64)
        for ri, run_key in enumerate(runs):
            for pi, pair in enumerate(pairs):
                match = next(
                    (row for row in distance_rows if row["run"] == run_key and row["pc_variant"] == variant and row["pair"] == pair),
                    None,
                )
                if match is not None:
                    mat[ri, pi] = float(match["normalized_distance"])
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pairs)))
        ax.set_xticklabels([p.replace("_", "\n") for p in pairs], fontsize=7)
        ax.set_yticks(np.arange(len(runs)))
        ax.set_yticklabels(runs, fontsize=8)
        ax.set_title(variant, fontsize=9)
        for ri in range(len(runs)):
            for pi in range(len(pairs)):
                if np.isfinite(mat[ri, pi]):
                    ax.text(pi, ri, f"{mat[ri, pi]:.2f}", ha="center", va="center", color="white", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Normalized pairwise distances (runs × pairs)", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_s_space_centroids(
    result: dict[str, Any],
    out_path: Path,
    *,
    distance_key: str = "distance",
    annotation_note: str = "all-PC distances annotated",
) -> None:
    features = result["features"]
    labels = result["labels"]
    stim_names: tuple[str, ...] = result["stim_names"]
    colors: list[str] = result["colors"]
    centroids = result["variants"]["all"]["centroids"]
    dists = result["variants"]["all"]["pairwise"]

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for stim_ix, name in enumerate(stim_names):
        mask = labels == stim_ix
        col = colors[stim_ix % len(colors)]
        ax.scatter(
            features[mask, 0],
            features[mask, 1],
            s=28,
            alpha=0.65,
            color=col,
            edgecolors="none",
            label=name,
        )
        cx, cy = centroids[name][0], centroids[name][1]
        ax.scatter([cx], [cy], s=120, color=col, edgecolors="black", linewidths=1.2, zorder=5)
        ax.annotate(name, (cx, cy), textcoords="offset points", xytext=(5, 5), fontsize=8)

    centroid_xy = {name: (centroids[name][0], centroids[name][1]) for name in stim_names}
    for pair in PAIR_NAMES:
        a_name, b_name = pair.split("_", 1)
        x0, y0 = centroid_xy[a_name]
        x1, y1 = centroid_xy[b_name]
        ax.plot([x0, x1], [y0, y1], color="0.35", linewidth=1.0, zorder=2)
        dist = dists[pair][distance_key]
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{dist:.2f}", fontsize=7, ha="center", va="bottom")

    ax.set_xlabel("s-dPC1 (window mean)")
    ax.set_ylabel("s-dPC2 (window mean)")
    ax.set_title(f"{result['run_key']}: trial centroids in native s-space ({annotation_note})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_condition_distance_outputs(
    run_results: dict[str, dict[str, Any]],
    *,
    out_root: Path,
    reference_run: str,
    metric: str,
    n_pcs: int,
    frame_window: tuple[int, int],
    run_keys: tuple[str, ...],
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    is_mahal = metric == "mahalanobis"
    distance_rows = _build_distance_rows(run_results, reference_run=reference_run)
    summary_rows = _build_summary_rows(run_results)

    _write_csv(distance_rows, out_root / "condition_distances.csv")
    _write_csv(summary_rows, out_root / "separability_summary.csv")
    notes = [
        "Centroids are the mean of all trial feature vectors per condition (no trial resampling).",
        "distance_sem is the SEM of each pairwise distance, propagated from per-trial scatter.",
        "normalized_distance divides each pairwise distance by the mean of all three pairs within the same run and pc_variant.",
        "normalized_distance_sem approximates SEM by dividing raw distance_sem by the same normalization denominator.",
        "fold_change_vs_reference uses Gaussian-approximate CIs from propagated SEMs.",
    ]
    if is_mahal:
        notes.extend(
            [
                "Mahalanobis distances use pooled trial covariance within each run's native s-space.",
                f"Covariance diagonal ridge regularization: {COV_REGULARIZATION}.",
                "Single-PC Mahalanobis uses pooled variance along that axis.",
                "Raw Mahalanobis values are still defined per-run; use normalized_distance for relative cross-run trends.",
            ]
        )
    else:
        notes.append("Single-PC distances are absolute centroid separation along that axis; all uses 5D Euclidean distance.")

    metadata = {
        "analysis": ANALYSIS,
        "metric": metric,
        "run_keys": list(run_keys),
        "reference_run": reference_run,
        "n_pcs": n_pcs,
        "frame_window": list(frame_window),
        "pc_variants": list(PC_VARIANTS),
        "pairs": list(PAIR_NAMES),
        "notes": notes,
        "runs": {run_key: run_results[run_key]["meta"] for run_key in run_results},
    }
    if is_mahal:
        metadata["covariance_regularization"] = COV_REGULARIZATION
        metadata["pooled_covariance_by_run"] = {
            run_key: {
                variant: run_results[run_key]["variants"][variant].get("pooled_covariance")
                for variant in PC_VARIANTS
            }
            for run_key in run_results
        }

    (out_root / "condition_distances.json").write_text(
        json.dumps({"metadata": metadata, "distance_rows": distance_rows, "summary_rows": summary_rows}, indent=2),
        encoding="utf-8",
    )

    bar_ylabel = "Mahalanobis distance" if is_mahal else "Pairwise centroid distance"
    trend_ylabel = "Normalized Mahalanobis distance" if is_mahal else "Normalized pairwise distance"
    sep_title = (
        "Mean pairwise Mahalanobis separability across runs (native s-space)"
        if is_mahal
        else "Mean pairwise separability across runs (native s-space)"
    )
    sep_ylabel = "Mean pairwise Mahalanobis distance" if is_mahal else "Mean pairwise distance"

    for variant in PC_VARIANTS:
        _plot_pair_distances_by_run(
            distance_rows,
            out_root / f"pair_distances_by_run_{variant}.png",
            pc_variant=variant,
            metric_label=bar_ylabel,
        )
        _plot_normalized_pair_trends(
            distance_rows,
            out_root / f"normalized_pair_trends_{variant}.png",
            pc_variant=variant,
            metric_label=trend_ylabel,
        )
    _plot_separability_across_runs(summary_rows, out_root / "separability_across_runs.png", title=sep_title, y_label=sep_ylabel)
    _plot_pair_distance_heatmap(distance_rows, out_root / "pair_distance_heatmap.png")

    for run_key, result in run_results.items():
        _plot_s_space_centroids(
            result,
            out_root / f"s_space_centroids_{run_key}.png",
            distance_key="distance",
            annotation_note="5D Mahalanobis distances annotated" if is_mahal else "all-PC distances annotated",
        )

    print(f"Wrote {metric} condition distance analysis to {out_root}")


def run_condition_distances(
    *,
    run_keys: tuple[str, ...] = base.RUN_KEYS,
    n_pcs: int = DEFAULT_N_PCS,
    frame_window: tuple[int, int] = base.SCATTER_MEAN_FRAME_WINDOW,
    reference_run: str = REFERENCE_RUN,
    out_root: Path = OUTPUT_ROOT,
) -> None:
    run_results: dict[str, dict[str, Any]] = {}

    for run_key in run_keys:
        print(f"[condition_distances/euclidean] analyzing {run_key}/{ANALYSIS} ...")
        result = _analyze_run(
            run_key,
            n_pcs=n_pcs,
            frame_window=frame_window,
            metric="euclidean",
        )
        if result is None:
            print(f"  skipped (no {ANALYSIS} outputs for {run_key})")
            continue
        run_results[run_key] = result

    _write_condition_distance_outputs(
        run_results,
        out_root=out_root,
        reference_run=reference_run,
        metric="euclidean",
        n_pcs=n_pcs,
        frame_window=frame_window,
        run_keys=run_keys,
    )


def run_mahalanobis_condition_distances(
    *,
    run_keys: tuple[str, ...] = base.RUN_KEYS,
    n_pcs: int = DEFAULT_N_PCS,
    frame_window: tuple[int, int] = base.SCATTER_MEAN_FRAME_WINDOW,
    reference_run: str = REFERENCE_RUN,
    out_root: Path = MAHALANOBIS_OUTPUT_ROOT,
) -> None:
    run_results: dict[str, dict[str, Any]] = {}

    for run_key in run_keys:
        print(f"[condition_distances/mahalanobis] analyzing {run_key}/{ANALYSIS} ...")
        result = _analyze_run(
            run_key,
            n_pcs=n_pcs,
            frame_window=frame_window,
            metric="mahalanobis",
        )
        if result is None:
            print(f"  skipped (no {ANALYSIS} outputs for {run_key})")
            continue
        run_results[run_key] = result

    _write_condition_distance_outputs(
        run_results,
        out_root=out_root,
        reference_run=reference_run,
        metric="mahalanobis",
        n_pcs=n_pcs,
        frame_window=frame_window,
        run_keys=run_keys,
    )


def run_all_condition_distances(
    *,
    run_keys: tuple[str, ...] = base.RUN_KEYS,
    n_pcs: int = DEFAULT_N_PCS,
    frame_window: tuple[int, int] = base.SCATTER_MEAN_FRAME_WINDOW,
    reference_run: str = REFERENCE_RUN,
) -> None:
    run_condition_distances(
        run_keys=run_keys,
        n_pcs=n_pcs,
        frame_window=frame_window,
        reference_run=reference_run,
    )
    run_mahalanobis_condition_distances(
        run_keys=run_keys,
        n_pcs=n_pcs,
        frame_window=frame_window,
        reference_run=reference_run,
    )


if __name__ == "__main__":
    run_all_condition_distances()
