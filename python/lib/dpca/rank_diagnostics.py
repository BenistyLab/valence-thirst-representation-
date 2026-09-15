"""
Rank-sweep diagnostics for dPCA analyses.

For each requested run and label set, this script fits one fresh high-rank dPCA
model and writes cumulative component-rank diagnostics under the existing dPCA
output directories.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dPCA.dPCA import dPCA

from lib.dpca import analysis as base
from lib.io.npz import load_npz_pack

DEFAULT_ANALYSES = ("3conditions", "hedonic_valence", "consumption_mode")
DEFAULT_MAX_RANK = 15
MARGINALIZATIONS = ("t", "s", "st")


def _center_by_observable(X: np.ndarray) -> np.ndarray:
    """Match dPCA centering: subtract each neuron's mean over all task/time bins."""
    Xf = np.asarray(X, dtype=np.float64)
    return Xf - np.mean(Xf.reshape((Xf.shape[0], -1)), axis=1).reshape((Xf.shape[0], 1, 1))


def _relative_error(target: np.ndarray, reconstructed: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    denom = float(np.sum(target * target))
    if denom <= 0.0:
        return float("nan")
    residual = target - reconstructed
    return float(np.sqrt(np.sum(residual * residual) / denom))


def _paper_explained_variance_from_error(relative_error: float) -> float:
    """Paper-style cumulative explained variance: 1 - ||residual||^2 / ||target||^2."""
    err = float(relative_error)
    if not np.isfinite(err):
        return float("nan")
    return float(1.0 - err * err)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _read_existing_regularizer(out_dir: Path) -> float | None:
    meta_path = out_dir / "trial_counts.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("optimal_regularizer")
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    return None


def _regularizer_for_diagnostics(out_dir: Path, mode: str) -> float | str | None:
    if mode == "none":
        return None
    if mode == "auto":
        return "auto"
    if mode == "meta_or_auto":
        return _read_existing_regularizer(out_dir) or "auto"
    raise ValueError(f"Unknown regularizer mode: {mode}")


def _reconstruction_metrics(model: dPCA, X_fit: np.ndarray, rank: int) -> dict[str, float]:
    X0 = _center_by_observable(X_fit).reshape((X_fit.shape[0], -1))
    target_marginals = model._marginalize(X_fit)  # noqa: SLF001 - matches dPCA's own objective.

    out: dict[str, float] = {}
    total_reconstruction = np.zeros_like(X0)
    for marg in MARGINALIZATIONS:
        if marg not in model.P or marg not in model.D or marg not in target_marginals:
            out[f"reconstruction_error_{marg}"] = float("nan")
            continue
        P = np.asarray(model.P[marg], dtype=np.float64)
        D = np.asarray(model.D[marg], dtype=np.float64)
        n_use = min(rank, P.shape[1], D.shape[1])
        reconstruction = P[:, :n_use] @ (D[:, :n_use].T @ X0)
        total_reconstruction += reconstruction
        marg_error = _relative_error(target_marginals[marg], reconstruction)
        out[f"reconstruction_error_{marg}"] = marg_error
        out[f"paper_explained_variance_{marg}"] = _paper_explained_variance_from_error(marg_error)

    total_error = _relative_error(X0, total_reconstruction)
    out["reconstruction_error_total"] = total_error
    out["paper_explained_variance_total"] = _paper_explained_variance_from_error(total_error)
    return out


def _explained_variance_metrics(model: dPCA, rank: int) -> dict[str, float]:
    ev = getattr(model, "explained_variance_ratio_", {})
    out: dict[str, float] = {}
    total = 0.0
    for marg in MARGINALIZATIONS:
        vals = np.asarray(ev.get(marg, []), dtype=np.float64).ravel()
        value = float(np.nansum(vals[:rank])) if vals.size else float("nan")
        out[f"explained_variance_{marg}"] = value
        if np.isfinite(value):
            total += value
    out["explained_variance_total"] = total
    return out


def _fit_max_rank_model(
    X_fit: np.ndarray,
    trialX_fit: np.ndarray,
    max_rank: int,
    regularizer: float | str | None,
) -> tuple[dPCA, float | str | None]:
    model = dPCA(labels="st", n_components=max_rank, regularizer=regularizer)
    model.debug = 0
    model.protect = ["t"]
    model.fit_transform(X_fit, trialX=trialX_fit)
    return model, getattr(model, "regularizer", regularizer)


def _prepare_analysis_tensors(pack: dict[str, Any], spec: base.AnalysisSpec) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    segs_by_s, min_T, n_neurons = base._collect_segments(pack, spec.stimulus_levels)  # noqa: SLF001
    trialX, X_mean, counts, candidate_counts, ignored_short_counts = base._build_mean_and_trialx(  # noqa: SLF001
        segs_by_s,
        spec.stimulus_levels,
        base.ANALYSIS_T_FRAMES,
        n_neurons,
    )
    X_fit, trialX_fit = base._zscore_mean_and_trialx(X_mean, trialX)  # noqa: SLF001
    meta = {
        "stimulus_levels": [name for name, _ in spec.stimulus_levels],
        "trial_counts": counts,
        "candidate_trial_counts": candidate_counts,
        "ignored_short_trial_counts": ignored_short_counts,
        "ignored_short_trial_total": int(sum(ignored_short_counts.values())),
        "frame_window": base.ANALYSIS_T_FRAMES,
        "min_candidate_trial_frames": min_T,
        "n_neurons": n_neurons,
        "tensor_shape_mean": list(X_mean.shape),
        "labels": "st",
    }
    return X_fit, trialX_fit, meta


def _diagnostic_fieldnames() -> list[str]:
    fields = ["rank", "regularizer", "warning"]
    for prefix in ("explained_variance", "paper_explained_variance", "reconstruction_error"):
        fields.extend([f"{prefix}_{marg}" for marg in MARGINALIZATIONS])
        fields.append(f"{prefix}_total")
    return fields


def _write_rank_diagnostics_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_diagnostic_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_ready_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ready: list[dict[str, object]] = []
    for row in rows:
        converted: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, float):
                converted[key] = _finite_or_none(value)
            else:
                converted[key] = value
        ready.append(converted)
    return ready


def _write_rank_diagnostics_json(payload: dict[str, object], out_path: Path) -> None:
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _row_paper_explained_variance(row: dict[str, object], marginalization: str) -> float:
    value = row.get(f"paper_explained_variance_{marginalization}")
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    err = row.get(f"reconstruction_error_{marginalization}")
    if isinstance(err, (int, float)) and np.isfinite(float(err)):
        return _paper_explained_variance_from_error(float(err))
    return float("nan")


def _plot_rank_diagnostics(rows: list[dict[str, object]], out_path: Path, title: str) -> None:
    ok_rows = [row for row in rows if not row.get("warning")]
    if not ok_rows:
        return

    ranks = np.asarray([int(row["rank"]) for row in ok_rows], dtype=np.int64)
    explained_pct = 100.0 * np.asarray(
        [_row_paper_explained_variance(row, "total") for row in ok_rows],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(
        ranks,
        explained_pct,
        marker="o",
        markersize=3.5,
        linewidth=2.0,
        color="#b22222",
        label="dPCA",
    )
    ax.set_xlabel("Component")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_xticks(ranks[:: max(1, len(ranks) // 10)])
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"{title}\nreconstruction-based; no signal-noise correction", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_rank_diagnostics_for_analysis(
    run_key: str,
    spec: base.AnalysisSpec,
    pack: dict[str, Any],
    *,
    max_rank: int,
    regularizer_mode: str,
) -> None:
    analysis_out_dir = base.DPCA_OUT_ROOT / run_key / spec.subdir
    out_dir = base.DPCA_OUT_ROOT / "rank_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = f"{run_key}_{spec.subdir}_rank_diagnostics"
    X_fit, trialX_fit, tensor_meta = _prepare_analysis_tensors(pack, spec)
    regularizer = _regularizer_for_diagnostics(analysis_out_dir, regularizer_mode)
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    fitted_regularizer: float | str | None = regularizer

    try:
        model, fitted_regularizer = _fit_max_rank_model(X_fit, trialX_fit, max_rank, regularizer)
    except Exception as exc:  # noqa: BLE001 - write a diagnostic file with the failure details.
        warning = f"max-rank fit failed: {type(exc).__name__}: {exc}"
        warnings.append(warning)
        for rank in range(1, max_rank + 1):
            rows.append({"rank": rank, "regularizer": "", "warning": warning})
    else:
        for rank in range(1, max_rank + 1):
            row: dict[str, object] = {
                "rank": rank,
                "regularizer": "" if fitted_regularizer is None else str(fitted_regularizer),
                "warning": "",
            }
            metrics = {}
            metrics.update(_explained_variance_metrics(model, rank))
            metrics.update(_reconstruction_metrics(model, X_fit, rank))
            row["regularizer"] = "" if fitted_regularizer is None else str(fitted_regularizer)
            for key, value in metrics.items():
                row[key] = _finite_or_none(value)
            rows.append(row)

    payload = {
        "run_key": run_key,
        "analysis": spec.subdir,
        "title": spec.title,
        "rank_min": 1,
        "rank_max_requested": max_rank,
        "rank_fit": max_rank,
        "regularizer_mode": regularizer_mode,
        "regularizer_requested": None if regularizer is None else str(regularizer),
        "regularizer_fit": None if fitted_regularizer is None else str(fitted_regularizer),
        "diagnostic_definition": {
            "explained_variance": "cumulative sum of explained_variance_ratio_ entries 1..rank from one max-rank fit",
            "paper_explained_variance": "1 - reconstruction_error^2; matches the dPCA paper cumulative explained variance formula before signal-noise correction",
            "reconstruction_error": "sqrt(sum((target - reconstruction)^2) / sum(target^2))",
            "total_reconstruction_error": "relative error between centered X_fit and summed marginal reconstructions using components 1..rank",
        },
        "tensor_meta": tensor_meta,
        "warnings": warnings,
        "rows": _json_ready_rows(rows),
    }

    _write_rank_diagnostics_csv(rows, out_dir / f"{out_prefix}.csv")
    _write_rank_diagnostics_json(payload, out_dir / f"{out_prefix}.json")
    _plot_rank_diagnostics(rows, out_dir / f"{out_prefix}.png", f"{spec.title} — dPCA rank diagnostics")
    print(f"[{run_key}/{spec.subdir}] wrote rank diagnostics for ranks 1..{max_rank} to {out_dir / out_prefix}.*")
    for warning in warnings:
        print(f"  warning: {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="dPCA rank-sweep diagnostics.")
    parser.add_argument("--run", choices=base.RUN_KEYS, action="append", help="Run key to process. May be repeated.")
    parser.add_argument(
        "--analysis",
        choices=DEFAULT_ANALYSES,
        action="append",
        help="Analysis subdir to process. May be repeated.",
    )
    parser.add_argument("--max-rank", type=int, default=DEFAULT_MAX_RANK, help="Maximum tested dPCA rank.")
    parser.add_argument(
        "--regularizer-mode",
        choices=("meta_or_auto", "auto", "none"),
        default="meta_or_auto",
        help="Use existing optimal regularizer when available, optimize automatically, or fit unregularized.",
    )
    args = parser.parse_args()

    if args.max_rank < 1:
        raise ValueError("--max-rank must be at least 1")

    run_keys = tuple(args.run) if args.run else base.RUN_KEYS
    analyses = set(args.analysis or DEFAULT_ANALYSES)
    for run_key in run_keys:
        pack = load_npz_pack(run_key)
        specs = [spec for spec in base.analyses_for_run(run_key, pack) if spec.subdir in analyses]
        for spec in specs:
            run_rank_diagnostics_for_analysis(
                run_key,
                spec,
                pack,
                max_rank=args.max_rank,
                regularizer_mode=args.regularizer_mode,
            )


if __name__ == "__main__":
    main()
