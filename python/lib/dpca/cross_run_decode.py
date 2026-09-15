"""
Decode analysis labels in a fixed dPCA space learned from a reference phase.

This is a sanity/stability diagnostic:
1. Can labels be decoded from reference-phase trials projected onto reference-phase dPCA axes?
2. Can labels be decoded from another phase projected onto those same fixed axes?
3. Does a classifier trained on reference-phase features transfer to the other phase?

Run after run_dpca.py so reference-phase loadings contain the desired number of dPCs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib.dpca import analysis as base
from lib.io.npz import load_npz_pack

DEFAULT_TRAIN_RUN = base.RUN_KEYS[0]
DEFAULT_TEST_RUN = base.RUN_KEYS[1] if len(base.RUN_KEYS) > 1 else base.RUN_KEYS[0]
DEFAULT_ANALYSES = ("3conditions", "hedonic_valence", "consumption_mode")
DEFAULT_N_PCS = 5
DEFAULT_CV_REPEATS = 20
DEFAULT_BOOTSTRAPS = 1000
DEFAULT_RUN_PAIRS = tuple(
    (train_run, test_run)
    for train_run in base.RUN_KEYS
    for test_run in base.RUN_KEYS
    if train_run != test_run
)
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "s": ("s",),
    "st": ("st",),
    "t": ("t",),
    "all": ("s", "st", "t"),
}
OUTPUT_ROOT = base.DPCA_OUT_ROOT / "cross_run_decode"


def _load_decoder_matrices(run_key: str, analysis: str) -> dict[str, np.ndarray]:
    path = base.DPCA_OUT_ROOT / run_key / analysis / "loadings" / "encoder_decoder_matrices.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_dpca.py first so decoder matrices are saved."
        )
    with np.load(path) as data:
        return {str(k): np.asarray(data[k], dtype=np.float64) for k in data.files}


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
        "min_candidate_trial_frames": min_t,
        "n_neurons": n_neurons,
        "tensor_shape_mean": list(x_mean.shape),
    }
    return trial_x_fit, center, meta


def _project_trials(
    trial_x: np.ndarray,
    center: np.ndarray,
    decoder_matrices: dict[str, np.ndarray],
    *,
    marginalizations: tuple[str, ...],
    n_pcs: int,
    frame_window: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    start, stop = frame_window
    if stop <= start:
        raise ValueError(f"Invalid frame window: {frame_window}")
    stop = min(stop, int(trial_x.shape[3]))
    if stop <= start:
        raise ValueError(f"Frame window {frame_window} is outside trial length {trial_x.shape[3]}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    feature_names: list[str] = []
    warnings: list[str] = []

    decoder_by_marg: dict[str, np.ndarray] = {}
    for marg in marginalizations:
        key = f"decoder_D_{marg}"
        if key not in decoder_matrices:
            warnings.append(f"Missing {key}; skipping marginalization {marg}.")
            continue
        decoder = np.asarray(decoder_matrices[key], dtype=np.float64)
        n_use = min(n_pcs, int(decoder.shape[1]))
        if n_use < n_pcs:
            warnings.append(f"{key} has {decoder.shape[1]} PCs; requested {n_pcs}, using {n_use}.")
        if n_use < 1:
            warnings.append(f"{key} has no usable PCs; skipping marginalization {marg}.")
            continue
        decoder_by_marg[marg] = decoder[:, :n_use]
        feature_names.extend([f"{marg}_dpc{pc + 1}" for pc in range(n_use)])

    if not decoder_by_marg:
        raise RuntimeError("No usable decoder matrices for requested feature set.")

    for stim_ix in range(trial_x.shape[2]):
        for trial_ix in range(trial_x.shape[0]):
            xi = np.asarray(trial_x[trial_ix, :, stim_ix, :], dtype=np.float64)
            if np.any(np.isnan(xi)):
                continue
            x0 = xi - center[:, None]
            row_parts: list[np.ndarray] = []
            for decoder in decoder_by_marg.values():
                projected = decoder.T @ x0
                row_parts.append(np.mean(projected[:, start:stop], axis=1))
            features.append(np.concatenate(row_parts))
            labels.append(int(stim_ix))

    if not features:
        raise RuntimeError("No finite trial features were assembled.")
    return np.vstack(features), np.asarray(labels, dtype=np.int64), feature_names, warnings


def _classifier() -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=0),
    )


def _majority_accuracy(y: np.ndarray) -> float:
    counts = np.bincount(np.asarray(y, dtype=np.int64))
    return float(np.max(counts) / np.sum(counts)) if counts.size else float("nan")


def _sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _stratified_bootstrap_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstraps: int,
    random_state: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    class_indices = [np.flatnonzero(y_true == cls) for cls in np.unique(y_true)]
    acc: list[float] = []
    bacc: list[float] = []
    for _ in range(n_bootstraps):
        sample_ix = np.concatenate(
            [rng.choice(ix, size=ix.size, replace=True) for ix in class_indices if ix.size > 0]
        )
        acc.append(float(accuracy_score(y_true[sample_ix], y_pred[sample_ix])))
        bacc.append(float(balanced_accuracy_score(y_true[sample_ix], y_pred[sample_ix])))
    return {
        "accuracy_sem": _sem(acc),
        "balanced_accuracy_sem": _sem(bacc),
        "n_bootstraps": int(n_bootstraps),
    }


def _cv_scores(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | str]:
    counts = np.bincount(y)
    nonzero = counts[counts > 0]
    if nonzero.size < 2:
        return {"warning": "Need at least two classes for decoding."}
    n_splits = min(5, int(np.min(nonzero)))
    if n_splits < 2:
        return {"warning": "Need at least two trials per class for cross-validation."}
    acc: list[float] = []
    bacc: list[float] = []
    for repeat_ix in range(DEFAULT_CV_REPEATS):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat_ix)
        for train_ix, test_ix in cv.split(x, y):
            clf = _classifier()
            clf.fit(x[train_ix], y[train_ix])
            pred = clf.predict(x[test_ix])
            acc.append(float(accuracy_score(y[test_ix], pred)))
            bacc.append(float(balanced_accuracy_score(y[test_ix], pred)))
    return {
        "n_splits": n_splits,
        "n_repeats": DEFAULT_CV_REPEATS,
        "n_score_samples": int(len(bacc)),
        "accuracy": float(np.mean(acc)),
        "accuracy_sem": _sem(acc),
        "balanced_accuracy": float(np.mean(bacc)),
        "balanced_accuracy_sem": _sem(bacc),
        "majority_accuracy": _majority_accuracy(y),
        "chance_balanced_accuracy": float(1.0 / nonzero.size),
        "warning": "",
    }


def _train_test_scores(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float | str]:
    train_classes = set(int(v) for v in np.unique(y_train))
    test_classes = set(int(v) for v in np.unique(y_test))
    if train_classes != test_classes:
        return {"warning": f"Train/test class mismatch: train={sorted(train_classes)}, test={sorted(test_classes)}"}
    clf = _classifier()
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    bootstrap = _stratified_bootstrap_scores(
        y_test,
        pred,
        n_bootstraps=DEFAULT_BOOTSTRAPS,
        random_state=0,
    )
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "majority_accuracy": _majority_accuracy(y_test),
        "chance_balanced_accuracy": float(1.0 / len(test_classes)),
        **bootstrap,
        "warning": "",
    }


def _prefixed(prefix: str, values: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _plot_analysis_rows(
    rows: list[dict[str, object]],
    out_path: Path,
    title: str,
    *,
    train_run: str,
    test_run: str,
) -> None:
    feature_sets = [str(row["feature_set"]) for row in rows]
    metrics = (
        (
            f"{train_run} CV",
            f"{train_run}_cv_balanced_accuracy",
            f"{train_run}_cv_balanced_accuracy_sem",
            "#0b3d91",
        ),
        (
            f"{test_run} CV on {train_run} axes",
            f"{test_run}_fixed_axes_cv_balanced_accuracy",
            f"{test_run}_fixed_axes_cv_balanced_accuracy_sem",
            "#2e8b57",
        ),
        (
            f"train {train_run} -> test {test_run}",
            f"train_{train_run}_test_{test_run}_balanced_accuracy",
            f"train_{train_run}_test_{test_run}_balanced_accuracy_sem",
            "#b22222",
        ),
    )
    x = np.arange(len(feature_sets), dtype=np.float64)
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for idx, (label, key, err_key, color) in enumerate(metrics):
        vals = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        errs = np.asarray([row.get(err_key, np.nan) for row in rows], dtype=np.float64)
        ax.bar(
            x + (idx - 1) * width,
            vals,
            width=width,
            yerr=errs,
            capsize=3,
            color=color,
            label=label,
        )
    chance = rows[0].get(f"train_{train_run}_test_{test_run}_chance_balanced_accuracy", np.nan)
    if isinstance(chance, (int, float)) and np.isfinite(float(chance)):
        ax.axhline(float(chance), color="0.35", linestyle="--", linewidth=0.9, label="balanced chance")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_sets)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel(f"Fixed {train_run} dPCA feature set")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_s_fixed_axes_decode_summary(
    rows: list[dict[str, object]],
    out_path: Path,
    *,
    train_run: str,
    test_run: str,
    analyses: tuple[str, ...],
) -> None:
    """Balanced accuracy for s features only: train-run CV vs test-run CV on train axes."""
    analysis_labels = {
        "3conditions": "3 conditions",
        "hedonic_valence": "hedonic valence",
        "consumption_mode": "consumption mode",
    }
    metrics = (
        (f"{train_run} CV", f"{train_run}_cv_balanced_accuracy", f"{train_run}_cv_balanced_accuracy_sem", "#0b3d91"),
        (
            f"{test_run} CV on {train_run} axes",
            f"{test_run}_fixed_axes_cv_balanced_accuracy",
            f"{test_run}_fixed_axes_cv_balanced_accuracy_sem",
            "#2e8b57",
        ),
    )
    x = np.arange(len(analyses), dtype=np.float64)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for idx, (label, key, err_key, color) in enumerate(metrics):
        vals: list[float] = []
        errs: list[float] = []
        for analysis in analyses:
            match = next(
                (row for row in rows if row.get("analysis") == analysis and row.get("feature_set") == "s"),
                None,
            )
            vals.append(float(match.get(key, np.nan)) if match else float("nan"))
            errs.append(float(match.get(err_key, np.nan)) if match else float("nan"))
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            yerr=errs,
            capsize=3,
            color=color,
            label=label,
        )
    chance_labeled = False
    for xi, analysis in enumerate(analyses):
        match = next(
            (row for row in rows if row.get("analysis") == analysis and row.get("feature_set") == "s"),
            None,
        )
        if match is None:
            continue
        chance = match.get(f"{train_run}_cv_chance_balanced_accuracy", np.nan)
        if not isinstance(chance, (int, float)) or not np.isfinite(float(chance)):
            continue
        chance = float(chance)
        span = 0.42
        ax.plot(
            [xi - span, xi + span],
            [chance, chance],
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            label="balanced chance" if not chance_labeled else None,
            zorder=0,
        )
        chance_labeled = True
    ax.set_xticks(x)
    ax.set_xticklabels([analysis_labels.get(a, a) for a in analyses])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Label type")
    ax.set_title(f"s-space decode on fixed {train_run} axes ({train_run} vs {test_run})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _load_decode_rows(out_root: Path, train_run: str, test_run: str) -> list[dict[str, object]]:
    json_path = out_root / f"{train_run}_to_{test_run}" / "fixed_axes_decode.json"
    if not json_path.exists():
        return []
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    return rows if isinstance(rows, list) else []


def _s_row_for_analysis(rows: list[dict[str, object]], analysis: str) -> dict[str, object] | None:
    return next((row for row in rows if row.get("analysis") == analysis and row.get("feature_set") == "s"), None)


def plot_reference_s_fixed_axes_decode_combined(
    out_root: Path = OUTPUT_ROOT,
    *,
    train_run: str = DEFAULT_TRAIN_RUN,
    test_runs: tuple[str, ...] = tuple(rk for rk in base.RUN_KEYS if rk != DEFAULT_TRAIN_RUN),
    analyses: tuple[str, ...] = DEFAULT_ANALYSES,
) -> None:
    """One figure: reference CV plus each test phase decoded on fixed reference s-axes."""
    pair_rows = {test_run: _load_decode_rows(out_root, train_run, test_run) for test_run in test_runs}
    if not all(pair_rows.values()):
        missing = [f"{train_run}_to_{tr}" for tr in test_runs if not pair_rows[tr]]
        print(f"[cross_run_decode] skipping combined reference s summary; missing: {', '.join(missing)}")
        return

    analysis_labels = {
        "3conditions": "3 conditions",
        "hedonic_valence": "hedonic valence",
        "consumption_mode": "consumption mode",
    }
    cmap = plt.get_cmap("tab10")
    test_colors = {rk: cmap(i % 10) for i, rk in enumerate(test_runs)}
    bar_specs: list[tuple[str, str, str, str]] = [
        (f"{train_run} CV", f"{train_run}_cv_balanced_accuracy", f"{train_run}_cv_balanced_accuracy_sem", "#0b3d91"),
    ]
    for test_run in test_runs:
        bar_specs.append(
            (
                f"{test_run} on {train_run} axes",
                f"{test_run}_fixed_axes_cv_balanced_accuracy",
                f"{test_run}_fixed_axes_cv_balanced_accuracy_sem",
                test_colors.get(test_run, "0.45"),
            )
        )

    x = np.arange(len(analyses), dtype=np.float64)
    n_bars = len(bar_specs)
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for bar_ix, (label, key, err_key, color) in enumerate(bar_specs):
        vals: list[float] = []
        errs: list[float] = []
        for analysis in analyses:
            if bar_ix == 0:
                row = _s_row_for_analysis(pair_rows[test_runs[0]], analysis)
            else:
                row = _s_row_for_analysis(pair_rows[test_runs[bar_ix - 1]], analysis)
            vals.append(float(row.get(key, np.nan)) if row else float("nan"))
            errs.append(float(row.get(err_key, np.nan)) if row else float("nan"))
        offset = (bar_ix - (n_bars - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, yerr=errs, capsize=3, color=color, label=label)

    chance_labeled = False
    for xi, analysis in enumerate(analyses):
        row = _s_row_for_analysis(pair_rows[test_runs[0]], analysis)
        if row is None:
            continue
        chance = row.get(f"{train_run}_cv_chance_balanced_accuracy", np.nan)
        if not isinstance(chance, (int, float)) or not np.isfinite(float(chance)):
            continue
        chance = float(chance)
        ax.plot(
            [xi - 0.48, xi + 0.48],
            [chance, chance],
            color="0.35",
            linestyle="--",
            linewidth=1.0,
            label="balanced chance" if not chance_labeled else None,
            zorder=0,
        )
        chance_labeled = True

    test_label = " & ".join(test_runs)
    ax.set_xticks(x)
    ax.set_xticklabels([analysis_labels.get(a, a) for a in analyses])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Label type")
    ax.set_title(f"s-space decode on fixed {train_run} axes ({train_run} vs {test_label})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path = out_root / f"s_fixed_{train_run}_axes_decode_combined.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote combined s-space decode summary to {out_path}")


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


def run_cross_run_decode(
    *,
    train_run: str,
    test_run: str,
    analyses: tuple[str, ...],
    n_pcs: int,
    frame_window: tuple[int, int],
) -> None:
    out_dir = OUTPUT_ROOT / f"{train_run}_to_{test_run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "train_axes_run": train_run,
        "test_run": test_run,
        "n_pcs_requested_per_marginalization": n_pcs,
        "frame_window": list(frame_window),
        "feature_sets": {key: list(value) for key, value in FEATURE_SETS.items()},
        "notes": [
            "All features are trial-window means after projection onto decoder_D matrices learned from the train run.",
            "Each run is z-scored and centered with its own trial tensor before projection, removing run-specific baseline offsets.",
            f"Within-run bars show mean balanced accuracy across {DEFAULT_CV_REPEATS} repeated stratified CV runs; error bars are SEM across folds.",
            "The transfer score trains a classifier on train-run projections and evaluates on test-run projections in the same fixed dPCA space.",
            f"Transfer error bars are stratified bootstrap SEM over test-run trials ({DEFAULT_BOOTSTRAPS} resamples).",
        ],
    }

    train_pack = load_npz_pack(train_run)
    test_pack = load_npz_pack(test_run)
    train_specs = {spec.subdir: spec for spec in base.analyses_for_run(train_run, train_pack)}
    test_specs = {spec.subdir: spec for spec in base.analyses_for_run(test_run, test_pack)}
    for analysis in analyses:
        if analysis not in train_specs or analysis not in test_specs:
            rows.append({"analysis": analysis, "warning": f"Analysis not available for both {train_run} and {test_run}."})
            continue

        train_spec = train_specs[analysis]
        test_spec = test_specs[analysis]
        decoder_matrices = _load_decoder_matrices(train_run, analysis)
        train_trial_x, train_center, train_meta = _prepare_trial_tensor(train_run, train_spec)
        test_trial_x, test_center, test_meta = _prepare_trial_tensor(test_run, test_spec)
        metadata[f"{analysis}_train_tensor"] = train_meta
        metadata[f"{analysis}_test_tensor"] = test_meta
        analysis_rows: list[dict[str, object]] = []

        for feature_set, marginalizations in FEATURE_SETS.items():
            x_train, y_train, feature_names, train_warnings = _project_trials(
                train_trial_x,
                train_center,
                decoder_matrices,
                marginalizations=marginalizations,
                n_pcs=n_pcs,
                frame_window=frame_window,
            )
            x_test, y_test, _test_feature_names, test_warnings = _project_trials(
                test_trial_x,
                test_center,
                decoder_matrices,
                marginalizations=marginalizations,
                n_pcs=n_pcs,
                frame_window=frame_window,
            )
            row: dict[str, object] = {
                "analysis": analysis,
                "feature_set": feature_set,
                "marginalizations": "+".join(marginalizations),
                "n_features": int(x_train.shape[1]),
                "features": ";".join(feature_names),
                "train_run": train_run,
                "test_run": test_run,
                "n_train_trials": int(x_train.shape[0]),
                "n_test_trials": int(x_test.shape[0]),
                "train_counts": ";".join(str(int(v)) for v in np.bincount(y_train)),
                "test_counts": ";".join(str(int(v)) for v in np.bincount(y_test)),
                "warning": "; ".join(train_warnings + test_warnings),
            }
            row.update(_prefixed(f"{train_run}_cv", _cv_scores(x_train, y_train)))
            row.update(_prefixed(f"{test_run}_fixed_axes_cv", _cv_scores(x_test, y_test)))
            row.update(_prefixed(f"train_{train_run}_test_{test_run}", _train_test_scores(x_train, y_train, x_test, y_test)))
            rows.append(row)
            analysis_rows.append(row)

        _plot_analysis_rows(
            analysis_rows,
            out_dir / f"{analysis}_fixed_{train_run}_axes_decode.png",
            f"{analysis}: fixed {train_run} dPCA axes",
            train_run=train_run,
            test_run=test_run,
        )

    _plot_s_fixed_axes_decode_summary(
        rows,
        out_dir / f"s_fixed_{train_run}_axes_decode_summary.png",
        train_run=train_run,
        test_run=test_run,
        analyses=analyses,
    )

    _write_csv(rows, out_dir / "fixed_axes_decode.csv")
    (out_dir / "fixed_axes_decode.json").write_text(
        json.dumps({"metadata": metadata, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote cross-run dPCA decoding diagnostics to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decode labels in fixed dPCA axes across runs. By default, runs all ordered "
            "pairs in run_dpca RUN_KEYS; pass both --train-run and --test-run for one pair."
        )
    )
    parser.add_argument("--train-run", choices=base.RUN_KEYS)
    parser.add_argument("--test-run", choices=base.RUN_KEYS)
    parser.add_argument(
        "--analysis",
        choices=DEFAULT_ANALYSES,
        action="append",
        help="Analysis to process. May be repeated. Default: all core analyses.",
    )
    parser.add_argument("--n-pcs", type=int, default=DEFAULT_N_PCS, help="dPCs per marginalization to use.")
    parser.add_argument(
        "--window",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        default=base.SCATTER_MEAN_FRAME_WINDOW,
        help="Frame window used to average each projected dPC time course.",
    )
    args = parser.parse_args()
    if args.n_pcs < 1:
        raise ValueError("--n-pcs must be at least 1")
    analyses = tuple(args.analysis or DEFAULT_ANALYSES)
    if (args.train_run is None) != (args.test_run is None):
        parser.error("Pass both --train-run and --test-run, or neither to run all default pairs.")
    run_pairs = ((args.train_run, args.test_run),) if args.train_run else DEFAULT_RUN_PAIRS
    for train_run, test_run in run_pairs:
        run_cross_run_decode(
            train_run=train_run,
            test_run=test_run,
            analyses=analyses,
            n_pcs=args.n_pcs,
            frame_window=(int(args.window[0]), int(args.window[1])),
        )
    if len(base.RUN_KEYS) > 1:
        plot_reference_s_fixed_axes_decode_combined(OUTPUT_ROOT, analyses=analyses)


if __name__ == "__main__":
    main()
