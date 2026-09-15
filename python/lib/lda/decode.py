"""
Binary decoding and pooled LDA on normalized Livneh trial packs.

Train on pooled train set (phase packs excluding SLM),
evaluate source CV folds and full transfer on configured phase packs.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from lib.config import DECODE_BINARY_DIR, EVAL_RUN_KEYS, LDA_DIR, LDA_FOUR_CLASS_DIR, LDA_POOLED_BINARY_DIR
from lib.io.npz import load_npz_pack
from lib.tasks import TASK_SPECS, COND_IDS as _COND_IDS, cond_name_map as _cond_name_map, normalize_label as _normalize_label

RANDOM_STATE = 42
N_FOLDS = 5
TRAIN_T0 = 60
TRAIN_T1 = 120
PCA_DIR = DECODE_BINARY_DIR / "pca"
LDA_TOP_K = 30
TARGET_RUN_KEYS = tuple(EVAL_RUN_KEYS)


def mean_slice(seg: np.ndarray, t0: int, t1: int) -> np.ndarray:
    L = seg.shape[1]
    if t0 >= L:
        return np.full(seg.shape[0], np.nan, dtype=np.float32)
    return np.mean(seg[:, t0:min(t1, L)], axis=1)


def features_for_pack(key: str) -> tuple[np.ndarray, np.ndarray]:
    pack = load_npz_pack(key)
    segments = pack["segments"]
    y_cond = pack["trial_type"].astype(int)
    X = np.zeros((len(segments), pack["n_neurons"]), dtype=np.float32)
    for i, seg in enumerate(segments):
        X[i] = mean_slice(seg, TRAIN_T0, TRAIN_T1)
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    return X, y_cond


def build_binary_labels(y_cond: np.ndarray, cond_name_map: dict[int, str], task_key: str) -> tuple[np.ndarray, np.ndarray]:
    spec = TASK_SPECS[task_key]
    y_bin = np.full_like(y_cond, fill_value=-1, dtype=np.int64)
    for cid in np.unique(y_cond):
        name = cond_name_map.get(int(cid), f"cond_{int(cid)}")
        n = _normalize_label(name)
        assigned = None
        for cls, tokens in spec["rules"]:
            if any(tok in n for tok in tokens):
                assigned = cls
                break
        if assigned is None:
            ci = int(cid)
            if task_key == "hedonic_valence":
                if ci in _COND_IDS["water"]:
                    assigned = 1
                elif ci in (_COND_IDS["nacl"] | _COND_IDS["airpuff"]):
                    assigned = 0
            else:
                if ci in (_COND_IDS["water"] | _COND_IDS["nacl"]):
                    assigned = 1
                elif ci in (_COND_IDS["airpuff"] | _COND_IDS["waterfree"]):
                    assigned = 0
        if assigned is not None:
            y_bin[y_cond == cid] = assigned
    keep = y_bin >= 0
    return y_bin, keep


def source_feature_sets(n_neurons: int) -> list[tuple[str, list[int]]]:
    return [("All", list(range(n_neurons)))]



def _run_markers(run_keys: tuple[str, ...]) -> dict[str, str]:
    marker_cycle = ("o", "s", "^", "D", "v", "P", "X")
    return {rk: marker_cycle[i % len(marker_cycle)] for i, rk in enumerate(run_keys)}


def _classifier_sort_key(name: str) -> tuple[int, int]:
    if name == "All":
        return (0, 0)
    m = re.match(r"Rank(\d+)$", name)
    if m:
        return (1, int(m.group(1)))
    return (2, 999)


def plot_accuracy_for_source_combined(agg: pd.DataFrame, source_key: str) -> None:
    target_runs = TARGET_RUN_KEYS
    sub = agg[(agg["source"] == source_key) & (agg["target"].isin(target_runs))].copy()
    if sub.empty:
        return
    classifiers = sorted(list(sub["classifier"].drop_duplicates()), key=_classifier_sort_key)
    n_cls = len(classifiers)
    if n_cls == 0:
        return
    nfeat_map = {}
    for clf_name in classifiers:
        rows = sub[sub["classifier"] == clf_name]
        nfeat_map[clf_name] = int(rows["n_features"].iloc[0]) if not rows.empty else 0

    tasks = list(TASK_SPECS.keys())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(11, 4.2 * len(tasks)), squeeze=False, sharex=True)
    x = np.arange(len(target_runs), dtype=float)
    width = min(0.82 / max(n_cls, 1), 0.18)
    offsets = np.linspace(-(n_cls - 1) * width / 2.0, (n_cls - 1) * width / 2.0, n_cls)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_cls)]

    for ax_i, task_key in enumerate(tasks):
        ax = axes[ax_i, 0]
        d_task = sub[sub["task"] == task_key]
        if d_task.empty:
            continue
        for j, clf_name in enumerate(classifiers):
            d = d_task[d_task["classifier"] == clf_name]
            means = []
            sems = []
            for tgt in target_runs:
                row = d[d["target"] == tgt]
                if row.empty:
                    means.append(np.nan)
                    sems.append(0.0)
                else:
                    means.append(float(row["mean_accuracy"].iloc[0]))
                    sems.append(float(row["sem_accuracy"].iloc[0]))
            means_arr = np.array(means, dtype=float)
            sems_arr = np.array(sems, dtype=float)
            bar_x = x + offsets[j]
            bars = ax.bar(
                bar_x,
                means_arr,
                width=width,
                yerr=sems_arr,
                capsize=3,
                color=colors[j],
                label=f"{clf_name} (n={nfeat_map.get(clf_name, 0)})" if ax_i == 0 else None,
            )
            for b, m, se in zip(bars, means_arr, sems_arr):
                if np.isnan(m):
                    continue
                y_text = min(1.08, float(m + se) + 0.02)
                ax.text(b.get_x() + b.get_width() / 2.0, y_text, f"{m:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
        ax.set_ylabel("Accuracy (Mean +- SEM)")
        ax.set_title(TASK_SPECS[task_key]["display"])
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(list(target_runs))
        ax.tick_params(axis="x", labelbottom=True)

    axes[0, 0].set_xlabel("Test division")
    axes[-1, 0].set_xlabel("Test division")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Predictor", loc="upper right", fontsize=8)
    fig.suptitle(f"{source_key}: transfer accuracy across divisions (global groups)", y=1.01)
    fig.tight_layout()
    fig.savefig(DECODE_BINARY_DIR / f"{source_key}_across_runs_predictors_mean_sem.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_pooled_by_task(task_key: str, X_by_run: dict[str, np.ndarray], ycond_by_run: dict[str, np.ndarray]) -> None:
    pooled_chunks = []
    for rk in TARGET_RUN_KEYS:
        yb_t, keep_t = build_binary_labels(ycond_by_run[rk], _cond_name_map(rk), task_key)
        Xt = X_by_run[rk][keep_t]
        if len(Xt) > 0:
            pooled_chunks.append(Xt)
    if not pooled_chunks:
        return

    X_pool = np.vstack(pooled_chunks)
    scaler = StandardScaler()
    Xz_pool = scaler.fit_transform(X_pool)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(Xz_pool)

    colors = {0: "#b22222", 1: "#0b3d91"}
    markers = _run_markers(TARGET_RUN_KEYS)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    spec = TASK_SPECS[task_key]

    for rk in TARGET_RUN_KEYS:
        yb_t, keep_t = build_binary_labels(ycond_by_run[rk], _cond_name_map(rk), task_key)
        Xt = X_by_run[rk][keep_t]
        ybt = yb_t[keep_t]
        if len(ybt) == 0:
            continue
        Xt2 = pca.transform(scaler.transform(Xt))
        for cls in (0, 1):
            m = ybt == cls
            if np.any(m):
                ax.scatter(
                    Xt2[m, 0],
                    Xt2[m, 1],
                    c=colors[cls],
                    marker=markers.get(rk, "o"),
                    alpha=0.72,
                    s=15,
                )

    ax.set_title(f"pooled_{task_key} PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors[0],
            markeredgecolor="0.25",
            markersize=8,
            label=spec["class0_name"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors[1],
            markeredgecolor="0.25",
            markersize=8,
            label=spec["class1_name"],
        ),
    ]
    run_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[rk],
            linestyle="None",
            color="0.35",
            markerfacecolor="0.35",
            markeredgecolor="0.2",
            markersize=7,
            label=rk,
        )
        for rk in TARGET_RUN_KEYS
    ]
    leg_class = ax.legend(handles=color_handles, loc="upper left", fontsize=8, title="Class")
    ax.add_artist(leg_class)
    ax.legend(handles=run_handles, loc="upper right", fontsize=8, title="Division")
    fig.tight_layout()
    fig.savefig(PCA_DIR / f"pooled_{task_key}_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_source_by_task(
    source_key: str,
    task_key: str,
    X_by_run: dict[str, np.ndarray],
    ycond_by_run: dict[str, np.ndarray],
) -> None:
    y_src_bin, keep_src = build_binary_labels(ycond_by_run[source_key], _cond_name_map(source_key), task_key)
    X_src = X_by_run[source_key][keep_src]
    if len(X_src) == 0:
        return

    scaler = StandardScaler()
    Xz_src = scaler.fit_transform(X_src)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(Xz_src)

    colors = {0: "#b22222", 1: "#0b3d91"}
    markers = _run_markers(TARGET_RUN_KEYS)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for rk in TARGET_RUN_KEYS:
        yb_t, keep_t = build_binary_labels(ycond_by_run[rk], _cond_name_map(rk), task_key)
        Xt = X_by_run[rk][keep_t]
        ybt = yb_t[keep_t]
        if len(ybt) == 0:
            continue
        Xt2 = pca.transform(scaler.transform(Xt))
        is_source = rk == source_key
        point_alpha = 0.85 if is_source else 0.30
        point_size = 18 if is_source else 13
        for cls in (0, 1):
            m = ybt == cls
            if np.any(m):
                ax.scatter(
                    Xt2[m, 0],
                    Xt2[m, 1],
                    c=colors[cls],
                    marker=markers.get(rk, "o"),
                    alpha=point_alpha,
                    s=point_size,
                    label=rk if cls == 0 else None,
                )

    ax.set_title(f"{source_key}_{task_key} PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    handles, labels = ax.get_legend_handles_labels()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in uniq_l:
            continue
        uniq_h.append(h)
        uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PCA_DIR / f"{source_key}_{task_key}_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def pooled_xy_for_lda(
    task_key: str, X_by_run: dict[str, np.ndarray], ycond_by_run: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pooled standardized features with per-row stimulus class, run group (0=reference, 1=other), and division name."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    yr: list[np.ndarray] = []
    rks: list[str] = []
    for rk in TARGET_RUN_KEYS:
        yb, keep = build_binary_labels(ycond_by_run[rk], _cond_name_map(rk), task_key)
        Xt = X_by_run[rk][keep]
        ybt = yb[keep]
        if len(ybt) == 0:
            continue
        n = len(ybt)
        xs.append(Xt.astype(np.float64, copy=False))
        ys.append(ybt.astype(np.int64, copy=False))
        yr.append(np.full(n, 0 if rk == TARGET_RUN_KEYS[0] else 1, dtype=np.int64))
        rks.extend([rk] * n)
    if not xs:
        return None
    X = np.vstack(xs)
    y_stim = np.concatenate(ys)
    y_run = np.concatenate(yr)
    run_key = np.array(rks, dtype=object)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    return Xz, y_stim, y_run, run_key


def pooled_xy_for_four_class_lda(
    task_key: str, X_by_run: dict[str, np.ndarray], ycond_by_run: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pooled z-scored X with class 0..3 = (reference|other)×(binary outcome), run_key per row for markers."""
    xs: list[np.ndarray] = []
    y4: list[np.ndarray] = []
    rks: list[str] = []
    for rk in TARGET_RUN_KEYS:
        yb, keep = build_binary_labels(ycond_by_run[rk], _cond_name_map(rk), task_key)
        Xt = X_by_run[rk][keep]
        ybt = yb[keep]
        if len(ybt) == 0:
            continue
        n = len(ybt)
        run_grp = 0 if rk == TARGET_RUN_KEYS[0] else 1
        y_four_row = (run_grp * 2 + ybt.astype(np.int64, copy=False)).astype(np.int64, copy=False)
        xs.append(Xt.astype(np.float64, copy=False))
        y4.append(y_four_row)
        rks.extend([rk] * n)
    if not xs:
        return None
    X = np.vstack(xs)
    y_four = np.concatenate(y4)
    run_key = np.array(rks, dtype=object)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    return Xz, y_four, run_key


def four_class_labels(task_key: str) -> list[str]:
    spec = TASK_SPECS[task_key]
    a0, a1 = spec["class0_name"], spec["class1_name"]
    return [
        f"{TARGET_RUN_KEYS[0]} | {a0}",
        f"{TARGET_RUN_KEYS[0]} | {a1}",
        f"{'+'.join(TARGET_RUN_KEYS[1:]) if len(TARGET_RUN_KEYS) > 1 else 'other'} | {a0}",
        f"{'+'.join(TARGET_RUN_KEYS[1:]) if len(TARGET_RUN_KEYS) > 1 else 'other'} | {a1}",
    ]


def _bar_topk_lda(ax: Axes, loadings: np.ndarray, title: str, *, show_ylabel: bool = False) -> None:
    v = np.asarray(loadings, dtype=float).ravel()
    if v.size == 0 or np.all(np.isnan(v)):
        ax.set_title(f"{title} (n/a)")
        ax.axis("off")
        return
    v_plot = np.nan_to_num(v, nan=0.0)
    k = min(LDA_TOP_K, v.size)
    ix = np.argsort(-np.abs(v_plot))[:k]
    vals = v_plot[ix]
    y_pos = np.arange(k)
    ax.barh(y_pos, vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(int(i)) for i in ix], fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0.0, color="gray", lw=0.8, alpha=0.7)
    ax.set_xlabel("LDA weight (per neuron, z-scored activity)")
    ax.set_title(title, fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Neuron index\n(largest |weight| at top)")


def export_lda_pooled_task(task_key: str, X_by_run: dict[str, np.ndarray], ycond_by_run: dict[str, np.ndarray]) -> None:
    packed = pooled_xy_for_lda(task_key, X_by_run, ycond_by_run)
    if packed is None:
        return
    Xz, y_stim, y_run, run_key = packed
    if len(np.unique(y_run)) < 2 or len(np.unique(y_stim)) < 2:
        return

    lda_run = LinearDiscriminantAnalysis(solver="svd", n_components=1)
    lda_run.fit(Xz, y_run)
    v_run_raw = np.asarray(lda_run.scalings_.ravel(), dtype=np.float64)

    lda_stim = LinearDiscriminantAnalysis(solver="svd", n_components=1)
    lda_stim.fit(Xz, y_stim)
    v_stim_raw = np.asarray(lda_stim.scalings_.ravel(), dtype=np.float64)

    x_lda_run = lda_run.transform(Xz)[:, 0]
    y_lda_stim = lda_stim.transform(Xz)[:, 0]

    mask_slm = y_run == 1
    v_stim_slm = np.full(Xz.shape[1], np.nan, dtype=np.float64)
    if int(np.sum(mask_slm)) > 1 and len(np.unique(y_stim[mask_slm])) >= 2:
        lda_slm = LinearDiscriminantAnalysis(solver="svd", n_components=1)
        lda_slm.fit(Xz[mask_slm], y_stim[mask_slm])
        v_stim_slm = np.asarray(lda_slm.scalings_.ravel(), dtype=np.float64)

    n_neurons = Xz.shape[1]
    df_lda = pd.DataFrame(
        {
            "neuron_index": np.arange(n_neurons, dtype=np.int64),
            "run_lda_loading": v_run_raw,
            "stim_lda_loading": v_stim_raw,
            "stim_lda_loading_slm_only": v_stim_slm,
        }
    )
    LDA_POOLED_BINARY_DIR.mkdir(parents=True, exist_ok=True)
    df_lda.to_csv(LDA_POOLED_BINARY_DIR / f"pooled_{task_key}_lda_loadings.csv", index=False)

    colors = {0: "#b22222", 1: "#0b3d91"}
    markers = _run_markers(TARGET_RUN_KEYS)
    spec = TASK_SPECS[task_key]
    outcome = spec.get("lda_outcome_short", "Outcome")
    scatter_slug = spec.get("lda_scatter_slug", "outcome")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for rk in TARGET_RUN_KEYS:
        m = run_key == rk
        if not np.any(m):
            continue
        for cls in (0, 1):
            mc = m & (y_stim == cls)
            if np.any(mc):
                ax.scatter(
                    x_lda_run[mc],
                    y_lda_stim[mc],
                    c=colors[cls],
                    marker=markers.get(rk, "o"),
                    alpha=0.72,
                    s=15,
                )
    ax.set_title(f"{task_key} LDA")
    ax.set_xlabel(f"LD1 run ({TARGET_RUN_KEYS[0]} vs other phases)")
    ax.set_ylabel(f"LD1 {outcome} (binary LDA; axes not orthogonalized)")
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors[0],
            markeredgecolor="0.25",
            markersize=8,
            label=spec["class0_name"],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=colors[1],
            markeredgecolor="0.25",
            markersize=8,
            label=spec["class1_name"],
        ),
    ]
    run_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[rk],
            linestyle="None",
            color="0.35",
            markerfacecolor="0.35",
            markeredgecolor="0.2",
            markersize=7,
            label=rk,
        )
        for rk in TARGET_RUN_KEYS
    ]
    leg_class = ax.legend(handles=color_handles, loc="upper left", fontsize=8, title="Class")
    ax.add_artist(leg_class)
    ax.legend(handles=run_handles, loc="upper right", fontsize=8, title="Division")
    fig.tight_layout()
    fig.savefig(LDA_POOLED_BINARY_DIR / f"pooled_{task_key}_lda_run_vs_{scatter_slug}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, axes = plt.subplots(1, 2, figsize=(11.5, 7.2), squeeze=False)
    loadings_panels: list[tuple[str, np.ndarray]] = [
        (f"Run\n(top {LDA_TOP_K} by |weight|)", v_run_raw),
        (f"{outcome}\n(top {LDA_TOP_K} by |weight|)", v_stim_raw),
    ]
    for i, (ax, (title, vec)) in enumerate(zip(axes.ravel(), loadings_panels)):
        _bar_topk_lda(ax, vec, title, show_ylabel=(i == 0))
    fig2.suptitle(f"{task_key} LDA", y=1.0, fontsize=11)
    fig2.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    fig2.savefig(LDA_POOLED_BINARY_DIR / f"pooled_{task_key}_lda_top_loadings.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)




def export_lda_four_class_pooled_task(
    task_key: str, X_by_run: dict[str, np.ndarray], ycond_by_run: dict[str, np.ndarray]
) -> None:
    """Pooled 4-class LDA: (reference vs other phases) × binary task label; LD1 vs LD2 scatter."""
    packed = pooled_xy_for_four_class_lda(task_key, X_by_run, ycond_by_run)
    if packed is None:
        return
    Xz, y_four, run_key = packed
    n_classes = int(len(np.unique(y_four)))
    n_comp = min(2, n_classes - 1)
    if n_comp < 2:
        print(
            f"[four_class LDA] {task_key}: need ≥3 distinct (run×outcome) classes for two LDA axes "
            f"(found {n_classes}); skipping."
        )
        return

    lda_fc = LinearDiscriminantAnalysis(solver="svd", n_components=n_comp)
    lda_fc.fit(Xz, y_four)
    Z = lda_fc.transform(Xz)
    scalings = np.asarray(lda_fc.scalings_, dtype=np.float64)
    n_neurons = Xz.shape[1]
    v_ld1 = scalings[:, 0].ravel()
    v_ld2 = scalings[:, 1].ravel()

    LDA_FOUR_CLASS_DIR.mkdir(parents=True, exist_ok=True)
    class_names = four_class_labels(task_key)
    pd.DataFrame({"class_id": np.arange(4, dtype=np.int64), "label": class_names}).to_csv(
        LDA_FOUR_CLASS_DIR / f"pooled_{task_key}_four_class_lda_class_key.csv", index=False
    )
    pd.DataFrame({"class_id": np.arange(4, dtype=np.int64), "n_trials": [int(np.sum(y_four == c)) for c in range(4)]}).to_csv(
        LDA_FOUR_CLASS_DIR / f"pooled_{task_key}_four_class_lda_class_counts.csv", index=False
    )

    pd.DataFrame(
        {
            "neuron_index": np.arange(n_neurons, dtype=np.int64),
            "lda_ld1_loading": v_ld1,
            "lda_ld2_loading": v_ld2,
        }
    ).to_csv(LDA_FOUR_CLASS_DIR / f"pooled_{task_key}_four_class_lda_loadings.csv", index=False)

    cmap = plt.get_cmap("tab10")
    class_colors = {c: cmap(int(c) % 10) for c in range(4)}
    markers = _run_markers(TARGET_RUN_KEYS)
    spec = TASK_SPECS[task_key]

    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    for rk in TARGET_RUN_KEYS:
        for cls in range(4):
            m = (run_key == rk) & (y_four == cls)
            if not np.any(m):
                continue
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                c=[class_colors[cls]],
                marker=markers.get(rk, "o"),
                alpha=0.75,
                s=18,
                edgecolors="0.25",
                linewidths=0.25,
            )

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=class_colors[c],
            markeredgecolor="0.25",
            markersize=8,
            label=class_names[c],
        )
        for c in range(4)
    ]
    run_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[rk],
            linestyle="None",
            color="0.35",
            markerfacecolor="0.35",
            markeredgecolor="0.2",
            markersize=7,
            label=rk,
        )
        for rk in TARGET_RUN_KEYS
    ]
    ax.set_xlabel("LD1 (4-class pooled LDA)")
    ax.set_ylabel("LD2 (4-class pooled LDA)")
    ax.set_title(f"{task_key}: run × {spec['lda_outcome_short']}")
    leg_c = ax.legend(handles=class_handles, loc="upper left", fontsize=7, title="Class (run | outcome)")
    ax.add_artist(leg_c)
    ax.legend(handles=run_handles, loc="upper right", fontsize=8, title="Division")
    fig.tight_layout()
    fig.savefig(LDA_FOUR_CLASS_DIR / f"pooled_{task_key}_four_class_lda_ld1_vs_ld2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, axes = plt.subplots(1, 2, figsize=(11.5, 7.2), squeeze=False)
    loadings_panels: list[tuple[str, np.ndarray]] = [
        (f"LD1\n(top {LDA_TOP_K} by |weight|)", v_ld1),
        (f"LD2\n(top {LDA_TOP_K} by |weight|)", v_ld2),
    ]
    for i, (ax_b, (title, vec)) in enumerate(zip(axes.ravel(), loadings_panels)):
        _bar_topk_lda(ax_b, vec, title, show_ylabel=(i == 0))
    fig2.suptitle(f"{task_key} four-class LDA", y=1.0, fontsize=11)
    fig2.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    fig2.savefig(LDA_FOUR_CLASS_DIR / f"pooled_{task_key}_four_class_lda_top_loadings.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)



def main() -> None:
    DECODE_BINARY_DIR.mkdir(parents=True, exist_ok=True)
    PCA_DIR.mkdir(parents=True, exist_ok=True)
    LDA_DIR.mkdir(parents=True, exist_ok=True)
    LDA_POOLED_BINARY_DIR.mkdir(parents=True, exist_ok=True)
    LDA_FOUR_CLASS_DIR.mkdir(parents=True, exist_ok=True)

    X_by_run = {}
    ycond_by_run = {}
    for rk in TARGET_RUN_KEYS:
        Xr, yr = features_for_pack(rk)
        X_by_run[rk] = Xr
        ycond_by_run[rk] = yr

    n_neurons = next(iter(X_by_run.values())).shape[1]
    feature_sets = source_feature_sets(n_neurons)

    folds_all = []
    confusion_all = []

    def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
        yt = y_true.astype(int)
        yp = y_pred.astype(int)
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tp = int(np.sum((yt == 1) & (yp == 1)))
        return tn, fp, fn, tp

    for source_key in TARGET_RUN_KEYS:
        X_src = X_by_run[source_key]
        y_src_cond = ycond_by_run[source_key]
        for task_key in TASK_SPECS:
            y_src_bin, keep = build_binary_labels(y_src_cond, _cond_name_map(source_key), task_key)
            X = X_src[keep]
            y = y_src_bin[keep]
            if len(np.unique(y)) < 2:
                continue
            min_class = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
            n_splits = min(N_FOLDS, max(2, min_class))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

            for fold_idx, (tr, te) in enumerate(cv.split(X, y), start=1):
                for clf_name, cols in feature_sets:
                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(X[tr][:, cols])
                    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
                    clf.fit(Xtr, y[tr])

                    Xte = scaler.transform(X[te][:, cols])
                    yhat_src = clf.predict(Xte)
                    acc_src = float(np.mean(yhat_src == y[te]))
                    folds_all.append(
                        {"source": source_key, "target": source_key, "task": task_key, "classifier": clf_name, "n_features": len(cols), "fold": fold_idx, "accuracy": acc_src}
                    )
                    tn, fp, fn, tp = _binary_counts(y[te], yhat_src)
                    confusion_all.append(
                        {
                            "source": source_key,
                            "target": source_key,
                            "task": task_key,
                            "classifier": clf_name,
                            "n_features": len(cols),
                            "fold": fold_idx,
                            "tn": tn,
                            "fp": fp,
                            "fn": fn,
                            "tp": tp,
                        }
                    )

                    for tgt_key in TARGET_RUN_KEYS:
                        if tgt_key == source_key:
                            continue
                        Xt_raw = X_by_run[tgt_key]
                        yt_cond = ycond_by_run[tgt_key]
                        yb_t, keep_t = build_binary_labels(yt_cond, _cond_name_map(tgt_key), task_key)
                        Xt = Xt_raw[keep_t]
                        ybt = yb_t[keep_t]
                        if len(ybt) == 0 or len(np.unique(ybt)) < 2:
                            continue
                        yhat_t = clf.predict(scaler.transform(Xt[:, cols]))
                        acc_t = float(np.mean(yhat_t == ybt))
                        folds_all.append(
                            {"source": source_key, "target": tgt_key, "task": task_key, "classifier": clf_name, "n_features": len(cols), "fold": fold_idx, "accuracy": acc_t}
                        )
                        tn, fp, fn, tp = _binary_counts(ybt, yhat_t)
                        confusion_all.append(
                            {
                                "source": source_key,
                                "target": tgt_key,
                                "task": task_key,
                                "classifier": clf_name,
                                "n_features": len(cols),
                                "fold": fold_idx,
                                "tn": tn,
                                "fp": fp,
                                "fn": fn,
                                "tp": tp,
                            }
                        )

    if not folds_all:
        raise RuntimeError("No binary decoding results produced.")
    folds_df = pd.DataFrame(folds_all)
    agg_df = (
        folds_df.groupby(["source", "target", "task", "classifier", "n_features"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"), n_folds=("accuracy", "count"))
        .fillna({"std_accuracy": 0.0})
    )
    agg_df["sem_accuracy"] = agg_df["std_accuracy"] / np.sqrt(agg_df["n_folds"].clip(lower=1))
    conf_df = pd.DataFrame(confusion_all)
    conf_agg = conf_df.groupby(["source", "target", "task", "classifier", "n_features"], as_index=False)[["tn", "fp", "fn", "tp"]].sum()
    folds_df.to_csv(DECODE_BINARY_DIR / "fold_level_accuracy.csv", index=False)
    agg_df.to_csv(DECODE_BINARY_DIR / "aggregated_accuracy_mean_sem.csv", index=False)
    conf_df.to_csv(DECODE_BINARY_DIR / "confusion_counts_fold_level.csv", index=False)
    conf_agg.to_csv(DECODE_BINARY_DIR / "confusion_counts_aggregated.csv", index=False)
    for source_key in TARGET_RUN_KEYS:
        plot_accuracy_for_source_combined(agg_df, source_key)
    for source_key in TARGET_RUN_KEYS:
        for task_key in TASK_SPECS:
            plot_pca_source_by_task(source_key, task_key, X_by_run, ycond_by_run)
    for task_key in TASK_SPECS:
        plot_pca_pooled_by_task(task_key, X_by_run, ycond_by_run)
    for task_key in TASK_SPECS:
        export_lda_pooled_task(task_key, X_by_run, ycond_by_run)
    for task_key in TASK_SPECS:
        export_lda_four_class_pooled_task(task_key, X_by_run, ycond_by_run)
    summary = [
        "Livneh binary decoding summary",
        f"Frame window: [{TRAIN_T0}, {TRAIN_T1})",
        f"Tasks: {', '.join(TASK_SPECS.keys())}",
        f"Sources: {', '.join(TARGET_RUN_KEYS)} (models trained per source run)",
        f"Targets: {', '.join(sorted(set(folds_df['target'])))}",
        f"Saved: {DECODE_BINARY_DIR / 'fold_level_accuracy.csv'}",
        f"Saved: {DECODE_BINARY_DIR / 'aggregated_accuracy_mean_sem.csv'}",
        f"Saved: {DECODE_BINARY_DIR / 'confusion_counts_fold_level.csv'}",
        f"Saved: {DECODE_BINARY_DIR / 'confusion_counts_aggregated.csv'}",
        f"Saved: per-source transfer plots for {', '.join(TARGET_RUN_KEYS)}",
        f"Saved: {PCA_DIR / 'pooled_hedonic_valence_pca.png'}",
        f"Saved: {PCA_DIR / 'pooled_consumption_mode_pca.png'}",
        f"Two-binary pooled LDA (reference vs other phases × task LD1), subdir: {LDA_POOLED_BINARY_DIR}",
        f"Saved: {LDA_POOLED_BINARY_DIR / 'pooled_hedonic_valence_lda_loadings.csv'}",
        f"Saved: {LDA_POOLED_BINARY_DIR / 'pooled_hedonic_valence_lda_run_vs_valence.png'}",
        f"Saved: {LDA_POOLED_BINARY_DIR / 'pooled_hedonic_valence_lda_top_loadings.png'}",
        f"Saved: {LDA_POOLED_BINARY_DIR / 'pooled_consumption_mode_lda_loadings.csv'}",
        f"Saved: {LDA_POOLED_BINARY_DIR / 'pooled_consumption_mode_lda_run_vs_consumption_mode.png'}",
        f"Saved: {LDA_POOLED_BINARY_DIR / 'pooled_consumption_mode_lda_top_loadings.png'}",
        f"Four-class LDA (reference vs other phases × binary outcome), subdir: {LDA_FOUR_CLASS_DIR}",
        f"Saved: {LDA_FOUR_CLASS_DIR / 'pooled_hedonic_valence_four_class_lda_class_key.csv'}",
        f"Saved: {LDA_FOUR_CLASS_DIR / 'pooled_hedonic_valence_four_class_lda_ld1_vs_ld2.png'}",
        f"Saved: {LDA_FOUR_CLASS_DIR / 'pooled_hedonic_valence_four_class_lda_loadings.csv'}",
        f"Saved: {LDA_FOUR_CLASS_DIR / 'pooled_consumption_mode_four_class_lda_class_key.csv'}",
        f"Saved: {LDA_FOUR_CLASS_DIR / 'pooled_consumption_mode_four_class_lda_ld1_vs_ld2.png'}",
        f"Saved: {LDA_FOUR_CLASS_DIR / 'pooled_consumption_mode_four_class_lda_loadings.csv'}",
    ]
    (DECODE_BINARY_DIR / "summary.txt").write_text("\n".join(summary))
    print(f"Wrote {DECODE_BINARY_DIR}")


if __name__ == "__main__":
    main()
