"""Four-class LDA top-neuron deep-dive plots (run after lib.lda.decode.main)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lib.lda.decode as dcb
from lib.config import LDA_FOUR_CLASS_DIR, LDA_OUT
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score


TOP_DIVE_DIR = LDA_OUT / "top_neuron_dive"
TOP_KS = (5, 10, 20, 30)
TOP_NEURONS_FOR_ACTIVITY = 12
RANDOM_BASELINE_REPEATS = 40
RANDOM_STATE = 42


def _keep_only_core_four_class_outputs() -> None:
    """Drop stray files; keep PNGs and CSVs needed by run_trajectories / downstream."""
    keep = {
        "pooled_consumption_mode_four_class_lda_ld1_vs_ld2.png",
        "pooled_consumption_mode_four_class_lda_top_loadings.png",
        "pooled_hedonic_valence_four_class_lda_ld1_vs_ld2.png",
        "pooled_hedonic_valence_four_class_lda_top_loadings.png",
        "pooled_consumption_mode_four_class_lda_class_key.csv",
        "pooled_consumption_mode_four_class_lda_class_counts.csv",
        "pooled_consumption_mode_four_class_lda_loadings.csv",
        "pooled_hedonic_valence_four_class_lda_class_key.csv",
        "pooled_hedonic_valence_four_class_lda_class_counts.csv",
        "pooled_hedonic_valence_four_class_lda_loadings.csv",
    }
    for path in LDA_FOUR_CLASS_DIR.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()


def _load_top_neurons(task_key: str, k_max: int) -> dict[str, np.ndarray]:
    loadings_path = LDA_FOUR_CLASS_DIR / f"pooled_{task_key}_four_class_lda_loadings.csv"
    df = pd.read_csv(loadings_path)
    idx = df["neuron_index"].to_numpy(dtype=np.int64)
    ld1 = df["lda_ld1_loading"].to_numpy(dtype=np.float64)
    ld2 = df["lda_ld2_loading"].to_numpy(dtype=np.float64)
    top_ld1 = idx[np.argsort(-np.abs(ld1))[:k_max]]
    top_ld2 = idx[np.argsort(-np.abs(ld2))[:k_max]]
    return {"LD1": top_ld1, "LD2": top_ld2}


def _pooled_feature_pack(task_key: str, X_by_run: dict[str, np.ndarray], ycond_by_run: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed = dcb.pooled_xy_for_four_class_lda(task_key, X_by_run, ycond_by_run)
    if packed is None:
        raise RuntimeError(f"No pooled four-class data for task={task_key}.")
    Xz, y_four, run_key = packed
    return Xz.astype(np.float64), y_four.astype(np.int64), run_key


def _trial_segments_by_task(task_key: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    segs: list[np.ndarray] = []
    y_bins: list[int] = []
    run_names: list[str] = []
    for rk in dcb.TARGET_RUN_KEYS:
        pack = dcb.load_npz_pack(rk)
        yb, keep = dcb.build_binary_labels(pack["trial_type"], dcb._cond_name_map(rk), task_key)
        keep_idx = np.flatnonzero(keep)
        if keep_idx.size == 0:
            continue
        yk = yb[keep].astype(np.int64)
        for local_i, trial_i in enumerate(keep_idx):
            segs.append(np.asarray(pack["segments"][int(trial_i)], dtype=np.float64))
            y_bins.append(int(yk[local_i]))
            run_names.append(rk)
    if not segs:
        raise RuntimeError(f"No trial segments available for task={task_key}.")
    return segs, np.asarray(y_bins, dtype=np.int64), np.asarray(run_names, dtype=object)


def _plot_activity_profiles(task_key: str, task_dir: Path, top_neurons: dict[str, np.ndarray]) -> None:
    segments, y_bin, run_name = _trial_segments_by_task(task_key)
    combos = [(rk, cls) for rk in dcb.TARGET_RUN_KEYS for cls in (0, 1)]
    class_names = {0: dcb.TASK_SPECS[task_key]["class0_name"], 1: dcb.TASK_SPECS[task_key]["class1_name"]}
    combo_labels = [f"{rk} | {class_names[cls]}" for rk, cls in combos]
    class_colors = {0: "#b22222", 1: "#0b3d91"}
    run_colors = {"run2": "#2ca02c", "run3_4": "#9467bd", "run5": "#ff7f0e"}
    run_styles = {"run2": "-", "run3_4": "--", "run5": ":"}
    class_styles = {0: "-", 1: "--"}

    axis_payload: dict[str, dict[str, object]] = {}
    for axis_name in ("LD1", "LD2"):
        top_idx = np.asarray(top_neurons[axis_name][:TOP_NEURONS_FOR_ACTIVITY], dtype=np.int64)
        if top_idx.size == 0:
            continue

        # Heatmap: neuron x (run,class), value is train-window mean response.
        heat = np.full((top_idx.size, len(combos)), np.nan, dtype=np.float64)
        traces: list[np.ndarray] = []
        max_trace_len = 0
        for c_i, (rk, cls) in enumerate(combos):
            idxs = np.flatnonzero((run_name == rk) & (y_bin == cls))
            if idxs.size == 0:
                traces.append(np.array([], dtype=np.float64))
                continue
            win_vals = []
            trial_traces = []
            for ii in idxs:
                seg = segments[int(ii)]
                t1 = min(dcb.TRAIN_T1, seg.shape[1])
                if dcb.TRAIN_T0 < t1:
                    win_vals.append(seg[top_idx, dcb.TRAIN_T0:t1].mean(axis=1))
                else:
                    win_vals.append(np.full(top_idx.size, np.nan, dtype=np.float64))
                trial_traces.append(seg[top_idx, :].mean(axis=0))
            heat[:, c_i] = np.nanmean(np.vstack(win_vals), axis=0)

            local_max = max(len(v) for v in trial_traces)
            buf = np.full((len(trial_traces), local_max), np.nan, dtype=np.float64)
            for r_i, tr in enumerate(trial_traces):
                buf[r_i, : len(tr)] = tr
            combo_trace = np.nanmean(buf, axis=0)
            traces.append(combo_trace)
            max_trace_len = max(max_trace_len, combo_trace.size)

        axis_payload[axis_name] = {"top_idx": top_idx, "heat": heat, "traces": traces, "max_trace_len": max_trace_len}

    if not axis_payload:
        return

    def _axis_code_suffix(ax_name: str) -> str:
        if task_key == "hedonic_valence":
            if ax_name == "LD1":
                return "(codes run2 vs run3_4+run5)"
            return "(codes aversive vs appetitive)"
        return ""

    # Figure 1: both left panels (heatmaps) together.
    fig_h, axes_h = plt.subplots(1, 2, figsize=(13.2, 5.0))
    for i, axis_name in enumerate(("LD1", "LD2")):
        ax = axes_h[i]
        if axis_name not in axis_payload:
            ax.axis("off")
            continue
        payload = axis_payload[axis_name]
        top_idx = np.asarray(payload["top_idx"], dtype=np.int64)
        heat = np.asarray(payload["heat"], dtype=np.float64)
        im = ax.imshow(heat, aspect="auto", interpolation="nearest")
        ax.set_title(f"{task_key} {axis_name}: top-{top_idx.size} window mean {_axis_code_suffix(axis_name)}")
        ax.set_yticks(np.arange(top_idx.size))
        ax.set_yticklabels([str(int(v)) for v in top_idx], fontsize=7)
        ax.set_xticks(np.arange(len(combo_labels)))
        ax.set_xticklabels(combo_labels, rotation=35, ha="right", fontsize=8)
        if i == 0:
            ax.set_ylabel("Neuron index")
        fig_h.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig_h.tight_layout()
    fig_h.savefig(task_dir / f"{task_key}_ld1_ld2_activity_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig_h)

    # Figure 2: both right panels (traces) together.
    fig_t, axes_t = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    for i, axis_name in enumerate(("LD1", "LD2")):
        ax = axes_t[i]
        if axis_name not in axis_payload:
            ax.axis("off")
            continue
        payload = axis_payload[axis_name]
        traces = payload["traces"]
        max_trace_len = int(payload["max_trace_len"])
        t = np.arange(max_trace_len, dtype=np.int64)
        for c_i, (rk, cls) in enumerate(combos):
            tr = traces[c_i]
            if tr.size == 0 or np.isnan(tr).all():
                continue
            line_color = run_colors.get(rk, "#333333") if axis_name == "LD1" else class_colors[cls]
            line_style = class_styles[cls] if axis_name == "LD1" else run_styles.get(rk, "-")
            ax.plot(
                t[: tr.size],
                tr,
                lw=1.7,
                alpha=0.90,
                color=line_color,
                linestyle=line_style,
                label=f"{rk} | {class_names[cls]}",
            )
        ax.axvspan(dcb.TRAIN_T0, dcb.TRAIN_T1, alpha=0.12)
        ax.set_title(f"{axis_name} {_axis_code_suffix(axis_name)}")
        ax.set_xlabel("Frame")
        if i == 0:
            ax.set_ylabel("Mean activity")
        ax.legend(fontsize=7, ncol=2, title="Run | Class")
    fig_t.suptitle("Hedonic valence: mean activity traces of top LD-selected neurons", y=1.03, fontsize=11)
    fig_t.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig_t.savefig(task_dir / f"{task_key}_ld1_ld2_activity_traces.png", dpi=150, bbox_inches="tight")
    plt.close(fig_t)


def _plot_hedonic_ld_axis_summary(task_dir: Path, Xz: np.ndarray, y: np.ndarray) -> None:
    """More intuitive LD summary: class centroids on LD1 and LD2."""
    lda = LinearDiscriminantAnalysis(solver="svd", n_components=2)
    xy = lda.fit_transform(Xz, y)
    labels = dcb.four_class_labels("hedonic_valence")

    # y coding follows decode_cross_binary.four_class_labels order:
    # 0 run2|Aversive, 1 run2|Appetitive, 2 run3_4+run5|Aversive, 3 run3_4+run5|Appetitive
    run_colors = {0: "#2ca02c", 1: "#9467bd", 2: "#2ca02c", 3: "#9467bd"}  # run2 vs run3_4+run5
    valence_hatch = {0: "///", 1: "\\\\\\", 2: "///", 3: "\\\\\\"}  # aversive vs appetitive

    means = []
    sems = []
    for cls in range(4):
        m = y == cls
        if np.any(m):
            means.append(np.nanmean(xy[m, :], axis=0))
            if int(np.sum(m)) > 1:
                sems.append(np.nanstd(xy[m, :], axis=0, ddof=1) / np.sqrt(int(np.sum(m))))
            else:
                sems.append(np.zeros(2, dtype=float))
        else:
            means.append(np.array([np.nan, np.nan], dtype=float))
            sems.append(np.array([np.nan, np.nan], dtype=float))
    mu = np.vstack(means)
    se = np.vstack(sems)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=False)
    for ax_i, ax_name in enumerate(("LD1", "LD2")):
        ax = axes[ax_i]
        x = np.arange(4, dtype=float)
        for cls in range(4):
            ax.bar(
                x[cls],
                mu[cls, ax_i],
                yerr=se[cls, ax_i],
                color=run_colors[cls],
                edgecolor="black",
                linewidth=0.8,
                hatch=valence_hatch[cls],
                capsize=3,
            )
        ax.axhline(0.0, color="0.4", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(f"{ax_name} score")
        ax.set_title(f"Hedonic valence: class means on {ax_name}")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Intuitive LD summary (hedonic valence)\n"
        "Bar color = run bucket (run2 vs run3_4+run5), hatch = valence (aversive vs appetitive)",
        y=1.03,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(task_dir / "hedonic_valence_ld_axis_intuitive_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cv_scores(X: np.ndarray, y: np.ndarray, cols: np.ndarray, cv: StratifiedKFold) -> np.ndarray:
    clf = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)
    return cross_val_score(clf, X[:, cols], y, cv=cv, scoring="accuracy")


def _labels_from_y_four(y_four: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From 4-class labels -> (run-bucket binary, valence binary)."""
    y_four = np.asarray(y_four, dtype=np.int64).ravel()
    # 0: run2|Aversive, 1: run2|Appetitive, 2: run3_4+run5|Aversive, 3: run3_4+run5|Appetitive
    y_run = np.where(np.isin(y_four, [0, 1]), 0, 1).astype(np.int64)   # 0=run2, 1=run3_4+run5
    y_stim = np.where(np.isin(y_four, [0, 2]), 0, 1).astype(np.int64)  # 0=Aversive, 1=Appetitive
    return y_run, y_stim


def _plot_binary_axis_decoding_benchmarks(
    task_key: str,
    task_dir: Path,
    Xz: np.ndarray,
    y_four: np.ndarray,
    top_neurons: dict[str, np.ndarray],
) -> None:
    """Two binary benchmarks: run from LD1 and valence from LD2."""
    y_run, y_stim = _labels_from_y_four(y_four)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_idx = np.arange(Xz.shape[1], dtype=np.int64)

    panels = [
        ("LD1", y_run, "run bucket", ("run2", "run3_4+run5")),
        ("LD2", y_stim, "stimulus valence", ("Aversive", "Appetitive")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, (axis_name, y_bin, target_name, class_names) in zip(axes, panels):
        top_order = np.asarray(top_neurons[axis_name], dtype=np.int64)
        top_means, top_sems = [], []
        rand_means, rand_sems = [], []
        comp_match_means, comp_match_sems = [], []

        for k in TOP_KS:
            k_eff = min(int(k), int(top_order.size))
            idx_top = top_order[:k_eff]
            idx_comp = np.setdiff1d(all_idx, idx_top, assume_unique=False)

            scores_top = _cv_scores(Xz, y_bin, idx_top, cv)
            rng = np.random.default_rng(RANDOM_STATE + k_eff + (101 if axis_name == "LD2" else 0))

            rand_scores = []
            comp_match_scores = []
            for _ in range(RANDOM_BASELINE_REPEATS):
                idx_rand = rng.choice(all_idx, size=k_eff, replace=False)
                idx_comp_match = rng.choice(idx_comp, size=k_eff, replace=False)
                rand_scores.append(float(_cv_scores(Xz, y_bin, idx_rand, cv).mean()))
                comp_match_scores.append(float(_cv_scores(Xz, y_bin, idx_comp_match, cv).mean()))

            top_means.append(float(scores_top.mean()))
            top_sems.append(float(scores_top.std(ddof=0) / np.sqrt(len(scores_top))))
            rand_arr = np.asarray(rand_scores, dtype=np.float64)
            comp_arr = np.asarray(comp_match_scores, dtype=np.float64)
            rand_means.append(float(rand_arr.mean()))
            rand_sems.append(float(rand_arr.std(ddof=0)))
            comp_match_means.append(float(comp_arr.mean()))
            comp_match_sems.append(float(comp_arr.std(ddof=0)))

        ks = np.asarray(TOP_KS, dtype=np.int64)
        tm, ts = np.asarray(top_means), np.asarray(top_sems)
        rm, rs = np.asarray(rand_means), np.asarray(rand_sems)
        cm, cs = np.asarray(comp_match_means), np.asarray(comp_match_sems)

        ax.plot(ks, tm, marker="o", label=f"Top-K ({axis_name})")
        ax.fill_between(ks, tm - ts, tm + ts, alpha=0.2)
        ax.plot(ks, rm, marker="^", label="Random-K")
        ax.fill_between(ks, rm - rs, rm + rs, alpha=0.2)
        ax.plot(ks, cm, marker="d", label="Complement matched-K")
        ax.fill_between(ks, cm - cs, cm + cs, alpha=0.2)
        ax.set_title(f"{task_key} {axis_name}: decode {target_name}")
        ax.set_xlabel("K top neurons")
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        ax.text(
            0.02,
            0.98,
            f"Classes: {class_names[0]} vs {class_names[1]}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
        )
    axes[0].set_ylabel("CV accuracy")
    axes[0].set_ylim(0.0, 1.0)
    fig.suptitle("Binary decoding benchmarks using LD-axis-ranked neurons", y=1.02, fontsize=10)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(task_dir / f"{task_key}_binary_ld_axis_decoding_benchmarks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_ld1_ld2_decoding(
    task_key: str,
    task_dir: Path,
    Xz: np.ndarray,
    y_four: np.ndarray,
    top_neurons: dict[str, np.ndarray],
) -> None:
    """4-class decoding with union(top-K LD1, top-K LD2) vs matched random total-K."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_idx = np.arange(Xz.shape[1], dtype=np.int64)
    top_ld1 = np.asarray(top_neurons["LD1"], dtype=np.int64)
    top_ld2 = np.asarray(top_neurons["LD2"], dtype=np.int64)

    k_pairs = [5, 10, 20, 30]
    total_ks = []
    top_means, top_sems = [], []
    rand_means, rand_sems = [], []
    comp_match_means, comp_match_sems = [], []

    for k in k_pairs:
        k1 = min(int(k), int(top_ld1.size))
        k2 = min(int(k), int(top_ld2.size))
        union_idx = np.union1d(top_ld1[:k1], top_ld2[:k2]).astype(np.int64)
        total_k = int(union_idx.size)
        total_ks.append(total_k)
        if total_k == 0:
            top_means.append(np.nan)
            top_sems.append(np.nan)
            rand_means.append(np.nan)
            rand_sems.append(np.nan)
            comp_match_means.append(np.nan)
            comp_match_sems.append(np.nan)
            continue

        scores_top = _cv_scores(Xz, y_four, union_idx, cv)
        comp_idx = np.setdiff1d(all_idx, union_idx, assume_unique=False)
        rng = np.random.default_rng(RANDOM_STATE + 700 + total_k)
        rand_scores = []
        comp_match_scores = []
        for _ in range(RANDOM_BASELINE_REPEATS):
            idx_rand = rng.choice(all_idx, size=total_k, replace=False)
            rand_scores.append(float(_cv_scores(Xz, y_four, idx_rand, cv).mean()))
            if comp_idx.size >= total_k:
                idx_comp = rng.choice(comp_idx, size=total_k, replace=False)
                comp_match_scores.append(float(_cv_scores(Xz, y_four, idx_comp, cv).mean()))

        top_means.append(float(scores_top.mean()))
        top_sems.append(float(scores_top.std(ddof=0) / np.sqrt(len(scores_top))))
        rand_arr = np.asarray(rand_scores, dtype=np.float64)
        rand_means.append(float(rand_arr.mean()))
        rand_sems.append(float(rand_arr.std(ddof=0)))
        if comp_match_scores:
            comp_arr = np.asarray(comp_match_scores, dtype=np.float64)
            comp_match_means.append(float(comp_arr.mean()))
            comp_match_sems.append(float(comp_arr.std(ddof=0)))
        else:
            comp_match_means.append(np.nan)
            comp_match_sems.append(np.nan)

    x = np.arange(len(k_pairs), dtype=float)
    tick_labels = [f"{k}+{k} -> {tk}" for k, tk in zip(k_pairs, total_ks)]
    tm, ts = np.asarray(top_means), np.asarray(top_sems)
    rm, rs = np.asarray(rand_means), np.asarray(rand_sems)
    cm, cs = np.asarray(comp_match_means), np.asarray(comp_match_sems)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(x, tm, marker="o", lw=1.8, label="Union(Top-K LD1, Top-K LD2)")
    ax.fill_between(x, tm - ts, tm + ts, alpha=0.2)
    ax.plot(x, rm, marker="^", lw=1.6, label="Random total-K")
    ax.fill_between(x, rm - rs, rm + rs, alpha=0.2)
    ax.plot(x, cm, marker="d", lw=1.6, label="Complement matched-total-K")
    ax.fill_between(x, cm - cs, cm + cs, alpha=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Top-K per axis (resulting total unique neurons)")
    ax.set_ylabel("CV accuracy (4-class)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{task_key}: combined LD1+LD2 top-neuron decoding")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    ax.text(
        0.01,
        0.01,
        "4 pooled classes: run2|Aversive, run2|Appetitive, run3_4+run5|Aversive, run3_4+run5|Appetitive",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
    )
    fig.tight_layout()
    fig.savefig(task_dir / f"{task_key}_combined_ld1_ld2_topk_decoding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_topk_decoding(task_key: str, task_dir: Path, Xz: np.ndarray, y: np.ndarray, top_neurons: dict[str, np.ndarray]) -> tuple[str, int, np.ndarray]:
    n_neurons = Xz.shape[1]
    all_idx = np.arange(n_neurons, dtype=np.int64)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_axis = "LD1"
    best_k = int(TOP_KS[0])
    best_score = -np.inf
    best_cols = top_neurons[best_axis][:best_k]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for a_i, axis_name in enumerate(("LD1", "LD2")):
        top_order = np.asarray(top_neurons[axis_name], dtype=np.int64)
        top_means, top_sems = [], []
        rand_means, rand_sems = [], []
        comp_match_means, comp_match_sems = [], []

        for k in TOP_KS:
            k_eff = min(int(k), int(top_order.size))
            idx_top = top_order[:k_eff]
            idx_comp = np.setdiff1d(all_idx, idx_top, assume_unique=False)

            scores_top = _cv_scores(Xz, y, idx_top, cv)

            rng = np.random.default_rng(RANDOM_STATE + k_eff + (11 if axis_name == "LD2" else 0))
            rand_scores = []
            comp_match_scores = []
            for _ in range(RANDOM_BASELINE_REPEATS):
                idx_rand = rng.choice(all_idx, size=k_eff, replace=False)
                idx_comp_match = rng.choice(idx_comp, size=k_eff, replace=False)
                rand_scores.append(float(_cv_scores(Xz, y, idx_rand, cv).mean()))
                comp_match_scores.append(float(_cv_scores(Xz, y, idx_comp_match, cv).mean()))

            top_means.append(float(scores_top.mean()))
            top_sems.append(float(scores_top.std(ddof=0) / np.sqrt(len(scores_top))))
            rand_arr = np.asarray(rand_scores, dtype=np.float64)
            comp_match_arr = np.asarray(comp_match_scores, dtype=np.float64)
            rand_means.append(float(rand_arr.mean()))
            rand_sems.append(float(rand_arr.std(ddof=0)))
            comp_match_means.append(float(comp_match_arr.mean()))
            comp_match_sems.append(float(comp_match_arr.std(ddof=0)))

            if top_means[-1] > best_score:
                best_score = top_means[-1]
                best_axis = axis_name
                best_k = k_eff
                best_cols = idx_top.copy()

        ks = np.array(TOP_KS, dtype=np.int64)
        ax = axes[a_i]
        top_means_arr = np.asarray(top_means)
        top_sems_arr = np.asarray(top_sems)
        rand_means_arr = np.asarray(rand_means)
        rand_sems_arr = np.asarray(rand_sems)
        comp_match_means_arr = np.asarray(comp_match_means)
        comp_match_sems_arr = np.asarray(comp_match_sems)

        ax.plot(ks, top_means_arr, marker="o", label="Top-K")
        ax.fill_between(ks, top_means_arr - top_sems_arr, top_means_arr + top_sems_arr, alpha=0.2)
        ax.plot(ks, rand_means_arr, marker="^", label="Random-K")
        ax.fill_between(ks, rand_means_arr - rand_sems_arr, rand_means_arr + rand_sems_arr, alpha=0.2)
        ax.plot(ks, comp_match_means_arr, marker="d", label="Complement matched-K")
        ax.fill_between(ks, comp_match_means_arr - comp_match_sems_arr, comp_match_means_arr + comp_match_sems_arr, alpha=0.2)

        ax.set_title(f"{task_key} {axis_name}: decoding benchmarks")
        ax.set_xlabel("K top neurons")
        if a_i == 0:
            ax.set_ylabel("CV accuracy")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(ks)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=7)

    class_note = "Decoding 4 pooled classes: run2|Aversive, run2|Appetitive, run3_4+run5|Aversive, run3_4+run5|Appetitive"
    feat_note = "Features: per-trial mean activity in frames 60-120 (one feature per neuron)"
    fig.suptitle(f"{class_note}\n{feat_note}", y=1.03, fontsize=9)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.savefig(task_dir / f"{task_key}_topk_decoding_benchmarks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return best_axis, best_k, best_cols


def _plot_confusion_and_errors(task_key: str, task_dir: Path, Xz: np.ndarray, y: np.ndarray, run_key: np.ndarray, best_axis: str, best_k: int, best_cols: np.ndarray) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    clf = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)
    y_pred = cross_val_predict(clf, Xz[:, best_cols], y, cv=cv, method="predict")
    labels = np.arange(4, dtype=np.int64)
    class_names = dcb.four_class_labels(task_key)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    axes_flat = axes.ravel()
    partitions = [("pooled", np.ones_like(y, dtype=bool))] + [(rk, run_key == rk) for rk in dcb.TARGET_RUN_KEYS]
    for ax, (name, mask) in zip(axes_flat, partitions):
        cm = confusion_matrix(y[mask], y_pred[mask], labels=labels)
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(name)
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{task_key} confusion (best top-K: {best_axis}, K={best_k})", y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(task_dir / f"{task_key}_confusion_best_topk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm_pool = confusion_matrix(y, y_pred, labels=labels)
    pair_labels = []
    pair_vals = []
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            pair_labels.append(f"{class_names[i]} -> {class_names[j]}")
            pair_vals.append(int(cm_pool[i, j]))
    order = np.argsort(-np.asarray(pair_vals))[:8]
    top_labels = [pair_labels[int(i)] for i in order]
    top_vals = [pair_vals[int(i)] for i in order]
    fig2, ax2 = plt.subplots(figsize=(10.0, 4.2))
    y_pos = np.arange(len(top_labels))
    ax2.barh(y_pos, top_vals)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_labels, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Misclassification count")
    ax2.set_title(f"{task_key} pooled error concentration (best {best_axis} top-{best_k})")
    ax2.grid(True, axis="x", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(task_dir / f"{task_key}_error_concentration_best_topk.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


def _run_top_neuron_dive(X_by_run: dict[str, np.ndarray], ycond_by_run: dict[str, np.ndarray]) -> None:
    TOP_DIVE_DIR.mkdir(parents=True, exist_ok=True)
    max_k = max(TOP_KS + (TOP_NEURONS_FOR_ACTIVITY,))
    for task_key in dcb.TASK_SPECS:
        task_dir = TOP_DIVE_DIR / task_key
        task_dir.mkdir(parents=True, exist_ok=True)
        top_neurons = _load_top_neurons(task_key, max_k)
        Xz, y_four, run_key = _pooled_feature_pack(task_key, X_by_run, ycond_by_run)
        _plot_activity_profiles(task_key, task_dir, top_neurons)
        if task_key == "hedonic_valence":
            _plot_hedonic_ld_axis_summary(task_dir, Xz, y_four)
            _plot_binary_axis_decoding_benchmarks(task_key, task_dir, Xz, y_four, top_neurons)
            _plot_combined_ld1_ld2_decoding(task_key, task_dir, Xz, y_four, top_neurons)
        best_axis, best_k, best_cols = _plot_topk_decoding(task_key, task_dir, Xz, y_four, top_neurons)
        _plot_confusion_and_errors(task_key, task_dir, Xz, y_four, run_key, best_axis, best_k, best_cols)


def main() -> None:
    LDA_FOUR_CLASS_DIR.mkdir(parents=True, exist_ok=True)
    TOP_DIVE_DIR.mkdir(parents=True, exist_ok=True)

    X_by_run: dict[str, object] = {}
    ycond_by_run: dict[str, object] = {}
    for run_key in dcb.TARGET_RUN_KEYS:
        X_run, y_run = dcb.features_for_pack(run_key)
        X_by_run[run_key] = X_run
        ycond_by_run[run_key] = y_run

    _run_top_neuron_dive(X_by_run, ycond_by_run)
    _keep_only_core_four_class_outputs()

    print(f"Four-class LDA loadings: {LDA_FOUR_CLASS_DIR}")
    print(f"Top-neuron deep-dive PNGs: {TOP_DIVE_DIR}")


if __name__ == "__main__":
    main()
