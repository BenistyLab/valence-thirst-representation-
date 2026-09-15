# -*- coding: utf-8 -*-
"""Per-run condition averages: all neurons vs SLM-targeted subsets.

Writes figures to outputs/exploration/slm_activity/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib.config import EXPLORATION_OUT, EVAL_RUN_KEYS, default_session_dir
from lib.io.npz import load_npz_pack, safe_filename
from lib.explore.inspect_slm import load_slm_runs_detailed

OUT_DIR = EXPLORATION_OUT / "slm_activity"


def cond_label(cond_id: int, run_key: str) -> str:
    if cond_id == 3:
        return "water"
    if cond_id == 4:
        return "nacl"
    if cond_id == 18:
        return "airpuff"
    if cond_id == 26:
        return "slm"
    if run_key in ("run6", "run7") and cond_id == 7:
        return "waters"
    if run_key in ("run6", "run7") and cond_id == 8:
        return "waterfree"
    return f"cond_{cond_id}"


def _rois_1based_to_0based(rois: list[int], n_neurons: int) -> set[int]:
    out: set[int] = set()
    for r in rois:
        z = int(r) - 1
        if 0 <= z < n_neurons:
            out.add(z)
    return out


def slm_sets(session_dir: Path, n_neurons: int) -> tuple[np.ndarray, np.ndarray]:
    runs = load_slm_runs_detailed(session_dir)
    by_run: dict[int, list[int]] = {}
    for rr in runs:
        by_run[int(rr["run_number"])] = list(rr.get("targeted_rois") or [])
    s3 = _rois_1based_to_0based(by_run.get(3, []), n_neurons)
    s4 = _rois_1based_to_0based(by_run.get(4, []), n_neurons)
    s5 = _rois_1based_to_0based(by_run.get(5, []), n_neurons)
    slm34 = np.array(sorted(s3 | s4), dtype=int)
    slm5 = np.array(sorted(s5), dtype=int)
    return slm34, slm5


def mean_sem_from_traces(traces: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n_max = max((t.shape[0] for t in traces), default=0)
    mean_line = np.full(n_max, np.nan, dtype=float)
    sem_line = np.full(n_max, np.nan, dtype=float)
    for t in range(n_max):
        vals = [float(tr[t]) for tr in traces if tr.shape[0] > t]
        if vals:
            v = np.asarray(vals, dtype=float)
            mean_line[t] = float(np.mean(v))
            sem_line[t] = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return mean_line, sem_line


def _plot_line(ax, x: np.ndarray, m: np.ndarray, s: np.ndarray, color: str, label: str) -> None:
    valid = ~np.isnan(m)
    if not np.any(valid):
        return
    ax.plot(x[valid], m[valid], color=color, lw=1.8, label=label)
    sv = valid & ~np.isnan(s)
    if np.any(sv):
        ax.fill_between(x[sv], m[sv] - s[sv], m[sv] + s[sv], color=color, alpha=0.22)


def main() -> None:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get neuron count from first available run pack.
    n_neurons = None
    for rk in EVAL_RUN_KEYS:
        try:
            p = load_npz_pack(rk)
        except FileNotFoundError:
            continue
        segs = p["segments"]
        if segs:
            n_neurons = int(segs[0].shape[0])
            break
    if n_neurons is None:
        raise RuntimeError("No run packs found in normalized_data.")

    slm34_ix, slm5_ix = slm_sets(default_session_dir(), n_neurons)
    all_ix = np.arange(n_neurons, dtype=int)

    # (run_key, condition_label) -> subset key -> list of per-trial mean traces
    store: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}

    for rk in EVAL_RUN_KEYS:
        try:
            pack = load_npz_pack(rk)
        except FileNotFoundError:
            print(f"{rk}: missing pack, skipped")
            continue
        segs = pack["segments"]
        y = pack["trial_type"].astype(int)
        for i, seg in enumerate(segs):
            lab = cond_label(int(y[i]), rk)
            key = (rk, lab)
            if key not in store:
                store[key] = {"all_neurons": [], "slm_run3_4": [], "slm_run5": []}

            tr_all = np.mean(seg[all_ix, :], axis=0).astype(float)
            store[key]["all_neurons"].append(tr_all)

            if slm34_ix.size > 0:
                tr_34 = np.mean(seg[slm34_ix, :], axis=0).astype(float)
                store[key]["slm_run3_4"].append(tr_34)
            if slm5_ix.size > 0:
                tr_5 = np.mean(seg[slm5_ix, :], axis=0).astype(float)
                store[key]["slm_run5"].append(tr_5)

    if not store:
        raise RuntimeError("No trials found to plot.")

    summary_lines = [
        "Per-run condition averages: all neurons vs SLM subsets",
        f"output_dir: {out_dir}",
        f"output_dir: {out_dir}",
        f"runs_used: {', '.join(EVAL_RUN_KEYS)}",
        f"n_neurons: {n_neurons}",
        f"slm_run3_4_size: {int(slm34_ix.size)}",
        f"slm_run5_size: {int(slm5_ix.size)}",
        "",
    ]

    colors = {"all_neurons": "black", "slm_run3_4": "#1f77b4", "slm_run5": "#ff7f0e"}

    for rk, lab in sorted(store):
        all_m, all_s = mean_sem_from_traces(store[(rk, lab)]["all_neurons"])
        s34_m, s34_s = mean_sem_from_traces(store[(rk, lab)]["slm_run3_4"])
        s5_m, s5_s = mean_sem_from_traces(store[(rk, lab)]["slm_run5"])
        n_max = max(all_m.shape[0], s34_m.shape[0], s5_m.shape[0])
        x = np.arange(n_max)

        fig, ax = plt.subplots(figsize=(10, 5))
        _plot_line(ax, x, all_m, all_s, colors["all_neurons"], f"all_neurons (n={n_neurons})")
        _plot_line(ax, x, s34_m, s34_s, colors["slm_run3_4"], f"slm_run3_4 (n={slm34_ix.size})")
        _plot_line(ax, x, s5_m, s5_s, colors["slm_run5"], f"slm_run5 (n={slm5_ix.size})")

        ax.set_title(f"{rk} | {lab} | average response")
        ax.set_xlabel("Frames from cue")
        ax.set_ylabel("Mean z-scored activity")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out_path = out_dir / f"response_{safe_filename(rk)}_{safe_filename(lab)}_all_vs_slm34_vs_slm5.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        summary_lines.append(
            f"{rk} | {lab}: "
            f"trials(all/slm34/slm5)={len(store[(rk, lab)]['all_neurons'])}/"
            f"{len(store[(rk, lab)]['slm_run3_4'])}/{len(store[(rk, lab)]['slm_run5'])}"
        )

    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote plots under {out_dir}")


if __name__ == "__main__":
    main()

