"""
plot_mean_ff_results.py — Visualize OOF results from train_mean_ff_multhead.py.

Produces four figures:

  Figure 1 — Ternary simplex (pre trials through pre decoder)
  Figure 2 — Ternary simplex (SLM-water trials through pre decoder)
  Figure 3 — Ternary simplex (SLM-airpuff trials through pre decoder)
  Figure 4 — Grouped bar chart (accuracy across all trial-source × decoder-head pairs)

Usage (from python/):
    python plot_mean_ff_results.py
    python plot_mean_ff_results.py --latent-dim 64
    python plot_mean_ff_results.py --save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from lib import config as _cfg

# ── Constants ─────────────────────────────────────────────────────────────────

PHASES      = ("pre", "slm_water", "slm_airpuff")
PHASE_DISP  = {"pre": "Pre", "slm_water": "SLM-water", "slm_airpuff": "SLM-airpuff"}
LABELS      = ("water", "salt", "airpuff")
PROTOCOLS   = ("Mock", "1Hz", "20Hz")

COLORS = ["#4472C4", "#ED7D31", "#C00000"]

SQRT3_2 = np.sqrt(3) / 2

# ── Helpers ───────────────────────────────────────────────────────────────────

def _npz_path(mouse_id: str, latent_dim: int) -> Path:
    protocol   = _cfg.SESSION_PROTOCOL[mouse_id]
    mouse_name = _cfg.SESSION_MOUSE[mouse_id]
    return (
        Path(__file__).resolve().parent
        / "outputs" / protocol / mouse_name
        / f"mean_ff_multhead_enc{latent_dim}"
        / "mean_ff_oof_predictions.npz"
    )


def _load_session(mouse_id: str, latent_dim: int) -> dict | None:
    p = _npz_path(mouse_id, latent_dim)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    return {
        "mouse_id":   mouse_id,
        "protocol":   _cfg.SESSION_PROTOCOL[mouse_id],
        "phase_names": list(d["phase_names"]),
        "label_names": list(d["label_names"]),
        **{k: d[k] for k in d.files if k not in ("phase_names", "label_names")},
    }


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = [np.mean(y_pred[y_true == c] == c)
               for c in range(n_classes) if np.any(y_true == c)]
    return float(np.mean(recalls))


def _session_accuracies(sess: dict) -> dict[tuple[str, str], float]:
    out = {}
    phase_names = sess["phase_names"]
    for si, src in enumerate(phase_names):
        probs_key = f"{src}_probabilities_by_decoder"
        label_key = f"{src}_true_labels"
        probs_all = sess[probs_key]
        y_true    = sess[label_key]
        for di, dec in enumerate(phase_names):
            probs = probs_all[:, di, :]
            y_pred = probs.argmax(axis=1)
            out[(src, dec)] = float(np.mean(y_pred == y_true))
    return out


# ── Ternary helpers ───────────────────────────────────────────────────────────

def _to_xy(probs: np.ndarray) -> np.ndarray:
    p_w, p_s, p_a = probs[:, 0], probs[:, 1], probs[:, 2]
    x = p_s + 0.5 * p_a
    y = SQRT3_2 * p_a
    return np.stack([x, y], axis=1)


def _draw_ternary_frame(ax: plt.Axes) -> None:
    tri = np.array([[0, 0], [1, 0], [0.5, SQRT3_2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.2, zorder=4)

    grid_kw = dict(color="#cccccc", lw=0.6, zorder=2)
    for f in (1/3, 2/3):
        ax.plot([0.5*f, 1 - 0.5*f], [SQRT3_2*f, SQRT3_2*f], **grid_kw)
        ax.plot([f, f + 0.5*(1-f)], [0, SQRT3_2*(1-f)], **grid_kw)
        ax.plot([1-f, 0.5*(1-f)], [0, SQRT3_2*(1-f)], **grid_kw)

    kw = dict(fontsize=8.5, ha="center")
    ax.text(0,    -0.07,            "Water",   va="top",    **kw)
    ax.text(1,    -0.07,            "NaCl",    va="top",    **kw)
    ax.text(0.5,  SQRT3_2 + 0.05,  "Airpuff", va="bottom", **kw)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.14, SQRT3_2 + 0.10)


def _ternary_legend_and_note(fig: plt.Figure) -> None:
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[i],
               markersize=7, label=lbl.capitalize())
        for i, lbl in enumerate(("Water", "NaCl", "Airpuff"))
    ]
    legend_elements.append(Line2D([0], [0], color="w", label=""))
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=3, fontsize=8.5,
        frameon=False, bbox_to_anchor=(0.35, 0.01),
    )
    fig.text(0.62, 0.04,
             "Small dots = individual held-out trials; large diamonds = per-mouse class means.",
             ha="center", fontsize=8, color="#555555")


# ── Figure 1: Ternary (pre trials, pre decoder) ───────────────────────────────

def plot_ternary(sessions: list[dict], latent_dim: int, save_dir: Path | None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle(
        f"Direct {latent_dim}D: pre-run trials through the pre-run decoder",
        fontsize=13, fontweight="bold", y=0.97,
    )
    fig.text(0.5, 0.91,
             "Five-fold out-of-fold predictions; color is the true trial class.",
             ha="center", fontsize=8.5, color="#555555")

    by_protocol: dict[str, list[dict]] = {p: [] for p in PROTOCOLS}
    for s in sessions:
        by_protocol[s["protocol"]].append(s)

    for ax, protocol in zip(axes, PROTOCOLS):
        proto_sessions = by_protocol[protocol]
        _draw_ternary_frame(ax)

        all_ba, all_n = [], 0
        for sess in proto_sessions:
            probs_all = sess["pre_probabilities_by_decoder"]
            probs     = probs_all[:, 0, :]
            y_true    = sess["pre_true_labels"]
            y_pred    = probs.argmax(axis=1)

            xy = _to_xy(probs)
            all_n += len(y_true)
            all_ba.append(_balanced_accuracy(y_true, y_pred, len(LABELS)))

            for ci, color in enumerate(COLORS):
                mask = y_true == ci
                if mask.any():
                    ax.scatter(xy[mask, 0], xy[mask, 1],
                               c=color, s=14, alpha=0.35,
                               linewidths=0, zorder=3)

            for ci, color in enumerate(COLORS):
                mask = y_true == ci
                if mask.any():
                    mu = xy[mask].mean(axis=0)
                    ax.scatter(mu[0], mu[1],
                               c=color, s=110, marker="D",
                               edgecolors="white", linewidths=0.8,
                               zorder=5)

        ba_mean = float(np.mean(all_ba)) if all_ba else float("nan")
        ax.set_title(
            f"{protocol}  |  BA={ba_mean*100:.1f}%  |  N={all_n}",
            fontsize=10, fontweight="bold", pad=10,
        )

    _ternary_legend_and_note(fig)
    plt.tight_layout(rect=[0, 0.07, 1, 0.91])

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"ternary_pre_direct{latent_dim}.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved ternary figure → {save_dir / f'ternary_pre_direct{latent_dim}.png'}")
    else:
        plt.show()
    plt.close(fig)


# ── Figures 2 & 3: Ternary cross-phase ───────────────────────────────────────

def plot_ternary_cross(
    sessions: list[dict],
    source_phase: str,
    decoder_phase: str,
    latent_dim: int,
    save_dir: Path | None,
) -> None:
    """
    Ternary plot: trials from `source_phase` evaluated by the `decoder_phase` head.

    Parameters
    ----------
    source_phase :  which trials to show  (e.g. "slm_water")
    decoder_phase : which decoder head    (e.g. "pre")
    """
    decoder_idx = list(PHASES).index(decoder_phase)
    src_disp    = PHASE_DISP[source_phase]
    dec_disp    = PHASE_DISP[decoder_phase]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle(
        f"Direct {latent_dim}D: {src_disp} trials through the {dec_disp} decoder",
        fontsize=13, fontweight="bold", y=0.97,
    )
    fig.text(0.5, 0.91,
             "Five-fold out-of-fold predictions; color is the true trial class.",
             ha="center", fontsize=8.5, color="#555555")

    by_protocol: dict[str, list[dict]] = {p: [] for p in PROTOCOLS}
    for s in sessions:
        by_protocol[s["protocol"]].append(s)

    for ax, protocol in zip(axes, PROTOCOLS):
        proto_sessions = by_protocol[protocol]
        _draw_ternary_frame(ax)

        all_ba, all_n = [], 0
        for sess in proto_sessions:
            probs_key = f"{source_phase}_probabilities_by_decoder"
            label_key = f"{source_phase}_true_labels"
            if probs_key not in sess or label_key not in sess:
                continue

            probs_all = sess[probs_key]
            probs     = probs_all[:, decoder_idx, :]
            y_true    = sess[label_key]
            y_pred    = probs.argmax(axis=1)

            xy = _to_xy(probs)
            all_n += len(y_true)
            all_ba.append(_balanced_accuracy(y_true, y_pred, len(LABELS)))

            for ci, color in enumerate(COLORS):
                mask = y_true == ci
                if mask.any():
                    ax.scatter(xy[mask, 0], xy[mask, 1],
                               c=color, s=14, alpha=0.35,
                               linewidths=0, zorder=3)

            for ci, color in enumerate(COLORS):
                mask = y_true == ci
                if mask.any():
                    mu = xy[mask].mean(axis=0)
                    ax.scatter(mu[0], mu[1],
                               c=color, s=110, marker="D",
                               edgecolors="white", linewidths=0.8,
                               zorder=5)

        ba_mean = float(np.mean(all_ba)) if all_ba else float("nan")
        ax.set_title(
            f"{protocol}  |  BA={ba_mean*100:.1f}%  |  N={all_n}",
            fontsize=10, fontweight="bold", pad=10,
        )

    _ternary_legend_and_note(fig)
    plt.tight_layout(rect=[0, 0.07, 1, 0.91])

    fname = f"ternary_{source_phase}_thru_{decoder_phase}_direct{latent_dim}.png"
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
        print(f"Saved cross ternary → {save_dir / fname}")
    else:
        plt.show()
    plt.close(fig)


# ── Figure 4: Bar chart ───────────────────────────────────────────────────────

def plot_bar_chart(sessions: list[dict], latent_dim: int, save_dir: Path | None) -> None:
    from collections import defaultdict
    acc_data: dict[tuple, list[float]] = defaultdict(list)

    for sess in sessions:
        proto = sess["protocol"]
        accs  = _session_accuracies(sess)
        for (src, dec), val in accs.items():
            acc_data[(proto, src, dec)].append(val)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    fig.suptitle(
        f"Direct{latent_dim}: decoding accuracy across trial groups and decoder heads",
        fontsize=13, fontweight="bold", y=0.97,
    )
    fig.text(0.5, 0.91,
             "Mean ± SEM across mice; session values averaged within mouse",
             ha="center", fontsize=8.5, color="#555555")

    n_sources  = len(PHASES)
    n_decoders = len(PHASES)
    bar_w      = 0.22
    group_gap  = 0.08
    offsets    = np.arange(n_decoders) * bar_w - (n_decoders - 1) * bar_w / 2

    x_centers  = np.arange(n_sources) * (n_decoders * bar_w + group_gap + 0.1)
    x_ticks    = x_centers
    x_labels   = [PHASE_DISP[p] for p in PHASES]

    for ax, protocol in zip(axes, PROTOCOLS):
        for di, dec_phase in enumerate(PHASES):
            means, sems = [], []
            for src_phase in PHASES:
                vals = acc_data.get((protocol, src_phase, dec_phase), [])
                if vals:
                    arr  = np.array(vals)
                    means.append(arr.mean())
                    sems.append(arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0)
                else:
                    means.append(np.nan)
                    sems.append(np.nan)

            xs = x_centers + offsets[di]
            ax.bar(xs, means, width=bar_w, color=COLORS[di],
                   label=PHASE_DISP[dec_phase] + " head", zorder=3)
            ax.errorbar(xs, means, yerr=sems,
                        fmt="none", ecolor="black", elinewidth=1.2,
                        capsize=3.5, capthick=1.2, zorder=4)

        ax.set_title(protocol, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        ax.tick_params(axis="both", labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=0)
        if ax is axes[0]:
            ax.set_ylabel("Accuracy", fontsize=9)

    legend_handles = [
        mpatches.Patch(color=COLORS[di], label=f"{PHASE_DISP[dec]} head")
        for di, dec in enumerate(PHASES)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=3, fontsize=8.5,
        frameon=False, bbox_to_anchor=(0.5, 0.01),
    )
    fig.text(0.5, 0.06,
             "Each group is a trial source; bars show predictions from all three decoder heads.",
             ha="center", fontsize=8, color="#555555")

    plt.tight_layout(rect=[0, 0.1, 1, 0.90])

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"bar_accuracy_direct{latent_dim}.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved bar figure   → {save_dir / f'bar_accuracy_direct{latent_dim}.png'}")
    else:
        plt.show()
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="Encoder width used when training (default 32).")
    parser.add_argument("--save", action="store_true",
                        help="Write PNGs to outputs/figures/ instead of showing.")
    args = parser.parse_args()

    sessions = []
    missing  = []
    for mouse_id in _cfg.KNOWN_MICE:
        sess = _load_session(mouse_id, args.latent_dim)
        if sess is None:
            missing.append(mouse_id)
        else:
            sessions.append(sess)

    if not sessions:
        print("No trained sessions found. Run train_mean_ff_multhead.py first.")
        return
    if missing:
        print(f"No output found for: {missing}  (skipping)")

    print(f"Loaded {len(sessions)} session(s): "
          + ", ".join(s["mouse_id"] for s in sessions))

    save_dir = (
        Path(__file__).resolve().parent / "outputs" / "figures"
        if args.save else None
    )

    # Figure 1: pre trials → pre decoder
    plot_ternary(sessions, args.latent_dim, save_dir)

    # Figure 2: SLM-water trials → pre decoder
    plot_ternary_cross(sessions, "slm_water", "pre", args.latent_dim, save_dir)

    # Figure 3: SLM-airpuff trials → pre decoder
    plot_ternary_cross(sessions, "slm_airpuff", "pre", args.latent_dim, save_dir)

    # Figure 4: full accuracy bar chart
    plot_bar_chart(sessions, args.latent_dim, save_dir)


if __name__ == "__main__":
    main()
