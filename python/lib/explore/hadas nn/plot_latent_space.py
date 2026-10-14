"""Plot held-out latent representations for each cross-validation fold."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data_loader import load_mouse
from train_mean_ff_multhead import LABELS, PHASES, SharedEncoderMultiHead


PHASE_COLORS = {"pre": "#0072B2", "slm_water": "#E69F00", "slm_airpuff": "#009E73"}
LABEL_MARKERS = {0: "circle", 1: "square", 2: "triangle"}
DISCARD_SAMPLES = 60


def _pca2(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = X - X.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    variance = singular[:2] ** 2 / max(np.sum(singular ** 2), 1e-12)
    return coords, variance


def _shape_svg(marker: str, x: float, y: float, color: str, title: str) -> str:
    safe_title = (title.replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;"))
    if marker == "circle":
        shape = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" fill-opacity="0.82" stroke="#222" stroke-width="0.5"/>'
    elif marker == "square":
        shape = f'<rect x="{x - 4.5:.2f}" y="{y - 4.5:.2f}" width="9" height="9" fill="{color}" fill-opacity="0.82" stroke="#222" stroke-width="0.5"/>'
    else:
        shape = f'<path d="M {x:.2f} {y - 5.5:.2f} L {x + 5.5:.2f} {y + 4.5:.2f} L {x - 5.5:.2f} {y + 4.5:.2f} Z" fill="{color}" fill-opacity="0.82" stroke="#222" stroke-width="0.5"/>'
    return f"<g>{shape}<title>{safe_title}</title></g>"


def _build_svg(rows: list[dict], variances: dict[int, np.ndarray],
               output_path: Path) -> None:
    width, height = 1220, 960
    panel_w, panel_h = 570, 390
    origins = [(25, 35), (625, 35), (25, 465), (625, 465)]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:12px}.title{font-size:18px;font-weight:bold}</style>',
             '<text x="610" y="23" text-anchor="middle" class="title">Held-out shared encoder representations by fold</text>']
    for fold, (ox, oy) in enumerate(origins, start=1):
        subset = [r for r in rows if r["fold"] == fold]
        if not subset:
            continue
        coords = np.asarray([[r["pc1"], r["pc2"]] for r in subset])
        # Recover plotted coordinates as already PCA-projected; scale the panel consistently.
        x_values, y_values = coords[:, 0], coords[:, 1]
        xmin, xmax = float(x_values.min()), float(x_values.max())
        ymin, ymax = float(y_values.min()), float(y_values.max())
        xpad = max((xmax - xmin) * 0.12, 0.5)
        ypad = max((ymax - ymin) * 0.12, 0.5)
        xmin, xmax = xmin - xpad, xmax + xpad
        ymin, ymax = ymin - ypad, ymax + ypad
        left, top = ox + 55, oy + 44
        plot_w, plot_h = panel_w - 85, panel_h - 100
        parts.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" rx="8" fill="#fff" stroke="#aaa"/>')
        v = variances[fold]
        parts.append(f'<text x="{ox + 15}" y="{oy + 25}" class="title">Fold {fold} held-out trials</text>')
        parts.append(f'<text x="{ox + panel_w - 12}" y="{oy + 24}" text-anchor="end" class="small">PC1 {v[0]*100:.1f}% · PC2 {v[1]*100:.1f}%</text>')
        parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#555"/>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#555"/>')
        parts.append(f'<text x="{left + plot_w/2:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="small">PC1</text>')
        parts.append(f'<text x="{ox + 18}" y="{top + plot_h/2:.1f}" transform="rotate(-90 {ox+18} {top+plot_h/2:.1f})" text-anchor="middle" class="small">PC2</text>')
        for row in subset:
            px = left + (row["pc1"] - xmin) / (xmax - xmin) * plot_w
            py = top + plot_h - (row["pc2"] - ymin) / (ymax - ymin) * plot_h
            label = LABELS[row["label"]]
            title = (f"{row['phase']} | {label} | source trial "
                     f"{row['source_trial_index_zero_based']}")
            parts.append(_shape_svg(LABEL_MARKERS[row["label"]], px, py,
                                    PHASE_COLORS[row["phase"]], title))

    # Shared legends: color identifies run group; marker identifies stimulus label.
    legend_y = 900
    parts.append('<text x="30" y="925" class="small" font-weight="bold">Run group:</text>')
    x = 105
    for phase in PHASES:
        parts.append(f'<circle cx="{x}" cy="921" r="6" fill="{PHASE_COLORS[phase]}"/>')
        parts.append(f'<text x="{x+11}" y="925" class="small">{phase.replace("_", "-")}</text>')
        x += 175
    parts.append('<text x="690" y="925" class="small" font-weight="bold">Trial label:</text>')
    x = 775
    for label_i, label in enumerate(LABELS):
        parts.append(_shape_svg(LABEL_MARKERS[label_i], x, 920, "#888", label))
        parts.append(f'<text x="{x+12}" y="925" class="small">{label}</text>')
        x += 130
    parts.append('</svg>')
    output_path.write_text("\n".join(parts), encoding="utf-8")


def plot(session_dir: Path, model_dir: Path, output_dir: Path) -> Path:
    runs, condition_names = load_mouse(session_dir, min_duration=300, to_zscore=False)
    name_to_id = {name: cid for cid, name in condition_names.items()}
    label_by_id = {name_to_id["Water-Op"]: 0, name_to_id["NaCl"]: 1,
                   name_to_id["Pav-Airpuff"]: 2}
    by_name = {run.name: run for run in runs}
    group_data = {}
    for phase in PHASES:
        run = by_name[phase]
        keep = np.isin(run.condition, list(label_by_id))
        X = run.X[:, DISCARD_SAMPLES:, keep].mean(axis=1, dtype=np.float64).T.astype(np.float32)
        y = np.asarray([label_by_id[int(c)] for c in run.condition[keep]], dtype=np.int64)
        source_indices = run.trial_index[keep]
        group_data[phase] = (X, y, source_indices)

    saved = np.load(model_dir / "mean_ff_oof_predictions.npz", allow_pickle=False)
    saved_phases = tuple(saved["phase_names"].astype(str).tolist())
    if saved_phases != PHASES:
        raise ValueError(f"Model output phase order mismatch: {saved_phases}")
    folds_by_phase = {p: saved[f"{p}_fold_ids"].astype(np.int64) for p in PHASES}
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    variances: dict[int, np.ndarray] = {}
    latent_csv = output_dir / "heldout_latents_pca.csv"

    for fold in range(1, 5):
        checkpoint_path = model_dir / "fold_checkpoints" / f"fold_{fold}.pt"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = SharedEncoderMultiHead(
            n_cells=next(iter(group_data.values()))[0].shape[1],
            latent_dim=int(state["latent_dim"]),
            encoder_hidden=int(state["encoder_hidden"]),
            decoder_hidden=int(state["decoder_hidden"]),
        )
        model.load_state_dict(state["model_state"])
        model.eval()
        mean = np.asarray(state["mean"], dtype=np.float32)
        std = np.asarray(state["std"], dtype=np.float32)
        fold_rows = []
        latent_arrays = []
        for phase in PHASES:
            X, y, source_indices = group_data[phase]
            test_idx = np.flatnonzero(folds_by_phase[phase] == fold)
            normalized = ((X[test_idx] - mean) / np.maximum(std, 1e-6)).astype(np.float32)
            with torch.no_grad():
                z = model.encode(torch.from_numpy(normalized)).numpy()
            latent_arrays.append(z)
            for local_i, point in zip(test_idx, z):
                fold_rows.append({
                    "fold": fold,
                    "phase": phase,
                    "label": int(y[local_i]),
                    "source_trial_index_zero_based": int(source_indices[local_i]),
                    "latent": point,
                })
        coords, variance = _pca2(np.concatenate(latent_arrays, axis=0))
        variances[fold] = variance
        offset = 0
        for row in fold_rows:
            row["pc1"], row["pc2"] = map(float, coords[offset])
            offset += 1
            rows.append(row)

    _build_svg(rows, variances, output_dir / "latent_space_by_fold.svg")
    with latent_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["fold", "phase", "label", "source_trial_index_zero_based", "pc1", "pc2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    with (output_dir / "pca_variance.txt").open("w", encoding="utf-8") as f:
        f.write("Fold-specific PCA is fit to that fold's held-out latent points.\n")
        for fold, variance in variances.items():
            f.write(f"Fold {fold}: PC1={variance[0]:.4f}, PC2={variance[1]:.4f}, "
                    f"combined={variance.sum():.4f}\n")
    return output_dir / "latent_space_by_fold.svg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent / "outputs" / args.session_dir.name
    model_dir = args.model_dir or base / "mean_ff_multhead_wd001"
    output_dir = args.output_dir or model_dir / "latent_visualization"
    plot_path = plot(args.session_dir, model_dir, output_dir)
    print(f"Saved latent PCA plot to: {plot_path}")


if __name__ == "__main__":
    main()
