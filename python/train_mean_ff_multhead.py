"""Train a shared FF encoder and three run-group decoder heads.

Features are per-trial mean activity across frames 61–300 (MATLAB indexing).
The model is trained jointly on pre-run, SLM-water, and SLM-airpuff trials;
each trial contributes classification loss through its own run-group head.

Architecture:
  Shared encoder : n_cells  →  Linear(latent_dim)  →  ReLU
  Per-phase heads: latent_dim  →  Linear(3)        (one per phase)

At test time every decoder head is applied to every phase's test trials,
producing a (source_phase × decoder_head) accuracy matrix.

Usage (from the python/ directory, with LIVNEH_MOUSE_ID and LIVNEH_DATA_ROOT
set as environment variables or via the .env mechanism):

    python train_mean_ff_multhead.py                 # uses LIVNEH_MOUSE_ID
    python train_mean_ff_multhead.py --mouse 20Hz/AL45
    python train_mean_ff_multhead.py --all           # train every known session
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Add python/ to sys.path so that lib/ and data_loader are importable
# regardless of where PyCharm or the shell invokes this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader import load_mouse


PHASES = ("pre", "slm_water", "slm_airpuff")
LABELS = ("water", "salt", "airpuff")
DISCARD_SAMPLES = 60     # skip first 60 frames (MATLAB 1:60)
FRAME_CAP       = 300    # use frames 61–300 (MATLAB 61:300, Python [60:300])
LATENT_DIM = 32          # fixed encoder width
MIN_TRIALS_PER_CLASS = 5 # minimum trials of each label per phase; sessions below this are skipped


class InsufficientTrialsError(Exception):
    """Raised when a session lacks enough trials of some class in some phase."""


class SharedEncoderMultiHead(nn.Module):
    """Shared feed-forward encoder (n_cells → latent_dim, ReLU) and linear heads per phase."""

    def __init__(self, n_cells: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_cells, latent_dim),
            nn.ReLU(),
        )
        self.decoders = nn.ModuleDict({
            phase: nn.Linear(latent_dim, len(LABELS)) for phase in PHASES
        })

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, phase: str) -> torch.Tensor:
        return self.decoders[phase](z)

    def forward(self, x: torch.Tensor, phase: str) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z, phase)


def _stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for fold_i, subset in enumerate(np.array_split(indices, n_folds)):
            folds[fold_i].extend(subset.tolist())
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def _train_val_split(y: np.ndarray, indices: np.ndarray, fraction: float,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit, val = [], []
    for label in np.unique(y[indices]):
        selected = indices[y[indices] == label].copy()
        rng.shuffle(selected)
        n_val = max(1, int(round(len(selected) * fraction)))
        n_val = min(n_val, len(selected) - 1)
        val.extend(selected[:n_val].tolist())
        fit.extend(selected[n_val:].tolist())
    return np.asarray(fit, dtype=np.int64), np.asarray(val, dtype=np.int64)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recall = [np.mean(y_pred[y_true == c] == c)
              for c in range(len(LABELS)) if np.any(y_true == c)]
    return float(np.mean(recall))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    result = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(result, (y_true, y_pred), 1)
    return result


def _make_datasets(runs, condition_names: dict[int, str]) -> dict[str, dict]:
    name_to_id = {name: cid for cid, name in condition_names.items()}
    required = {"Water-Op": 0, "NaCl": 1, "Pav-Airpuff": 2}
    missing = [name for name in required if name not in name_to_id]
    if missing:
        raise ValueError(f"Session is missing class conditions: {missing}")
    by_run_name = {run.name: run for run in runs}
    datasets = {}
    for phase in PHASES:
        if phase not in by_run_name:
            raise ValueError(f"Loader returned no nonempty {phase} group")
        run = by_run_name[phase]
        label_by_id = {name_to_id[name]: label for name, label in required.items()}
        keep = np.isin(run.condition, list(label_by_id))
        # MATLAB sample indices 61:300 (Python [60:300]): discard first 60 frames
        # and use exactly FRAME_CAP (300) frames. data_loader drops any trial
        # shorter than min_duration and trims the rest to exactly min_duration,
        # so every trial in run.X has exactly 300 frames — no zero-padded tail.
        features = run.X[:, DISCARD_SAMPLES:FRAME_CAP, keep].mean(
            axis=1, dtype=np.float64
        ).T.astype(np.float32)
        labels = np.asarray([label_by_id[int(cid)] for cid in run.condition[keep]],
                            dtype=np.int64)
        datasets[phase] = {
            "X": features,
            "y": labels,
            "source_trial_index": run.trial_index[keep].astype(np.int64),
        }
    return datasets


def _get_folds(datasets: dict[str, dict], fold_csv: Path | None,
               n_folds: int, seed: int) -> dict[str, np.ndarray]:
    folds = {}
    if fold_csv is not None and fold_csv.is_file():
        with fold_csv.open(newline="", encoding="utf-8") as f:
            pre_fold = list(csv.DictReader(f))
        pre_data = datasets["pre"]
        if not np.array_equal(
                np.asarray([int(row["source_trial_index_zero_based"])
                            for row in pre_fold]),
                pre_data["source_trial_index"]):
            raise ValueError("Pre-run trial order differs from the existing Python CV folds")
        expected_labels = [LABELS[i] for i in pre_data["y"]]
        if [row["label"] for row in pre_fold] != expected_labels:
            raise ValueError("Pre-run labels differ from the existing Python CV folds")
        folds["pre"] = np.asarray([int(row["fold"]) for row in pre_fold], dtype=np.int64)
    else:
        folds["pre"] = None
    for phase_i, phase in enumerate(PHASES):
        y = datasets[phase]["y"]
        counts = np.bincount(y, minlength=len(LABELS))
        if np.any(counts < n_folds):
            raise ValueError(
                f"{phase} needs at least {n_folds} trials per class; "
                f"counts={dict(zip(LABELS, counts.tolist()))}"
            )
        if phase == "pre" and folds[phase] is not None:
            continue
        test_folds = _stratified_folds(y, n_folds, seed + phase_i)
        ids = np.empty(len(y), dtype=np.int64)
        for fold_i, test_idx in enumerate(test_folds):
            ids[test_idx] = fold_i + 1
        folds[phase] = ids
    return folds


def _transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def _predict_logits(model: SharedEncoderMultiHead, X: np.ndarray,
                    phase: str, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).to(device)
        return model.decode(model.encode(x), phase).cpu().numpy()


def train_session(mouse_id: str, output_dir: Path, *, folds: int = 5,
                  epochs: int = 250, patience: int = 25,
                  batch_size: int = 16, latent_dim: int = LATENT_DIM,
                  weight_decay: float = 1e-2,
                  normalize: bool = False,
                  seed: int = 2026, pre_fold_csv: Path | None = None) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to this Python environment; stopping rather "
            "than silently falling back to CPU."
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"Training device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Mouse: {mouse_id}")
    print(f"Encoder width: {latent_dim}; weight decay: {weight_decay:g}; "
          f"fold-local normalise: {normalize}")

    runs, condition_names = load_mouse(mouse_id=mouse_id, min_duration=300)
    datasets = _make_datasets(runs, condition_names)
    for phase in PHASES:
        counts = np.bincount(datasets[phase]["y"], minlength=len(LABELS))
        print(f"{phase}: {len(datasets[phase]['y'])} labeled trials; "
              f"counts={dict(zip(LABELS, counts.tolist()))}")

    # Require at least MIN_TRIALS_PER_CLASS trials of every label in every phase.
    # If any phase falls short, the entire session is unusable — raise so the
    # caller can skip it cleanly (especially important in --all mode).
    for phase in PHASES:
        counts = np.bincount(datasets[phase]["y"], minlength=len(LABELS))
        short = {LABELS[i]: int(counts[i])
                 for i in range(len(LABELS)) if counts[i] < MIN_TRIALS_PER_CLASS}
        if short:
            raise InsufficientTrialsError(
                f"{mouse_id}: phase '{phase}' has fewer than {MIN_TRIALS_PER_CLASS} "
                f"trials for class(es) {short} — session dropped."
            )

    fold_ids = _get_folds(datasets, pre_fold_csv, folds, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "fold_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    oof_probs = {
        source: np.full((len(datasets[source]["y"]), len(PHASES), len(LABELS)),
                        np.nan, dtype=np.float32)
        for source in PHASES
    }
    fold_metrics: list[dict] = []

    for fold_i in range(1, folds + 1):
        torch.manual_seed(seed + fold_i)
        torch.cuda.manual_seed_all(seed + fold_i)
        split = {}
        for phase in PHASES:
            data = datasets[phase]
            test = np.flatnonzero(fold_ids[phase] == fold_i)
            train = np.flatnonzero(fold_ids[phase] != fold_i)
            fit, val = _train_val_split(data["y"], train, 0.20,
                                        seed + 100 * fold_i + PHASES.index(phase))
            split[phase] = {"test": test, "fit": fit, "val": val}

        # Fold-local normalization: fit on joint training partition only so test
        # data is never seen during normalisation (prevents leakage).
        # NOTE: prepare_data.py already z-scores the NPZ data globally, so
        # pass normalize=False if you want to skip this second pass.
        if normalize:
            fit_features = np.concatenate([
                datasets[p]["X"][split[p]["fit"]] for p in PHASES
            ], axis=0)
            mean = fit_features.mean(axis=0, dtype=np.float64).astype(np.float32)
            std  = fit_features.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
            std  = np.maximum(std, 1e-6)
            normalized = {
                p: {part: _transform(datasets[p]["X"][split[p][part]], mean, std)
                    for part in ("fit", "val", "test")}
                for p in PHASES
            }
        else:
            # Data already globally z-scored by prepare_data.py; pass through as-is.
            mean = np.zeros(datasets["pre"]["X"].shape[1], dtype=np.float32)
            std  = np.ones(datasets["pre"]["X"].shape[1],  dtype=np.float32)
            normalized = {
                p: {part: datasets[p]["X"][split[p][part]]
                    for part in ("fit", "val", "test")}
                for p in PHASES
            }

        loaders = {}
        criteria = {}
        for phase in PHASES:
            fit_indices = split[phase]["fit"]
            y_fit = datasets[phase]["y"][fit_indices]
            class_counts = np.bincount(y_fit, minlength=len(LABELS))
            class_weights = len(y_fit) / (len(LABELS) * class_counts)
            criteria[phase] = nn.CrossEntropyLoss(
                weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
            )
            loaders[phase] = DataLoader(
                TensorDataset(torch.from_numpy(normalized[phase]["fit"]),
                              torch.from_numpy(y_fit)),
                batch_size=batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(seed + fold_i * 10 + PHASES.index(phase)),
            )

        model = SharedEncoderMultiHead(
            datasets["pre"]["X"].shape[1], latent_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                      weight_decay=weight_decay)
        best_val, best_epoch, best_state = float("inf"), 0, None
        stale = 0
        max_batches = max(len(loader) for loader in loaders.values())

        for epoch in range(1, epochs + 1):
            model.train()
            iterators = {p: iter(loaders[p]) for p in PHASES}
            for _ in range(max_batches):
                phase_losses = []
                for phase in PHASES:
                    try:
                        xb, yb = next(iterators[phase])
                    except StopIteration:
                        continue
                    xb, yb = xb.to(device), yb.to(device)
                    phase_losses.append(criteria[phase](model(xb, phase)[1], yb))
                if not phase_losses:
                    continue
                loss = torch.stack(phase_losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            validation_losses = []
            with torch.no_grad():
                for phase in PHASES:
                    x_val = torch.from_numpy(normalized[phase]["val"]).to(device)
                    y_val = torch.from_numpy(
                        datasets[phase]["y"][split[phase]["val"]]
                    ).to(device)
                    validation_losses.append(
                        criteria[phase](model(x_val, phase)[1], y_val)
                    )
                val_loss = float(torch.stack(validation_losses).mean().item())
            if val_loss < best_val:
                best_val, best_epoch, stale = val_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError(f"No checkpoint saved for fold {fold_i}")
        model.load_state_dict(best_state)
        model.to(device).eval()

        fold_record = {"fold": fold_i, "best_epoch": best_epoch,
                       "joint_validation_loss": best_val}
        for source in PHASES:
            test_idx = split[source]["test"]
            source_X = normalized[source]["test"]
            for target_i, target in enumerate(PHASES):
                logits = _predict_logits(model, source_X, target, device)
                probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
                oof_probs[source][test_idx, target_i, :] = probs
                pred = probs.argmax(axis=1)
                y_test = datasets[source]["y"][test_idx]
                fold_record[f"{source}_to_{target}_accuracy"] = float(np.mean(pred == y_test))
                fold_record[f"{source}_to_{target}_balanced_accuracy"] = _balanced_accuracy(y_test, pred)
        fold_metrics.append(fold_record)
        torch.save({
            "model_state": best_state,
            "mean": mean,
            "std": std,
            "fold": fold_i,
            "latent_dim": latent_dim,
            "normalize": normalize,
            "phases": PHASES,
            "labels": LABELS,
        }, ckpt_dir / f"fold_{fold_i}.pt")
        print(f"Fold {fold_i}/{folds}: best epoch {best_epoch}, "
              f"validation loss {best_val:.4f}")

    metric_summary = {}
    confusion_matrices = {}
    for source in PHASES:
        y_true = datasets[source]["y"]
        for target_i, target in enumerate(PHASES):
            probs = oof_probs[source][:, target_i, :]
            if np.isnan(probs).any():
                raise RuntimeError(f"Missing OOF predictions for {source} -> {target}")
            y_pred = probs.argmax(axis=1)
            key = f"{source}_to_{target}"
            metric_summary[key] = {
                "accuracy": float(np.mean(y_pred == y_true)),
                "balanced_accuracy": _balanced_accuracy(y_true, y_pred),
            }
            confusion_matrices[key] = _confusion(y_true, y_pred)
            print(f"{key}: accuracy={metric_summary[key]['accuracy']:.3f}, "
                  f"balanced accuracy={metric_summary[key]['balanced_accuracy']:.3f}")

    npz_arrays = {}
    for phase in PHASES:
        npz_arrays[f"{phase}_probabilities_by_decoder"] = oof_probs[phase]
        npz_arrays[f"{phase}_true_labels"] = datasets[phase]["y"]
        npz_arrays[f"{phase}_source_trial_indices"] = datasets[phase]["source_trial_index"]
        npz_arrays[f"{phase}_fold_ids"] = fold_ids[phase]
    npz_arrays["phase_names"] = np.asarray(PHASES)
    npz_arrays["label_names"] = np.asarray(LABELS)
    np.savez_compressed(output_dir / "mean_ff_oof_predictions.npz", **npz_arrays)

    with (output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_metrics[0]))
        writer.writeheader()
        writer.writerows(fold_metrics)
    with (output_dir / "fold_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("phase", "source_trial_index_zero_based", "label", "fold"))
        for phase in PHASES:
            data = datasets[phase]
            for source_idx, label, fold_id in zip(
                    data["source_trial_index"], data["y"], fold_ids[phase]):
                writer.writerow((phase, int(source_idx), LABELS[int(label)], int(fold_id)))
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Mouse: {mouse_id}\nDevice: {torch.cuda.get_device_name(device)}\n")
        f.write(f"Feature: trial mean activity over frames {DISCARD_SAMPLES + 1}:{FRAME_CAP} "
                f"({FRAME_CAP - DISCARD_SAMPLES} samples)\n")
        f.write(f"Architecture: n_cells → Linear({latent_dim}) → ReLU "
                f"→ [Linear(3) per phase]\n")
        f.write(f"AdamW weight decay: {weight_decay:g}; fold-local normalise: {normalize}\n")
        f.write(f"Phase order: {PHASES}\nLabel order: {LABELS}\n\n")
        for key, metrics in metric_summary.items():
            f.write(f"{key}: accuracy={metrics['accuracy']:.4f}, "
                    f"balanced_accuracy={metrics['balanced_accuracy']:.4f}\n")
            f.write(f"confusion rows=true/columns=predicted:\n{confusion_matrices[key]}\n\n")
    return {"metrics": metric_summary, "confusion_matrices": confusion_matrices,
            "device": torch.cuda.get_device_name(device)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mouse", default=None,
        help=(
            "Session id, e.g. '20Hz/AL45'. Defaults to the LIVNEH_MOUSE_ID "
            "environment variable. Pass 'all' to train every known session."
        ),
    )
    parser.add_argument(
        "--all", dest="all_mice", action="store_true",
        help="Train every known session (shorthand for --mouse all).",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override the output directory for a single-session run.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM,
                        help="Width of the shared encoder layer (default 32).")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--normalize", action="store_true",
                        help="Apply fold-local z-score on top of prepare_data.py's global "
                             "z-score. Off by default (data is already normalised).")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pre-fold-csv", type=Path, default=None,
                        help="Existing pre-run fold CSV, to reuse the LSTM/SVM splits.")
    args = parser.parse_args()

    from lib import config as _cfg

    # Resolve the list of sessions to train.
    if args.all_mice or (args.mouse and args.mouse.lower() == "all"):
        mice = list(_cfg.KNOWN_MICE)
    else:
        mice = [_cfg.resolve_mouse_id(args.mouse)]   # falls back to LIVNEH_MOUSE_ID

    for mouse_id in mice:
        protocol   = _cfg.SESSION_PROTOCOL[mouse_id]
        mouse_name = _cfg.SESSION_MOUSE[mouse_id]
        base = Path(__file__).resolve().parent / "outputs" / protocol / mouse_name
        output = (
            args.output_dir
            if (args.output_dir and len(mice) == 1)
            else base / f"mean_ff_multhead_enc{args.latent_dim}"
        )
        fold_csv = args.pre_fold_csv

        if len(mice) > 1:
            sep = "=" * 60
            print(f"\n{sep}\nMouse: {mouse_id}\n{sep}")

        try:
            train_session(
                mouse_id, output,
                folds=args.folds, epochs=args.epochs, patience=args.patience,
                latent_dim=args.latent_dim, weight_decay=args.weight_decay,
                normalize=args.normalize,
                seed=args.seed, pre_fold_csv=fold_csv,
            )
        except InsufficientTrialsError as exc:
            print(f"[SKIPPED] {exc}")


if __name__ == "__main__":
    main()