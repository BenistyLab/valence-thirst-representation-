"""
Train and evaluate the shared LSTM encoder + phase decoders.

Usage (from the python/ directory, with LIVNEH_MOUSE_ID and LIVNEH_DATA_ROOT
set as environment variables or via the .env mechanism used by the rest of
the pipeline):

    python run_lstm.py                          # train & evaluate (default mouse)
    python run_lstm.py --mouse 20Hz/AL45        # specific mouse/session
    python run_lstm.py --eval-only              # skip training, load checkpoint
    python run_lstm.py --phases pre airpuff water
    python run_lstm.py --epochs 300 --patience 30

The script:
  1. Loads NPZ trial packs for each phase (produced by prepare_data.py).
  2. Builds a PhaseDecoderModel (shared LSTM encoder + per-phase decoders).
  3. Trains it with early stopping, saving the best checkpoint.
  4. Evaluates within-phase decoding accuracy.
  5. Runs cross-phase decoding (pre trials → SLM decoder, etc.).
  6. Saves evaluation results to outputs/<protocol>/<mouse>/lstm/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate LSTM phase decoder.")
    p.add_argument("--mouse", default=None, help="Session id, e.g. '20Hz/AL45'.")
    p.add_argument(
        "--phases",
        nargs="+",
        default=["pre", "airpuff", "water"],
        help="Phase names to include (must match NPZ keys from prepare_data.py).",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load the existing checkpoint and evaluate.",
    )
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--lstm-hidden", type=int, default=8,
                   help="LSTM hidden-state size.")
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[],
                   help="Hidden layer sizes for the encoder MLP projector "
                        "(space-separated, e.g. --mlp-hidden 32). "
                        "Independent of --latent-dim.")
    p.add_argument("--latent-dim", type=int, default=8,
                   help="Output size of the encoder MLP = input size of each "
                        "phase decoder (the shared latent space size).")
    p.add_argument("--dec-hidden", type=int, nargs="+", default=[],
                   help="Hidden layer sizes for each phase decoder "
                        "(space-separated, e.g. --dec-hidden 8). "
                        "Architecture: latent_dim -> dec_hidden -> n_classes.")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------ env
    if args.mouse:
        os.environ["LIVNEH_MOUSE_ID"] = args.mouse
    if args.data_root:
        os.environ["LIVNEH_DATA_ROOT"] = str(args.data_root.expanduser())

    # Import after setting env vars so lib/config.py resolves correctly
    from lib import config as cfg
    from lib.lstm.dataset import (
        load_phase_datasets,
        make_dataloaders,
        compute_class_weights,
        N_CLASSES,
    )
    from lib.lstm.model import PhaseDecoderModel
    from lib.lstm.train import train, load_checkpoint, _get_device
    from lib.lstm.evaluate import (
        evaluate_within_phase,
        evaluate_cross_phase,
        print_interesting_evaluation,
        extract_latents,
        save_results,
    )

    mouse_id = cfg.resolve_mouse_id()
    output_dir = cfg.OUTPUTS_DIR / "lstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "lstm_model.pt"

    print(f"Mouse / session : {mouse_id}")
    print(f"Phases          : {args.phases}")
    print(f"Output dir      : {output_dir}")
    print()

    # ----------------------------------------------------------- data
    print("Loading trial data …")
    datasets = load_phase_datasets(phases=args.phases)
    for ph, ds in datasets.items():
        print(f"  {ph:8s}: {len(ds)} trials  ({ds.n_neurons} neurons, T_max={ds.t_max})")

    # Infer n_neurons from the first dataset
    n_neurons = next(iter(datasets.values())).n_neurons

    # Compute class weights to counter label imbalance
    class_weights = compute_class_weights(datasets)
    print(f"\nClass weights (inverse-frequency, mean-normalised):")
    from lib.lstm.dataset import LABEL_NAMES
    for c, name in LABEL_NAMES.items():
        print(f"  {name:8s}: {class_weights[c]:.3f}")

    train_loaders, val_loaders = make_dataloaders(
        datasets,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print()

    # ----------------------------------------------------------- model
    device = _get_device()
    print(f"Device: {device}")

    model = PhaseDecoderModel(
        n_neurons=n_neurons,
        phases=args.phases,
        lstm_hidden=args.lstm_hidden,
        mlp_hidden=tuple(args.mlp_hidden),
        latent_dim=args.latent_dim,
        dec_hidden=tuple(args.dec_hidden),
        n_classes=N_CLASSES,
        dropout=args.dropout,
    )

    if args.eval_only:
        if checkpoint_path.exists():
            print(f"\nLoading checkpoint: {checkpoint_path}")
            load_checkpoint(model, checkpoint_path, device=device)
        else:
            print("--eval-only requested but no checkpoint found; training first.")
            args.eval_only = False
    elif checkpoint_path.exists():
        print(
            f"\nNote: an old checkpoint exists at {checkpoint_path} but --eval-only "
            "was not passed, so it will be overwritten by a fresh training run."
        )

    # ----------------------------------------------------------- train
    if not args.eval_only:
        print(f"\nTraining (max {args.epochs} epochs, patience={args.patience}) …\n")
        history = train(
            model,
            train_loaders,
            val_loaders,
            output_dir=output_dir,
            checkpoint_name="lstm_model.pt",
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.epochs,
            patience=args.patience,
            device=device,
            class_weights=class_weights,
            verbose=True,
        )

        # Reload the best checkpoint after training
        load_checkpoint(model, checkpoint_path, device=device)

    # ----------------------------------------------------------- evaluate
    print("\n=== Within-phase decoding ===")
    within_results = evaluate_within_phase(model, val_loaders)
    print_interesting_evaluation(within_results)

    print("\n=== Cross-phase decoding ===")
    cross_results = evaluate_cross_phase(model, val_loaders)
    print_interesting_evaluation(cross_results)

    # ----------------------------------------------------------- latents
    print("\nExtracting latent embeddings for all validation sets …")
    latents = extract_latents(model, val_loaders)
    latent_path = output_dir / "latents.npz"
    import numpy as np
    np.savez_compressed(
        latent_path,
        **{
            f"{ph}_latents": v["latents"]
            for ph, v in latents.items()
        },
        **{
            f"{ph}_labels": v["labels"]
            for ph, v in latents.items()
        },
    )
    print(f"Latents saved to: {latent_path}")

    # ----------------------------------------------------------- save
    save_results(
        {"within_phase": within_results, "cross_phase": cross_results},
        output_dir=output_dir,
        filename="lstm_results.npz",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
