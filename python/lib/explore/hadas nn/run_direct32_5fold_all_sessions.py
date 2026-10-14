"""Train the selected direct32 multi-head decoder on every session.

For each session this runs five-fold cross-validation with one shared
cells-to-32 ReLU encoder and three linear decoder heads (Pre, SLM-water,
SLM-airpuff). Every held-out trial is evaluated through all three heads.

Example (PowerShell):
    python run_direct32_5fold_all_sessions.py `
        --data-root "D:/Ca_mice_Livne" `
        --output-root "D:/direct32_results"

The data root must contain Mock, 1Hz, and 20Hz directories, each containing
one directory per session. The trainer requires a CUDA-enabled PyTorch install.
"""
from __future__ import annotations

import argparse
import csv
import json
import traceback
from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_mean_ff_multhead import LABELS, PHASES, train_session  # noqa: E402


PROTOCOLS = ("Mock", "1Hz", "20Hz")
ARCHITECTURE = {
    "latent_dim": 32,
    "encoder_hidden": 0,  # direct cells -> 32 ReLU encoder
    "decoder_hidden": 0,  # three linear 32 -> 3 heads
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct32 five-fold cross-validation for all mouse sessions."
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Directory containing Mock/, 1Hz/, and 20Hz/ session folders.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=SCRIPT_DIR / "outputs" / "direct32_5fold_all_sessions",
        help="Where to save checkpoints, predictions, logs, and summary CSVs.",
    )
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS),
                        help="Protocols to run (default: Mock 1Hz 20Hz).")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This training pipeline requires a CUDA-enabled "
            "PyTorch environment and will not silently fall back to CPU."
        )
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "data_root": str(data_root),
        "protocols": args.protocols,
        "output_root": str(output_root),
        "architecture": ARCHITECTURE,
        "folds": args.folds,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "labels": LABELS,
        "run_groups": PHASES,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    sessions: list[tuple[str, Path]] = []
    ignored: list[str] = []
    for protocol in args.protocols:
        protocol_dir = data_root / protocol
        if not protocol_dir.is_dir():
            raise FileNotFoundError(f"Missing protocol directory: {protocol_dir}")
        for session_dir in sorted(p for p in protocol_dir.iterdir() if p.is_dir()):
            if (session_dir / "exp_data.mat").is_file() and (
                session_dir / "motion_corrected_dfFvalid.mat"
            ).is_file():
                sessions.append((protocol, session_dir))
            else:
                ignored.append(str(session_dir))
    if not sessions:
        raise RuntimeError(f"No session folders with the required MAT files found under {data_root}")

    print(f"Found {len(sessions)} sessions; protocols: {', '.join(args.protocols)}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Saving outputs under: {output_root}")

    result_rows: list[dict] = []
    failure_rows: list[dict] = []
    for i, (protocol, session_dir) in enumerate(sessions, start=1):
        session_id = session_dir.name
        session_output = output_root / protocol / session_id
        log_path = output_root / "logs" / f"{protocol}__{session_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(sessions)}] {protocol} / {session_id}", flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
                result = train_session(
                    session_dir,
                    session_output,
                    folds=args.folds,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    latent_dim=ARCHITECTURE["latent_dim"],
                    encoder_hidden=ARCHITECTURE["encoder_hidden"],
                    decoder_hidden=ARCHITECTURE["decoder_hidden"],
                    weight_decay=args.weight_decay,
                    seed=args.seed,
                    pre_fold_csv=None,
                )
            with np.load(session_output / "mean_ff_oof_predictions.npz", allow_pickle=False) as saved:
                for source_i, source in enumerate(PHASES):
                    y = saved[f"{source}_true_labels"].astype(np.int64)
                    counts = np.bincount(y, minlength=len(LABELS))
                    for head_i, head in enumerate(PHASES):
                        score = result["metrics"][f"{source}_to_{head}"]
                        result_rows.append({
                            "protocol": protocol,
                            "session": session_id,
                            "source_run_group": source,
                            "decoder_head": head,
                            "n_trials": int(len(y)),
                            "n_water": int(counts[0]),
                            "n_salt": int(counts[1]),
                            "n_airpuff": int(counts[2]),
                            "accuracy": score["accuracy"],
                            "balanced_accuracy": score["balanced_accuracy"],
                        })
            print(f"    complete; log: {log_path}", flush=True)
        except Exception as exc:
            failure_rows.append({
                "protocol": protocol,
                "session": session_id,
                "error": f"{type(exc).__name__}: {exc}",
                "log": str(log_path),
            })
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"    FAILED: {type(exc).__name__}: {exc}; see {log_path}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metric_columns = ["protocol", "session", "source_run_group", "decoder_head",
                      "n_trials", "n_water", "n_salt", "n_airpuff",
                      "accuracy", "balanced_accuracy"]
    failure_columns = ["protocol", "session", "error", "log"]
    write_csv(output_root / "per_session_metrics.csv", result_rows, metric_columns)
    write_csv(output_root / "failures.csv", failure_rows, failure_columns)
    print(f"Finished {len(sessions)-len(failure_rows)}/{len(sessions)} sessions.")
    print(f"Metrics: {output_root / 'per_session_metrics.csv'}")
    print(f"Failures: {output_root / 'failures.csv'}")
    if ignored:
        print(f"Ignored {len(ignored)} folders without both required MAT files.")
    return 0 if not failure_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
