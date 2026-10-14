"""Collect pre-head OOF predictions from all sessions into one CSV.

Walks the output root produced by run_direct32_5fold_all_sessions.py,
reads each session's mean_ff_oof_predictions.npz, and writes a single
combined CSV that plot_multhead_results.py expects as --predictions.

Example (PowerShell):
    python collect_prehead_predictions.py `
        --results-root "D:\\direct32_results" `
        --out "D:\\direct32_results\\direct32_allruns_prehead_oof_predictions.csv"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

PROTOCOLS = ("Mock", "1Hz", "20Hz")
PHASES = ("pre", "slm_water", "slm_airpuff")
LABELS = ("water", "salt", "airpuff")
# Index of the pre-run head within the decoder axis
PRE_HEAD_INDEX = 0

LABEL_DISPLAY = {0: "Water", 1: "NaCl", 2: "Airpuff"}


def mouse_id(session: str) -> str:
    return session.split("-", 1)[0]


def collect(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    missing: list[str] = []

    for protocol in PROTOCOLS:
        protocol_dir = results_root / protocol
        if not protocol_dir.is_dir():
            print(f"  [skip] protocol folder not found: {protocol_dir}")
            continue

        for session_dir in sorted(p for p in protocol_dir.iterdir() if p.is_dir()):
            npz_path = session_dir / "mean_ff_oof_predictions.npz"
            if not npz_path.is_file():
                missing.append(str(npz_path))
                continue

            session_id = session_dir.name
            mouse = mouse_id(session_id)

            with np.load(npz_path, allow_pickle=False) as data:
                # Verify expected keys exist
                phase_names = tuple(data["phase_names"].astype(str).tolist())
                label_names = tuple(data["label_names"].astype(str).tolist())

                for phase in PHASES:
                    if f"{phase}_probabilities_by_decoder" not in data:
                        print(f"  [warn] {protocol}/{session_id}: missing {phase} arrays, skipping phase")
                        continue

                    # Shape: (n_trials, n_decoder_heads, n_classes)
                    probs_all = data[f"{phase}_probabilities_by_decoder"]
                    true_labels = data[f"{phase}_true_labels"].astype(np.int64)

                    # Pre-head probabilities only (index 0 = pre decoder head)
                    probs = probs_all[:, PRE_HEAD_INDEX, :]  # (n_trials, 3)
                    predicted = probs.argmax(axis=1)

                    for i in range(len(true_labels)):
                        true_cls = int(true_labels[i])
                        rows.append({
                            "protocol": protocol,
                            "session": session_id,
                            "mouse": mouse,
                            "source_run_group": phase,
                            "true_class": LABEL_DISPLAY[true_cls],
                            "predicted_class": LABEL_DISPLAY[int(predicted[i])],
                            "p_water": f"{probs[i, 0]:.6f}",
                            "p_salt": f"{probs[i, 1]:.6f}",
                            "p_airpuff": f"{probs[i, 2]:.6f}",
                        })

    if missing:
        print(f"\n  [warn] {len(missing)} session(s) had no npz file:")
        for p in missing:
            print(f"    {p}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, required=True,
        help="Output root from run_direct32_5fold_all_sessions.py "
             "(contains Mock/, 1Hz/, 20Hz/ subfolders).",
    )
    parser.add_argument(
        "--out", type=Path,
        default=None,
        help="Path for the combined CSV. Defaults to <results-root>/direct32_allruns_prehead_oof_predictions.csv",
    )
    args = parser.parse_args()

    results_root = args.results_root.expanduser().resolve()
    out_path = args.out or results_root / "direct32_allruns_prehead_oof_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {results_root}")
    rows = collect(results_root)

    if not rows:
        print("No rows collected — check that results-root contains protocol subfolders with trained sessions.")
        return

    fieldnames = ["protocol", "session", "mouse", "source_run_group",
                  "true_class", "predicted_class", "p_water", "p_salt", "p_airpuff"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to: {out_path}")


if __name__ == "__main__":
    main()
