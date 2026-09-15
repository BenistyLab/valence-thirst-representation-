"""
Inspect calcium imaging data without opening .mat files in MATLAB.

Run from repo root: python inspect_ca_data.py
"""

import numpy as np
import pandas as pd

from lib.config import EXPLORATION_OUT, default_session_dir
from lib.io.raw import load_dfFvalid, load_exp_data


def main():
    session_dir = default_session_dir()
    if not session_dir.exists():
        print(f"Dataset not found at: {session_dir}")
        print("Check that the path exists (e.g. Dropbox synced).")
        return

    print("=" * 60)
    print("Ca_mice_Livne – motion_corrected_dfFvalid.mat")
    print("=" * 60)

    # Load dfF
    dfF = load_dfFvalid(session_dir)
    n_neurons, n_frames = dfF.shape
    print(f"\nShape: {dfF.shape}")
    print(f"  Rows = neurons (cells): {n_neurons}")
    print(f"  Cols = time (frames):  {n_frames}")
    print(f"\nMeaning: dfF[i, t] = ΔF/F at frame t for neuron i")
    print("  (F corrected for neuropil; F0 from 5000-frame window; dfF = (F-F0)/F0)")

    # Basic stats
    print("\n--- Summary statistics (across all neurons & frames) ---")
    print(f"  Min:   {np.nanmin(dfF):.4f}")
    print(f"  Max:   {np.nanmax(dfF):.4f}")
    print(f"  Mean:  {np.nanmean(dfF):.4f}")
    print(f"  Std:   {np.nanstd(dfF):.4f}")
    print(f"  NaN count: {np.isnan(dfF).sum()}")

    # Per-neuron stats (first few)
    print("\n--- Per-neuron (first 5): mean ΔF/F, std, min, max ---")
    for i in range(min(5, n_neurons)):
        row = dfF[i]
        print(f"  Neuron {i}: mean={np.nanmean(row):.4f}, std={np.nanstd(row):.4f}, "
              f"min={np.nanmin(row):.4f}, max={np.nanmax(row):.4f}")

    # Export a small readable sample
    out_dir = EXPLORATION_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    n_show_neurons = min(10, n_neurons)
    n_show_frames = min(500, n_frames)
    sample = dfF[:n_show_neurons, :n_show_frames]
    df_sample = pd.DataFrame(
        sample.T,
        columns=[f"neuron_{i}" for i in range(n_show_neurons)],
        index=pd.Index(range(n_show_frames), name="frame"),
    )
    csv_path = out_dir / "dfF_sample_first10neurons_first500frames.csv"
    df_sample.to_csv(csv_path)
    print(f"\nSaved a small sample to: {csv_path}")
    print(f"  (first {n_show_neurons} neurons, first {n_show_frames} frames)")

    # Optional: exp_data summary
    print("\n" + "=" * 60)
    print("exp_data.mat (experiment info)")
    print("=" * 60)
    try:
        exp = load_exp_data(session_dir)
        if "fs" in exp:
            fs = np.asarray(exp["fs"]).reshape(-1)
            if fs.size >= 1:
                print(f"\n  fs (frames per second): {fs.flat[0]}")
        if "lickFr" in exp:
            lick = np.asarray(exp["lickFr"]).reshape(-1)
            print(f"  lickFr: length {len(lick)}, sum(licks) = {np.sum(lick)}")
        if "frame_cond_trial" in exp:
            fct = np.asarray(exp["frame_cond_trial"])
            if fct.ndim == 2:
                print(f"  frame_cond_trial: {fct.shape} (trials x 4)")
            else:
                print(f"  frame_cond_trial: shape {fct.shape}")
        print(f"  Other keys in exp_data: {[k for k in exp if not k.startswith('_')]}")
    except Exception as e:
        print(f"  Could not load exp_data: {e}")

    # Simple plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        t = np.arange(n_show_frames)
        for i in range(min(5, n_neurons)):
            ax[0].plot(t, dfF[i, :n_show_frames], alpha=0.7, label=f"neuron {i}")
        ax[0].set_ylabel("ΔF/F")
        ax[0].set_title("First 5 neurons, first 500 frames")
        ax[0].legend(loc="upper right", fontsize=8)
        ax[0].grid(True, alpha=0.3)
        # Mean across neurons
        ax[1].plot(t, np.nanmean(dfF[:, :n_show_frames], axis=0), color="black", alpha=0.8)
        ax[1].set_ylabel("Mean ΔF/F")
        ax[1].set_xlabel("Frame")
        ax[1].set_title("Population mean (all neurons)")
        ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / "dfF_preview.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"\nSaved preview plot: {plot_path}")
    except ImportError:
        print("\n(Install matplotlib to save a preview plot: pip install matplotlib)")

    print("\nDone. You can open the CSV in Excel/Numbers or use pandas to explore further.")


if __name__ == "__main__":
    main()
