# livneh-mice

Analysis pipeline for **Livneh mouse calcium imaging** data: load raw `.mat` files, explore trials, prepare normalized NPZs, run **dPCA** and **LDA** analyses.

## Setup

1. Clone or copy this folder
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Point to your dataset (the `Ca_mice_Livne` Dropbox folder):

```bash
# Linux/macOS
export LIVNEH_DATA_ROOT="/path/to/Ca_mice_Livne"

# Windows PowerShell
$env:LIVNEH_DATA_ROOT = "C:\path\to\Ca_mice_Livne"
```

See `.env.example` for variable names (the file is not loaded automatically).

## Multi-mouse quick start

Sessions are keyed as `protocol/mouse` (e.g. `20Hz/AL45`, `1Hz/AL48`, `Mock/AL42`). Phase-to-run maps live in `lib/config.py` (`MOUSE_PHASE_RUNS`).

```bash
# Run one session
python run_mouse.py --mouse 20Hz/AL45

# Run every 20Hz session
python run_mouse.py --protocol 20Hz

# Run all catalogued sessions
python run_mouse.py --all

# Subset of steps (skip data prep if NPZs already exist)
python run_mouse.py --mouse 20Hz/AL45 --skip-prep --steps dpca,dpca_downstream,svm_axes,axes_trajectories,lda,trajectories
```

`run_mouse.py` auto-detects the Dropbox path on this machine if `LIVNEH_DATA_ROOT` is unset. Override with `--data-root` if needed.

To run a single step manually for one session, set both env vars first:

```bash
$env:LIVNEH_DATA_ROOT = "C:\path\to\Ca_mice_Livne"
$env:LIVNEH_MOUSE_ID = "20Hz/AL45"
python prepare_data.py
```

## Single-mouse quick start

From the repo root (defaults to `20Hz/AL45` when `LIVNEH_MOUSE_ID` is unset):

```bash
# 1. Build normalized trial packs (required for all analyses)
python prepare_data.py

# 2. Explore raw data (optional)
python inspect_ca_data.py
python inspect_trials.py
python plot_slm_averages.py

# 3. dPCA (per-run stimulus demixing)
python run_dpca.py
python run_dpca_downstream.py
python run_svm_axes.py
python run_axes_trajectories.py

# 4. LDA + trajectories
python run_lda.py
python run_trajectories.py
```

**Minimum for dPCA:** steps 1 + 3.  
**Minimum for LDA:** steps 1 + 4 (`run_lda.py`).  
**Full trajectories with `lda_proj`:** steps 1 + 4, then `run_trajectories.py` (needs LDA CSVs from `run_lda.py`).

## Directory layout

```text
livneh-mice/
├── run_mouse.py, prepare_data.py, run_dpca.py, ...  # entry scripts
├── lib/                                             # library code
│   ├── io/          # raw .mat + NPZ loaders
│   ├── explore/     # inspection & SLM plots
│   ├── prep/        # prepare_data
│   ├── dpca/        # dPCA + trajectories
│   └── lda/         # LDA fitting + plots
├── normalized_data/
│   └── 20Hz/AL45/   # also 1Hz/…, Mock/…; packs: pre.npz, airpuff.npz, water.npz, …
└── outputs/
    └── 20Hz/AL45/
        ├── exploration/
        ├── dpca/
        ├── lda/
        ├── trajectories/
        └── axes_trajectories/
```

## Analysis notes

- **Train pool z-score:** `prepare_data.py` fits per-neuron mean/std on phase packs (`pre`, `airpuff`, `water`), excluding SLM (condition id 26).
- **LDA trial features:** mean activity over frames `[60, 120)` from cue, pooled across phases.
- **dPCA:** 300 cue-relative frames per trial; analyses on `pre`, `airpuff`, `water`.
- **Thirst / valence axes** (`run_thirst_valence_axes.py`, also part of `run_dpca_downstream.py`):
  - Fixed from **pre** `3conditions` s-space: thirst = nacl − water; valence (centroid) = mean(nacl, airpuff) − water.
  - Second valence axis = pre `hedonic_valence` s-dPC1 (sign flipped so aversive is positive).
  - Later phases (including SLM) are projected onto those fixed axes.
  - Two normalization modes (run both by default; pass `--mode per_phase|shared|both`):
    - **per_phase** → `outputs/<protocol>/<mouse>/dpca/thirst_valence_axes/within_phase/`: each phase z-scored/centered on itself. Isolates within-phase condition separation (removes cross-phase drift).
    - **shared** → `outputs/<protocol>/<mouse>/dpca/thirst_valence_axes/drift/`: all phases normalized with the `pre` tensor's stats, so cross-phase drift is preserved. Adds `drift_summary.csv`, `drift_grand_mean_by_phase.png`, `drift_water_centroid_by_phase.png`.
  - Outputs: trial/centroid CSVs, SLM-shift and separability summaries, faceted + overlay centroid plots.

## Dependencies

`requirements.txt`: numpy, scipy, h5py, pandas, matplotlib, scikit-learn, dpca, numexpr.
