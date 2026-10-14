# Python trial decoder groundwork

This folder is a new, separate implementation area for the trial decoder. Its initial loader ports the grouping and extraction behavior of the repository's MATLAB `loadMouse.m` and `uniteRuns.m`.

## Load a session

Dependencies: Python, NumPy, h5py, and PyTorch with CUDA support. The MATLAB files must be v7.3 MAT files. On Windows, install the general requirements and the CUDA wheel separately:

```powershell
python -m pip install numpy h5py
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

```python
from data_loader import load_mouse

runs, condition_names = load_mouse(
    r"C:\path\to\Ca_mice_Livne\Mock\AL48-20250930",
    min_duration=300,
    to_zscore=False,
)

for run in runs:
    print(run.name, run.type, run.X.shape, run.condition.shape)
```

`Run.X` has shape `(cells, time, trials)`. `Run.condition` contains the original condition IDs, and `condition_names` maps IDs to names. `Run.trial_index` points back to zero-based rows in `frame_cond_trial`. Groups follow the legacy MATLAB output order: pre-SLM, SLM-airpuff, SLM-water, any raw runs after the final water-SLM run, then all retained raw runs. Empty groups are omitted.

The loader infers the SLM target intent from the `SLM_cell_airpuff_` or `SLM_cell_water_` XML filename recorded in `slm_info.runs`, which is the human-readable information associated with MATLAB's `target_intent` metadata in these v7.3 files.

## Initial pre-run classifier

`train_pre_classifier.py` trains a sequence-to-one LSTM for water, salt, and airpuff using only the pre-run trials. It uses stratified four-fold out-of-fold evaluation, fits per-cell normalization from each fold's training partition, and writes per-fold checkpoints, out-of-fold probabilities, metrics, and a confusion matrix under `outputs/<session-folder>/`.

The script requires CUDA and exits with an error instead of silently training on CPU. Example:

```powershell
python train_pre_classifier.py "C:\path\to\Ca_mice_Livne\Mock\AL48-20250930"
```

## Shared feed-forward encoder with run-group heads

`train_mean_ff_multhead.py` averages activity over samples 61:300 of each trial (matching the MATLAB pairwise loader's 60-sample discard), fits a fold-local per-cell standardization, and trains separate water/salt/airpuff heads for pre-run, SLM-water, and SLM-airpuff. Defaults are a shared `347 → 16 → 8` ReLU encoder and linear `8 → 3` decoder heads; set `--encoder-hidden 0 --decoder-hidden 4` to reproduce the small-hidden-decoder variant. It reuses the LSTM/SVM pre-run folds when the saved `fold_assignments.csv` is available, creates stratified folds for the two SLM groups, and evaluates every source group through every decoder head. This script requires CUDA and writes probabilities, fold assignments, fold metrics, confusion matrices, and checkpoints under `outputs/<session-folder>/mean_ff_multhead_encoder16/` by default.

For the weight-decay comparison, use `--encoder-hidden 0 --decoder-hidden 0` to select the original direct 347-to-8 encoder and linear heads, and set `--weight-decay` to the value being evaluated.

```powershell
.\.venv\Scripts\python.exe train_mean_ff_multhead.py "C:\path\to\Ca_mice_Livne\Mock\AL48-20250930"
```
