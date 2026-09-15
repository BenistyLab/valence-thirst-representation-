"""
Load Ca_mice_Livne calcium imaging data from .mat files.

Requires LIVNEH_DATA_ROOT env var pointing to the Ca_mice_Livne folder.
  - session1/motion_corrected_dfFvalid.mat  -> dfFvalid [neurons x frames]
  - session1/exp_data.mat                  -> fs, lickFr, frame_cond_trial, etc.

MAT v7.0 and older: scipy.io.loadmat. MAT v7.3 (HDF5): h5py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lib.config import default_session_dir


def _load_mat_array_v73(mat_path, variable_name):
    """Load a single array from a MATLAB v7.3 (HDF5) .mat file."""
    try:
        import h5py
    except ImportError:
        raise ImportError("MATLAB v7.3 .mat files require h5py. Install with: pip install h5py")

    with h5py.File(str(mat_path), "r") as f:
        if variable_name not in f:
            keys = [k for k in f.keys() if not k.startswith("#")]
            raise KeyError(f"Variable '{variable_name}' not in {mat_path}. Keys: {keys}")
        node = f[variable_name]
        if isinstance(node, h5py.Dataset):
            dset = node
        elif isinstance(node, h5py.Group) and "value" in node:
            dset = node["value"]
        else:
            raise KeyError(f"Variable '{variable_name}' is not a simple array. Keys: {list(node.keys())}")
        arr = np.array(dset)
        if arr.ndim == 2:
            arr = arr.T
        return arr.astype(float)


def load_dfFvalid(session_dir=None):
    """Load motion-corrected dF/F from motion_corrected_dfFvalid.mat."""
    if session_dir is None:
        session_dir = default_session_dir()
    session_dir = Path(session_dir)
    mat_path = session_dir / "motion_corrected_dfFvalid.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Not found: {mat_path}")

    try:
        from scipy.io import loadmat
        data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        if "dfFvalid" in data:
            return np.asarray(data["dfFvalid"], dtype=float)
        skip = {"__header__", "__version__", "__globals__"}
        for key in data:
            if key in skip:
                continue
            arr = data[key]
            if isinstance(arr, np.ndarray):
                return np.asarray(arr, dtype=float)
        raise KeyError(f"No array found in {mat_path}. Keys: {[k for k in data if k not in skip]}")
    except NotImplementedError:
        return _load_mat_array_v73(mat_path, "dfFvalid")


def load_exp_data(session_dir=None):
    """Load experiment metadata from exp_data.mat."""
    if session_dir is None:
        session_dir = default_session_dir()
    session_dir = Path(session_dir)
    mat_path = session_dir / "exp_data.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Not found: {mat_path}")

    try:
        from scipy.io import loadmat
        raw = loadmat(
            str(mat_path),
            squeeze_me=True,
            struct_as_record=True,
            mat_dtype=True,
        )
    except NotImplementedError:
        raw = _load_mat_v73_to_dict(mat_path)

    skip = {"__header__", "__version__", "__globals__"}
    return {key: raw[key] for key in raw if key not in skip}


def _load_mat_v73_to_dict(mat_path):
    """Load a v7.3 .mat file into a dict of arrays (top-level datasets only)."""
    try:
        import h5py
    except ImportError:
        raise ImportError("MATLAB v7.3 .mat files require h5py. Install with: pip install h5py")

    out = {}
    with h5py.File(str(mat_path), "r") as f:
        for key in f.keys():
            if key.startswith("#"):
                continue
            node = f[key]
            if isinstance(node, h5py.Dataset):
                dset = node
            elif isinstance(node, h5py.Group) and "value" in node:
                dset = node["value"]
            else:
                continue
            arr = np.array(dset)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.T.ravel()
            elif arr.ndim == 2:
                arr = arr.T
            out[key] = arr
    return out


