"""
Read slm_info.runs from exp_data.mat: target_intent and targeted_rois per run.
Outputs which run numbers have which targets and which neurons were activated.
Results saved to str(EXPLORATION_OUT)/slm_runs_targets.txt.
"""

from pathlib import Path

import numpy as np

from lib.config import EXPLORATION_OUT, default_session_dir


def _decode_matlab_string(arr):
    """Decode MATLAB char array (scipy or h5py) to Python str."""
    if arr is None:
        return ""
    arr = np.asarray(arr)
    if arr.size == 0:
        return ""
    # Scipy: may be (n,) or (1,n) uint8 or uint16
    if arr.dtype.kind in ("U", "S"):
        return "".join(np.atleast_1d(arr).ravel().astype(str))
    flat = np.atleast_1d(arr).ravel()
    if flat.dtype.kind == "O":
        # Object array (e.g. cell of strings)
        try:
            return str(flat.flat[0])
        except Exception:
            return ""
    try:
        return flat.astype(np.uint16).tobytes().decode("utf-16", errors="replace").strip("\x00")
    except Exception:
        try:
            return flat.astype(np.uint8).tobytes().decode("ascii", errors="replace").strip("\x00")
        except Exception:
            return str(arr)


def _get_run_number(r, run_index):
    """Extract run_number from a single run struct (scipy)."""
    for name in ("run_number", "Run_number", "run_Number"):
        val = getattr(r, name, None)
        if val is None and hasattr(r, "dtype") and getattr(r.dtype, "names", None) and name in (r.dtype.names or ()):
            try:
                val = r[name]
            except (KeyError, IndexError):
                pass
        if val is not None:
            v = np.asarray(val).ravel()
            if v.size > 0:
                return int(v.flat[0])
    return run_index + 1  # fallback


def _get_target_intent(r):
    """Extract target_intent string from a single run struct (scipy)."""
    val = getattr(r, "target_intent", None)
    if val is None and hasattr(r, "dtype") and getattr(r.dtype, "names", None) and "target_intent" in (r.dtype.names or ()):
        try:
            val = r["target_intent"]
        except (KeyError, IndexError):
            pass
    return _decode_matlab_string(val) if val is not None else ""


def _get_targeted_rois(r):
    """Extract targeted_rois as list of int (cells chosen for SLM; matches row number in dfFvalid)."""
    val = getattr(r, "targeted_rois", None)
    if val is None and hasattr(r, "dtype") and getattr(r.dtype, "names", None) and "targeted_rois" in (r.dtype.names or ()):
        try:
            val = r["targeted_rois"]
        except (KeyError, IndexError):
            pass
    if val is None:
        return []
    arr = np.asarray(val).ravel()
    return arr.astype(int).tolist()


def _parse_slm_runs_h5py(f, runs_node):
    """Parse slm_info.runs from an open h5py File. Returns list of dicts or []."""
    import h5py
    out = []
    if isinstance(runs_node, h5py.Group):
        keys = sorted([k for k in runs_node.keys() if not k.startswith("#")], key=lambda x: int(x) if x.isdigit() else 0)
        for key in keys:
            elem = runs_node[key]
            if not isinstance(elem, h5py.Group):
                continue
            rn = None
            if "run_number" in elem:
                rn = int(np.array(elem["run_number"]).ravel().flat[0])
            intent = ""
            if "target_intent" in elem:
                ref = elem["target_intent"]
                if isinstance(ref, h5py.Dataset):
                    arr = np.array(ref)
                    if arr.dtype == np.dtype("O") or (arr.dtype.kind == "O"):
                        try:
                            arr = np.array(f[arr.flat[0]])
                        except Exception:
                            arr = arr
                    intent = _decode_matlab_string(arr)
                else:
                    try:
                        arr = np.array(f[ref[0]])
                        intent = _decode_matlab_string(arr)
                    except Exception:
                        pass
            rois = []
            if "targeted_rois" in elem:
                rois = np.array(elem["targeted_rois"]).ravel().astype(int).tolist()
            if rn is None:
                rn = len(out) + 1
            out.append({"run_number": rn, "target_intent": intent, "targeted_rois": rois})
    elif isinstance(runs_node, h5py.Dataset):
        refs = np.array(runs_node).ravel()
        for i, ref in enumerate(refs):
            elem = f[ref]
            rn = int(np.array(elem["run_number"]).ravel().flat[0]) if "run_number" in elem else i + 1
            intent = ""
            if "target_intent" in elem:
                intent = _decode_matlab_string(np.array(elem["target_intent"]))
            rois = np.array(elem["targeted_rois"]).ravel().astype(int).tolist() if "targeted_rois" in elem else []
            out.append({"run_number": rn, "target_intent": intent, "targeted_rois": rois})
    return out


def load_slm_runs_detailed(session_dir=None):
    """
    Load slm_info.runs with run_number, target_intent, targeted_rois per run.
    Returns list of dicts: [{"run_number": int, "target_intent": str, "targeted_rois": [int, ...]}, ...]
    """
    if session_dir is None:
        session_dir = default_session_dir()
    session_dir = Path(session_dir)
    mat_path = session_dir / "exp_data.mat"
    if not mat_path.exists():
        return []

    # Try h5py first (v7.3 / HDF5). Avoids scipy.loadmat hanging on v7.3 files.
    try:
        import h5py
        with h5py.File(str(mat_path), "r") as f:
            if "slm_info" in f and "runs" in f["slm_info"]:
                si = f["slm_info"]
                runs_node = si["runs"]
                out = _parse_slm_runs_h5py(f, runs_node)
                if out:
                    return out
    except (OSError, Exception):
        pass

    # Try scipy (v7.0 and older)
    try:
        from scipy.io import loadmat
        raw = loadmat(
            str(mat_path),
            squeeze_me=True,
            struct_as_record=True,
            mat_dtype=True,
        )
        if "slm_info" not in raw:
            return []
        si = raw["slm_info"]
        # Unwrap (1,1) or 0-d with max iterations to avoid infinite loop
        if hasattr(si, "shape"):
            for _ in range(20):
                if not (hasattr(si, "shape") and si.shape in ((), (1,), (1, 1)) and hasattr(si, "flat")):
                    break
                si = si.flat[0]
        runs = getattr(si, "runs", None)
        if runs is None and hasattr(si, "dtype") and getattr(si.dtype, "names", None) and "runs" in (si.dtype.names or ()):
            runs = si["runs"]
        if runs is None:
            return []
        runs = np.atleast_1d(runs)
        out = []
        for i, r in enumerate(runs.ravel()):
            rn = _get_run_number(r, i)
            intent = _get_target_intent(r)
            rois = _get_targeted_rois(r)
            out.append({"run_number": rn, "target_intent": intent, "targeted_rois": rois})
        return out
    except NotImplementedError:
        pass
    except Exception:
        pass
    return []


def main():
    session_dir = default_session_dir()
    out_dir = EXPLORATION_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "slm_runs_targets.txt"

    runs = load_slm_runs_detailed(session_dir)
    if not runs:
        with open(out_path, "w") as f:
            f.write("Session: {}\n".format(session_dir))
            f.write("No SLM runs found or could not read slm_info.runs.\n")
        print("Wrote (no runs):", out_path)
        return

    lines = []
    lines.append("Session: {}".format(session_dir))
    lines.append("")
    lines.append("Run number -> target intent, and neurons (ROIs) activated in that run")
    lines.append("(targeted_rois = cells chosen for SLM, matches row number in dfFvalid)")
    lines.append("")
    for r in runs:
        rn = r["run_number"]
        intent = r["target_intent"] or "(unknown)"
        rois = r["targeted_rois"]
        lines.append("Run {}: target_intent = {}".format(rn, intent))
        lines.append("  targeted_rois (neurons activated): {} (n={})".format(rois, len(rois)))
        lines.append("")
    text = "\n".join(lines)

    with open(out_path, "w") as f:
        f.write(text)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
