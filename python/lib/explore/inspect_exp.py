"""
Inspect exp_data.mat structure so we can correctly read slm_info (and other nested structs).

Run from repo root: python inspect_exp_data.py

Prints:
- How the file was opened (scipy vs h5py)
- All top-level keys and their types/shapes
- Full drill-down of slm_info (runs, run_number, etc.)
"""

import numpy as np

from lib.config import default_session_dir


def _describe(obj, indent=0):
    """Return a string description of a numpy/scipy object for debugging."""
    pref = "  " * indent
    if obj is None:
        return f"{pref}None"
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        sh = getattr(obj, "shape", None)
        dt = getattr(obj, "dtype", None)
        return f"{pref}ndarray shape={sh} dtype={dt}"
    if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        return f"{pref}record array names={obj.dtype.names}"
    t = type(obj).__name__
    return f"{pref}{t}"


def _inspect_struct_recursive(obj, name="", indent=0, max_depth=5):
    """Recursively print struct fields (scipy loadmat struct_as_record)."""
    if indent > max_depth:
        print("  " * indent + "... (max depth)")
        return
    pref = "  " * indent
    if obj is None:
        print(f"{pref}{name}: None")
        return
    if hasattr(obj, "shape"):
        sh = obj.shape
        if sh == () or (len(sh) == 1 and sh[0] == 1) or (len(sh) == 2 and sh[0] == 1 and sh[1] == 1):
            try:
                scalar = np.asarray(obj).flat[0]
                print(f"{pref}{name}: scalar/0-d -> {type(scalar).__name__}")
                _inspect_struct_recursive(scalar, name + ".flat[0]", indent + 1, max_depth)
                return
            except Exception:
                pass
        if hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
            print(f"{pref}{name}: record array shape={sh} names={obj.dtype.names}")
            for n in (obj.dtype.names or [])[:10]:
                _inspect_struct_recursive(obj[n], f"{name}.{n}", indent + 1, max_depth)
            if obj.dtype.names and len(obj.dtype.names) > 10:
                print(f"{pref}  ... and {len(obj.dtype.names) - 10} more fields")
            return
        print(f"{pref}{name}: array shape={sh} dtype={getattr(obj, 'dtype', None)}")
        if obj.size > 0 and obj.size <= 4:
            print(f"{pref}  value: {obj}")
        return
    # Python object (e.g. numpy.void or matlab struct as record)
    if hasattr(obj, "_fieldnames"):
        fields = obj._fieldnames
    elif hasattr(obj, "dtype") and getattr(obj.dtype, "names", None):
        fields = obj.dtype.names
    else:
        fields = [a for a in dir(obj) if not a.startswith("_") and not callable(getattr(obj, a, None))]
    if not fields:
        print(f"{pref}{name}: {type(obj).__name__} (no fields)")
        return
    print(f"{pref}{name}: {type(obj).__name__} fields={fields}")
    for f in fields[:15]:
        try:
            val = getattr(obj, f, None) if hasattr(obj, f) else (obj[f] if hasattr(obj, "__getitem__") else None)
            if f == "runs" and val is not None:
                _inspect_struct_recursive(val, f"{name}.runs", indent + 1, max_depth)
                # Also show first run's run_number if present
                vflat = np.atleast_1d(val)
                if vflat.size > 0:
                    r0 = vflat.flat[0]
                    for rn_name in ("run_number", "Run_number", "run_Number"):
                        rn = getattr(r0, rn_name, None)
                        if rn is not None:
                            print(f"{pref}  -> first run.{rn_name}: {np.asarray(rn).ravel()}")
                            break
            else:
                _inspect_struct_recursive(val, f"{name}.{f}", indent + 1, max_depth)
        except Exception as e:
            print(f"{pref}  {f}: <error {e}>")
    if len(fields) > 15:
        print(f"{pref}  ... and {len(fields) - 15} more fields")


def main():
    session_dir = default_session_dir()
    mat_path = session_dir / "exp_data.mat"
    if not mat_path.exists():
        print(f"File not found: {mat_path}")
        return

    print("=" * 70)
    print("exp_data.mat inspection")
    print("=" * 70)
    print(f"Path: {mat_path}")
    print()

    # ---- Try scipy ----
    print("--- Loading with scipy.io.loadmat ---")
    try:
        from scipy.io import loadmat
        raw = loadmat(
            str(mat_path),
            squeeze_me=True,
            struct_as_record=True,
            mat_dtype=True,
        )
        print("Success (v7.0 or older).")
        skip = {"__header__", "__version__", "__globals__"}
        keys = [k for k in raw.keys() if k not in skip]
        print(f"Top-level keys ({len(keys)}): {keys}")
        for k in keys:
            v = raw[k]
            print(f"  {k}: {_describe(v, indent=0).strip()}")
        print()

        if "slm_info" in raw:
            print("--- slm_info (scipy) ---")
            si = raw["slm_info"]
            _inspect_struct_recursive(si, "slm_info", indent=0, max_depth=6)
        else:
            # Look for any key containing slm
            slm_keys = [k for k in keys if "slm" in k.lower()]
            if slm_keys:
                print(f"--- Found keys containing 'slm': {slm_keys} ---")
                for k in slm_keys:
                    _inspect_struct_recursive(raw[k], k, indent=0, max_depth=6)
            else:
                print("No 'slm_info' or 'slm*' key in file.")
    except NotImplementedError as e:
        print(f"NotImplementedError (likely v7.3 / HDF5): {e}")
        raw = None
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        raw = None
    print()

    # ---- Try h5py ----
    print("--- Loading with h5py ---")
    try:
        import h5py
        with h5py.File(str(mat_path), "r") as f:
            keys = [k for k in f.keys() if not k.startswith("#")]
            print(f"Top-level keys: {keys}")
            if "slm_info" in f:
                print("slm_info present. Structure:")
                si = f["slm_info"]
                for k in si.keys():
                    print(f"  slm_info.{k}: {type(si[k]).__name__}")
                    if k == "runs":
                        runs = si[k]
                        if isinstance(runs, h5py.Group):
                            for rk in runs.keys():
                                print(f"    runs.{rk}: {type(runs[rk]).__name__} shape={np.array(runs[rk]).shape}")
                        elif isinstance(runs, h5py.Dataset):
                            arr = np.array(runs)
                            print(f"    runs dataset: shape={arr.shape} dtype={arr.dtype}")
            else:
                print("No 'slm_info' in HDF5 root.")
    except OSError as e:
        print(f"OSError (not HDF5 / file signature): {e}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
