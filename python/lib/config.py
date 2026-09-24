"""Central path and settings for the livneh-mice repo."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

# sklearn KMeans on Windows/MKL warns (and can leak) when the dataset has
# fewer chunks than OpenMP threads. dPCA hits this during regularizer="auto".
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings(
    "ignore",
    message="KMeans is known to have a memory leak on Windows with MKL",
    category=UserWarning,
    module=r"sklearn\.cluster\._kmeans",
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Original three 20Hz phase maps (pre-protocol catalog).
#
# MOUSE_PHASE_RUNS: dict[str, dict[str, tuple[int, ...]]] = {
#     "session1": {
#         "pre": (2,),
#         "airpuff": (3, 4),
#         "water": (5,),
#     },
#     "AL45": {
#         "pre": (2, 3),
#         "airpuff": (4,),
#         "water": (5,),
#     },
#     "AL48": {
#         "pre": (2,),
#         "airpuff": (3,),
#         "water": (4,),
#     },
# }

# ---------------------------------------------------------------------------
# Phase-run patterns
#
# Semantic phases used by prepare_data / dPCA / LDA. Values are raw run
# numbers from exp_data.trialRunMap.
#
# SLM airpuff vs water comes from slm_info.runs.xml_source filenames
# (SLM_cell_airpuff_ / SLM_cell_water_). Pre is the core
# water/nacl/airpuff block(s) after warmup (run 1) and before the first
# SLM run, with no condition-26 trials.
# ---------------------------------------------------------------------------
PHASE_PATTERN_STANDARD: dict[str, tuple[int, ...]] = {
    "pre": (2,),
    "airpuff": (3,),
    "water": (4,),
    "free-consumption": (5,)
}
PHASE_PATTERN_SPLIT_PRE: dict[str, tuple[int, ...]] = {
    "pre": (2, 3),
    "airpuff": (4,),
    "water": (5,),
    "free-consumption": (6,),
}
PHASE_PATTERN_SPLIT_AIRPUFF: dict[str, tuple[int, ...]] = {
    "pre": (2,),
    "airpuff": (3, 4),
    "water": (5,),
    "free-consumption": (6,),
}
PHASE_PATTERN_SPLIT_WATER: dict[str, tuple[int, ...]] = {
    "pre": (2,),
    "airpuff": (3,),
    "water": (4, 5),
    "free-consumption": (6,),
}

# Canonical session id is "{protocol}/{mouse}", e.g. "20Hz/AL45".
# folder = dated directory under Ca_mice_Livne/{protocol}/.
# AL49_1 and AL49_2 are two recording rounds of the same animal AL49.
_SESSIONS: tuple[tuple[str, str, str, dict[str, tuple[int, ...]]], ...] = (
    # protocol, mouse, folder, phase map
    # 20Hz
    ("20Hz", "AL41", "AL41-20250506", PHASE_PATTERN_STANDARD),
    ("20Hz", "AL42", "AL42-20250507", PHASE_PATTERN_SPLIT_AIRPUFF),
    ("20Hz", "AL45", "AL45-20250513", PHASE_PATTERN_SPLIT_PRE),
    ("20Hz", "AL48", "AL48-20250909", PHASE_PATTERN_STANDARD),
    ("20Hz", "AL49_1", "AL49_1-20260320", PHASE_PATTERN_STANDARD),
    ("20Hz", "AL49_2", "AL49_2-20260505", PHASE_PATTERN_STANDARD),
    # 1Hz
    ("1Hz", "AL41", "AL41-20250514", PHASE_PATTERN_STANDARD),
    ("1Hz", "AL42", "AL42-20250520", PHASE_PATTERN_STANDARD),
    ("1Hz", "AL45", "AL45-20250527", PHASE_PATTERN_STANDARD),
    ("1Hz", "AL48", "AL48-20250916", PHASE_PATTERN_SPLIT_PRE),
    ("1Hz", "AL49_1", "AL49_1-20260414", PHASE_PATTERN_STANDARD),
    # Mock
    ("Mock", "AL42", "AL42-20251015", PHASE_PATTERN_SPLIT_WATER),
    ("Mock", "AL48", "AL48-20250930", PHASE_PATTERN_STANDARD),
    ("Mock", "AL49_1", "AL49_1-20260331", PHASE_PATTERN_STANDARD),
    ("Mock", "AL49_2", "AL49_2-20260512", PHASE_PATTERN_STANDARD),
)

MOUSE_SESSION_REL: dict[str, Path] = {}
MOUSE_PHASE_RUNS: dict[str, dict[str, tuple[int, ...]]] = {}
SESSION_MOUSE: dict[str, str] = {}
SESSION_PROTOCOL: dict[str, str] = {}

for _protocol, _mouse, _folder, _phases in _SESSIONS:
    _sid = f"{_protocol}/{_mouse}"
    MOUSE_SESSION_REL[_sid] = Path(_protocol) / _folder
    MOUSE_PHASE_RUNS[_sid] = _phases
    SESSION_MOUSE[_sid] = _mouse
    SESSION_PROTOCOL[_sid] = _protocol

KNOWN_MICE: tuple[str, ...] = tuple(
    f"{protocol}/{mouse}" for protocol, mouse, _folder, _phases in _SESSIONS
)
PROTOCOLS: tuple[str, ...] = ("20Hz", "1Hz", "Mock")
DEFAULT_MOUSE_ID = "20Hz/AL45"


def resolve_mouse_id(mouse_id: str | None = None) -> str:
    mid = mouse_id or os.environ.get("LIVNEH_MOUSE_ID", DEFAULT_MOUSE_ID)
    if mid not in MOUSE_SESSION_REL:
        known = ", ".join(KNOWN_MICE)
        raise ValueError(
            f"Unknown session id {mid!r}. Use protocol/mouse "
            f"(e.g. '20Hz/AL45'). Known sessions: {known}"
        )
    return mid


def raw_data_root() -> Path:
    root = os.environ.get("LIVNEH_DATA_ROOT")
    if not root:
        raise EnvironmentError(
            "Set LIVNEH_DATA_ROOT to your Ca_mice_Livne folder "
            "(e.g. export LIVNEH_DATA_ROOT=/path/to/Ca_mice_Livne). See .env.example."
        )
    return Path(root)


def default_session_dir(mouse_id: str | None = None) -> Path:
    mid = resolve_mouse_id(mouse_id)
    return raw_data_root() / MOUSE_SESSION_REL[mid]


def phase_run_map(mouse_id: str | None = None) -> dict[str, tuple[int, ...]]:
    mid = resolve_mouse_id(mouse_id)
    return MOUSE_PHASE_RUNS[mid]


_mouse_id = resolve_mouse_id()
PHASE_RUN_MAP = phase_run_map(_mouse_id)
PHASE_KEYS: tuple[str, ...] = tuple(PHASE_RUN_MAP.keys())
_protocol = SESSION_PROTOCOL[_mouse_id]
_mouse = SESSION_MOUSE[_mouse_id]
NORMALIZED_DIR = REPO_ROOT / "normalized_data" / _protocol / _mouse
OUTPUTS_DIR = REPO_ROOT / "outputs" / _protocol / _mouse
EXPLORATION_OUT = OUTPUTS_DIR / "exploration"
DPCA_OUT = OUTPUTS_DIR / "dpca"
TRAJECTORIES_OUT = OUTPUTS_DIR / "trajectories"
AXES_TRAJECTORIES_OUT = OUTPUTS_DIR / "axes_trajectories"
AXES_TRAJECTORIES_PRE_PCA_OUT = AXES_TRAJECTORIES_OUT / "pre_pca"
AXES_TRAJECTORIES_AIRPUFF_PCA_OUT = AXES_TRAJECTORIES_OUT / "airpuff_pca"
AXES_TRAJECTORIES_WATER_PCA_OUT = AXES_TRAJECTORIES_OUT / "water_pca"
AXES_TRAJECTORIES_SVM_AXES_OUT = AXES_TRAJECTORIES_OUT / "svm_axes"
AXES_TRAJECTORIES_SVM_AXES_WATER_AIRPUFF_VALENCE_OUT = (
    AXES_TRAJECTORIES_SVM_AXES_OUT / "water_vs_airpuff_valence"
)
AXES_TRAJECTORIES_SVM_NO_DPCA_OUT = AXES_TRAJECTORIES_OUT / "svm_no_dpca"
AXES_TRAJECTORIES_SVM_NO_DPCA_WATER_AIRPUFF_VALENCE_OUT = (
    AXES_TRAJECTORIES_SVM_NO_DPCA_OUT / "water_vs_airpuff_valence"
)
LDA_OUT = OUTPUTS_DIR / "lda"

# LDA / binary decoding outputs
DECODE_BINARY_DIR = LDA_OUT / "cross_binary"
LDA_DIR = DECODE_BINARY_DIR / "lda"
LDA_POOLED_BINARY_DIR = LDA_DIR / "two_binary_ldas_pooled"
LDA_FOUR_CLASS_DIR = LDA_DIR / "four_class_run_by_outcome"

EVAL_RUN_KEYS = PHASE_KEYS

SLM_COND_ID = 26
