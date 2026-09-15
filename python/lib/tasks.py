"""Shared task definitions and condition mappings."""
from __future__ import annotations

import re

TASK_SPECS = {
    "hedonic_valence": {
        "display": "HedonicValence",
        "class0_name": "Aversive",
        "class1_name": "Appetitive",
        "rules": ((0, ("nacl", "airpuff")), (1, ("water",))),
        "lda_outcome_short": "Valence",
        "lda_scatter_slug": "valence",
    },
    "consumption_mode": {
        "display": "ConsumptionMode",
        "class0_name": "NonConsumption",
        "class1_name": "Consumption",
        "rules": ((0, ("airpuff", "pav", "waterfree")), (1, ("water", "nacl", "salt"))),
        "lda_outcome_short": "Consumption mode",
        "lda_scatter_slug": "consumption_mode",
    },
}

COND_IDS = {"water": {3, 7}, "nacl": {4}, "airpuff": {18}, "waterfree": {8}}

_BASE_COND_NAMES = {
    3: "water",
    4: "nacl",
    7: "waters",
    8: "waterfree",
    18: "airpuff",
    26: "slm",
}

_STATIC_COND_MAP = {
    "pre": {3: "water", 4: "nacl", 18: "airpuff", 26: "slm"},
    "airpuff": {3: "water", 4: "nacl", 18: "airpuff", 26: "slm"},
    "water": {3: "water", 4: "nacl", 18: "airpuff", 26: "slm", 7: "waters", 8: "waterfree"},
}


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def cond_name_map(run_key: str) -> dict[int, str]:
    """Map condition ids to names for a run pack (data-driven when NPZ exists)."""
    try:
        from lib.io.npz import load_npz_pack

        pack = load_npz_pack(run_key)
        ids = {int(x) for x in pack["trial_type"]}
        mapped = {cid: _BASE_COND_NAMES[cid] for cid in ids if cid in _BASE_COND_NAMES}
        if mapped:
            return mapped
    except FileNotFoundError:
        pass
    return _STATIC_COND_MAP.get(run_key, {})
