"""
Centralized PPG-DaLiA subject loading.

IMPORTANT: the wrist ACC array inside SX.pkl is already in physical g units
(confirmed empirically: mean |acc| = 1.013g during a known-stationary window,
matching Earth's gravity). The "1/64g" unit note in the dataset's own readme
refers to the SEPARATE raw SX_E4.zip files, not this pre-synchronized field.
Do NOT divide this array by 64 - that was a bug present in earlier phases.
"""
from __future__ import annotations

import pickle
from pathlib import Path

FS_BVP = 64
FS_ACC = 32
FS_ACTIVITY = 4
WINDOW_LEN_S = 8
WINDOW_SHIFT_S = 2

ACTIVITY_NAMES = {
    0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer",
    4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working",
}


def load_subject(subject_id: int, data_root: Path) -> dict:
    """Load one subject's raw SX.pkl dictionary, unmodified."""
    pkl_path = data_root / f"S{subject_id}" / f"S{subject_id}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_subject_signals(subject_id: int, data_root: Path) -> dict:
    """Load and extract the signals actually used in this project, with
    correct units applied. ACC is used as-is (already in g) - NOT divided
    by 64.
    """
    d = load_subject(subject_id, data_root)
    return {
        "bvp": d["signal"]["wrist"]["BVP"].flatten(),
        "acc_g": d["signal"]["wrist"]["ACC"],  # already in g units, no conversion
        "hr_label": d["label"].flatten(),
        "activity": d["activity"].flatten(),
        "subject": subject_id,
    }
