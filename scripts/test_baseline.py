import sys
import pickle
from pathlib import Path

import numpy as np

sys.path.append(".")
from src.ppg_processing import preprocess_ppg
from src.baseline import estimate_hr_spectral

DATA_ROOT = Path("data/raw/PPG_FieldStudy")
SUBJECT = 1
FS_BVP = 64
WINDOW_LEN_S = 8
WINDOW_SHIFT_S = 2

def load_subject(subject_id, data_root):
    pkl_path = data_root / f"S{subject_id}" / f"S{subject_id}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")

d = load_subject(SUBJECT, DATA_ROOT)
bvp = d["signal"]["wrist"]["BVP"].flatten()
hr_label = d["label"].flatten()

bvp_filtered = preprocess_ppg(bvp, fs=FS_BVP)

def test_window(start_s, label_name):
    label_idx = int(start_s / WINDOW_SHIFT_S)
    start_sample = int(start_s * FS_BVP)
    end_sample = start_sample + int(WINDOW_LEN_S * FS_BVP)
    window = bvp_filtered[start_sample:end_sample]
    predicted_hr = estimate_hr_spectral(window, fs=FS_BVP)
    true_hr = hr_label[label_idx]
    print(f"{label_name} (t={start_s}s): predicted={predicted_hr:.1f} BPM, "
          f"true={true_hr:.1f} BPM, error={abs(predicted_hr-true_hr):.1f} BPM")

test_window(300, "Sitting")
test_window(2050, "Cycling")
