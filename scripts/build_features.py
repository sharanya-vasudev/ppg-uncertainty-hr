import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(".")
from src.ppg_processing import preprocess_ppg
from src.feature_extraction import extract_features

DATA_ROOT = Path("data/raw/PPG_FieldStudy")
SUBJECTS = list(range(1, 16))
FS_BVP = 64
FS_ACC = 32
FS_ACTIVITY = 4
WINDOW_LEN_S = 8
WINDOW_SHIFT_S = 2

def load_subject(subject_id, data_root):
    pkl_path = data_root / f"S{subject_id}" / f"S{subject_id}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def process_subject(subject_id):
    d = load_subject(subject_id, DATA_ROOT)
    bvp = d["signal"]["wrist"]["BVP"].flatten()
    acc = d["signal"]["wrist"]["ACC"] / 64.0  # raw units -> g
    hr_label = d["label"].flatten()
    activity = d["activity"].flatten()
    bvp_filtered = preprocess_ppg(bvp, fs=FS_BVP)

    n_windows = len(hr_label)
    rows = []
    n_dropped = 0

    for i in range(n_windows):
        start_s = i * WINDOW_SHIFT_S
        bvp_start = int(start_s * FS_BVP)
        bvp_end = bvp_start + int(WINDOW_LEN_S * FS_BVP)
        acc_start = int(start_s * FS_ACC)
        acc_end = acc_start + int(WINDOW_LEN_S * FS_ACC)

        if bvp_end > len(bvp_filtered) or acc_end > len(acc):
            n_dropped += 1
            continue

        bvp_raw_win = bvp[bvp_start:bvp_end]
        bvp_filt_win = bvp_filtered[bvp_start:bvp_end]
        acc_win = acc[acc_start:acc_end]

        feats = extract_features(bvp_raw_win, bvp_filt_win, acc_win, FS_BVP, FS_ACC)

        act_start = int(start_s * FS_ACTIVITY)
        act_end = min(int((start_s + WINDOW_LEN_S) * FS_ACTIVITY), len(activity))
        act_id = -1
        if act_start < act_end:
            vals, counts = np.unique(activity[act_start:act_end], return_counts=True)
            act_id = int(vals[np.argmax(counts)])

        row = {"subject": subject_id, "window_idx": i, "true_hr": hr_label[i], "activity": act_id}
        row.update(feats)
        rows.append(row)

    return rows, n_dropped

print("Building features for all 15 subjects...")
all_rows = []
total_dropped = 0
for sid in tqdm(SUBJECTS):
    rows, dropped = process_subject(sid)
    all_rows.extend(rows)
    total_dropped += dropped
    if dropped > 0:
        print(f"  S{sid}: dropped {dropped} windows (insufficient trailing signal)")

df = pd.DataFrame(all_rows)
print(f"\nTotal feature rows: {len(df)}")
print(f"Total windows dropped: {total_dropped}")
print(f"Feature columns ({len(df.columns) - 4}): {[c for c in df.columns if c not in ('subject','window_idx','true_hr','activity')]}")

nan_counts = df.isna().sum()
nan_cols = nan_counts[nan_counts > 0]
print(f"\nColumns with NaN values:")
if len(nan_cols) > 0:
    print(nan_cols)
else:
    print("  none")

out_path = Path("data/processed/ppg_acc_features_all_subjects.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({df.shape[0]} rows x {df.shape[1]} columns)")
