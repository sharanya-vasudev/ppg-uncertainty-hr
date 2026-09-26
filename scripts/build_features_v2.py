import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(".")
from src.data_loader import load_subject_signals, FS_BVP, FS_ACC, FS_ACTIVITY, WINDOW_LEN_S, WINDOW_SHIFT_S
from src.ppg_processing import preprocess_ppg
from src.feature_extraction import extract_features_with_heterogeneity

DATA_ROOT = Path("data/raw/PPG_FieldStudy")
SUBJECTS = list(range(1, 16))

def process_subject(subject_id):
    sig = load_subject_signals(subject_id, DATA_ROOT)
    bvp, acc, hr_label, activity = sig["bvp"], sig["acc_g"], sig["hr_label"], sig["activity"]
    bvp_filtered = preprocess_ppg(bvp, fs=FS_BVP)

    rows = []
    for i in range(len(hr_label)):
        start_s = i * WINDOW_SHIFT_S
        bvp_start, bvp_end = int(start_s * FS_BVP), int(start_s * FS_BVP) + int(WINDOW_LEN_S * FS_BVP)
        acc_start, acc_end = int(start_s * FS_ACC), int(start_s * FS_ACC) + int(WINDOW_LEN_S * FS_ACC)
        if bvp_end > len(bvp_filtered) or acc_end > len(acc):
            break
        feats = extract_features_with_heterogeneity(
            bvp[bvp_start:bvp_end], bvp_filtered[bvp_start:bvp_end],
            acc[acc_start:acc_end], FS_BVP, FS_ACC
        )
        act_start, act_end = int(start_s * FS_ACTIVITY), min(int((start_s + WINDOW_LEN_S) * FS_ACTIVITY), len(activity))
        act_id = -1
        if act_start < act_end:
            vals, counts = np.unique(activity[act_start:act_end], return_counts=True)
            act_id = int(vals[np.argmax(counts)])
        row = {"subject": subject_id, "window_idx": i, "true_hr": hr_label[i], "activity": act_id}
        row.update(feats)
        rows.append(row)
    return rows

print("Rebuilding feature table with heterogeneity features...")
all_rows = []
for sid in tqdm(SUBJECTS):
    all_rows.extend(process_subject(sid))

df = pd.DataFrame(all_rows)
print(f"Total rows: {len(df)}, columns: {len(df.columns)}")
df.to_csv("data/processed/ppg_acc_features_v2.csv", index=False)
print("Saved: data/processed/ppg_acc_features_v2.csv")
