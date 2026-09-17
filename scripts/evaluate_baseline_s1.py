import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.append(".")
from src.ppg_processing import preprocess_ppg
from src.baseline import estimate_hr_spectral

DATA_ROOT = Path("data/raw/PPG_FieldStudy")
SUBJECT = 1
FS_BVP = 64
FS_ACTIVITY = 4
WINDOW_LEN_S = 8
WINDOW_SHIFT_S = 2

ACTIVITY_NAMES = {
    0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer",
    4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working",
}

def load_subject(subject_id, data_root):
    pkl_path = data_root / f"S{subject_id}" / f"S{subject_id}.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")

print(f"Loading S{SUBJECT}...")
d = load_subject(SUBJECT, DATA_ROOT)
bvp = d["signal"]["wrist"]["BVP"].flatten()
hr_label = d["label"].flatten()
activity = d["activity"].flatten()

print("Filtering PPG...")
bvp_filtered = preprocess_ppg(bvp, fs=FS_BVP)

n_windows = len(hr_label)
predictions = np.full(n_windows, np.nan)
window_activity = np.full(n_windows, -1, dtype=int)

print(f"Estimating HR for {n_windows} windows...")
for i in range(n_windows):
    start_s = i * WINDOW_SHIFT_S
    start_sample = int(start_s * FS_BVP)
    end_sample = start_sample + int(WINDOW_LEN_S * FS_BVP)

    if end_sample > len(bvp_filtered):
        break  # last window may run past the signal end

    window = bvp_filtered[start_sample:end_sample]
    try:
        predictions[i] = estimate_hr_spectral(window, fs=FS_BVP)
    except ValueError:
        predictions[i] = np.nan  # window too short, shouldn't happen here but guard anyway

    # Determine the dominant activity ID during this window
    act_start = int(start_s * FS_ACTIVITY)
    act_end = int((start_s + WINDOW_LEN_S) * FS_ACTIVITY)
    act_end = min(act_end, len(activity))
    if act_start < act_end:
        vals, counts = np.unique(activity[act_start:act_end], return_counts=True)
        window_activity[i] = vals[np.argmax(counts)]

valid = ~np.isnan(predictions)
pred_valid = predictions[valid]
true_valid = hr_label[valid]
act_valid = window_activity[valid]

mae = np.mean(np.abs(pred_valid - true_valid))
rmse = np.sqrt(np.mean((pred_valid - true_valid) ** 2))
corr, _ = pearsonr(pred_valid, true_valid)

print(f"\n=== Overall (S{SUBJECT}, {valid.sum()} windows) ===")
print(f"MAE:  {mae:.2f} BPM")
print(f"RMSE: {rmse:.2f} BPM")
print(f"Correlation: {corr:.3f}")

print(f"\n=== Per-activity breakdown ===")
print(f"{'Activity':<15} {'N windows':<12} {'MAE':<10} {'RMSE':<10}")
for act_id in sorted(set(act_valid)):
    if act_id == -1:
        continue
    mask = act_valid == act_id
    if mask.sum() < 5:
        continue  # skip activities with too few windows to be meaningful
    act_mae = np.mean(np.abs(pred_valid[mask] - true_valid[mask]))
    act_rmse = np.sqrt(np.mean((pred_valid[mask] - true_valid[mask]) ** 2))
    name = ACTIVITY_NAMES.get(act_id, f"unknown_{act_id}")
    print(f"{name:<15} {mask.sum():<12} {act_mae:<10.2f} {act_rmse:<10.2f}")

# Save results for later comparison against fusion models (Phase 6+)
results_dir = Path("results/metrics")
results_dir.mkdir(parents=True, exist_ok=True)
np.savez(
    results_dir / f"baseline_ppg_only_S{SUBJECT}.npz",
    predictions=pred_valid,
    true=true_valid,
    activity=act_valid,
)
print(f"\nSaved to {results_dir / f'baseline_ppg_only_S{SUBJECT}.npz'}")
