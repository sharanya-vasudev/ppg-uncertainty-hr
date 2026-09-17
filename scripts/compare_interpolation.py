import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.append(".")
from src.ppg_processing import preprocess_ppg
from src.baseline import estimate_hr_spectral, estimate_hr_spectral_interpolated

DATA_ROOT = Path("data/raw/PPG_FieldStudy")
SUBJECTS = list(range(1, 16))
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

def evaluate_subject(subject_id):
    d = load_subject(subject_id, DATA_ROOT)
    bvp = d["signal"]["wrist"]["BVP"].flatten()
    hr_label = d["label"].flatten()
    activity = d["activity"].flatten()
    bvp_filtered = preprocess_ppg(bvp, fs=FS_BVP)

    n_windows = len(hr_label)
    pred_naive = np.full(n_windows, np.nan)
    pred_interp = np.full(n_windows, np.nan)
    window_activity = np.full(n_windows, -1, dtype=int)

    for i in range(n_windows):
        start_s = i * WINDOW_SHIFT_S
        start_sample = int(start_s * FS_BVP)
        end_sample = start_sample + int(WINDOW_LEN_S * FS_BVP)
        if end_sample > len(bvp_filtered):
            break
        window = bvp_filtered[start_sample:end_sample]
        try:
            pred_naive[i] = estimate_hr_spectral(window, fs=FS_BVP)
            pred_interp[i] = estimate_hr_spectral_interpolated(window, fs=FS_BVP)
        except ValueError:
            pass

        act_start = int(start_s * FS_ACTIVITY)
        act_end = min(int((start_s + WINDOW_LEN_S) * FS_ACTIVITY), len(activity))
        if act_start < act_end:
            vals, counts = np.unique(activity[act_start:act_end], return_counts=True)
            window_activity[i] = vals[np.argmax(counts)]

    valid = ~np.isnan(pred_naive) & ~np.isnan(pred_interp)
    return {
        "pred_naive": pred_naive[valid], "pred_interp": pred_interp[valid],
        "true": hr_label[valid], "activity": window_activity[valid],
    }

print("Evaluating all 15 subjects (both methods)...")
all_results = [evaluate_subject(sid) for sid in tqdm(SUBJECTS)]

all_naive = np.concatenate([r["pred_naive"] for r in all_results])
all_interp = np.concatenate([r["pred_interp"] for r in all_results])
all_true = np.concatenate([r["true"] for r in all_results])
all_act = np.concatenate([r["activity"] for r in all_results])

def report(name, pred, true):
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    corr, _ = pearsonr(pred, true)
    print(f"{name:<25} MAE={mae:.2f}  RMSE={rmse:.2f}  corr={corr:.3f}")
    return mae

print("\n=== Overall (pooled) ===")
naive_mae = report("Naive (bin-quantized)", all_naive, all_true)
interp_mae = report("Interpolated", all_interp, all_true)
print(f"MAE improvement: {naive_mae - interp_mae:+.2f} BPM")

print("\n=== Per-activity MAE comparison ===")
print(f"{'Activity':<15} {'Naive':<10} {'Interp':<10} {'Improvement':<12}")
for act_id in sorted(set(all_act)):
    if act_id == -1:
        continue
    mask = all_act == act_id
    if mask.sum() < 20:
        continue
    n_mae = np.mean(np.abs(all_naive[mask] - all_true[mask]))
    i_mae = np.mean(np.abs(all_interp[mask] - all_true[mask]))
    name = ACTIVITY_NAMES.get(act_id, str(act_id))
    print(f"{name:<15} {n_mae:<10.2f} {i_mae:<10.2f} {n_mae-i_mae:+.2f}")

ratio = all_interp / all_true
oct_down = ((ratio > 0.45) & (ratio < 0.55)).sum()
print(f"\nOctave-down rate (interpolated method): {100*oct_down/len(all_true):.1f}% "
      f"(was 7.9% for naive method)")

results_dir = Path("results/metrics")
np.savez(results_dir / "baseline_interpolation_comparison.npz",
         pred_naive=all_naive, pred_interp=all_interp, true=all_true, activity=all_act)
print(f"\nSaved to {results_dir / 'baseline_interpolation_comparison.npz'}")
