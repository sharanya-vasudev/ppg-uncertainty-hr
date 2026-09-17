import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.append(".")
from src.ppg_processing import preprocess_ppg
from src.baseline import estimate_hr_spectral

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
    predictions = np.full(n_windows, np.nan)
    window_activity = np.full(n_windows, -1, dtype=int)

    for i in range(n_windows):
        start_s = i * WINDOW_SHIFT_S
        start_sample = int(start_s * FS_BVP)
        end_sample = start_sample + int(WINDOW_LEN_S * FS_BVP)
        if end_sample > len(bvp_filtered):
            break
        window = bvp_filtered[start_sample:end_sample]
        try:
            predictions[i] = estimate_hr_spectral(window, fs=FS_BVP)
        except ValueError:
            predictions[i] = np.nan

        act_start = int(start_s * FS_ACTIVITY)
        act_end = min(int((start_s + WINDOW_LEN_S) * FS_ACTIVITY), len(activity))
        if act_start < act_end:
            vals, counts = np.unique(activity[act_start:act_end], return_counts=True)
            window_activity[i] = vals[np.argmax(counts)]

    valid = ~np.isnan(predictions)
    return {
        "subject": subject_id,
        "predictions": predictions[valid],
        "true": hr_label[valid],
        "activity": window_activity[valid],
    }

all_results = []
print("Evaluating all 15 subjects...")
for sid in tqdm(SUBJECTS):
    result = evaluate_subject(sid)
    all_results.append(result)
    mae = np.mean(np.abs(result["predictions"] - result["true"]))
    print(f"  S{sid}: {len(result['true'])} windows, MAE={mae:.2f} BPM")

all_pred = np.concatenate([r["predictions"] for r in all_results])
all_true = np.concatenate([r["true"] for r in all_results])
all_act = np.concatenate([r["activity"] for r in all_results])
all_subj = np.concatenate([np.full(len(r["true"]), r["subject"]) for r in all_results])

micro_mae = np.mean(np.abs(all_pred - all_true))
micro_rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))
micro_corr, _ = pearsonr(all_pred, all_true)

per_subject_mae = [np.mean(np.abs(r["predictions"] - r["true"])) for r in all_results]
macro_mae = np.mean(per_subject_mae)
macro_std = np.std(per_subject_mae)

print("\n=== Pooled / micro-average (all windows equal) ===")
print(f"MAE:  {micro_mae:.2f} BPM")
print(f"RMSE: {micro_rmse:.2f} BPM")
print(f"Correlation: {micro_corr:.3f}")

print("\n=== Subject-averaged / macro-average (each subject equal) ===")
print(f"Mean of per-subject MAE: {macro_mae:.2f} BPM (std across subjects: {macro_std:.2f})")

print("\n=== Per-subject MAE ===")
for r, mae in zip(all_results, per_subject_mae):
    print(f"  S{r['subject']:<3} n={len(r['true']):<5} MAE={mae:.2f}")

print("\n=== Per-activity breakdown (pooled across all subjects) ===")
print(f"{'Activity':<15} {'N windows':<12} {'MAE':<10} {'RMSE':<10}")
for act_id in sorted(set(all_act)):
    if act_id == -1:
        continue
    mask = all_act == act_id
    if mask.sum() < 20:
        continue
    act_mae = np.mean(np.abs(all_pred[mask] - all_true[mask]))
    act_rmse = np.sqrt(np.mean((all_pred[mask] - all_true[mask]) ** 2))
    name = ACTIVITY_NAMES.get(act_id, f"unknown_{act_id}")
    print(f"{name:<15} {mask.sum():<12} {act_mae:<10.2f} {act_rmse:<10.2f}")

results_dir = Path("results/metrics")
results_dir.mkdir(parents=True, exist_ok=True)
np.savez(
    results_dir / "baseline_ppg_only_all_subjects.npz",
    predictions=all_pred, true=all_true, activity=all_act, subject=all_subj,
)
print(f"\nSaved to {results_dir / 'baseline_ppg_only_all_subjects.npz'}")
