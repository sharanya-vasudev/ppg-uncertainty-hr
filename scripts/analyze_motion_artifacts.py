import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.append(".")
from src.data_loader import load_subject_signals, FS_BVP, FS_ACC, FS_ACTIVITY, WINDOW_LEN_S, WINDOW_SHIFT_S, ACTIVITY_NAMES
from src.ppg_processing import preprocess_ppg, ppg_sqi
from src.baseline import estimate_hr_spectral_interpolated
from src.accelerometer import motion_features

DATA_ROOT = Path("data/raw/PPG_FieldStudy")
SUBJECTS = list(range(1, 16))

def evaluate_subject(subject_id):
    sig = load_subject_signals(subject_id, DATA_ROOT)
    bvp, acc, hr_label, activity = sig["bvp"], sig["acc_g"], sig["hr_label"], sig["activity"]
    bvp_filtered = preprocess_ppg(bvp, fs=FS_BVP)

    n_windows = len(hr_label)
    rows = []
    for i in range(n_windows):
        start_s = i * WINDOW_SHIFT_S
        bvp_start, bvp_end = int(start_s * FS_BVP), int(start_s * FS_BVP) + int(WINDOW_LEN_S * FS_BVP)
        acc_start, acc_end = int(start_s * FS_ACC), int(start_s * FS_ACC) + int(WINDOW_LEN_S * FS_ACC)
        if bvp_end > len(bvp_filtered) or acc_end > len(acc):
            break
        bvp_raw_win, bvp_filt_win = bvp[bvp_start:bvp_end], bvp_filtered[bvp_start:bvp_end]
        acc_win = acc[acc_start:acc_end]
        try:
            pred = estimate_hr_spectral_interpolated(bvp_filt_win, fs=FS_BVP)
        except ValueError:
            continue
        sqi = ppg_sqi(bvp_raw_win, fs=FS_BVP)
        mfeat = motion_features(acc_win, fs=FS_ACC)
        act_start, act_end = int(start_s * FS_ACTIVITY), min(int((start_s + WINDOW_LEN_S) * FS_ACTIVITY), len(activity))
        act_id = -1
        if act_start < act_end:
            vals, counts = np.unique(activity[act_start:act_end], return_counts=True)
            act_id = vals[np.argmax(counts)]
        rows.append({"subject": subject_id, "true": hr_label[i], "pred": pred,
                      "abs_error": abs(pred - hr_label[i]), "activity": act_id, "sqi": sqi, **mfeat})
    return rows

print("Rebuilding Phase 4 analysis with corrected ACC units...")
all_rows = []
for sid in tqdm(SUBJECTS):
    all_rows.extend(evaluate_subject(sid))

df = pd.DataFrame(all_rows)
print(f"\nTotal windows: {len(df)}")
print(f"\nacc_rms_mag stats (should now center near 1.0, not 0.016):")
print(df["rms_mag"].describe())

print("\n=== Pooled correlations with |HR error| (should match previous run - scale invariant) ===")
for feat in ["rms_mag", "std_mag", "dominant_freq_hz", "cardiac_band_energy_frac", "sqi"]:
    pr, _ = pearsonr(df[feat], df["abs_error"])
    print(f"{feat:<28} Pearson r = {pr:.3f}")

figures_dir = Path("results/figures")
def binned_error_plot(feature, xlabel, filename, n_bins=10):
    bins = pd.qcut(df[feature], q=n_bins, duplicates="drop")
    grouped = df.groupby(bins, observed=True)["abs_error"].agg(["mean", "std", "count"])
    x = [interval.mid for interval in grouped.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(x, grouped["mean"], yerr=grouped["std"] / np.sqrt(grouped["count"]), marker="o", capsize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean |HR error| (BPM)")
    ax.set_title(f"{xlabel} vs HR error (CORRECTED units)")
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150)
    plt.close()
    print(f"Saved: results/figures/{filename}")

binned_error_plot("rms_mag", "Accelerometer RMS magnitude (g)", "motion_rms_vs_error.png")

df.to_csv("results/metrics/motion_analysis_all_subjects.csv", index=False)
print("\nSaved corrected results/metrics/motion_analysis_all_subjects.csv")
