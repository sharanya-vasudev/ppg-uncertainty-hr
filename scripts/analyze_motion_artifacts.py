import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.append(".")
from src.ppg_processing import preprocess_ppg, ppg_sqi
from src.baseline import estimate_hr_spectral_interpolated
from src.accelerometer import motion_features

DATA_ROOT = Path("data/raw/PPG_FieldStudy")
SUBJECTS = list(range(1, 16))
FS_BVP = 64
FS_ACC = 32
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
    acc = d["signal"]["wrist"]["ACC"] / 64.0  # convert to g
    hr_label = d["label"].flatten()
    activity = d["activity"].flatten()
    bvp_filtered = preprocess_ppg(bvp, fs=FS_BVP)

    n_windows = len(hr_label)
    rows = []

    for i in range(n_windows):
        start_s = i * WINDOW_SHIFT_S
        bvp_start = int(start_s * FS_BVP)
        bvp_end = bvp_start + int(WINDOW_LEN_S * FS_BVP)
        acc_start = int(start_s * FS_ACC)
        acc_end = acc_start + int(WINDOW_LEN_S * FS_ACC)

        if bvp_end > len(bvp_filtered) or acc_end > len(acc):
            break

        bvp_raw_win = bvp[bvp_start:bvp_end]
        bvp_filt_win = bvp_filtered[bvp_start:bvp_end]
        acc_win = acc[acc_start:acc_end]

        try:
            pred = estimate_hr_spectral_interpolated(bvp_filt_win, fs=FS_BVP)
        except ValueError:
            continue

        sqi = ppg_sqi(bvp_raw_win, fs=FS_BVP)
        mfeat = motion_features(acc_win, fs=FS_ACC)

        act_start = int(start_s * FS_ACTIVITY)
        act_end = min(int((start_s + WINDOW_LEN_S) * FS_ACTIVITY), len(activity))
        act_id = -1
        if act_start < act_end:
            vals, counts = np.unique(activity[act_start:act_end], return_counts=True)
            act_id = vals[np.argmax(counts)]

        rows.append({
            "subject": subject_id, "true": hr_label[i], "pred": pred,
            "abs_error": abs(pred - hr_label[i]), "activity": act_id,
            "sqi": sqi, **mfeat,
        })
    return rows

print("Processing all 15 subjects...")
all_rows = []
for sid in tqdm(SUBJECTS):
    all_rows.extend(evaluate_subject(sid))

import pandas as pd
df = pd.DataFrame(all_rows)
print(f"\nTotal windows: {len(df)}")

# --- Pooled correlations: motion/quality features vs abs error ---
print("\n=== Pooled correlations with |HR error| ===")
print(f"{'Feature':<28} {'Pearson r':<12} {'Spearman r':<12}")
features = ["rms_mag", "std_mag", "dominant_freq_hz", "cardiac_band_energy_frac", "sqi"]
for feat in features:
    pr, _ = pearsonr(df[feat], df["abs_error"])
    sr, _ = spearmanr(df[feat], df["abs_error"])
    print(f"{feat:<28} {pr:<12.3f} {sr:<12.3f}")

# --- Subject-level (macro) correlation check for the two headline features ---
print("\n=== Subject-level correlation check (mean +/- std across 15 subjects) ===")
for feat in ["rms_mag", "sqi"]:
    per_subj_r = []
    for sid in SUBJECTS:
        sub = df[df["subject"] == sid]
        if len(sub) > 10 and sub[feat].std() > 0:
            r, _ = pearsonr(sub[feat], sub["abs_error"])
            per_subj_r.append(r)
    print(f"{feat:<28} mean r = {np.mean(per_subj_r):+.3f}  (std = {np.std(per_subj_r):.3f}, n_subjects={len(per_subj_r)})")

# --- Binned plots ---
figures_dir = Path("results/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

def binned_error_plot(feature, xlabel, filename, n_bins=10):
    bins = pd.qcut(df[feature], q=n_bins, duplicates="drop")
    grouped = df.groupby(bins, observed=True)["abs_error"].agg(["mean", "std", "count"])
    x = [interval.mid for interval in grouped.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(x, grouped["mean"], yerr=grouped["std"] / np.sqrt(grouped["count"]),
                marker="o", capsize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean |HR error| (BPM)")
    ax.set_title(f"{xlabel} vs HR error (binned into deciles, error bars = SEM)")
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150)
    plt.close()
    print(f"Saved: results/figures/{filename}")

binned_error_plot("rms_mag", "Accelerometer RMS magnitude (g)", "motion_rms_vs_error.png")
binned_error_plot("sqi", "PPG Signal Quality Index (band power ratio)", "sqi_vs_error.png")
binned_error_plot("cardiac_band_energy_frac", "ACC cardiac-band energy fraction", "acc_cardiac_overlap_vs_error.png")

# --- Bar chart: HR error by activity (interpolated baseline, saved as figure this time) ---
act_summary = df[df["activity"] != -1].groupby("activity")["abs_error"].mean().sort_values()
act_labels = [ACTIVITY_NAMES.get(a, str(a)) for a in act_summary.index]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(act_labels, act_summary.values, color="steelblue")
ax.set_ylabel("Mean |HR error| (BPM)")
ax.set_title("HR error by activity (interpolated baseline, all subjects)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(figures_dir / "hr_error_by_activity.png", dpi=150)
plt.close()
print("Saved: results/figures/hr_error_by_activity.png")

# Save the full feature table for later phases (feature engineering will build on this)
results_dir = Path("results/metrics")
df.to_csv(results_dir / "motion_analysis_all_subjects.csv", index=False)
print(f"\nSaved: {results_dir / 'motion_analysis_all_subjects.csv'} ({len(df)} rows)")
