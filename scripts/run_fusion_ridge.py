import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(".")
from src.models import get_model
from src.evaluation import run_loso_cv

df = pd.read_csv("data/processed/ppg_acc_features_all_subjects.csv")

PPG_FEATURES = ["candidate_hr_spectral", "ppg_sqi", "ppg_spectral_peak_dominance",
                "n_peaks", "mean_peak_interval_s", "std_peak_interval_s",
                "ppg_std", "ppg_rms", "ppg_peak_to_peak"]
ACC_FEATURES = ["acc_mean_mag", "acc_std_mag", "acc_rms_mag", "acc_dominant_freq_hz",
                 "acc_cardiac_band_energy_frac", "acc_mean_ax", "acc_std_ax",
                 "acc_mean_ay", "acc_std_ay", "acc_mean_az", "acc_std_az"]
ALL_FEATURES = PPG_FEATURES + ACC_FEATURES

FEATURE_SETS = {
    "ppg_only": (PPG_FEATURES, "PPG-only (Model A)"),
    "acc_only": (ACC_FEATURES, "ACC-only (Model B)"),
    "ppg_acc": (ALL_FEATURES, "PPG+ACC (Model C)"),
}

print(f"Loaded {len(df)} windows, {df['subject'].nunique()} subjects\n")

summary = []
for key, (cols, label) in FEATURE_SETS.items():
    print(f"Running LOSO CV: {label} ({len(cols)} features) with Ridge...")
    result = run_loso_cv(df, cols, lambda: get_model("ridge"))

    pooled_mae = result["abs_error"].mean()
    pooled_rmse = np.sqrt((result["abs_error"] ** 2).mean())
    macro_mae = result.groupby("subject")["abs_error"].mean().mean()

    print(f"  Pooled MAE: {pooled_mae:.2f} BPM | RMSE: {pooled_rmse:.2f} BPM | Macro MAE: {macro_mae:.2f} BPM\n")
    summary.append({"feature_set": label, "model": "ridge",
                     "pooled_mae": pooled_mae, "pooled_rmse": pooled_rmse, "macro_mae": macro_mae})
    result.to_csv(f"results/metrics/phase6_ridge_{key}.csv", index=False)

summary_df = pd.DataFrame(summary)
print("=== Summary ===")
print(summary_df.to_string(index=False))
summary_df.to_csv("results/metrics/phase6_ridge_summary.csv", index=False)

print(f"\nFor comparison - Phase 3 PPG-only spectral baseline (no ML): MAE 18.33 BPM")
