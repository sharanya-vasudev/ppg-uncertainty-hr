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
    "ppg_only": (PPG_FEATURES, "PPG-only"),
    "acc_only": (ACC_FEATURES, "ACC-only"),
    "ppg_acc": (ALL_FEATURES, "PPG+ACC"),
}

# Plain Linear Regression skipped deliberately: with standardized features and
# modest Ridge regularization (alpha=1.0), results are essentially identical.
# Compute is spent on the two model families that can actually test whether
# Ridge's LINEARITY (not features) was the limiting factor.
MODELS = ["ridge", "random_forest", "gradient_boosting"]

print(f"Loaded {len(df)} windows, {df['subject'].nunique()} subjects")
print(f"Running {len(MODELS)} models x {len(FEATURE_SETS)} feature sets = "
      f"{len(MODELS)*len(FEATURE_SETS)} combinations x 15 LOSO folds each.\n"
      f"Random Forest / Gradient Boosting are slower than Ridge - this will take a while.\n")

summary = []
for model_name in MODELS:
    for key, (cols, label) in FEATURE_SETS.items():
        print(f"Running: {label} ({len(cols)} features) + {model_name}...")
        result = run_loso_cv(df, cols, lambda: get_model(model_name))

        pooled_mae = result["abs_error"].mean()
        pooled_rmse = np.sqrt((result["abs_error"] ** 2).mean())
        macro_mae = result.groupby("subject")["abs_error"].mean().mean()

        print(f"  Pooled MAE: {pooled_mae:.2f} | RMSE: {pooled_rmse:.2f} | Macro MAE: {macro_mae:.2f}\n")
        summary.append({"feature_set": label, "model": model_name,
                         "pooled_mae": pooled_mae, "pooled_rmse": pooled_rmse, "macro_mae": macro_mae})
        result.to_csv(f"results/metrics/phase6_{model_name}_{key}.csv", index=False)

summary_df = pd.DataFrame(summary).sort_values(["feature_set", "pooled_mae"])
print("=== Full comparison ===")
print(summary_df.to_string(index=False))
summary_df.to_csv("results/metrics/phase6_full_comparison.csv", index=False)
print(f"\nFor reference - Phase 3 PPG-only spectral baseline (no ML): MAE 18.33 BPM")
