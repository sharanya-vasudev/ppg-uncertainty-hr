import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(".")
from src.models import get_model
from src.evaluation import run_loso_cv

df = pd.read_csv("data/processed/ppg_acc_features_v2.csv")

PPG_FEATURES = ["candidate_hr_spectral", "ppg_sqi", "ppg_spectral_peak_dominance",
                "n_peaks", "mean_peak_interval_s", "std_peak_interval_s",
                "ppg_std", "ppg_rms", "ppg_peak_to_peak"]
ACC_FEATURES = ["acc_mean_mag", "acc_std_mag", "acc_rms_mag", "acc_dominant_freq_hz",
                 "acc_cardiac_band_energy_frac", "acc_mean_ax", "acc_std_ax",
                 "acc_mean_ay", "acc_std_ay", "acc_mean_az", "acc_std_az"]
HETERO_FEATURES = ["acc_subwin_rms_std", "acc_subwin_rms_range"]

FEATURE_SETS = {
    "ppg_only": PPG_FEATURES,
    "ppg_acc": PPG_FEATURES + ACC_FEATURES,
    "ppg_acc_hetero": PPG_FEATURES + ACC_FEATURES + HETERO_FEATURES,
}

print(f"Loaded {len(df)} windows\n")
summary = []
for model_name in ["random_forest", "gradient_boosting"]:
    for key, cols in FEATURE_SETS.items():
        result = run_loso_cv(df, cols, lambda: get_model(model_name), show_progress=False)
        mae = result["abs_error"].mean()
        macro_mae = result.groupby("subject")["abs_error"].mean().mean()
        print(f"{model_name:<20} {key:<18} pooled MAE={mae:.2f}  macro MAE={macro_mae:.2f}")
        summary.append({"model": model_name, "feature_set": key, "mae": mae, "macro_mae": macro_mae})
        result.to_csv(f"results/metrics/phase7_test_{model_name}_{key}.csv", index=False)

pd.DataFrame(summary).to_csv("results/metrics/phase7_heterogeneity_test.csv", index=False)
