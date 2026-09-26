import sys
from pathlib import Path

import pandas as pd

sys.path.append(".")
from src.models import get_model
from src.evaluation import run_loso_cv

df = pd.read_csv("data/processed/ppg_acc_features_v2.csv")

PPG_FEATURES = ["candidate_hr_spectral", "ppg_sqi", "ppg_spectral_peak_dominance",
                "n_peaks", "mean_peak_interval_s", "std_peak_interval_s",
                "ppg_std", "ppg_rms", "ppg_peak_to_peak"]

# Original ACC set (includes raw per-axis MEANS - orientation-encoding, static)
ACC_FULL = ["acc_mean_mag", "acc_std_mag", "acc_rms_mag", "acc_dominant_freq_hz",
            "acc_cardiac_band_energy_frac", "acc_mean_ax", "acc_std_ax",
            "acc_mean_ay", "acc_std_ay", "acc_mean_az", "acc_std_az"]

# DYNAMIC-only ACC set: drops raw per-axis means (acc_mean_ax/ay/az), which
# primarily encode static wrist orientation - a person/device-fit-specific
# nuisance variable, not a motion or physiological signal. Keeps acc_mean_mag
# (orientation-invariant: |acc| stays ~1g regardless of orientation).
ACC_DYNAMIC = ["acc_mean_mag", "acc_std_mag", "acc_rms_mag", "acc_dominant_freq_hz",
               "acc_cardiac_band_energy_frac", "acc_std_ax", "acc_std_ay", "acc_std_az"]

FEATURE_SETS = {
    "ppg_only": PPG_FEATURES,
    "ppg_acc_full": PPG_FEATURES + ACC_FULL,
    "ppg_acc_dynamic_only": PPG_FEATURES + ACC_DYNAMIC,
}

print(f"Loaded {len(df)} windows\n")
for model_name in ["random_forest", "gradient_boosting"]:
    for key, cols in FEATURE_SETS.items():
        result = run_loso_cv(df, cols, lambda: get_model(model_name), show_progress=False)
        mae = result["abs_error"].mean()
        macro_mae = result.groupby("subject")["abs_error"].mean().mean()
        print(f"{model_name:<20} {key:<22} pooled MAE={mae:.2f}  macro MAE={macro_mae:.2f}")
