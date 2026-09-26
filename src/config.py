"""
Canonical, finalized configuration - feature sets and constants shared across
Phase 6+ scripts.

ACC feature set history (Phase 6):
    Originally included raw per-axis means (acc_mean_ax/ay/az). Feature
    importance analysis + a controlled experiment showed these encode static
    wrist orientation (person/device-fit specific), not motion dynamics -
    causing PPG+ACC fusion to UNDERPERFORM PPG-only (verified across Ridge,
    Random Forest, Gradient Boosting). Removing them and keeping only dynamic
    ACC features REVERSED this: fusion now beats PPG-only consistently
    (Random Forest: 9.25 -> 8.91 BPM MAE; Gradient Boosting: 9.41 -> 9.08 BPM).
    acc_mean_mag is kept (orientation-invariant: |acc| ~1g regardless of
    device orientation).
"""
from __future__ import annotations

PPG_FEATURES = [
    "candidate_hr_spectral", "ppg_sqi", "ppg_spectral_peak_dominance",
    "n_peaks", "mean_peak_interval_s", "std_peak_interval_s",
    "ppg_std", "ppg_rms", "ppg_peak_to_peak",
]

# Official ACC feature set as of Phase 6 - see module docstring.
ACC_FEATURES_FINAL = [
    "acc_mean_mag", "acc_std_mag", "acc_rms_mag", "acc_dominant_freq_hz",
    "acc_cardiac_band_energy_frac", "acc_std_ax", "acc_std_ay", "acc_std_az",
]

# Deprecated: static-orientation features, kept in the raw feature CSV for
# reference/ablation, excluded from the official feature set.
ACC_FEATURES_DEPRECATED_ORIENTATION = ["acc_mean_ax", "acc_mean_ay", "acc_mean_az"]

ALL_FEATURES_FINAL = PPG_FEATURES + ACC_FEATURES_FINAL

BEST_MODEL_NAME = "random_forest"

ACTIVITY_NAMES = {
    0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer",
    4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working",
}
