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
ACC_DYNAMIC = ["acc_mean_mag", "acc_std_mag", "acc_rms_mag", "acc_dominant_freq_hz",
               "acc_cardiac_band_energy_frac", "acc_std_ax", "acc_std_ay", "acc_std_az"]

ACTIVITY_NAMES = {0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer",
                  4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working"}

FEATURE_SETS = {"ppg_only": PPG_FEATURES, "ppg_acc_dynamic": PPG_FEATURES + ACC_DYNAMIC}

for model_name in ["random_forest", "gradient_boosting"]:
    results = {}
    for key, cols in FEATURE_SETS.items():
        result = run_loso_cv(df, cols, lambda: get_model(model_name), show_progress=False)
        results[key] = result
        result.to_csv(f"results/metrics/phase6_final_{model_name}_{key}.csv", index=False)

    a = results["ppg_only"].groupby("subject")["abs_error"].mean()
    b = results["ppg_acc_dynamic"].groupby("subject")["abs_error"].mean()
    improved = (b < a).sum()
    print(f"\n=== {model_name} ===")
    print(f"Subjects improved by dynamic-ACC fusion: {improved}/15")
    print(f"Per-subject delta (fusion - ppg_only), positive = fusion worse:")
    print((b - a).round(2).to_string())

    print(f"\n{'Activity':<15}{'N':<8}{'PPG-only':<12}{'PPG+ACC(dyn)':<14}{'Delta':<8}")
    for act_id in sorted(results["ppg_only"]["activity"].unique()):
        if act_id == -1:
            continue
        mask_a = results["ppg_only"]["activity"] == act_id
        mask_b = results["ppg_acc_dynamic"]["activity"] == act_id
        n = mask_a.sum()
        if n < 100:
            continue
        mae_a = results["ppg_only"].loc[mask_a, "abs_error"].mean()
        mae_b = results["ppg_acc_dynamic"].loc[mask_b, "abs_error"].mean()
        name = ACTIVITY_NAMES.get(act_id, str(act_id))
        print(f"{name:<15}{n:<8}{mae_a:<12.2f}{mae_b:<14.2f}{mae_b-mae_a:+.2f}")
