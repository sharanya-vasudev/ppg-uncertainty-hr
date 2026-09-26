import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(".")
from src.models import get_model

df = pd.read_csv("data/processed/ppg_acc_features_v2.csv")

PPG_FEATURES = ["candidate_hr_spectral", "ppg_sqi", "ppg_spectral_peak_dominance",
                "n_peaks", "mean_peak_interval_s", "std_peak_interval_s",
                "ppg_std", "ppg_rms", "ppg_peak_to_peak"]
ACC_FEATURES = ["acc_mean_mag", "acc_std_mag", "acc_rms_mag", "acc_dominant_freq_hz",
                 "acc_cardiac_band_energy_frac", "acc_mean_ax", "acc_std_ax",
                 "acc_mean_ay", "acc_std_ay", "acc_mean_az", "acc_std_az",
                 "acc_subwin_rms_std", "acc_subwin_rms_range"]
ALL_FEATURES = PPG_FEATURES + ACC_FEATURES

print("DIAGNOSTIC ONLY - fit on full data, not a performance claim.\n")

for model_name in ["random_forest", "gradient_boosting"]:
    X = StandardScaler().fit_transform(df[ALL_FEATURES].values)
    y = df["true_hr"].values
    model = get_model(model_name)
    model.fit(X, y)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X, y, n_repeats=3, random_state=42, n_jobs=-1)
        importances = r.importances_mean

    imp_df = pd.DataFrame({"feature": ALL_FEATURES, "importance": importances})
    imp_df["type"] = imp_df["feature"].apply(lambda f: "ACC" if f.startswith("acc") else "PPG")
    imp_df = imp_df.sort_values("importance", ascending=False)

    print(f"=== {model_name} feature importances ===")
    print(imp_df.to_string(index=False))
    print(f"\nTotal importance - PPG: {imp_df[imp_df['type']=='PPG']['importance'].sum():.3f}  "
          f"ACC: {imp_df[imp_df['type']=='ACC']['importance'].sum():.3f}\n")
