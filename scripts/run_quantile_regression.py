import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(".")
from src.config import ALL_FEATURES_FINAL
from src.uncertainty import run_loso_quantile_cv

df = pd.read_csv("data/processed/ppg_acc_features_v2.csv")

LOW_Q, HIGH_Q = 0.05, 0.95  # a 90% nominal prediction interval: 95th - 5th percentile

print(f"Loaded {len(df)} windows, {df['subject'].nunique()} subjects")
print(f"Training quantile regression: [{LOW_Q}, {HIGH_Q}] -> nominal {round((HIGH_Q-LOW_Q)*100)}% prediction interval\n")

result = run_loso_quantile_cv(df, ALL_FEATURES_FINAL, LOW_Q, HIGH_Q)

median_mae = result["abs_error_median"].mean()
empirical_coverage = result["covered"].mean()
mean_width = result["interval_width"].mean()

print(f"Median (q=0.5) point-estimate MAE: {median_mae:.2f} BPM "
      f"(compare: Random Forest point estimate was 8.91 BPM)")
print(f"\nNominal coverage target: {round((HIGH_Q-LOW_Q)*100)}%")
print(f"Empirical coverage (fraction of true HR values actually inside the interval): {empirical_coverage*100:.1f}%")
print(f"Mean prediction interval width: {mean_width:.2f} BPM")
print(f"\n(This is a quick sanity check only - full calibration analysis across")
print(f" multiple nominal levels, with proper reliability diagrams, is Phase 10.)")

result.to_csv("results/metrics/predictions/phase8_quantile_regression_90.csv", index=False)
print(f"\nSaved: results/metrics/predictions/phase8_quantile_regression_90.csv")
