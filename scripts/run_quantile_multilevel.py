import sys
import pandas as pd

sys.path.append(".")
from src.config import ALL_FEATURES_FINAL
from src.uncertainty import run_loso_quantile_cv

df = pd.read_csv("data/processed/ppg_acc_features_v2.csv")
LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]

print(f"Loaded {len(df)} windows. Training quantile regression at {len(LEVELS)} levels "
      f"(reuses the Phase 8 code, so each level retrains its own low/high/median models).\n")

rows = []
for level in LEVELS:
    low_q, high_q = round(0.5 - level / 2, 4), round(0.5 + level / 2, 4)
    print(f"Level {level:.0%}: training quantiles [{low_q}, {high_q}] ...", flush=True)
    result = run_loso_quantile_cv(df, ALL_FEATURES_FINAL, low_q, high_q)
    coverage = result["covered"].mean()
    width = result["interval_width"].mean()
    rows.append({"nominal": level, "empirical": coverage, "mean_width_bpm": width})
    print(f"  -> empirical coverage {coverage:.3f}, mean width {width:.2f} BPM")

calib = pd.DataFrame(rows).round(3)
print("\n=== Quantile regression calibration, all levels ===")
print(calib.to_string(index=False))
calib.to_csv("results/metrics/phase10_quantile_calibration_levels.csv", index=False)
print("\nSaved: results/metrics/phase10_quantile_calibration_levels.csv")
