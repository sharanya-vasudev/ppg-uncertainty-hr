import sys
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(".")
from src.config import ACTIVITY_NAMES

df = pd.read_csv("results/metrics/predictions/phase8_quantile_regression_90.csv")

print("=== 1. Quantile crossing check (low > high would be nonsensical) ===")
n_crossed = (df["pred_low"] > df["pred_high"]).sum()
print(f"Windows with pred_low > pred_high: {n_crossed} of {len(df)}")
print(f"Interval width - min: {df['interval_width'].min():.2f}, "
      f"median: {df['interval_width'].median():.2f}, max: {df['interval_width'].max():.2f}")

print("\n=== 2. Per-subject coverage (target 0.90) ===")
by_subj = df.groupby("subject").agg(
    coverage=("covered", "mean"),
    mean_width=("interval_width", "mean"),
    mae=("abs_error_median", "mean"),
).round(3)
print(by_subj.to_string())

print("\n=== 3. Per-activity coverage (target 0.90) ===")
by_act = df.groupby("activity").agg(
    n=("covered", "size"),
    coverage=("covered", "mean"),
    mean_width=("interval_width", "mean"),
    mae=("abs_error_median", "mean"),
).round(3)
by_act.index = [ACTIVITY_NAMES.get(a, str(a)) for a in by_act.index]
print(by_act.to_string())

rho, _ = spearmanr(df["interval_width"], df["abs_error_median"])
print("\n=== 4. Does interval width track difficulty? ===")
print(f"Spearman correlation, interval width vs |error|: {rho:.3f}")
print("(Preview only - the formal uncertainty-vs-error analysis is Phase 14.)")

by_subj.to_csv("results/metrics/phase8_quantile_by_subject.csv")
by_act.to_csv("results/metrics/phase8_quantile_by_activity.csv")
print("\nSaved small summary tables (safe to track in git).")
