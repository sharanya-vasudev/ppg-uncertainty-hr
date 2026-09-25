import numpy as np
import pandas as pd

for model in ["random_forest", "gradient_boosting"]:
    ppg_only = pd.read_csv(f"results/metrics/phase6_{model}_ppg_only.csv")
    ppg_acc = pd.read_csv(f"results/metrics/phase6_{model}_ppg_acc.csv")

    print(f"\n=== {model} ===")
    for label, mask_fn in [
        ("All windows", lambda df: pd.Series(True, index=df.index)),
        ("Excluding transient (activity=0)", lambda df: df["activity"] != 0),
    ]:
        mae_a = ppg_only.loc[mask_fn(ppg_only), "abs_error"].mean()
        mae_b = ppg_acc.loc[mask_fn(ppg_acc), "abs_error"].mean()
        n = mask_fn(ppg_only).sum()
        print(f"  {label:<35} n={n:<8} PPG-only={mae_a:.2f}  PPG+ACC={mae_b:.2f}  delta={mae_b-mae_a:+.2f}")
