import pandas as pd

ACTIVITY_NAMES = {
    0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer",
    4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working",
}

for model in ["random_forest", "gradient_boosting"]:
    ppg_only = pd.read_csv(f"results/metrics/phase6_{model}_ppg_only.csv")
    ppg_acc = pd.read_csv(f"results/metrics/phase6_{model}_ppg_acc.csv")

    print(f"\n=== {model}: PPG-only vs PPG+ACC, by activity ===")
    print(f"{'Activity':<15} {'N':<8} {'PPG-only MAE':<14} {'PPG+ACC MAE':<14} {'Fusion delta':<12}")

    for act_id in sorted(ppg_only["activity"].unique()):
        if act_id == -1:
            continue
        mask_a = ppg_only["activity"] == act_id
        mask_b = ppg_acc["activity"] == act_id
        n = mask_a.sum()
        if n < 100:
            continue
        mae_a = ppg_only.loc[mask_a, "abs_error"].mean()
        mae_b = ppg_acc.loc[mask_b, "abs_error"].mean()
        name = ACTIVITY_NAMES.get(act_id, str(act_id))
        delta = mae_b - mae_a
        marker = " <- fusion HELPS" if delta < -0.1 else ("" if delta < 0.1 else "")
        print(f"{name:<15} {n:<8} {mae_a:<14.2f} {mae_b:<14.2f} {delta:+.2f}{marker}")
