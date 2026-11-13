from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

P = Path("results/metrics/predictions")
cost = pd.read_csv("results/metrics/phase12_cost_measurements.csv").set_index("key")
rows = []


def add(label, inputs, key, y, pred, cov=None, width=None):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    c = cost.loc[key]
    rows.append({
        "Model": label, "Inputs": inputs,
        "MAE": f"{np.mean(np.abs(err)):.2f}", "RMSE": f"{np.sqrt(np.mean(err ** 2)):.2f}",
        "r": f"{np.corrcoef(y, pred)[0, 1]:.3f}",
        "Cov@90": "n/a" if cov is None else f"{cov:.3f}",
        "Width@90": "n/a" if width is None else f"{width:.1f}",
        "Parameters": c["params"], "Size (MB)": f"{c['size_mb']:.2f}",
        "Train (s)": f"{c['train_s']:.0f}",
        "Inference (ms)": "n/a" if pd.isna(c["infer_ms"]) else f"{c['infer_ms']:.2f}",
        "Pipeline (ms)": f"{c['pipeline_ms']:.2f}",
    })


def csv_point(name):
    d = pd.read_csv(P / name)
    return d["true_hr"], d["predicted_hr"]


base = np.load(P / "baseline_interpolation_comparison.npz")
add("Spectral baseline (interpolated)", "PPG", "spectral", base["true"], base["pred_interp"])
add("Ridge", "PPG", "ridge_ppg", *csv_point("phase6_ridge_ppg_only.csv"))
add("Random Forest", "PPG", "rf_ppg", *csv_point("phase6_final_random_forest_ppg_only.csv"))
add("Gradient Boosting", "PPG", "gb_ppg", *csv_point("phase6_final_gradient_boosting_ppg_only.csv"))
add("Random Forest, naive fusion", "PPG + all ACC", "rf_naive", *csv_point("phase6_random_forest_ppg_acc.csv"))
add("Random Forest (official Phase 6)", "PPG + dynamic ACC", "rf_dyn",
    *csv_point("phase6_final_random_forest_ppg_acc_dynamic.csv"))
add("Gradient Boosting", "PPG + dynamic ACC", "gb_dyn",
    *csv_point("phase6_final_gradient_boosting_ppg_acc_dynamic.csv"))

qr = pd.read_csv(P / "phase8_quantile_regression_90.csv")
add("Quantile regression (median, 5/95%)", "PPG + dynamic ACC", "qr_dyn",
    qr["true_hr"], qr["pred_median"], cov=qr["covered"].mean(), width=qr["interval_width"].mean())

gp = pd.read_csv(P / "phase9_gp_svgp.csv")
half = norm.ppf(0.95) * gp["gp_total_std"]
gp_cov = ((gp["true_hr"] >= gp["gp_mean"] - half) & (gp["true_hr"] <= gp["gp_mean"] + half)).mean()
add("Sparse variational GP", "PPG + dynamic ACC", "gp_dyn",
    gp["true_hr"], gp["gp_mean"], cov=gp_cov, width=(2 * half).mean())

cf = pd.read_csv(P / "phase11_conformal_loso_90.csv")
add("Conformalized quantile regression (all-subject calibration)", "PPG + dynamic ACC", "cqr_dyn",
    qr["true_hr"], qr["pred_median"], cov=cf["cov_conf"].mean(), width=cf["w_conf"].mean())

table = pd.DataFrame(rows)
not_built = {c: "not run" for c in table.columns}
for label in ["Deep fusion network", "Deep ensemble"]:
    table.loc[len(table)] = {**not_built, "Model": label, "Inputs": "PPG + ACC",
                             "Parameters": "not built (Phase 7 skipped by design)"}

notes = """
Notes
- MAE, RMSE (BPM) and Pearson r are pooled over 64,697 windows, leave-one-subject-out for every learned model.
- Cov@90 and Width@90 are for a nominal 90% prediction interval (width in BPM); n/a means a point estimate only.
- The conformalized row uses the quantile-regression median as its point estimate; only the interval differs.
- Parameters are in each family's own unit and are NOT like-for-like; compare Size (MB), the pickled model.
- Inference = model only; Pipeline = filtering + features + model (spectral row: filtering + FFT estimate).
- Timing: median of 200 single-window calls, ONE CPU thread, on the development laptop (see
  phase12_environment.txt). Models were fitted on all subjects except S1 (one example fold).
  This is NOT wearable hardware; it shows relative cost only.
- No significance testing yet (Phase 16).
"""


def to_md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


table.to_csv("results/tables/phase12_model_comparison.csv", index=False)
Path("results/tables/phase12_model_comparison.md").write_text(to_md(table) + "\n" + notes)
pd.set_option("display.width", 250, "display.max_columns", None, "display.max_colwidth", 48)
print(table.to_string(index=False))
print(notes)
print("Saved: results/tables/phase12_model_comparison.csv and .md")
