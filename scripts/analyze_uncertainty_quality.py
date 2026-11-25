import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(".")
from src.config import ACTIVITY_NAMES

P = "results/metrics/predictions/"
FRACS = [round(float(x), 2) for x in np.arange(0.50, 1.0001, 0.05)]   # retained share of windows
TABLE_FRACS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
REJECT_SHARE = 0.20
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

feat = pd.read_csv("data/processed/ppg_acc_features_v2.csv")
qr = pd.read_csv(P + "phase8_quantile_regression_90.csv")
cf = pd.read_csv(P + "phase11_conformal_loso_90.csv")

for name, d in [("quantile file", qr), ("conformal file", cf)]:
    ok = (len(d) == len(feat)
          and (d["subject"].values == feat["subject"].values).all()
          and np.allclose(d["true_hr"].values, feat["true_hr"].values))
    if not ok:
        raise SystemExit(f"Row alignment check FAILED for the {name}; stopping.")
print("Alignment check passed.\n")

df = pd.DataFrame({
    "subject": feat["subject"].values,
    "activity": feat["activity"].values,
    "acc_rms": feat["acc_rms_mag"].values,
    "sqi": feat["ppg_sqi"].values,
    "w": qr["interval_width"].values,            # uncertainty score: quantile interval width
    "abs_err": qr["abs_error_median"].values,
    "cov_conf": cf["cov_conf"].values.astype(bool),
    "w_conf": cf["w_conf"].values,
})
df["activity_name"] = df["activity"].map(ACTIVITY_NAMES)

# ---- 1. pooled rank correlations -------------------------------------------------
rho_w = spearmanr(df["w"], df["abs_err"])[0]
rho_rms = spearmanr(df["acc_rms"], df["abs_err"])[0]
rho_sqi = spearmanr(df["sqi"], df["abs_err"])[0]
print("=== 1. Pooled rank correlation with absolute error (all windows) ===")
print(f"interval width:           {rho_w:+.3f}   (Phase 8 reported 0.599 for the same quantity)")
print(f"accelerometer RMS:        {rho_rms:+.3f}")
print(f"PPG signal quality index: {rho_sqi:+.3f}   (negative expected: higher quality, lower error)")


# ---- 2. within-subject and within-activity ---------------------------------------
def within(col):
    rows = []
    for key, g in df.groupby(col):
        rows.append({col: key, "n": len(g),
                     "rho_width": spearmanr(g["w"], g["abs_err"])[0],
                     "rho_rms": spearmanr(g["acc_rms"], g["abs_err"])[0],
                     "rho_sqi": spearmanr(g["sqi"], g["abs_err"])[0]})
    return pd.DataFrame(rows).round(3)


by_subj = within("subject")
by_act = within("activity_name").sort_values("rho_width")
print("\n=== 2a. Within each subject (what one wearer's device would experience) ===")
print(by_subj.to_string(index=False))
print(f"median within-subject rho: width {by_subj['rho_width'].median():+.3f}, "
      f"RMS {by_subj['rho_rms'].median():+.3f}, SQI {by_subj['rho_sqi'].median():+.3f}")
print("\n=== 2b. Within each activity (does width rank errors beyond identifying the activity?) ===")
print(by_act.to_string(index=False))
print(f"median within-activity rho: width {by_act['rho_width'].median():+.3f}, "
      f"RMS {by_act['rho_rms'].median():+.3f}, SQI {by_act['rho_sqi'].median():+.3f}")

# ---- 3. selective prediction (risk-coverage) -------------------------------------
err = df["abs_err"].values
cov = df["cov_conf"].values.astype(float)
n = len(df)
orders = {   # most trusted first
    "width": np.argsort(df["w"].values, kind="stable"),
    "acc_rms": np.argsort(df["acc_rms"].values, kind="stable"),
    "ppg_sqi": np.argsort(-df["sqi"].values, kind="stable"),
    "oracle": np.argsort(err, kind="stable"),
}
curves = {}
for name, order in orders.items():
    ce, cc = np.cumsum(err[order]), np.cumsum(cov[order])
    curves[name] = {"mae": {}, "cov": {}}
    for f in sorted(set(FRACS) | set(TABLE_FRACS)):
        k = int(np.ceil(f * n))
        curves[name]["mae"][f] = ce[k - 1] / k
        curves[name]["cov"][f] = cc[k - 1] / k
overall = err.mean()

tbl = pd.DataFrame({
    "retained": TABLE_FRACS,
    "MAE_random": [overall] * len(TABLE_FRACS),
    "MAE_width": [curves["width"]["mae"][f] for f in TABLE_FRACS],
    "MAE_acc_rms": [curves["acc_rms"]["mae"][f] for f in TABLE_FRACS],
    "MAE_ppg_sqi": [curves["ppg_sqi"]["mae"][f] for f in TABLE_FRACS],
    "MAE_oracle": [curves["oracle"]["mae"][f] for f in TABLE_FRACS],
    "cov_width": [curves["width"]["cov"][f] for f in TABLE_FRACS],
}).round(3)
print("\n=== 3. Withholding the least-trusted windows: MAE (BPM) of what remains ===")
print(tbl.to_string(index=False))
print("(cov_width = conformal 90% coverage of the windows kept when ranking by width)")


def gain(name):
    m = np.mean([curves[name]["mae"][f] for f in FRACS])
    o = np.mean([curves["oracle"]["mae"][f] for f in FRACS])
    return (overall - m) / (overall - o)


print("\nShare of the oracle's MAE improvement captured, averaged over retention 50-100%:")
for name in ["width", "acc_rms", "ppg_sqi"]:
    print(f"  {name:<8} {gain(name):.3f}")
print("  (0 = no better than withholding at random, 1 = as good as knowing the true error)")

# ---- 4. fixed pooled threshold, per wearer ----------------------------------------
thr = float(np.quantile(df["w"].values, 1 - REJECT_SHARE))
rows = []
for s, g in df.groupby("subject"):
    keep = g["w"] <= thr
    rows.append({"subject": s, "n": len(g), "pct_withheld": 100 * (1 - keep.mean()),
                 "MAE_all": g["abs_err"].mean(), "MAE_kept": g.loc[keep, "abs_err"].mean(),
                 "cov_all": g["cov_conf"].mean(), "cov_kept": g.loc[keep, "cov_conf"].mean()})
sub_rej = pd.DataFrame(rows).round(3)
print(f"\n=== 4. One fixed rule for everyone: withhold windows with width above {thr:.1f} BPM "
      f"(the pooled {int(100 * (1 - REJECT_SHARE))}th percentile) ===")
print(sub_rej.to_string(index=False))
keep_all = df["w"] <= thr
print(f"Pooled: MAE {err.mean():.2f} -> {err[keep_all.values].mean():.2f}, "
      f"coverage {df['cov_conf'].mean():.3f} -> {df.loc[keep_all, 'cov_conf'].mean():.3f}, "
      f"{100 * (1 - keep_all.mean()):.1f}% withheld")

# ---- 5. Low / Medium / High label -------------------------------------------------
t1, t2 = np.quantile(df["w"].values, [1 / 3, 2 / 3])
df["label"] = pd.cut(df["w"], bins=[-np.inf, t1, t2, np.inf], labels=["Low", "Medium", "High"])
lab = (df.groupby("label", observed=True)
       .agg(n=("abs_err", "size"), MAE=("abs_err", "mean"), cov_conf=("cov_conf", "mean"),
            width_conf=("w_conf", "mean")).round(3))
print(f"\n=== 5. Uncertainty label from width tertiles (Low <= {t1:.1f} BPM < Medium <= {t2:.1f} BPM < High) ===")
print(lab.to_string())

by_subj.to_csv("results/metrics/phase14_correlations_by_subject.csv", index=False)
by_act.to_csv("results/metrics/phase14_correlations_by_activity.csv", index=False)
tbl.to_csv("results/metrics/phase14_selective_prediction.csv", index=False)
sub_rej.to_csv("results/metrics/phase14_fixed_threshold_by_subject.csv", index=False)
lab.to_csv("results/metrics/phase14_uncertainty_labels.csv")

# ---- figure -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
dec = pd.qcut(df["w"], 10, duplicates="drop")
g = df.groupby(dec, observed=True).agg(w=("w", "mean"), e=("abs_err", "mean"))
axes[0].plot(g["w"], g["e"], "o-")
axes[0].set_xlabel("Interval width (BPM), decile means")
axes[0].set_ylabel("Mean absolute error (BPM)")
axes[0].set_title("Error rises with predicted uncertainty")

for name, label in [("width", "Interval width"), ("acc_rms", "Accelerometer RMS"),
                    ("ppg_sqi", "PPG signal quality"), ("oracle", "Oracle (true error)")]:
    axes[1].plot(FRACS, [curves[name]["mae"][f] for f in FRACS], "o-" if name != "oracle" else "k:",
                 markersize=3, label=label)
axes[1].axhline(overall, color="gray", linestyle="--", label="Random withholding")
axes[1].set_xlabel("Share of windows retained")
axes[1].set_ylabel("MAE of retained windows (BPM)")
axes[1].set_title("Selective prediction")
axes[1].legend(fontsize=8)

x = np.arange(len(by_subj))
axes[2].bar(x - 0.2, by_subj["rho_width"], 0.4, label="Interval width")
axes[2].bar(x + 0.2, by_subj["rho_rms"], 0.4, label="Accelerometer RMS")
axes[2].set_xticks(x)
axes[2].set_xticklabels([f"S{s}" for s in by_subj["subject"]], rotation=90)
axes[2].set_ylabel("Within-subject Spearman with |error|")
axes[2].set_title("Rank agreement within each wearer")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/phase14_uncertainty_quality.png", dpi=150)
print("\nSaved: five small tables in results/metrics/ and results/figures/phase14_uncertainty_quality.png")
