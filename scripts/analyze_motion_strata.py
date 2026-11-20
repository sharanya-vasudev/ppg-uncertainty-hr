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
EXCLUDE = [5, 12]        # chosen in advance: flagged by three different methods
MIN_SUBJ_WINDOWS = 50    # per-subject coverage only where a subject has enough windows
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
print("Alignment check passed: all three files describe the same windows in the same order.")
agree = (cf["cov_raw"].values == qr["covered"].values).mean()
print(f"Uncalibrated coverage agrees between quantile and conformal files on "
      f"{agree:.4f} of windows (expect ~1.0)\n")

df = pd.DataFrame({
    "subject": feat["subject"].values,
    "activity": feat["activity"].values,
    "true_hr": feat["true_hr"].values,
    "acc_rms": feat["acc_rms_mag"].values,
    "pred": qr["pred_median"].values,
    "cov_raw": qr["covered"].values.astype(bool),
    "cov_conf": cf["cov_conf"].values.astype(bool),
    "w_conf": cf["w_conf"].values,
})
df["abs_err"] = (df["pred"] - df["true_hr"]).abs()
df["activity_name"] = df["activity"].map(ACTIVITY_NAMES)
COND = {1: "rest", 8: "rest", 5: "seated_daily", 6: "seated_daily", 7: "walking",
        4: "cycling", 2: "high_motion", 3: "high_motion", 0: "transient"}
df["condition"] = df["activity"].map(COND)
edges = [0, 1.00, 1.02, 1.06, 1.15, np.inf]
labels = ["<1.00 g", "1.00-1.02 g", "1.02-1.06 g", "1.06-1.15 g", ">1.15 g"]
df["motion_bin"] = pd.cut(df["acc_rms"], bins=edges, labels=labels, right=False)


def summarize(col):
    rows = []
    for key, g in df.groupby(col, observed=True):
        gx = g[~g["subject"].isin(EXCLUDE)]
        ps = g.groupby("subject")["cov_conf"].agg(["mean", "size"])
        ps = ps[ps["size"] >= MIN_SUBJ_WINDOWS]["mean"]
        mae = g["abs_err"].mean()
        rows.append({
            col: str(key), "n": len(g), "MAE": mae,
            "cov_raw": g["cov_raw"].mean(), "cov_conf": g["cov_conf"].mean(),
            "cov_conf_no_S5_S12": gx["cov_conf"].mean(),
            "width": g["w_conf"].mean(), "width_over_MAE": g["w_conf"].mean() / mae,
            "worst_subject_cov": ps.min() if len(ps) else np.nan, "n_subj": len(ps),
        })
    return pd.DataFrame(rows).round(3)


cond = summarize("condition").sort_values("MAE")
act = summarize("activity_name").sort_values("MAE")
mot = summarize("motion_bin")

print("=== By condition (sorted by MAE). cov_* are coverage of the 90% interval; width in BPM ===")
print(cond.to_string(index=False))
print("\n=== By activity (sorted by MAE) ===")
print(act.to_string(index=False))
print("\n=== By accelerometer RMS bin ===")
print(mot.to_string(index=False))

rho_a, _ = spearmanr(act["width"], act["MAE"])
rho_m, _ = spearmanr(mot["width"], mot["MAE"])
rho_c, _ = spearmanr(np.arange(len(mot)), mot["cov_conf"])
print(f"\nSpearman(width, MAE) across the {len(act)} activities: {rho_a:.2f}")
print(f"Spearman(width, MAE) across the {len(mot)} motion bins: {rho_m:.2f}")
print(f"Spearman(bin order, conformal coverage) across motion bins: {rho_c:.2f}")
print("(Rank agreement across a handful of conditions - descriptive, not a significance test.)")
print(f"Activities with conformal coverage below 0.85: {int((act['cov_conf'] < 0.85).sum())} of {len(act)}")

sub_act = pd.crosstab(df["subject"], df["activity_name"], values=df["cov_conf"], aggfunc="mean").round(3)
cond.to_csv("results/metrics/phase13_by_condition.csv", index=False)
act.to_csv("results/metrics/phase13_by_activity.csv", index=False)
mot.to_csv("results/metrics/phase13_by_motion_bin.csv", index=False)
sub_act.to_csv("results/metrics/phase13_subject_by_activity_coverage.csv")


def paired_bars(ax, names, a, b, la, lb, title, ylabel, hline=None, ylim=None):
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, a, w, label=la)
    ax.bar(x + w / 2, b, w, label=lb)
    if hline is not None:
        ax.axhline(hline, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
paired_bars(axes[0, 0], act["activity_name"], act["cov_raw"], act["cov_conf"],
            "Uncalibrated", "Conformal", "Coverage by activity", "Empirical coverage",
            hline=0.90, ylim=(0.4, 1.0))
paired_bars(axes[0, 1], act["activity_name"], act["MAE"], act["width"],
            "MAE (BPM)", "Mean 90% interval width (BPM)", "Error and interval width by activity", "BPM")
paired_bars(axes[1, 0], mot["motion_bin"], mot["cov_raw"], mot["cov_conf"],
            "Uncalibrated", "Conformal", "Coverage by accelerometer RMS", "Empirical coverage",
            hline=0.90, ylim=(0.4, 1.0))
paired_bars(axes[1, 1], mot["motion_bin"], mot["MAE"], mot["width"],
            "MAE (BPM)", "Mean 90% interval width (BPM)", "Error and interval width by accelerometer RMS", "BPM")
plt.tight_layout()
plt.savefig("results/figures/phase13_motion_specific.png", dpi=150)
print("\nSaved: four small tables in results/metrics/ and results/figures/phase13_motion_specific.png")
