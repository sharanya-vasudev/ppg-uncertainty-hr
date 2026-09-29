import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.append(".")
from src.config import ALL_FEATURES_FINAL, ACTIVITY_NAMES
from src.gp_model import fit_predict_svgp

SMOKE = "--smoke" in sys.argv
SUBSAMPLE_EVERY = 4   # 8s windows shifted by 2s overlap 75%; every 4th is non-overlapping
N_INDUCING = 500
N_EPOCHS = 5 if SMOKE else 40
LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]

df = pd.read_csv("data/processed/ppg_acc_features_v2.csv")
subjects = sorted(df["subject"].unique())
if SMOKE:
    subjects = subjects[:1]
    print("SMOKE TEST: one held-out subject, 5 epochs. Checks the code runs; NOT a result.\n")

parts = []
for s in subjects:
    train = df[(df["subject"] != s) & (df["window_idx"] % SUBSAMPLE_EVERY == 0)]
    test = df[df["subject"] == s]
    t0 = time.time()
    out = fit_predict_svgp(
        train[ALL_FEATURES_FINAL].values, train["true_hr"].values,
        test[ALL_FEATURES_FINAL].values,
        n_inducing=N_INDUCING, n_epochs=N_EPOCHS, verbose=SMOKE,
    )
    part = test[["subject", "activity", "true_hr"]].copy()
    part["gp_mean"] = out["mean"]
    part["gp_total_std"] = out["total_std"]
    part["gp_latent_std"] = out["latent_std"]
    part["gp_noise_std"] = out["noise_std"]
    parts.append(part)
    mae = np.abs(out["mean"] - test["true_hr"].values).mean()
    print(f"S{s}: train {len(train)} windows, test {len(test)}, MAE {mae:.2f} BPM, "
          f"noise sd {out['noise_std']:.2f}, mean latent sd {out['latent_std'].mean():.2f} BPM, "
          f"loss {out['losses'][0]:.2f}->{out['losses'][-1]:.2f} (drop over last 5 epochs: {out['losses'][-6] - out['losses'][-1]:.3f}), f{time.time() - t0:.0f}s", flush=True)

if SMOKE:
    print("\nSmoke test finished without errors.")
    print("Sanity checks: loss should trend downward, MAE should be under ~20 BPM,")
    print("and the latent sd should be positive and below ~23 BPM.")
    sys.exit(0)

res = pd.concat(parts, ignore_index=True)
res["abs_error"] = (res["gp_mean"] - res["true_hr"]).abs()

print(f"\nPoint estimate (GP mean): MAE {res['abs_error'].mean():.2f} BPM, "
      f"RMSE {np.sqrt((res['abs_error'] ** 2).mean()):.2f} BPM "
      f"(reference: quantile median 8.69, Random Forest 8.91)")

print("\n=== Calibration at five nominal levels (mean +/- z * total std) ===")
rows = []
for level in LEVELS:
    half = norm.ppf(0.5 + level / 2) * res["gp_total_std"]
    covered = (res["true_hr"] >= res["gp_mean"] - half) & (res["true_hr"] <= res["gp_mean"] + half)
    rows.append({"nominal": level, "empirical": covered.mean(), "mean_width_bpm": (2 * half).mean()})
calib = pd.DataFrame(rows).round(3)
print(calib.to_string(index=False))
print("(Reference, quantile regression at 90%: coverage 0.812, mean width 29.98 BPM)")

half90 = norm.ppf(0.95) * res["gp_total_std"]
res["covered90"] = (res["true_hr"] >= res["gp_mean"] - half90) & (res["true_hr"] <= res["gp_mean"] + half90)
res["width90"] = 2 * half90

print("\n=== Per subject at 90% (latent_sd = epistemic part, noise_sd = constant per fold) ===")
by_subj = res.groupby("subject").agg(
    coverage=("covered90", "mean"), mean_width=("width90", "mean"),
    latent_sd=("gp_latent_std", "mean"), noise_sd=("gp_noise_std", "mean"),
    mae=("abs_error", "mean"),
).round(3)
print(by_subj.to_string())

print("\n=== Per activity at 90% ===")
by_act = res.groupby("activity").agg(
    n=("covered90", "size"), coverage=("covered90", "mean"),
    mean_width=("width90", "mean"), mae=("abs_error", "mean"),
).round(3)
by_act.index = [ACTIVITY_NAMES.get(a, str(a)) for a in by_act.index]
print(by_act.to_string())

rho_total, _ = spearmanr(res["gp_total_std"], res["abs_error"])
rho_latent, _ = spearmanr(res["gp_latent_std"], res["abs_error"])
print(f"\nSpearman, total std vs |error|:  {rho_total:.3f}   (quantile regression: 0.599)")
print(f"Spearman, latent std vs |error|: {rho_latent:.3f}   (the epistemic part alone)")

res.to_csv("results/metrics/predictions/phase9_gp_svgp.csv", index=False)
calib.to_csv("results/metrics/phase9_gp_calibration_levels.csv", index=False)
by_subj.to_csv("results/metrics/phase9_gp_by_subject.csv")
by_act.to_csv("results/metrics/phase9_gp_by_activity.csv")
print("\nSaved: per-window file (gitignored) and three small summary tables (tracked).")
