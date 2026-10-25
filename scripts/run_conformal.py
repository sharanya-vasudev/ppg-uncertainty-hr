import sys
import time

import numpy as np
import pandas as pd

sys.path.append(".")
from src.config import ALL_FEATURES_FINAL, ACTIVITY_NAMES
from src.uncertainty import get_quantile_model
from src.conformal import cqr_scores, conformal_quantile

SMOKE = "--smoke" in sys.argv
LEVELS = [0.90] if SMOKE else [0.50, 0.70, 0.80, 0.90, 0.95]
N_CAL_SUBJECTS = 4        # scheme B: whole subjects held out for calibration
CAL_WINDOW_FRAC = 0.20    # scheme A: random 20% of windows held out
SEED = 42
FEATS = ALL_FEATURES_FINAL

df = pd.read_csv("data/processed/ppg_acc_features_v2.csv")
all_subjects = sorted(df["subject"].unique())
test_subjects = all_subjects[:1] if SMOKE else all_subjects
if SMOKE:
    print("SMOKE TEST: one held-out subject, 90% level only. Checks the code runs; NOT a result.\n")


def run_scheme(prop, cal, test, level):
    """Fit quantile models on prop (no feature scaling needed: tree models are
    invariant to it), conformalize on cal, apply to test."""
    alpha = 1 - level
    low_q, high_q = round(0.5 - level / 2, 4), round(0.5 + level / 2, 4)
    X_prop, y_prop = prop[FEATS].values, prop["true_hr"].values
    lo_m = get_quantile_model(low_q).fit(X_prop, y_prop)
    hi_m = get_quantile_model(high_q).fit(X_prop, y_prop)

    scores = cqr_scores(lo_m.predict(cal[FEATS].values), hi_m.predict(cal[FEATS].values),
                        cal["true_hr"].values)
    q = conformal_quantile(scores, alpha)

    lo, hi = lo_m.predict(test[FEATS].values), hi_m.predict(test[FEATS].values)
    out = test[["subject", "activity", "true_hr"]].copy()
    out["lo_raw"], out["hi_raw"] = lo, hi
    out["lo_conf"], out["hi_conf"] = lo - q, hi + q
    out["q"] = q
    return out


parts = []
for s in test_subjects:
    t0 = time.time()
    test = df[df["subject"] == s]
    others = [x for x in all_subjects if x != s]
    rng = np.random.default_rng(SEED + int(s))

    # Scheme B: calibrate on whole subjects the model never saw
    cal_subj = rng.choice(others, size=N_CAL_SUBJECTS, replace=False)
    prop_B = df[df["subject"].isin(others) & ~df["subject"].isin(cal_subj)]
    cal_B = df[df["subject"].isin(cal_subj)]

    # Scheme A: calibrate on random windows from the training subjects
    pool = df[df["subject"].isin(others)]
    is_cal = rng.random(len(pool)) < CAL_WINDOW_FRAC
    prop_A, cal_A = pool[~is_cal], pool[is_cal]

    msg = []
    for level in LEVELS:
        for scheme, prop, cal in [("A_window", prop_A, cal_A), ("B_subject", prop_B, cal_B)]:
            out = run_scheme(prop, cal, test, level)
            out["scheme"], out["level"] = scheme, level
            parts.append(out)
            if level == 0.90:
                msg.append(f"{scheme[0]}: q={out['q'].iloc[0]:+.1f} BPM")
    print(f"S{s}: {', '.join(msg)}  ({time.time() - t0:.0f}s)", flush=True)

res = pd.concat(parts, ignore_index=True)
res["cov_raw"] = (res["true_hr"] >= res["lo_raw"]) & (res["true_hr"] <= res["hi_raw"])
res["cov_conf"] = (res["true_hr"] >= res["lo_conf"]) & (res["true_hr"] <= res["hi_conf"])
res["w_raw"] = res["hi_raw"] - res["lo_raw"]
res["w_conf"] = res["hi_conf"] - res["lo_conf"]

if SMOKE:
    print(res.groupby("scheme")[["cov_raw", "cov_conf", "w_raw", "w_conf"]].mean().round(3))
    print("\nSmoke test finished without errors.")
    sys.exit(0)

print("\n=== Pooled coverage and mean width, by scheme and nominal level ===")
pooled = (res.groupby(["scheme", "level"])
          .agg(cov_raw=("cov_raw", "mean"), cov_conf=("cov_conf", "mean"),
               width_raw=("w_raw", "mean"), width_conf=("w_conf", "mean"))
          .round(3).reset_index())
print(pooled.to_string(index=False))
print("(Reference, Phase 10 quantile regression on 14 training subjects: "
      "0.437 / 0.610 / 0.709 / 0.812 / 0.878 at 50/70/80/90/95)")

r90 = res[res["level"] == 0.90]
rows = []
for s in sorted(r90["subject"].unique()):
    row = {"subject": s}
    for scheme in ["A_window", "B_subject"]:
        sub = r90[(r90["subject"] == s) & (r90["scheme"] == scheme)]
        tag = scheme[0]
        row[f"{tag}_raw"] = sub["cov_raw"].mean()
        row[f"{tag}_conf"] = sub["cov_conf"].mean()
        row[f"{tag}_q"] = sub["q"].iloc[0]
    rows.append(row)
per_subj = pd.DataFrame(rows).set_index("subject").round(3)
print("\n=== Per held-out subject at nominal 90%: coverage before (raw) and after (conf) ===")
print(per_subj.to_string())

print("\n=== How uneven is coverage across subjects? (nominal 0.90) ===")
for tag in ["A", "B"]:
    for kind in ["raw", "conf"]:
        c = per_subj[f"{tag}_{kind}"]
        within = ((c >= 0.85) & (c <= 0.95)).sum()
        print(f"Scheme {tag} {kind:<4}: mean {c.mean():.3f}, std {c.std():.3f}, "
              f"min {c.min():.3f} (S{c.idxmin()}), within 0.85-0.95: {within}/{len(c)}")

print("\n=== Per activity at 90%, scheme B (subject-level calibration) ===")
rb = r90[r90["scheme"] == "B_subject"]
by_act = (rb.groupby("activity")
          .agg(n=("cov_conf", "size"), cov_raw=("cov_raw", "mean"),
               cov_conf=("cov_conf", "mean"), width_conf=("w_conf", "mean"))
          .round(3))
by_act.index = [ACTIVITY_NAMES.get(a, str(a)) for a in by_act.index]
print(by_act.to_string())

neg = int((res["w_conf"] < 0).sum())
print(f"\nWindows where q shrank an interval below zero width: {neg} of {len(res)}")

pooled.to_csv("results/metrics/phase11_conformal_pooled.csv", index=False)
per_subj.to_csv("results/metrics/phase11_conformal_by_subject_90.csv")
by_act.to_csv("results/metrics/phase11_conformal_by_activity_90_schemeB.csv")
r90.to_csv("results/metrics/predictions/phase11_conformal_90.csv", index=False)
print("\nSaved three small summary tables (tracked) and the per-window file (gitignored).")
