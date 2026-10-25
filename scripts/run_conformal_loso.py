import sys
import time

import numpy as np
import pandas as pd

sys.path.append(".")
from src.config import ALL_FEATURES_FINAL
from src.uncertainty import get_quantile_model
from src.conformal import cqr_scores, conformal_quantile

SMOKE = "--smoke" in sys.argv
ALPHA = 0.10                 # nominal 90% interval
LOW_Q, HIGH_Q = 0.05, 0.95
FEATS = ALL_FEATURES_FINAL

df = pd.read_csv("data/processed/ppg_acc_features_v2.csv")
subjects = [int(s) for s in sorted(df["subject"].unique())]
test_subjects = subjects[:1] if SMOKE else subjects
if SMOKE:
    print("SMOKE TEST: one held-out subject. Checks the code runs; NOT a result.\n")


def fit_pair(train):
    X, y = train[FEATS].values, train["true_hr"].values
    return get_quantile_model(LOW_Q).fit(X, y), get_quantile_model(HIGH_Q).fit(X, y)


# Step 1: out-of-subject calibration scores. The model excluding subjects {a, b}
# gives scores for a (used when b is the test subject) and for b (used when a is).
needed = sorted({tuple(sorted((a, b))) for a in test_subjects for b in subjects if b != a})
print(f"Fitting {len(needed)} two-subjects-excluded model pairs...", flush=True)
t0 = time.time()
S = {}
for a, b in needed:
    lo_m, hi_m = fit_pair(df[~df["subject"].isin([a, b])])
    for held, other in [(a, b), (b, a)]:
        d = df[df["subject"] == held]
        S[(held, other)] = cqr_scores(lo_m.predict(d[FEATS].values),
                                      hi_m.predict(d[FEATS].values), d["true_hr"].values)
print(f"  done in {time.time() - t0:.0f}s", flush=True)

# Step 2: for each test subject, q from the scores of all 14 other subjects.
parts = []
for s in test_subjects:
    scores = np.concatenate([S[(j, s)] for j in subjects if j != s])
    q = conformal_quantile(scores, ALPHA)
    lo_m, hi_m = fit_pair(df[df["subject"] != s])   # same base model as Phase 10
    test = df[df["subject"] == s]
    lo, hi = lo_m.predict(test[FEATS].values), hi_m.predict(test[FEATS].values)
    out = test[["subject", "activity", "true_hr"]].copy()
    out["lo_raw"], out["hi_raw"] = lo, hi
    out["lo_conf"], out["hi_conf"] = lo - q, hi + q
    out["q"] = q
    parts.append(out)
    print(f"S{s}: q = {q:+.2f} BPM", flush=True)

res = pd.concat(parts, ignore_index=True)
res["cov_raw"] = (res["true_hr"] >= res["lo_raw"]) & (res["true_hr"] <= res["hi_raw"])
res["cov_conf"] = (res["true_hr"] >= res["lo_conf"]) & (res["true_hr"] <= res["hi_conf"])
res["w_raw"] = res["hi_raw"] - res["lo_raw"]
res["w_conf"] = res["hi_conf"] - res["lo_conf"]

print(f"\nPooled: raw coverage {res['cov_raw'].mean():.3f} (Phase 10 reference: 0.812), "
      f"conformal coverage {res['cov_conf'].mean():.3f}, "
      f"width {res['w_raw'].mean():.1f} -> {res['w_conf'].mean():.1f} BPM")
if SMOKE:
    print("\nSmoke test finished without errors.")
    sys.exit(0)

b = pd.read_csv("results/metrics/phase11_conformal_by_subject_90.csv", index_col="subject")
per = (res.groupby("subject")
       .agg(C_raw=("cov_raw", "mean"), C_conf=("cov_conf", "mean"), C_q=("q", "first"))
       .round(3).join(b[["B_conf", "B_q"]]))
print("\n=== Per held-out subject, nominal 90%: C = all 14 subjects calibrate, B = 4 subjects ===")
print(per.to_string())

print("\n=== Stability of q across folds ===")
for tag in ["B", "C"]:
    q = per[f"{tag}_q"]
    print(f"Scheme {tag}: q min {q.min():.1f}, median {q.median():.1f}, max {q.max():.1f}, std {q.std():.2f}")

print("\n=== Unevenness of per-subject coverage (nominal 0.90) ===")
for tag, col in [("B conf", "B_conf"), ("C raw", "C_raw"), ("C conf", "C_conf")]:
    c = per[col]
    print(f"{tag:<7}: mean {c.mean():.3f}, std {c.std():.3f}, min {c.min():.3f} (S{c.idxmin()}), "
          f"within 0.85-0.95: {((c >= 0.85) & (c <= 0.95)).sum()}/15, "
          f"above 0.95: {(c > 0.95).sum()}, below 0.85: {(c < 0.85).sum()}")

per.to_csv("results/metrics/phase11_conformal_loso_by_subject_90.csv")
res.to_csv("results/metrics/predictions/phase11_conformal_loso_90.csv", index=False)
print("\nSaved: small summary table (tracked) and per-window file (gitignored).")
