import io
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.append(".")
from src.config import (PPG_FEATURES, ACC_FEATURES_FINAL,
                        ACC_FEATURES_DEPRECATED_ORIENTATION, ALL_FEATURES_FINAL)
from src.data_loader import (load_subject_signals, FS_BVP, FS_ACC,
                             WINDOW_LEN_S, WINDOW_SHIFT_S)
from src.ppg_processing import preprocess_ppg
from src.baseline import estimate_hr_spectral_interpolated
from src.feature_extraction import extract_features
from src.models import get_model
from src.uncertainty import get_quantile_model
from src.gp_model import fit_svgp, predict_svgp

DATA_ROOT = Path("data/raw/PPG_FieldStudy")
HELD_OUT, N_LAT = 1, 200
ACC_FULL = ACC_FEATURES_FINAL + ACC_FEATURES_DEPRECATED_ORIENTATION

env = (f"platform: {platform.platform()}\nmachine: {platform.machine()}\n"
       f"cpu_count: {os.cpu_count()}\npython: {platform.python_version()}\n"
       f"scikit-learn: {sklearn.__version__}\ntorch: {torch.__version__}\n"
       "latency: ONE CPU thread, single window per call; training: default threads\n"
       f"models fitted on all subjects except S{HELD_OUT} (one example fold)")
print(env)
Path("results/metrics/phase12_environment.txt").write_text(env + "\n")

df = pd.read_csv("data/processed/ppg_acc_features_v2.csv")
train = df[df["subject"] != HELD_OUT]
test = df[df["subject"] == HELD_OUT]
ytr = train["true_hr"].values


def pickle_mb(obj):
    return len(pickle.dumps(obj)) / 1e6


def median_latency_ms(fn, X, n=N_LAT):
    fn(X[:1])  # warm-up
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(X[i:i + 1])
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def hgb_nodes(model):
    try:
        return int(sum(len(p.nodes) for it in model._predictors for p in it))
    except Exception:
        return None


# ---- feature-extraction cost (shared by every learned model) ----
sig = load_subject_signals(HELD_OUT, DATA_ROOT)
bvp, acc = sig["bvp"], sig["acc_g"]


def window_arrays(i):
    s = i * WINDOW_SHIFT_S
    b0 = int(s * FS_BVP)
    a0 = int(s * FS_ACC)
    return bvp[b0:b0 + int(WINDOW_LEN_S * FS_BVP)], acc[a0:a0 + int(WINDOW_LEN_S * FS_ACC)]


def time_pipeline(with_features):
    ts = []
    for i in range(50, 50 + N_LAT + 1):
        raw, acc_w = window_arrays(i)
        t0 = time.perf_counter()
        filt = preprocess_ppg(raw, fs=FS_BVP)
        if with_features:
            extract_features(raw, filt, acc_w, FS_BVP, FS_ACC)
        else:
            estimate_hr_spectral_interpolated(filt, fs=FS_BVP)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts[1:]))  # drop the warm-up call


with threadpool_limits(limits=1):
    feat_ms = time_pipeline(with_features=True)
    spectral_ms = time_pipeline(with_features=False)
print(f"\nFeature pipeline (filter + 20 features): {feat_ms:.2f} ms/window")
print(f"Spectral baseline pipeline (filter + FFT estimate): {spectral_ms:.2f} ms/window")

rows = [{"key": "spectral", "train_s": 0.0, "params": "0 learned (fixed filter)",
         "size_mb": 0.0, "infer_ms": np.nan, "pipeline_ms": spectral_ms}]

# ---- ridge / random forest / gradient boosting ----
configs = {
    "ridge_ppg": ("ridge", PPG_FEATURES),
    "rf_ppg": ("random_forest", PPG_FEATURES),
    "gb_ppg": ("gradient_boosting", PPG_FEATURES),
    "rf_naive": ("random_forest", PPG_FEATURES + ACC_FULL),
    "rf_dyn": ("random_forest", ALL_FEATURES_FINAL),
    "gb_dyn": ("gradient_boosting", ALL_FEATURES_FINAL),
}
for key, (name, cols) in configs.items():
    Xtr, Xte = train[cols].values, test[cols].values
    t0 = time.perf_counter()
    if name == "ridge":
        scaler = StandardScaler().fit(Xtr)
        model = get_model("ridge").fit(scaler.transform(Xtr), ytr)
        fit_s = time.perf_counter() - t0
        predict = lambda X, m=model, s=scaler: m.predict(s.transform(X))
        params = f"{model.coef_.size + 1 + 2 * len(cols)} (coefficients, intercept, scaler)"
        size = pickle_mb((scaler, model))
    else:
        model = get_model(name).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        if name == "random_forest":
            model.set_params(n_jobs=1)
            nodes = sum(e.tree_.node_count for e in model.estimators_)
            params = f"{nodes:,} tree nodes ({len(model.estimators_)} trees)"
        else:
            nodes = hgb_nodes(model)
            params = (f"{nodes:,} tree nodes ({model.n_iter_} trees)" if nodes
                      else f"{model.n_iter_} trees (node count unavailable)")
        predict = model.predict
        size = pickle_mb(model)
    with threadpool_limits(limits=1):
        infer = median_latency_ms(predict, Xte)
    rows.append({"key": key, "train_s": fit_s, "params": params, "size_mb": size,
                 "infer_ms": infer, "pipeline_ms": feat_ms + infer})
    print(f"{key}: fit {fit_s:.1f}s, {params}, {size:.2f} MB, {infer:.2f} ms/window")

# ---- quantile regression (median + 5% + 95% models) ----
Xtr, Xte = train[ALL_FEATURES_FINAL].values, test[ALL_FEATURES_FINAL].values
t0 = time.perf_counter()
qmods = [get_quantile_model(q).fit(Xtr, ytr) for q in (0.05, 0.5, 0.95)]
fit_s = time.perf_counter() - t0
nodes_list = [hgb_nodes(m) for m in qmods]
total_nodes = sum(nodes_list) if all(nodes_list) else None
params = f"{total_nodes:,} tree nodes (3 models)" if total_nodes else "3 models (node count unavailable)"
size = sum(pickle_mb(m) for m in qmods)
with threadpool_limits(limits=1):
    infer = median_latency_ms(lambda X: [m.predict(X) for m in qmods], Xte)
rows.append({"key": "qr_dyn", "train_s": fit_s, "params": params, "size_mb": size,
             "infer_ms": infer, "pipeline_ms": feat_ms + infer})
rows.append({"key": "cqr_dyn", "train_s": fit_s, "params": params + " + 1 scalar (q)",
             "size_mb": size, "infer_ms": infer, "pipeline_ms": feat_ms + infer})
print(f"qr_dyn / cqr_dyn: fit {fit_s:.1f}s, {params}, {size:.2f} MB, {infer:.2f} ms/window")

# ---- sparse variational GP ----
sub = train[train["window_idx"] % 4 == 0]
t0 = time.perf_counter()
state = fit_svgp(sub[ALL_FEATURES_FINAL].values, sub["true_hr"].values)
fit_s = time.perf_counter() - t0
n_par = (sum(p.numel() for p in state["model"].parameters())
         + sum(p.numel() for p in state["likelihood"].parameters()))
buf = io.BytesIO()
torch.save({"m": state["model"].state_dict(), "l": state["likelihood"].state_dict()}, buf)
size = buf.getbuffer().nbytes / 1e6
torch.set_num_threads(1)
infer = median_latency_ms(lambda X: predict_svgp(state, X), Xte)
rows.append({"key": "gp_dyn", "train_s": fit_s, "params": f"{n_par:,} trainable (500 inducing points)",
             "size_mb": size, "infer_ms": infer, "pipeline_ms": feat_ms + infer})
print(f"gp_dyn: fit {fit_s:.1f}s, {n_par:,} trainable, {size:.2f} MB, {infer:.2f} ms/window")

cost = pd.DataFrame(rows)
cost.to_csv("results/metrics/phase12_cost_measurements.csv", index=False)
print("\nSaved: results/metrics/phase12_cost_measurements.csv")
