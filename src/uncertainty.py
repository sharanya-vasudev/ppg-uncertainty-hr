"""
Uncertainty quantification (Phase 8+).

Quantile regression: instead of predicting the mean, a model is trained to
predict a specific quantile (percentile) of the target distribution, via the
pinball/quantile loss. Training a LOW quantile model (e.g. 5th percentile)
and a HIGH quantile model (e.g. 95th percentile) together define a
PREDICTION INTERVAL - not a confidence interval - for each individual window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


def get_quantile_model(quantile: float, random_state: int = 42):
    """A single quantile regressor. quantile=0.5 (the median) is also a
    valid MAE-optimal point estimator - minimizing pinball loss at q=0.5
    is mathematically equivalent to minimizing MAE.
    """
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile,
        max_iter=150, max_depth=8, random_state=random_state,
    )


def run_loso_quantile_cv(df: pd.DataFrame, feature_cols: list,
                          low_q: float, high_q: float,
                          target_col: str = "true_hr", group_col: str = "subject") -> pd.DataFrame:
    """LOSO CV producing a [low_q, high_q] prediction interval per window,
    via two separately-trained quantile models per fold."""
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df[group_col].values

    logo = LeaveOneGroupOut()
    pred_low = np.full(len(df), np.nan)
    pred_high = np.full(len(df), np.nan)
    pred_median = np.full(len(df), np.nan)

    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model_low = get_quantile_model(low_q)
        model_low.fit(X_train, y[train_idx])
        pred_low[test_idx] = model_low.predict(X_test)

        model_high = get_quantile_model(high_q)
        model_high.fit(X_train, y[train_idx])
        pred_high[test_idx] = model_high.predict(X_test)

        model_median = get_quantile_model(0.5)
        model_median.fit(X_train, y[train_idx])
        pred_median[test_idx] = model_median.predict(X_test)

    result = df[["subject", "activity", target_col]].copy()
    result["pred_median"] = pred_median
    result["pred_low"] = pred_low
    result["pred_high"] = pred_high
    result["interval_width"] = pred_high - pred_low
    result["covered"] = (result[target_col] >= pred_low) & (result[target_col] <= pred_high)
    result["abs_error_median"] = np.abs(pred_median - result[target_col])
    return result
