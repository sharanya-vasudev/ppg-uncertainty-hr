"""
Subject-aware cross-validation (Phase 6+).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def run_loso_cv(df: pd.DataFrame, feature_cols: list, model_factory,
                 target_col: str = "true_hr", group_col: str = "subject",
                 show_progress: bool = True) -> pd.DataFrame:
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df[group_col].values

    logo = LeaveOneGroupOut()
    all_preds = np.full(len(df), np.nan)
    n_folds = logo.get_n_splits(groups=groups)
    splits = logo.split(X, y, groups)
    if show_progress:
        splits = tqdm(splits, total=n_folds, leave=False)

    for train_idx, test_idx in splits:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = model_factory()
        model.fit(X_train, y[train_idx])
        all_preds[test_idx] = model.predict(X_test)

    result = df[["subject", "activity", target_col]].copy()
    result["predicted_hr"] = all_preds
    result["abs_error"] = np.abs(result["predicted_hr"] - result[target_col])
    return result
