"""
Classical ML models for HR estimation from PPG+ACC features (Phase 6).
"""
from __future__ import annotations

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def get_model(name: str, random_state: int = 42):
    """Return a fresh, unfitted sklearn regressor by name."""
    if name == "linear":
        return LinearRegression()
    elif name == "ridge":
        return Ridge(alpha=1.0, random_state=random_state)
    elif name == "random_forest":
        return RandomForestRegressor(n_estimators=100, max_depth=12,
                                      random_state=random_state, n_jobs=-1)
    elif name == "gradient_boosting":
        # HistGradientBoostingRegressor: sklearn's histogram-based GB - same
        # algorithm family as classic Gradient Boosting, much faster on our
        # ~65k-row table. Used here for that reason, not a different method.
        return HistGradientBoostingRegressor(max_iter=150, max_depth=8,
                                              random_state=random_state)
    else:
        raise ValueError(f"Unknown model name: {name}")
