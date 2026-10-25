"""
Split conformal prediction utilities (Phase 11).

Conformalized Quantile Regression (CQR):
  1. Fit a LOW and a HIGH quantile model on a PROPER-TRAINING set.
  2. On a separate CALIBRATION set (never used for fitting) compute the
     nonconformity score  s = max(low - y, y - high):
        s <= 0 : truth was inside the interval (by -s BPM)
        s  > 0 : truth missed the interval by s BPM
  3. Take the conformal quantile q of the calibration scores and use the
     interval [low - q, high + q].  q > 0 widens, q < 0 shrinks.

Guarantee (marginal): if calibration and test points are EXCHANGEABLE,
P(y in interval) >= 1 - alpha. In this project the exchangeable unit would
have to be the SUBJECT, not the window - see the Phase 11 experiment.
"""
from __future__ import annotations

import numpy as np


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(lo - y, y - hi)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile: the k-th smallest score
    with k = ceil((n + 1) * (1 - alpha))."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(scores, k - 1)[k - 1])
