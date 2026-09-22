"""
Feature extraction: build fixed-length feature vectors from synchronized
PPG + accelerometer windows, for use by classical ML models (Phase 6+).

Window convention: 8-second windows, shifted every 2 seconds - matches
PPG-DaLiA's own ground-truth HR labeling convention exactly (Phase 1),
so every feature vector aligns 1:1 with a true HR label.

Normalization is deliberately NOT applied here - it must be fit only on
training data (Phase 6), or subject-aware splits would leak information.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq

from src.ppg_processing import ppg_sqi
from src.baseline import estimate_hr_spectral_interpolated
from src.accelerometer import motion_features


def ppg_peak_features(ppg_filtered_window: np.ndarray, fs: float) -> dict:
    """Time-domain peak statistics - an independent, non-spectral estimate
    of pulse periodicity, complementing the frequency-domain candidate HR."""
    min_distance = max(1, int(fs / (220 / 60)))  # HR can't exceed ~220 BPM
    peaks, _ = find_peaks(ppg_filtered_window, distance=min_distance)

    n_peaks = len(peaks)
    if n_peaks >= 2:
        intervals_s = np.diff(peaks) / fs
        mean_peak_interval_s = float(intervals_s.mean())
        std_peak_interval_s = float(intervals_s.std())
    else:
        # Too few peaks for a meaningful interval - flag explicitly with
        # NaN rather than fabricating a 0.
        mean_peak_interval_s = np.nan
        std_peak_interval_s = np.nan

    return {
        "n_peaks": n_peaks,
        "mean_peak_interval_s": mean_peak_interval_s,
        "std_peak_interval_s": std_peak_interval_s,
        "ppg_std": float(ppg_filtered_window.std()),
        "ppg_rms": float(np.sqrt((ppg_filtered_window ** 2).mean())),
        "ppg_peak_to_peak": float(ppg_filtered_window.max() - ppg_filtered_window.min()),
    }


def spectral_peak_dominance(ppg_filtered_window: np.ndarray, fs: float,
                             low_hz: float = 0.5, high_hz: float = 4.0) -> float:
    """Fraction of in-band power concentrated in the single dominant peak.
    Distinct from SQI (band/total across whole spectrum): measures how much
    ONE frequency dominates, a proxy for a clean mono-periodic pulse."""
    n = len(ppg_filtered_window)
    freqs = rfftfreq(n, d=1 / fs)
    power = np.abs(rfft(ppg_filtered_window)) ** 2
    band_power = power[(freqs >= low_hz) & (freqs <= high_hz)]
    if band_power.sum() == 0:
        return 0.0
    return float(band_power.max() / band_power.sum())


def extract_features(bvp_raw_window: np.ndarray, bvp_filtered_window: np.ndarray,
                      acc_window_g: np.ndarray, fs_bvp: float, fs_acc: float) -> dict:
    """Build the full feature vector for one synchronized 8s window."""
    features = {}

    try:
        features["candidate_hr_spectral"] = estimate_hr_spectral_interpolated(
            bvp_filtered_window, fs=fs_bvp
        )
    except ValueError:
        features["candidate_hr_spectral"] = np.nan

    features["ppg_sqi"] = ppg_sqi(bvp_raw_window, fs=fs_bvp)
    features["ppg_spectral_peak_dominance"] = spectral_peak_dominance(bvp_filtered_window, fs=fs_bvp)
    features.update(ppg_peak_features(bvp_filtered_window, fs=fs_bvp))

    mfeat = motion_features(acc_window_g, fs=fs_acc)
    features.update({f"acc_{k}": v for k, v in mfeat.items()})

    for i, name in enumerate(["ax", "ay", "az"]):
        features[f"acc_mean_{name}"] = float(acc_window_g[:, i].mean())
        features[f"acc_std_{name}"] = float(acc_window_g[:, i].std())

    return features
