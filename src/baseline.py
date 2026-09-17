"""
Baseline PPG-only heart-rate estimation methods.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq


def estimate_hr_spectral(ppg_window: np.ndarray, fs: float,
                          low_hz: float = 0.5, high_hz: float = 4.0) -> float:
    """Dominant FFT bin HR estimate (quantized to fs/N * 60 BPM steps)."""
    n = len(ppg_window)
    freqs = rfftfreq(n, d=1 / fs)
    power = np.abs(rfft(ppg_window)) ** 2
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        raise ValueError("Window too short: no frequency bins fall within the band.")
    band_freqs = freqs[band_mask]
    band_power = power[band_mask]
    return band_freqs[np.argmax(band_power)] * 60.0


def estimate_hr_spectral_interpolated(ppg_window: np.ndarray, fs: float,
                                       low_hz: float = 0.5, high_hz: float = 4.0) -> float:
    """Dominant FFT bin HR estimate with parabolic interpolation for sub-bin precision."""
    n = len(ppg_window)
    freqs = rfftfreq(n, d=1 / fs)
    power = np.abs(rfft(ppg_window)) ** 2
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    band_indices = np.where(band_mask)[0]
    if len(band_indices) == 0:
        raise ValueError("Window too short: no frequency bins fall within the band.")
    band_power = power[band_indices]
    peak_idx = band_indices[np.argmax(band_power)]
    if peak_idx == 0 or peak_idx == len(power) - 1:
        return freqs[peak_idx] * 60.0
    alpha, beta, gamma = power[peak_idx - 1], power[peak_idx], power[peak_idx + 1]
    denom = alpha - 2 * beta + gamma
    p = 0.0 if denom == 0 else 0.5 * (alpha - gamma) / denom
    bin_width = freqs[1] - freqs[0]
    return (freqs[peak_idx] + p * bin_width) * 60.0
