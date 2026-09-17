"""
PPG signal preprocessing and quality assessment.

Physiological justification for filter band (0.5-4 Hz):
    Human heart rate range of interest: ~40-200 BPM
    40 BPM  = 40/60  = 0.667 Hz
    200 BPM = 200/60 = 3.33 Hz
    Band is set slightly wider (0.5-4 Hz) to avoid clipping the edges of this range.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq


def bandpass_filter(ppg: np.ndarray, fs: float, low_hz: float = 0.5,
                     high_hz: float = 4.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    nyquist = fs / 2.0
    if high_hz >= nyquist:
        raise ValueError(f"high_hz={high_hz} must be below Nyquist ({nyquist} Hz).")
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, ppg)


def detrend(ppg: np.ndarray) -> np.ndarray:
    """Remove linear trend from a PPG signal."""
    return signal.detrend(ppg, type="linear")


def preprocess_ppg(ppg: np.ndarray, fs: float) -> np.ndarray:
    """Full PPG preprocessing pipeline: detrend then band-pass filter."""
    return bandpass_filter(detrend(ppg), fs=fs)


def ppg_sqi(ppg_window: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 4.0) -> float:
    """Signal Quality Index: fraction of a RAW PPG window's spectral power
    inside the cardiac band. Higher = cleaner, more periodic pulse signal.
    Must be computed on raw (unfiltered) PPG, or it trivially returns ~1.0.
    """
    n = len(ppg_window)
    freqs = rfftfreq(n, d=1 / fs)
    power = np.abs(rfft(ppg_window - ppg_window.mean())) ** 2

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    total_power = power[freqs > 0].sum()
    band_power = power[band_mask].sum()
    return float(band_power / total_power) if total_power > 0 else 0.0
