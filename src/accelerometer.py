"""
Accelerometer signal processing and motion-descriptor extraction.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq


def acc_magnitude(acc_xyz: np.ndarray) -> np.ndarray:
    """Magnitude |a| = sqrt(ax^2+ay^2+az^2) from an (n_samples, 3) array in g units."""
    return np.sqrt((acc_xyz ** 2).sum(axis=1))


def motion_features(acc_window_xyz: np.ndarray, fs: float,
                     cardiac_low_hz: float = 0.5, cardiac_high_hz: float = 4.0) -> dict:
    """Compute motion-intensity descriptors for one accelerometer window.

    acc_window_xyz : (n_samples, 3) array, in g units (already /64 from raw).
    fs : accelerometer sampling frequency (Hz).
    """
    mag = acc_magnitude(acc_window_xyz)
    mag_centered = mag - mag.mean()  # remove gravity/DC before spectral analysis

    n = len(mag_centered)
    freqs = rfftfreq(n, d=1 / fs)
    power = np.abs(rfft(mag_centered)) ** 2

    nonzero_mask = freqs > 0
    total_power = power[nonzero_mask].sum()
    dominant_freq = freqs[nonzero_mask][np.argmax(power[nonzero_mask])] if total_power > 0 else 0.0

    cardiac_band_mask = (freqs >= cardiac_low_hz) & (freqs <= cardiac_high_hz)
    cardiac_band_power = power[cardiac_band_mask].sum()
    cardiac_band_energy_frac = (cardiac_band_power / total_power) if total_power > 0 else 0.0

    return {
        "mean_mag": float(mag.mean()),
        "std_mag": float(mag.std()),
        "rms_mag": float(np.sqrt((mag ** 2).mean())),
        "dominant_freq_hz": float(dominant_freq),
        "cardiac_band_energy_frac": float(cardiac_band_energy_frac),
    }
