# Feature Dictionary — Phase 5

## Windowing
- Window length: 8 seconds
- Shift: 2 seconds
- Fixed by PPG-DaLiA's own ground-truth HR labeling convention (Phase 1) — every
  feature row aligns 1:1 with one `label` value from the dataset.

## Missing data handling
- A window with insufficient trailing signal (recording-edge case) is dropped
  entirely, not imputed. 0 windows dropped across all 15 subjects.
- `mean_peak_interval_s` / `std_peak_interval_s` would be NaN if fewer than 2
  peaks were detected in a window; in practice this never occurred (see note
  below on why this doesn't imply peak detection is reliable during motion).

## Normalization
- Deliberately NOT applied at this stage. Must be fit only on each Phase 6
  training split, never on the full dataset, to avoid leaking information
  across subject-aware train/test splits.

## Features (20 total)

### PPG-derived
| Feature | Description |
|---|---|
| candidate_hr_spectral | Phase 3 baseline's own HR estimate (interpolated spectral method), included as an input, not just an evaluation output |
| ppg_sqi | Fraction of raw PPG spectral power inside the cardiac band (0.5-4 Hz) |
| ppg_spectral_peak_dominance | Fraction of in-band power in the single dominant frequency (distinct from SQI's whole-band ratio) |
| n_peaks | Count of detected peaks (time-domain) in the filtered PPG window |
| mean_peak_interval_s | Mean time between detected peaks |
| std_peak_interval_s | Std of time between detected peaks |
| ppg_std, ppg_rms, ppg_peak_to_peak | Basic amplitude statistics of filtered PPG |

Note: peak-based features are always defined (find_peaks locates *some* local
maxima even in motion-corrupted signal) but are not necessarily physiologically
meaningful during heavy motion — a downstream model is expected to learn this
distinction using the accelerometer features alongside them.

### Accelerometer-derived
| Feature | Description |
|---|---|
| acc_mean_mag, acc_std_mag, acc_rms_mag | Magnitude statistics of \|acc\| (Phase 4) |
| acc_dominant_freq_hz | Dominant frequency of motion |
| acc_cardiac_band_energy_frac | Fraction of accelerometer spectral energy inside 0.5-4 Hz |
| acc_mean_ax/ay/az, acc_std_ax/ay/az | Per-axis statistics (for later ablation, Phase 17) |

## Source
`src/feature_extraction.py`, built via `scripts/build_features.py`.
Output: `data/processed/ppg_acc_features_all_subjects.csv` (64,697 rows x 24 columns).
