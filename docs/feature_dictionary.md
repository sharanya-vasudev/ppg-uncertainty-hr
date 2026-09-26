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

## Phase 6 update: finalized ACC feature set

Feature importance analysis plus a controlled comparison found that the raw
per-axis accelerometer means (`acc_mean_ax`, `acc_mean_ay`, `acc_mean_az`)
primarily encode static wrist orientation - how an individual happens to wear
the device - rather than genuine motion dynamics. This let models partly
overfit to training-subject-specific device orientation, which does not
generalize to held-out subjects (the failure mode Leave-One-Subject-Out CV is
designed to catch).

**Removing these three features reversed the fusion result**: PPG+ACC went
from underperforming PPG-only (Random Forest: 9.25 -> 9.85 BPM MAE) to
beating it (9.25 -> 8.91 BPM), replicated with Gradient Boosting
(9.41 -> 9.08 BPM). The improvement concentrated exactly in the activities
most affected by orientation instability (stairs, walking), consistent with
the proposed mechanism.

`acc_mean_mag` is kept (orientation-invariant). Official ACC feature set,
final feature lists, and the chosen model are defined in `src/config.py`.

Two engineered heterogeneity features (sub-window RMS std/range) were also
tested and found to have negligible importance and no measurable effect -
not included in the final set.

**Caveat:** the improvement is a real average effect (replicated across two
model families) but not universal at the subject level - 12/15 subjects
improved with Random Forest, 9/15 with Gradient Boosting, with 2 subjects
showing consistent degradation. Formal statistical testing is deferred to
Phase 16.

**Official Phase 6 model: Random Forest, PPG + dynamic-ACC features, MAE
8.91 BPM pooled / 9.02 BPM macro.**
