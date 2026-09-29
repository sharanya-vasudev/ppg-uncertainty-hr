# Uncertainty-Aware PPG + Accelerometer Heart-Rate Estimation Under Motion Artifacts

Research project investigating whether PPG + accelerometer sensor fusion improves
heart-rate estimation robustness under motion, and whether a lightweight model can
produce calibrated predictive uncertainty — not just a point estimate.

## Research Question
Can a lightweight multimodal model combining PPG and accelerometer information
improve heart-rate estimation under motion while providing calibrated predictive
uncertainty?

## Dataset
[PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) (UCI Machine Learning
Repository) — 15 subjects, synchronized wrist PPG/BVP (64 Hz), wrist accelerometer
(32 Hz, g units), chest ECG-derived ground-truth HR (8s windows, 2s shift), 8
daily-life activities. **Not included in this repo** (~2.7 GB, licensed separately)
— see `data/README.md` for download instructions.

## Status: Phases 0-10 complete

**Phase 0 — Environment & structure.** Python 3.11 virtual environment, project
structure, dependency tracking.

**Phase 1 — Dataset understanding.** Verified dataset structure against official
documentation. Visual inspection confirmed clean PPG during sitting vs. visibly
distorted PPG during cycling/walking, alongside elevated accelerometer activity.

**Phase 2 — PPG signal processing.** 0.5-4 Hz Butterworth band-pass filter
(physiologically justified: 40-240 BPM maps to 0.67-4 Hz). Found band-pass
filtering alone does *not* meaningfully clean motion-corrupted PPG when motion
energy overlaps the cardiac frequency band — a key early motivation for fusion.

**Phase 3 — PPG-only baseline.** Spectral (dominant-frequency) HR estimator,
evaluated across all 15 subjects (~65,000 windows):
- Pooled MAE 18.50 BPM, RMSE 30.33 BPM, correlation 0.386
- Strong activity-dependent degradation: 3.6 BPM (sitting) -> 38.3 BPM (stairs)
- Root cause of high-motion error identified as octave/subharmonic confusion
  (predicting ~half true HR), not FFT frequency-resolution limits — confirmed by
  testing a resolution fix (parabolic interpolation), which only marginally
  helped (18.50 -> 18.33 BPM) and left the octave-error rate unchanged
- Large between-subject variability (per-subject MAE: 8.9-43.9 BPM)

**Phase 4 — Motion artifact analysis.** Quantified the relationship between
accelerometer-derived motion features and HR error across all subjects:
- Accelerometer RMS magnitude shows a threshold-like relationship with error:
  mean error stays under ~13 BPM below ~1.02g RMS (near-resting), rises sharply
  through 1.02-1.06g, and plateaus around 28-33 BPM above ~1.15g
- PPG signal-quality index (SQI) correlates with error, but most of the benefit
  is concentrated in the top ~20% of the quality range
- Raw motion-magnitude features correlate with error more strongly than
  cardiac-band-specific motion overlap
- Findings validated at the subject level, not just pooled across all windows

**Phase 5 — Feature engineering.** Built a 20-feature table (PPG + accelerometer)
for all 15 subjects, aligned 1:1 with PPG-DaLiA's own 8s/2s ground-truth HR
windows. See `docs/feature_dictionary.md` for full feature definitions,
windowing convention, and normalization/missing-data policy. Output:
`data/processed/ppg_acc_features_all_subjects.csv` (64,697 rows x 24 columns).

### Correction (post-Phase 5)
Discovered and fixed a units bug present since Phase 1: wrist accelerometer data
inside `SX.pkl` is already in g units, but code had applied an extra `/64`
conversion (intended for a different raw file in the dataset, not this one),
shrinking all accelerometer magnitude values by 64x. Ratio-based features and
all correlation coefficients were mathematically unaffected by this (confirmed
by exact reproduction of Phase 4's correlation values after the fix); absolute
magnitude values and plots have been corrected and regenerated. Subject loading
is now centralized in `src/data_loader.py` to prevent recurrence.

**Phase 6 — Sensor fusion baselines.** Trained Ridge, Random Forest, and
Gradient Boosting with Leave-One-Subject-Out CV comparing PPG-only, ACC-only,
and PPG+ACC feature sets:
- All ML models substantially beat Phase 3's spectral baseline (best:
  PPG-only Random Forest, MAE 9.25 BPM vs Phase 3's 18.33 BPM)
- ACC-only alone reaches ~15-16 BPM MAE - real signal, but likely reflects a
  population-level exercise-intensity-to-HR association rather than true
  cardiac information
- Initial PPG+ACC fusion (all accelerometer features) underperformed
  PPG-only. Root-cause investigation found raw per-axis accelerometer means
  (acc_mean_ax/ay/az) encode static wrist orientation - a person/device-fit
  specific artifact, not motion - causing overfitting that does not
  generalize across subjects
- Removing these orientation features and keeping only dynamic ACC features
  (magnitude/frequency/cardiac-band statistics) REVERSED the result: fusion
  now beats PPG-only, replicated across both tree-based models (Random
  Forest: 9.25 -> 8.91 BPM; Gradient Boosting: 9.41 -> 9.08 BPM MAE)
- The improvement concentrates exactly in the activities most affected by
  orientation instability - stairs (delta +1.06 -> -1.29 BPM) and walking
  (+2.04 -> -0.08 BPM) - strong mechanistic confirmation of the diagnosis
- Caveat: a real average improvement, but not universal - 12/15 subjects
  improved with Random Forest, 9/15 with Gradient Boosting; formal
  significance testing deferred to Phase 16 per the project's phased design

**Official Phase 6 model: Random Forest, PPG + dynamic-ACC features (see
`src/config.py`), MAE 8.91 BPM pooled / 9.02 BPM macro.**

**Decision: Phase 7 (neural network) skipped as not currently justified.**
Per the brief's own criterion ("only after classical models work, investigate
a neural network if justified"), a diagnosed, mechanistically-explained
feature fix already produced a genuine fusion improvement with simple
classical models - no current evidence the bottleneck is model capacity
rather than feature design.

**Phase 8 - Quantile regression.** Gradient-boosted quantile models
(5th/95th percentile, LOSO CV) producing per-window prediction intervals:
- Median-model point estimate: MAE 8.69 BPM
- 90% nominal interval: empirical coverage 81.2% (overconfident), mean width
  30.0 BPM
- Width is informative: Spearman 0.60 with absolute error, growing from
  ~10 BPM (sitting) to ~46 BPM (stairs)
- Miscalibration is subject-dependent: 9 of 15 held-out subjects land between
  0.89 and 0.93 coverage, while 6 are clearly under-covered, worst S5 (0.49)
  and S12 (0.61) - the same subjects where fusion underperformed in Phase 6

**Phase 9 - Gaussian Process regression.** Sparse variational GP (GPyTorch,
500 inducing points), used to test whether epistemic uncertainty explains
the S5/S12 pattern:
- Point estimate MAE 9.46 BPM (smaller training subsample than other models)
- Calibration crosses the diagonal: over-covers at low nominal levels
  (50% -> 63.8%) and under-covers at high ones (95% -> 89.8%), indicating a
  heavier-tailed real error distribution than the Gaussian assumption
- The epistemic term does not clearly flag S5 (its value is below-median for
  that subject); width is instead dominated by a near-constant per-fold
  noise term, so it barely varies across subjects (40-43 BPM) or activities
  despite true difficulty varying 5x (4-20 BPM MAE)
- Practical conclusion: quantile regression's interval is currently more
  useful (it adapts to context) despite the GP's more principled construction

**Phase 10 - Calibration comparison.** Reliability diagram across 5 nominal
levels (50/70/80/90/95%), see `results/figures/reliability_diagram_phase10.png`:
- The two methods fail in different, identifiable ways: the GP's curve
  crosses the diagonal near 80% (a shape mismatch - tails heavier than
  Gaussian), while quantile regression's curve stays below the diagonal at
  every level (a systematic ~7-9 point shortfall - no mechanism to widen for
  an unfamiliar subject)

### Upcoming
Conformal prediction (Phase 11) - tests whether a distribution-free method
closes these gaps, while explicitly checking its exchangeability assumption
against the subject-level distribution shift documented above -> motion-
specific evaluation -> final evaluation -> paper.

## Repository Structure
```
ppg-uncertainty-hr/
├── data/              # raw/processed data (gitignored, see data/README.md)
├── notebooks/         # exploratory notebooks, one per phase
├── src/               # reusable modules (config.py, data_loader.py, models.py,
│                       # uncertainty.py, gp_model.py, etc.)
├── scripts/           # experiment scripts
├── docs/              # feature dictionary and other project documentation
├── results/
│   ├── figures/        # saved plots
│   └── metrics/         # small summary CSVs (tracked)
│       └── predictions/  # large per-window prediction files (gitignored,
│                          # regenerate via the matching script)
└── paper/              # manuscript notes
```

## Setup
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Notes
All results in this repo come from real experiments run on PPG-DaLiA — no
fabricated numbers. Raw outputs backing the summary above live in
`results/metrics/` and `results/figures/`.
