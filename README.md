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

## Status: Phases 0-13 complete (Phase 7 skipped by design)

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

**Phase 11 - Conformal prediction.** Conformalized quantile regression (CQR)
on top of the Phase 8 quantile models, evaluated leave-one-subject-out,
comparing three ways of choosing the calibration data (nominal 90% interval
unless stated):
- Window-level calibration (random windows from the training subjects) barely
  helps: coverage 0.807 -> 0.835. Calibration windows resemble training
  windows, so the base model looks better calibrated than it is on a new person.
- Subject-level calibration with 4 held-out subjects brings pooled coverage
  close to nominal at all five levels (0.538 / 0.723 / 0.810 / 0.893 / 0.936
  for 50 / 70 / 80 / 90 / 95%), but the correction is unstable: q ranged from
  0.2 to 16.4 BPM depending on which 4 subjects were drawn.
- Leave-one-subject-out cross-conformal (all 14 other subjects calibrate) is
  stable (q 2.5-3.8 BPM) and reaches pooled coverage 0.900, with mean width
  30.0 -> 36.7 BPM. Its uncorrected coverage (0.812) reproduces Phase 10 exactly.
- Coverage is marginal (an average over wearers), not per-person: even with a
  stable q, per-subject coverage ranges from 0.56 (S5) to 0.975 (S7), with 8 of
  15 subjects above 0.95 and 4 below 0.85; the spread across subjects barely
  shrank (std 0.136 -> 0.115). A single global correction cannot remove
  person-to-person differences.
- By activity (4-subject scheme), coverage after correction ranges from 0.73
  (stairs) to 0.95 (driving) while interval width adapts from 21 to 54 BPM.

**Phase 12 - Final comparison.** Pooled leave-one-subject-out results over
64,697 windows (MAE and RMSE in BPM, r = Pearson correlation; Cov@90 and
Width@90 are for a nominal 90% prediction interval). The full table, with
parameter counts and training time, is in results/tables/phase12_model_comparison.md.

| Model | Inputs | MAE | RMSE | r | Cov@90 | Width@90 | Size (MB) | Pipeline (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spectral baseline (interpolated) | PPG | 18.33 | 30.25 | 0.388 | n/a | n/a | 0.00 | 0.45 |
| Ridge | PPG | 11.59 | 16.10 | 0.709 | n/a | n/a | 0.00 | 0.64 |
| Random Forest | PPG | 9.25 | 14.96 | 0.756 | n/a | n/a | 32.83 | 2.97 |
| Gradient Boosting | PPG | 9.41 | 14.94 | 0.757 | n/a | n/a | 0.55 | 0.98 |
| Random Forest, naive fusion | PPG + all ACC | 9.85 | 16.45 | 0.702 | n/a | n/a | 31.53 | 2.92 |
| Random Forest (official Phase 6) | PPG + dynamic ACC | 8.91 | 14.35 | 0.778 | n/a | n/a | 32.69 | 3.06 |
| Gradient Boosting | PPG + dynamic ACC | 9.08 | 14.16 | 0.785 | n/a | n/a | 0.57 | 0.98 |
| Quantile regression (median, 5/95%) | PPG + dynamic ACC | 8.69 | 14.13 | 0.787 | 0.812 | 30.0 | 1.71 | 1.96 |
| Sparse variational GP | PPG + dynamic ACC | 9.46 | 14.62 | 0.769 | 0.864 | 41.0 | 2.08 | 1.15 |
| Conformalized quantile regression (all-subject calibration) | PPG + dynamic ACC | 8.69 | 14.13 | 0.787 | 0.900 | 36.7 | 1.71 | 1.96 |

- Gradient-boosted quantile regression (median) is nominally best on all three
  point metrics (MAE 8.69, RMSE 14.13, r 0.787). One median model is about
  0.57 MB and under 0.5 ms per window, roughly 57x smaller and 6x faster than
  the Random Forest (8.91 / 14.35 / 0.778, 32.7 MB). Differences between the
  top models are small and not yet tested for significance (Phase 16).
- At a nominal 90% interval, conformalized quantile regression (calibrated on
  the other 14 subjects) has the best interval: coverage 0.900 at 36.7 BPM mean
  width, against 0.864 at 41.0 BPM for the sparse GP and 0.812 at 30.0 BPM for
  uncalibrated quantile regression.
- Naive fusion is also the worst tree model in RMSE (16.45 vs 14.96 for
  PPG-only); dynamic accelerometer features lower it to 14.35, so the
  orientation-feature diagnosis holds beyond MAE. RMSE gains from fusion
  (0.6-0.8 BPM) exceed MAE gains (0.2-0.3 BPM), consistent with fusion mostly
  removing large errors.
- Cost (one laptop CPU thread, one example fold, not wearable hardware):
  filtering plus features plus model takes 0.6-3.1 ms per window against a 2 s
  window cadence. No deep models were built (Phase 7 skipped by design), so
  those rows are not reported.

**Phase 13 - Motion-specific evaluation.** Primary configuration (quantile-regression
median plus a conformal 90% interval, leave-one-subject-out), reported by
condition and by accelerometer RMS bin. PPG-DaLiA contains no running, so that
condition from the original plan cannot be evaluated here. cov_raw and cov_conf
are coverage of the nominal 90% interval before and after conformal correction,
cov_conf_no_S5_S12 repeats the corrected coverage without subjects S5 and S12
(chosen in advance), width is in BPM, and worst_subject_cov is the lowest
per-subject corrected coverage in that condition. Full tables are in
results/metrics/phase13_*.csv and the figure is
results/figures/phase13_motion_specific.png.

| condition | n | MAE | cov_raw | cov_conf | cov_conf_no_S5_S12 | width | width_over_MAE | worst_subject_cov |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rest | 13066 | 4.424 | 0.826 | 0.943 | 0.968 | 22.629 | 5.115 | 0.749 |
| seated_daily | 20397 | 6.325 | 0.846 | 0.939 | 0.961 | 31.672 | 5.008 | 0.733 |
| transient | 17515 | 10.613 | 0.807 | 0.876 | 0.93 | 43.847 | 4.131 | 0.421 |
| walking | 4697 | 11.408 | 0.796 | 0.856 | 0.916 | 49.853 | 4.37 | 0.244 |
| cycling | 3473 | 11.65 | 0.758 | 0.863 | 0.905 | 41.933 | 3.599 | 0.172 |
| high_motion | 5549 | 17.206 | 0.722 | 0.792 | 0.845 | 51.16 | 2.973 | 0.144 |

| motion_bin | n | MAE | cov_raw | cov_conf | cov_conf_no_S5_S12 | width | width_over_MAE | worst_subject_cov |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <1.00 g | 10810 | 3.706 | 0.849 | 0.956 | 0.972 | 20.458 | 5.52 | 0.825 |
| 1.00-1.02 g | 19996 | 5.763 | 0.83 | 0.935 | 0.957 | 27.897 | 4.84 | 0.689 |
| 1.02-1.06 g | 13901 | 9.903 | 0.815 | 0.894 | 0.943 | 42.472 | 4.289 | 0.492 |
| 1.06-1.15 g | 13404 | 13.746 | 0.784 | 0.841 | 0.911 | 52.88 | 3.847 | 0.264 |
| >1.15 g | 6586 | 12.906 | 0.748 | 0.835 | 0.868 | 44.815 | 3.472 | 0.239 |

- Uncertainty moves in the right direction: rank agreement between interval
  width and MAE is 0.90 across the nine activities and 1.00 across the five
  motion bins, and coverage falls monotonically with accelerometer RMS
  (0.956 -> 0.835).
- It does not move enough: width grows 2.3-3.1x from the easiest to the hardest
  conditions while MAE grows 3.9-5.8x, so width/MAE falls from about 5.1 at rest
  to 3.0 for high-motion activities (2.6 for stairs; about 4.1 would be expected
  for correct Gaussian intervals). Easy conditions are over-covered (0.93-0.95)
  and the hardest under-covered (stairs 0.74).
- Part of the shortfall is a two-subject effect: excluding S5 and S12 lifts
  coverage by 4-6 points and brings walking (0.916), cycling (0.905) and table
  soccer (0.922) to nominal, while stairs (0.788) and the >1.15 g bin (0.868)
  stay below it.
- Pooled coverage hides individual failure: in motion conditions the worst
  single subject's corrected coverage is 0.11-0.25 (stairs 0.11, walking 0.24)
  against pooled 0.74-0.86.
- Rank agreement across 5-9 conditions is descriptive, not a significance test
  (Phase 16); the RMS bins use Phase 4 thresholds found on the same data.

### Upcoming
Uncertainty as a quality signal (Phase 14), lightweight/edge analysis
(Phase 15), statistical validation (Phase 16) -> ablations -> robustness ->
reproducibility -> paper.

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
