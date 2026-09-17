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
(32 Hz), chest ECG-derived ground-truth HR (8s windows, 2s shift), 8 daily-life
activities. **Not included in this repo** (~2.7 GB, licensed separately) — see
`data/README.md` for download instructions.

## Status: Phases 0-4 complete

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
- Accelerometer RMS magnitude shows a threshold-like relationship with error,
  not a smooth linear one
- PPG signal-quality index (SQI) correlates with error, but most of the benefit
  is concentrated in the top ~20% of the quality range
- Raw motion-magnitude features correlate with error more strongly than
  cardiac-band-specific motion overlap
- Findings validated at the subject level, not just pooled across all windows

### Upcoming
Phase 5 (feature engineering) -> Phase 6 (sensor fusion baselines) -> uncertainty
quantification (Gaussian Processes, quantile regression, conformal prediction) ->
calibration -> final evaluation -> paper.

## Repository Structure
ppg-uncertainty-hr/
├── data/ # raw/processed data (gitignored, see data/README.md)
├── notebooks/ # exploratory notebooks, one per phase
├── src/ # reusable, tested modules
├── scripts/ # one-off analysis scripts
├── results/
│ ├── figures/ # saved plots
│ └── metrics/ # saved numerical results (csv/npz)
└── paper/ # manuscript notes

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
