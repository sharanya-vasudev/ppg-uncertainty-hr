import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results/metrics")
figures_dir = Path("results/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

data = np.load(results_dir / "baseline_ppg_only_all_subjects.npz")
pred = data["predictions"]
true = data["true"]
activity = data["activity"]

ACTIVITY_NAMES = {
    0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer",
    4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working",
}

errors = pred - true

# --- Scatter: predicted vs true, colored by activity ---
all_vals = np.concatenate([true, pred])
lo = max(0, np.percentile(all_vals, 0.5) * 0.9)
hi = np.percentile(all_vals, 99.5) * 1.05
lims = [lo, hi]

fig, ax = plt.subplots(figsize=(8, 8))
cmap = plt.get_cmap("tab10")
for act_id in sorted(set(activity)):
    if act_id == -1:
        continue
    mask = activity == act_id
    ax.scatter(true[mask], pred[mask], s=3, alpha=0.3,
               color=cmap(act_id % 10), label=ACTIVITY_NAMES.get(act_id, str(act_id)))

ax.plot(lims, lims, 'k--', linewidth=1, label="y = x (perfect)")
ax.plot(lims, [2*l for l in lims], 'r:', linewidth=1, label="y = 2x (octave up)")
ax.plot(lims, [0.5*l for l in lims], 'b:', linewidth=1, label="y = 0.5x (octave down)")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("True HR (BPM)")
ax.set_ylabel("Predicted HR (BPM)")
ax.set_title("Predicted vs True HR, all subjects, colored by activity")
ax.legend(markerscale=3, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig(figures_dir / "baseline_scatter_pred_vs_true.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/baseline_scatter_pred_vs_true.png")

# --- Error histogram ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(errors, bins=100, color="steelblue", edgecolor="none")
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Prediction error (Predicted - True) BPM")
ax.set_ylabel("Count")
ax.set_title("Error distribution, all subjects, all windows")
plt.tight_layout()
plt.savefig(figures_dir / "baseline_error_histogram.png", dpi=150)
plt.close()
print("Saved: results/figures/baseline_error_histogram.png")

# --- Harmonic confusion check ---
ratio = pred / true
octave_up = (ratio > 1.8) & (ratio < 2.2)
octave_down = (ratio > 0.45) & (ratio < 0.55)

print(f"\nTotal windows: {len(pred)}")
print(f"Octave-up errors (predicted ~2x true):   {octave_up.sum()} ({100*octave_up.sum()/len(pred):.1f}%)")
print(f"Octave-down errors (predicted ~0.5x true): {octave_down.sum()} ({100*octave_down.sum()/len(pred):.1f}%)")

print("\nOctave-up errors by activity (>=20 windows):")
for act_id in sorted(set(activity)):
    if act_id == -1:
        continue
    mask = activity == act_id
    n_total = mask.sum()
    if n_total >= 20:
        n_oct = (mask & octave_up).sum()
        print(f"  {ACTIVITY_NAMES.get(act_id, act_id):<15} {n_oct}/{n_total} ({100*n_oct/n_total:.1f}%)")

print("\nOctave-down errors by activity (>=20 windows):")
for act_id in sorted(set(activity)):
    if act_id == -1:
        continue
    mask = activity == act_id
    n_total = mask.sum()
    if n_total >= 20:
        n_oct = (mask & octave_down).sum()
        print(f"  {ACTIVITY_NAMES.get(act_id, act_id):<15} {n_oct}/{n_total} ({100*n_oct/n_total:.1f}%)")

print(f"\nError mean: {errors.mean():.2f} BPM, std: {errors.std():.2f} BPM, median: {np.median(errors):.2f} BPM")
print(f"5th/95th percentile of error: {np.percentile(errors, 5):.2f} / {np.percentile(errors, 95):.2f} BPM")
