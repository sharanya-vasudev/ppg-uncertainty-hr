from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

quant = pd.read_csv("results/metrics/phase10_quantile_calibration_levels.csv")
gp = pd.read_csv("results/metrics/phase9_gp_calibration_levels.csv")
conf = pd.read_csv("results/metrics/phase11_conformal_pooled.csv")
conf_a = conf[conf["scheme"] == "A_window"]
conf_b = conf[conf["scheme"] == "B_subject"]

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
ax.plot(quant["nominal"], quant["empirical"], "o-", label="Quantile regression")
ax.plot(gp["nominal"], gp["empirical"], "s-", label="Gaussian Process")
ax.plot(conf_a["level"], conf_a["cov_conf"], "^:", color="gray",
        label="Conformal, window-level calibration")
ax.plot(conf_b["level"], conf_b["cov_conf"], "D-",
        label="Conformal, 4 calibration subjects")

loso_path = Path("results/metrics/predictions/phase11_conformal_loso_90.csv")
if loso_path.exists():
    loso = pd.read_csv(loso_path)
    ax.plot([0.90], [loso["cov_conf"].mean()], "*", markersize=14, color="red",
            label="Conformal, all 14 subjects (90% only)")
else:
    print("Note: per-window LOSO conformal file not found; skipping that marker.")

ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Empirical coverage")
ax.set_title("Reliability diagram: uncertainty methods compared")
ax.set_xlim(0.45, 1.0)
ax.set_ylim(0.4, 1.02)
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/reliability_diagram_phase11.png", dpi=150)
print("Saved: results/figures/reliability_diagram_phase11.png")
