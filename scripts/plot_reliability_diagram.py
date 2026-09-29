import pandas as pd
import matplotlib.pyplot as plt

quant = pd.read_csv("results/metrics/phase10_quantile_calibration_levels.csv")
gp = pd.read_csv("results/metrics/phase9_gp_calibration_levels.csv")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
ax.plot(quant["nominal"], quant["empirical"], "o-", label="Quantile regression")
ax.plot(gp["nominal"], gp["empirical"], "s-", label="Gaussian Process")
ax.set_xlabel("Nominal coverage")
ax.set_ylabel("Empirical coverage")
ax.set_title("Reliability diagram: nominal vs empirical coverage")
ax.set_xlim(0.45, 1.0)
ax.set_ylim(0.4, 1.02)
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/reliability_diagram_phase10.png", dpi=150)
print("Saved: results/figures/reliability_diagram_phase10.png")
