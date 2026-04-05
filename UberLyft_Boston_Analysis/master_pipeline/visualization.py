"""
Uber / Lyft Boston Ride Analysis — Visualization Suite
6 plots saved individually to ./plots/

Usage:
    python visualize_boston_rides.py

Outputs (saved to ./plots/):
    01_histogram_fare_distribution.png
    02_bar_avg_fare_by_surge.png
    03_qq_fare_normality.png
    04_boxplot_fare_by_ride_length.png
    05_scatter_distance_vs_fare.png
    06_heatmap_weather_surge_fare.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

# ── File paths ────────────────────────────────────────────
REG_PATH  = "../new_data/processed/regression_dataset_enriched.csv"
CLF_PATH  = "../new_data/processed/classification_dataset_enriched.csv"
PLOTS_DIR = "../new_data/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

def save(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {path}")

# ── Load data ─────────────────────────────────────────────
print("Loading data...")
reg = pd.read_csv(REG_PATH)
clf = pd.read_csv(CLF_PATH)

reg["Cab"] = reg["cab_type"].map({0: "Uber", 1: "Lyft"})
clf["Cab"] = clf["cab_type"].map({0: "Uber", 1: "Lyft"})

uber = reg[reg["Cab"] == "Uber"]["price"]
lyft = reg[reg["Cab"] == "Lyft"]["price"]

print("Generating plots...")

# ══════════════════════════════════════════════════════════
# 1. HISTOGRAM — Price distribution by cab type
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
bins = np.arange(0, 80, 2.5)
ax.hist(uber, bins=bins, alpha=0.65, color="steelblue", label=f"Uber (n={len(uber):,})")
ax.hist(lyft, bins=bins, alpha=0.65, color="crimson",   label=f"Lyft (n={len(lyft):,})")
ax.axvline(uber.mean(), color="steelblue", linestyle="--", linewidth=1.5, label=f"Uber mean ${uber.mean():.2f}")
ax.axvline(lyft.mean(), color="crimson",   linestyle="--", linewidth=1.5, label=f"Lyft mean ${lyft.mean():.2f}")
ax.set_title("Histogram — Fare Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Fare ($)")
ax.set_ylabel("Count")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.4)
fig.tight_layout()
save("01_histogram_fare_distribution.png")

# ══════════════════════════════════════════════════════════
# 2. BAR CHART — Average fare by ride service type
# ══════════════════════════════════════════════════════════
name_map = {
    0: "Lyft",      1: "Lyft XL",     8: "Lux",
    9: "Lux Black", 10: "Lux Black XL", 11: "Lyft Black",
    2: "UberX",     3: "UberXL",      4: "Black",
    5: "UberPool",  6: "Black SUV",   7: "UberWAV",
}
reg["service"] = reg["name"].map(name_map)

service_avg = (
    reg.groupby(["service", "Cab"])["price"]
    .mean()
    .reset_index()
    .sort_values("price", ascending=True)
)

uber_svcs = service_avg[service_avg["Cab"] == "Uber"].set_index("service")["price"]
lyft_svcs = service_avg[service_avg["Cab"] == "Lyft"].set_index("service")["price"]

# Sort each group by avg price
uber_svcs = uber_svcs.sort_values()
lyft_svcs = lyft_svcs.sort_values()

fig, (ax_u, ax_l) = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

for ax, data, color, label in [
    (ax_u, uber_svcs, "steelblue", "Uber"),
    (ax_l, lyft_svcs, "crimson",   "Lyft"),
]:
    bars = ax.barh(data.index, data.values, color=color, alpha=0.82, edgecolor="none")
    for bar, val in zip(bars, data.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"${val:.2f}", va="center", fontsize=9, color="dimgray")
    ax.set_title(f"{label} — Avg Fare by Service", fontsize=13, fontweight="bold")
    ax.set_xlabel("Avg Fare ($)")
    ax.set_xlim(0, data.max() * 1.22)
    ax.grid(axis="x", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Bar Chart — Avg Fare by Ride Service Type", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save("02_bar_avg_fare_by_service.png")

# ══════════════════════════════════════════════════════════
# 3. Q-Q PLOT — Are fares normally distributed?
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
sample_qq = reg["price"].sample(n=3000, random_state=42).sort_values()
(osm, osr), (slope, intercept, r) = stats.probplot(sample_qq, dist="norm")
ax.scatter(osm, osr, s=8, alpha=0.5, color="steelblue", label="Fare quantiles")
ax.plot(osm, slope * np.array(osm) + intercept, color="crimson",
        linewidth=2, label=f"Normal fit  r={r:.3f}")
ax.set_title("Q-Q Plot — Fare vs Normal Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles ($)")
ax.legend(fontsize=10)
ax.grid(alpha=0.4)
fig.tight_layout()
save("03_qq_fare_normality.png")

# ══════════════════════════════════════════════════════════
# 4. BOX PLOT — Fare by ride length and cab type
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

def ride_label(row):
    if row["short_ride"]:  return "Short (<1.5 mi)"
    if row["medium_ride"]: return "Medium (1.5-3 mi)"
    return "Long (>3 mi)"

reg["ride_type"] = reg.apply(ride_label, axis=1)
ride_order = ["Short (<1.5 mi)", "Medium (1.5-3 mi)", "Long (>3 mi)"]

box_data, positions, colors, tick_pos = [], [], [], []
for i, rt in enumerate(ride_order):
    for j, (cab, col) in enumerate([("Uber", "steelblue"), ("Lyft", "crimson")]):
        box_data.append(reg[(reg["ride_type"] == rt) & (reg["Cab"] == cab)]["price"].values)
        positions.append(i * 3 + j)
        colors.append(col)
    tick_pos.append(i * 3 + 0.5)

bp = ax.boxplot(box_data, positions=positions, widths=0.7,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticks(tick_pos)
ax.set_xticklabels(ride_order, fontsize=10)
ax.set_title("Box Plot — Fare by Ride Length & Cab Type", fontsize=14, fontweight="bold")
ax.set_ylabel("Fare ($)")
ax.legend(handles=[Patch(color="steelblue", label="Uber"),
                   Patch(color="crimson",   label="Lyft")], fontsize=10)
ax.grid(axis="y", alpha=0.4)
fig.tight_layout()
save("04_boxplot_fare_by_ride_length.png")

# ══════════════════════════════════════════════════════════
# 5. SCATTER PLOT — Distance vs Price coloured by is_premium
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
sample_sc = reg.sample(n=8000, random_state=42)
for is_p, col, lbl in [(0, "steelblue", "Standard"), (1, "crimson", "Premium")]:
    sub = sample_sc[sample_sc["is_premium"] == is_p]
    ax.scatter(sub["distance"], sub["price"], s=5, alpha=0.35,
               color=col, label=f"{lbl} (n={len(sub):,})", rasterized=True)
m, b, r, *_ = stats.linregress(sample_sc["distance"], sample_sc["price"])
xfit = np.linspace(0, 8, 200)
ax.plot(xfit, m * xfit + b, color="black", linewidth=2, linestyle="--",
        label=f"OLS  r2={r**2:.3f}")
ax.set_xlim(0, 8)
ax.set_ylim(0, 95)
ax.set_title("Scatter — Distance vs Fare (by Ride Class)", fontsize=14, fontweight="bold")
ax.set_xlabel("Distance (miles)")
ax.set_ylabel("Fare ($)")
ax.legend(fontsize=10, markerscale=3)
ax.grid(alpha=0.4)
fig.tight_layout()
save("05_scatter_distance_vs_fare.png")

# ══════════════════════════════════════════════════════════
# 6. HEATMAP — Avg fare: weather severity × surge level
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
surge_labels = {1.0:"1x",1.25:"1.25x",1.5:"1.5x",1.75:"1.75x",2.0:"2x",2.5:"2.5x",3.0:"3x"}
reg["surge_label"]  = reg["surge_multiplier"].map(surge_labels)
reg["weather_sev"]  = reg["weather_severity"].map({0:"Clear", 1:"Moderate", 2:"Severe"})

pivot = (
    reg.groupby(["weather_sev", "surge_label"])["price"]
    .mean()
    .unstack("surge_label")
    .reindex(["Clear", "Moderate", "Severe"])
)[["1x","1.25x","1.5x","1.75x","2x","2.5x","3x"]]

im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("Avg Fare ($)", fontsize=10)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=10)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=10)
for ri in range(pivot.shape[0]):
    for ci in range(pivot.shape[1]):
        val = pivot.values[ri, ci]
        ax.text(ci, ri, f"${val:.1f}", ha="center", va="center",
                fontsize=9, color="black" if val < 35 else "white")
ax.set_title("Heatmap — Avg Fare: Weather Severity x Surge", fontsize=14, fontweight="bold")
ax.set_xlabel("Surge Multiplier")
ax.set_ylabel("Weather Severity")
fig.tight_layout()
save("06_heatmap_weather_surge_fare.png")

print(f"\nDone. All 6 plots saved to ./{PLOTS_DIR}/")