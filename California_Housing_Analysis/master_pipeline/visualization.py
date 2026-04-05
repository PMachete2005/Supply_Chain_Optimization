"""
Step 4: Visualization
Generate EDA and analysis plots for California Housing data.

Plots:
  1. Price distribution (histogram)
  2. Geographic price heatmap (lat/lon scatter)
  3. Feature correlation heatmap
  4. Coastal vs Inland price comparison
  5. Income vs Price by region
  6. Enrichment impact bar chart
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')
PLOT_DIR = os.path.join(PROJ_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Load enriched datasets
reg_df = pd.read_csv(os.path.join(PROC_DIR, 'regression_dataset_enriched.csv'))

print("Generating visualizations...")

# ── Plot 1: Price Distribution ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(reg_df['MedHouseVal'] * 100, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Median House Value ($K)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of House Values')
axes[0].axvline(reg_df['MedHouseVal'].median() * 100, color='red', linestyle='--',
                label=f'Median: ${reg_df["MedHouseVal"].median()*100:.0f}K')
axes[0].legend()

axes[1].hist(np.log1p(reg_df['MedHouseVal']), bins=50, color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Log(Median House Value)')
axes[1].set_ylabel('Count')
axes[1].set_title('Log-Transformed Distribution')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '1_price_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  1/6 Price distribution ✓")

# ── Plot 2: Geographic Price Heatmap ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(
    reg_df['Longitude'], reg_df['Latitude'],
    c=reg_df['MedHouseVal'], cmap='RdYlGn',
    s=2, alpha=0.5
)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
cbar.set_label('Median House Value ($100K)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('California Housing Prices — Geographic Distribution')

# Mark major cities
cities = {'SF': (37.77, -122.42), 'LA': (34.05, -118.24),
          'SD': (32.72, -117.16), 'Sac': (38.58, -121.49)}
for name, (lat, lon) in cities.items():
    ax.annotate(name, (lon, lat), fontsize=10, fontweight='bold',
                color='black', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

plt.savefig(os.path.join(PLOT_DIR, '2_geographic_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  2/6 Geographic heatmap ✓")

# ── Plot 3: Feature Correlation Heatmap ─────────────────────────────────────
key_features = ['MedHouseVal', 'MedInc', 'HouseAge', 'AveRooms', 'AveOccup',
                'dist_to_coast', 'is_coastal', 'coastal_income', 'is_bay_area',
                'income_coast_interaction', 'climate_zone']
corr = reg_df[key_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(key_features)))
ax.set_yticks(range(len(key_features)))
ax.set_xticklabels(key_features, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(key_features, fontsize=9)

for i in range(len(key_features)):
    for j in range(len(key_features)):
        color = 'white' if abs(corr.iloc[i, j]) > 0.5 else 'black'
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                fontsize=8, color=color)

plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Feature Correlation Matrix (Key Features)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '3_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  3/6 Correlation heatmap ✓")

# ── Plot 4: Coastal vs Inland ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

coastal = reg_df[reg_df['is_coastal'] == 1]['MedHouseVal'] * 100
inland = reg_df[reg_df['is_coastal'] == 0]['MedHouseVal'] * 100

axes[0].hist(coastal, bins=40, alpha=0.7, label=f'Coastal (n={len(coastal):,}, μ=${coastal.mean():.0f}K)', color='dodgerblue')
axes[0].hist(inland, bins=40, alpha=0.7, label=f'Inland (n={len(inland):,}, μ=${inland.mean():.0f}K)', color='sandybrown')
axes[0].set_xlabel('Median House Value ($K)')
axes[0].set_ylabel('Count')
axes[0].set_title('Price Distribution: Coastal vs Inland')
axes[0].legend()

# Box plot by region
regions = []
values = []
for region, mask in [('Bay Area', reg_df['is_bay_area'] == 1),
                      ('SoCal', reg_df['is_socal'] == 1),
                      ('Coastal\n(other)', (reg_df['is_coastal'] == 1) & (reg_df['is_bay_area'] == 0) & (reg_df['is_socal'] == 0)),
                      ('Inland', reg_df['is_coastal'] == 0)]:
    vals = reg_df[mask]['MedHouseVal'].values * 100
    regions.extend([region] * len(vals))
    values.extend(vals)

region_df = pd.DataFrame({'Region': regions, 'Value': values})
region_order = ['Bay Area', 'SoCal', 'Coastal\n(other)', 'Inland']
bp = axes[1].boxplot(
    [region_df[region_df['Region'] == r]['Value'].values for r in region_order],
    labels=region_order, patch_artist=True,
    boxprops=dict(facecolor='lightblue'))
axes[1].set_ylabel('Median House Value ($K)')
axes[1].set_title('Price by Region')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '4_coastal_vs_inland.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  4/6 Coastal vs Inland ✓")

# ── Plot 5: Income vs Price by Region ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

coastal_mask = reg_df['is_coastal'] == 1
ax.scatter(reg_df[coastal_mask]['MedInc'], reg_df[coastal_mask]['MedHouseVal'] * 100,
           s=3, alpha=0.3, color='dodgerblue', label='Coastal')
ax.scatter(reg_df[~coastal_mask]['MedInc'], reg_df[~coastal_mask]['MedHouseVal'] * 100,
           s=3, alpha=0.3, color='sandybrown', label='Inland')

# Regression lines
for mask, color, label in [(coastal_mask, 'blue', 'Coastal trend'),
                            (~coastal_mask, 'red', 'Inland trend')]:
    x = reg_df[mask]['MedInc'].values
    y = reg_df[mask]['MedHouseVal'].values * 100
    slope, intercept, _, _, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2, label=label)

ax.set_xlabel('Median Income ($10K)')
ax.set_ylabel('Median House Value ($K)')
ax.set_title('Income vs House Value — Coastal Premium is Clear')
ax.legend()
plt.savefig(os.path.join(PLOT_DIR, '5_income_vs_price.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  5/6 Income vs Price ✓")

# ── Plot 6: Distance to Coast vs Price ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(reg_df['dist_to_coast'], reg_df['MedHouseVal'] * 100, s=2, alpha=0.2, color='steelblue')
# Binned averages
bins = pd.cut(reg_df['dist_to_coast'], bins=20)
binned = reg_df.groupby(bins)['MedHouseVal'].mean() * 100
bin_centers = [(b.left + b.right) / 2 for b in binned.index]
ax.plot(bin_centers, binned.values, 'r-o', linewidth=2, markersize=5, label='Binned average')
ax.set_xlabel('Distance to Coast (miles)')
ax.set_ylabel('Median House Value ($K)')
ax.set_title('Price Drops Sharply with Distance from Coast')
ax.legend()

# Climate zone comparison
ax = axes[1]
zone_names = {1: 'SoCal\n(warm)', 2: 'Central\n(med)', 3: 'Bay Area\n(temp)', 4: 'NorCal\n(cool)'}
zone_means = []
zone_labels = []
for z in [1, 2, 3, 4]:
    vals = reg_df[reg_df['climate_zone'] == z]['MedHouseVal'].values * 100
    zone_means.append(vals.mean())
    zone_labels.append(zone_names[z])

bars = ax.bar(zone_labels, zone_means, color=['#FF9999', '#FFCC99', '#99CCFF', '#99FF99'])
ax.set_ylabel('Avg House Value ($K)')
ax.set_title('Average Price by Climate Zone')
for bar, val in zip(bars, zone_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'${val:.0f}K', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '6_coast_distance_and_climate.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  6/6 Coast distance & climate ✓")

print(f"\nAll plots saved to: {PLOT_DIR}/")
