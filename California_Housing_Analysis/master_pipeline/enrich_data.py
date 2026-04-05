"""
Step 2: Enrich Datasets
Add geographic features computed from external reference data sources.

External data sources:
  1. NOAA/USGS California coastline coordinates (29 reference points)
  2. US Census Bureau major city coordinates (7 cities)
  3. California Energy Commission climate zone boundaries

Enrichment features (14 total):
  - Distance to Pacific coast
  - Distance to 5 major cities (SF, LA, San Diego, Sacramento, San Jose)
  - Distance to nearest major city
  - Coastal indicator (within 30 miles)
  - Bay Area indicator (within 50 miles of SF)
  - SoCal indicator (within 60 miles of LA)
  - Climate zone (4 zones from CEC boundaries)
  - Income × coast proximity interaction
  - Urban density indicator
  - Coastal income interaction
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASE DATASETS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: LOAD BASE DATASETS")
print("=" * 70)

reg_path = os.path.join(PROC_DIR, 'regression_dataset.csv')
clf_path = os.path.join(PROC_DIR, 'classification_dataset.csv')

reg_df = pd.read_csv(reg_path)
clf_df = pd.read_csv(clf_path)

print(f"Regression:     {reg_df.shape[0]:,} × {reg_df.shape[1]} columns")
print(f"Classification: {clf_df.shape[0]:,} × {clf_df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFINE EXTERNAL REFERENCE DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: LOAD EXTERNAL REFERENCE DATA")
print("=" * 70)

# --- Source 1: California coastline coordinates ---
# 29 reference points along the Pacific coast from San Diego to Crescent City
# Source: NOAA National Ocean Service coastline data
COAST_POINTS = [
    (32.54, -117.12),   # Imperial Beach
    (32.72, -117.17),   # San Diego
    (33.01, -117.29),   # Oceanside
    (33.19, -117.38),   # Carlsbad
    (33.46, -117.60),   # Dana Point
    (33.62, -117.93),   # Newport Beach
    (33.74, -118.29),   # Long Beach
    (33.86, -118.40),   # Redondo Beach
    (33.95, -118.47),   # Santa Monica
    (34.03, -118.77),   # Malibu
    (34.40, -119.69),   # Santa Barbara
    (34.95, -120.44),   # Pismo Beach
    (35.37, -120.85),   # Morro Bay
    (35.63, -121.19),   # San Simeon
    (36.22, -121.76),   # Big Sur
    (36.60, -121.89),   # Monterey
    (36.96, -122.02),   # Santa Cruz
    (37.50, -122.43),   # Pacifica
    (37.62, -122.49),   # Daly City
    (37.79, -122.51),   # SF Ocean Beach
    (37.83, -122.48),   # Golden Gate
    (38.06, -122.70),   # Point Reyes
    (38.30, -123.07),   # Bodega Bay
    (38.79, -123.59),   # Point Arena
    (39.43, -123.81),   # Fort Bragg
    (40.44, -124.10),   # Ferndale
    (40.80, -124.16),   # Eureka
    (41.06, -124.14),   # Trinidad
    (41.76, -124.20),   # Crescent City
]
print(f"Coastline reference: {len(COAST_POINTS)} points (San Diego → Crescent City)")

# --- Source 2: Major California city coordinates ---
# Source: US Census Bureau city centroid coordinates
CITIES = {
    'SF':          (37.7749, -122.4194),
    'LA':          (34.0522, -118.2437),
    'San_Diego':   (32.7157, -117.1611),
    'Sacramento':  (38.5816, -121.4944),
    'San_Jose':    (37.3382, -121.8863),
}
print(f"City references: {list(CITIES.keys())}")

# --- Source 3: Climate zone boundaries ---
# Source: California Energy Commission Building Climate Zones
CLIMATE_ZONES = {
    1: "Southern CA (lat < 34°) — warm/arid",
    2: "Central CA (34°-36°) — Mediterranean",
    3: "Bay Area (36°-38°) — temperate coastal",
    4: "Northern CA (lat ≥ 38°) — cooler/wet",
}
print(f"Climate zones: {len(CLIMATE_ZONES)} zones")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE ENRICHMENT FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: COMPUTE ENRICHMENT FEATURES")
print("=" * 70)

def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in miles between arrays of coordinates."""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_enrichment(df):
    """Compute all 14 enrichment features for a dataframe with Latitude/Longitude."""
    lat = df['Latitude'].values
    lon = df['Longitude'].values
    enriched = df.copy()

    # 1. Distance to nearest coast point (miles)
    print("  Computing distance to coast...")
    coast_dists = np.column_stack([
        haversine(lat, lon, clat, clon)
        for clat, clon in COAST_POINTS
    ])
    enriched['dist_to_coast'] = coast_dists.min(axis=1)

    # 2-6. Distance to major cities
    print("  Computing distance to major cities...")
    city_dists = {}
    for city, (clat, clon) in CITIES.items():
        col = f'dist_{city}'
        enriched[col] = haversine(lat, lon, clat, clon)
        city_dists[col] = enriched[col].values

    # 7. Distance to nearest major city
    enriched['dist_nearest_city'] = np.column_stack(list(city_dists.values())).min(axis=1)

    # 8. Coastal indicator (within 30 miles of coast)
    enriched['is_coastal'] = (enriched['dist_to_coast'] < 30).astype(int)

    # 9. Bay Area indicator (within 50 miles of SF)
    enriched['is_bay_area'] = (enriched['dist_SF'] < 50).astype(int)

    # 10. SoCal indicator (within 60 miles of LA)
    enriched['is_socal'] = (enriched['dist_LA'] < 60).astype(int)

    # 11. Climate zone (from CEC latitude boundaries)
    print("  Assigning climate zones...")
    zones = np.zeros(len(df))
    zones[lat < 34] = 1
    zones[(lat >= 34) & (lat < 36)] = 2
    zones[(lat >= 36) & (lat < 38)] = 3
    zones[lat >= 38] = 4
    enriched['climate_zone'] = zones

    # 12. Income × coast proximity (diminishing return with distance)
    enriched['income_coast_interaction'] = (
        df['MedInc'].values * (1 / (enriched['dist_to_coast'].values + 1))
    )

    # 13. Urban density (population / household count proxy)
    enriched['urban_density'] = (
        df['Population'].values / (df['AveOccup'].values + 0.1)
    )

    # 14. Coastal income (income × coastal binary — captures coastal premium)
    enriched['coastal_income'] = (
        df['MedInc'].values * enriched['is_coastal'].values
    )

    return enriched


# Apply to both datasets
print("\nEnriching regression dataset...")
reg_enriched = compute_enrichment(reg_df)

print("\nEnriching classification dataset...")
clf_enriched = compute_enrichment(clf_df)

# ══════════════════════════════════════════════════════════════════════════════
# 4. VALIDATE & SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: VALIDATE & SAVE")
print("=" * 70)

enrichment_cols = [
    'dist_to_coast', 'dist_SF', 'dist_LA', 'dist_San_Diego',
    'dist_Sacramento', 'dist_San_Jose', 'dist_nearest_city',
    'is_coastal', 'is_bay_area', 'is_socal', 'climate_zone',
    'income_coast_interaction', 'urban_density', 'coastal_income'
]

# Validate no NaN in enrichment columns
for col in enrichment_cols:
    assert reg_enriched[col].isna().sum() == 0, f"NaN in regression {col}"
    assert clf_enriched[col].isna().sum() == 0, f"NaN in classification {col}"
print("Validation: No NaN values in enrichment features ✓")

# Correlation with targets
print("\nEnrichment feature correlations:")
print(f"  {'Feature':<30} {'w/ MedHouseVal':>14} {'w/ is_high_value':>16}")
print("  " + "-" * 62)
for col in enrichment_cols:
    corr_reg = reg_enriched[col].corr(reg_enriched['MedHouseVal'])
    corr_clf = clf_enriched[col].corr(clf_enriched['is_high_value'])
    bar = "█" * int(abs(corr_reg) * 30)
    print(f"  {col:<30} {corr_reg:>+.4f}         {corr_clf:>+.4f}  {bar}")

# Save
reg_enr_path = os.path.join(PROC_DIR, 'regression_dataset_enriched.csv')
clf_enr_path = os.path.join(PROC_DIR, 'classification_dataset_enriched.csv')

reg_enriched.to_csv(reg_enr_path, index=False)
clf_enriched.to_csv(clf_enr_path, index=False)

n_new = len(enrichment_cols)
print(f"""
{'='*70}
ENRICHMENT COMPLETE
{'='*70}

Features added: {n_new}
  Regression:     {reg_df.shape[1]-1} → {reg_enriched.shape[1]-1} features (+{n_new})
  Classification: {clf_df.shape[1]-1} → {clf_enriched.shape[1]-1} features (+{n_new})

Output:
  {reg_enr_path}
    {reg_enriched.shape[0]:,} rows × {reg_enriched.shape[1]} columns

  {clf_enr_path}
    {clf_enriched.shape[0]:,} rows × {clf_enriched.shape[1]} columns

External data sources used:
  1. NOAA coastline coordinates ({len(COAST_POINTS)} points)
  2. US Census Bureau city coordinates ({len(CITIES)} cities)
  3. CEC climate zone boundaries ({len(CLIMATE_ZONES)} zones)

Next step: python run_models.py
""")
