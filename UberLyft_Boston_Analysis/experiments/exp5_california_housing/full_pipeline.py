"""
California Housing — Full Pipeline with Geographic Enrichment

Dataset: sklearn California Housing (20,640 samples, 8 features)
  Base: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

Enrichment: Geographic features computed from external reference data
  - Distance to Pacific coast (external coastline coordinates)
  - Distance to SF, LA, San Diego, Sacramento (external city coordinates)
  - Coastal indicator, Bay Area indicator
  - Climate zone from latitude bands
  - Income × location interactions

Regression target:  median house value ($100K units)
Classification target: is_high_value (above median price)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (r2_score, mean_absolute_error, accuracy_score,
                             f1_score, precision_score, recall_score)
import statsmodels.api as sm
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("STEP 1: LOAD CALIFORNIA HOUSING DATA")
print("=" * 80)

data = fetch_california_housing(as_frame=True)
df = data.frame
print(f"Loaded: {len(df):,} samples × {df.shape[1]} columns")
print(f"Target: MedHouseVal (median house value in $100K)")
print(f"Price range: ${df['MedHouseVal'].min()*100:.0f}K – ${df['MedHouseVal'].max()*100:.0f}K")
print(f"\nBase features: {data.feature_names}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: CREATE BASE DATASETS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 2: CREATE BASE DATASETS")
print("=" * 80)

# Regression: all 8 features → predict MedHouseVal
reg_base = df.copy()

# Classification: all 8 features → predict is_high_value
median_val = df['MedHouseVal'].median()
df['is_high_value'] = (df['MedHouseVal'] > median_val).astype(int)
clf_base = df.drop('MedHouseVal', axis=1).copy()

print(f"Regression:     {reg_base.shape[0]:,} × {reg_base.shape[1]-1} features → MedHouseVal")
print(f"Classification: {clf_base.shape[0]:,} × {clf_base.shape[1]-1} features → is_high_value")
print(f"Classification balance: {df['is_high_value'].mean()*100:.1f}% high value")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: GEOGRAPHIC ENRICHMENT (external reference data)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 3: GEOGRAPHIC ENRICHMENT")
print("=" * 80)

def haversine(lat1, lon1, lat2, lon2):
    """Distance in miles between two points."""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# === EXTERNAL DATA: California coastline reference points ===
# Source: NOAA/USGS coastline coordinates
coast_points = [
    (32.54, -117.12),   # San Diego / Imperial Beach
    (32.72, -117.17),   # San Diego downtown
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
    (37.62, -122.49),   # Daly City coast
    (37.79, -122.51),   # SF (Ocean Beach)
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

# === EXTERNAL DATA: Major city coordinates ===
cities = {
    'SF': (37.7749, -122.4194),
    'LA': (34.0522, -118.2437),
    'San_Diego': (32.7157, -117.1611),
    'Sacramento': (38.5816, -121.4944),
    'San_Jose': (37.3382, -121.8863),
    'Oakland': (37.8044, -122.2712),
    'Fresno': (36.7378, -119.7871),
}

lat = df['Latitude'].values
lon = df['Longitude'].values

# 1. Distance to nearest coast point
print("Computing distance to coast...")
coast_dists = np.column_stack([
    haversine(lat, lon, clat, clon)
    for clat, clon in coast_points
])
dist_to_coast = coast_dists.min(axis=1)

# 2. Distance to major cities
print("Computing distance to major cities...")
city_distances = {}
for city, (clat, clon) in cities.items():
    city_distances[f'dist_{city}'] = haversine(lat, lon, clat, clon)

# 3. Coastal indicator (within 30 miles of coast)
is_coastal = (dist_to_coast < 30).astype(int)

# 4. Bay Area indicator (within 50 miles of SF)
is_bay_area = (city_distances['dist_SF'] < 50).astype(int)

# 5. SoCal indicator (within 60 miles of LA)
is_socal = (city_distances['dist_LA'] < 60).astype(int)

# 6. Climate zone based on latitude (external climate zone boundaries)
# Source: California climate zones (CEC Building Climate Zones)
climate_zone = np.zeros(len(df))
climate_zone[lat < 34] = 1     # Southern CA (warm)
climate_zone[(lat >= 34) & (lat < 36)] = 2  # Central coast/valley
climate_zone[(lat >= 36) & (lat < 38)] = 3  # Bay Area / Central valley
climate_zone[lat >= 38] = 4    # Northern CA (cooler)

# 7. Inland distance interaction with income
income_coast = df['MedInc'].values * (1 / (dist_to_coast + 1))

# 8. Urban density proxy (pop / avg occupancy — neighborhood density)
urban_density = df['Population'].values / (df['AveOccup'].values + 0.1)

# 9. Distance to nearest major city
dist_nearest_city = np.column_stack(list(city_distances.values())).min(axis=1)

# 10. Coastal premium (income × coastal indicator)
coastal_income = df['MedInc'].values * is_coastal

# Build enriched datasets
enrichment_features = {
    'dist_to_coast': dist_to_coast,
    'dist_SF': city_distances['dist_SF'],
    'dist_LA': city_distances['dist_LA'],
    'dist_San_Diego': city_distances['dist_San_Diego'],
    'dist_Sacramento': city_distances['dist_Sacramento'],
    'dist_San_Jose': city_distances['dist_San_Jose'],
    'dist_nearest_city': dist_nearest_city,
    'is_coastal': is_coastal,
    'is_bay_area': is_bay_area,
    'is_socal': is_socal,
    'climate_zone': climate_zone,
    'income_coast_interaction': income_coast,
    'urban_density': urban_density,
    'coastal_income': coastal_income,
}

reg_enriched = reg_base.copy()
clf_enriched = clf_base.copy()

for feat_name, feat_values in enrichment_features.items():
    reg_enriched[feat_name] = feat_values
    clf_enriched[feat_name] = feat_values

print(f"\nEnrichment features added: {len(enrichment_features)}")
for name in enrichment_features:
    print(f"  {name}")

print(f"\nRegression:     {reg_base.shape[1]-1} base → {reg_enriched.shape[1]-1} enriched features")
print(f"Classification: {clf_base.shape[1]-1} base → {clf_enriched.shape[1]-1} enriched features")

# Quick correlation check
print("\nEnrichment feature correlations with MedHouseVal:")
for feat_name, feat_values in enrichment_features.items():
    corr = np.corrcoef(feat_values, df['MedHouseVal'].values)[0, 1]
    bar = "█" * int(abs(corr) * 40)
    print(f"  {feat_name:<30} {corr:>+.4f} {bar}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: RUN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def run_regression(dataframe, label):
    X = dataframe.drop('MedHouseVal', axis=1)
    y = dataframe['MedHouseVal']
    feat_names = X.columns.tolist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); Xtrs = sc.fit_transform(Xtr); Xtes = sc.transform(Xte)

    results = {}

    # 1. OLS
    ols = sm.OLS(ytr, sm.add_constant(Xtrs)).fit()
    pred = ols.predict(sm.add_constant(Xtes))
    results['OLS'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}

    # 2. Ridge
    m = Ridge(alpha=1.0, random_state=42).fit(Xtrs, ytr)
    pred = m.predict(Xtes)
    results['Ridge'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}
    ridge_coef = m.coef_

    # 3. Lasso
    m = Lasso(alpha=0.01, random_state=42, max_iter=10000).fit(Xtrs, ytr)
    pred = m.predict(Xtes)
    results['Lasso'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}

    # 4. Decision Tree
    m = DecisionTreeRegressor(max_depth=10, random_state=42).fit(Xtr, ytr)
    pred = m.predict(Xte)
    results['Decision Tree'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}
    dt_imp = dict(zip(feat_names, m.feature_importances_))

    # 5. Polynomial
    top5 = [feat_names[i] for i in np.argsort(np.abs(ridge_coef))[-5:][::-1]]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xtrp = poly.fit_transform(Xtr[top5]); Xtep = poly.transform(Xte[top5])
    m = Ridge(alpha=1.0, random_state=42).fit(Xtrp, ytr)
    pred = m.predict(Xtep)
    results['Polynomial'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}

    return results, ridge_coef, feat_names, dt_imp

def run_classification(dataframe, label):
    X = dataframe.drop('is_high_value', axis=1)
    y = dataframe['is_high_value']
    feat_names = X.columns.tolist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler(); Xtrs = sc.fit_transform(Xtr); Xtes = sc.transform(Xte)

    results = {}
    models = [
        ('Logistic Reg', LogisticRegression(max_iter=1000, random_state=42), True),
        ('Decision Tree', DecisionTreeClassifier(max_depth=10, random_state=42), False),
        ('Naive Bayes', GaussianNB(), True),
        ('LDA', LinearDiscriminantAnalysis(), True),
        ('Perceptron', Perceptron(max_iter=1000, random_state=42), True),
    ]

    details = {}
    for name, model, use_scaled in models:
        Xtr_use = Xtrs if use_scaled else Xtr
        Xte_use = Xtes if use_scaled else Xte
        model.fit(Xtr_use, ytr)
        pred = model.predict(Xte_use)
        results[name] = {
            'F1': f1_score(yte, pred), 'Acc': accuracy_score(yte, pred),
            'Prec': precision_score(yte, pred, zero_division=0),
            'Rec': recall_score(yte, pred),
        }
        if hasattr(model, 'feature_importances_'):
            details[name] = dict(zip(feat_names, model.feature_importances_))

    return results, details

print("\n" + "=" * 80)
print("STEP 4: RUNNING ALL MODELS")
print("=" * 80)

print("\n[1/4] Regression BASE...")
reg_b_res, rb_coef, rb_feats, rb_dt = run_regression(reg_base, "Base")
print("[2/4] Regression ENRICHED...")
reg_e_res, re_coef, re_feats, re_dt = run_regression(reg_enriched, "Enriched")
print("[3/4] Classification BASE...")
clf_b_res, cb_det = run_classification(clf_base, "Base")
print("[4/4] Classification ENRICHED...")
clf_e_res, ce_det = run_classification(clf_enriched, "Enriched")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("REGRESSION — Median House Value Prediction")
print("=" * 80)

print(f"\n{'Model':<18} {'Base R²':>8} {'Enr R²':>8} {'Δ R²':>8} {'Imp%':>7} {'Base MAE':>9} {'Enr MAE':>9}")
print("-" * 72)
reg_wins = 0
for name in ['OLS', 'Ridge', 'Lasso', 'Decision Tree', 'Polynomial']:
    b = reg_b_res[name]; e = reg_e_res[name]
    d = e['R2'] - b['R2']
    pct = d / abs(b['R2']) * 100 if b['R2'] != 0 else 0
    if d > 0.001: reg_wins += 1
    icon = "✅" if d > 0.001 else "➖" if abs(d) < 0.001 else "❌"
    print(f"{icon} {name:<16} {b['R2']:>8.4f} {e['R2']:>8.4f} {d:>+8.4f} {pct:>+6.1f}% {b['MAE']:>8.4f} {e['MAE']:>8.4f}")

avg_br = np.mean([v['R2'] for v in reg_b_res.values()])
avg_er = np.mean([v['R2'] for v in reg_e_res.values()])
print(f"\n   Average R²: {avg_br:.4f} → {avg_er:.4f} ({(avg_er-avg_br)/abs(avg_br)*100:+.2f}%)")
print(f"   Models improved: {reg_wins}/5")

# Top enriched features
print(f"\n   Top 5 Enriched DT importances:")
for feat, imp in sorted(re_dt.items(), key=lambda x: x[1], reverse=True)[:5]:
    new = " [NEW]" if feat in enrichment_features else ""
    print(f"     {feat:<30} {imp:.4f}{new}")

print(f"\n   Top 5 Enriched Ridge coefficients:")
for i in np.argsort(np.abs(re_coef))[-5:][::-1]:
    new = " [NEW]" if re_feats[i] in enrichment_features else ""
    print(f"     {re_feats[i]:<30} {re_coef[i]:>+.4f}{new}")

print("\n" + "=" * 80)
print("CLASSIFICATION — High-Value Housing Prediction")
print("=" * 80)

print(f"\n{'Model':<18} {'Base F1':>8} {'Enr F1':>8} {'Δ F1':>8} {'Imp%':>7} {'Base Acc':>9} {'Enr Acc':>9}")
print("-" * 72)
clf_wins = 0
for name in ['Logistic Reg', 'Decision Tree', 'Naive Bayes', 'LDA', 'Perceptron']:
    b = clf_b_res[name]; e = clf_e_res[name]
    d = e['F1'] - b['F1']
    pct = d / abs(b['F1']) * 100 if b['F1'] != 0 else 0
    if d > 0.001: clf_wins += 1
    icon = "✅" if d > 0.001 else "➖" if abs(d) < 0.001 else "❌"
    print(f"{icon} {name:<16} {b['F1']:>8.4f} {e['F1']:>8.4f} {d:>+8.4f} {pct:>+6.1f}% {b['Acc']:>8.4f} {e['Acc']:>8.4f}")

avg_bf = np.mean([v['F1'] for v in clf_b_res.values()])
avg_ef = np.mean([v['F1'] for v in clf_e_res.values()])
print(f"\n   Average F1: {avg_bf:.4f} → {avg_ef:.4f} ({(avg_ef-avg_bf)/abs(avg_bf)*100:+.2f}%)")
print(f"   Models improved: {clf_wins}/5")

if 'Decision Tree' in ce_det:
    print(f"\n   Top 5 Enriched DT importances:")
    for feat, imp in sorted(ce_det['Decision Tree'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        new = " [NEW]" if feat in enrichment_features else ""
        print(f"     {feat:<30} {imp:.4f}{new}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

best_reg = max(reg_e_res.items(), key=lambda x: x[1]['R2'])
best_clf = max(clf_e_res.items(), key=lambda x: x[1]['F1'])
reg_delta = avg_er - avg_br
clf_delta = avg_ef - avg_bf

print(f"""
  REGRESSION (enriched):
    Best:  {best_reg[0]} R²={best_reg[1]['R2']:.4f}, MAE={best_reg[1]['MAE']:.4f}
    Avg:   {avg_er:.4f} (base: {avg_br:.4f}, Δ={reg_delta:+.4f}, {reg_delta/abs(avg_br)*100:+.2f}%)
    Models improved by enrichment: {reg_wins}/5

  CLASSIFICATION (enriched):
    Best:  {best_clf[0]} F1={best_clf[1]['F1']:.4f}, Acc={best_clf[1]['Acc']:.4f}
    Avg:   {avg_ef:.4f} (base: {avg_bf:.4f}, Δ={clf_delta:+.4f}, {clf_delta/abs(avg_bf)*100:+.2f}%)
    Models improved by enrichment: {clf_wins}/5

  ENRICHMENT BENEFIT:
    Regression:     {reg_delta:+.4f} avg R² ({reg_delta/abs(avg_br)*100:+.2f}%) — {reg_wins}/5 models improved
    Classification: {clf_delta:+.4f} avg F1 ({clf_delta/abs(avg_bf)*100:+.2f}%) — {clf_wins}/5 models improved
""")

vs_uber_reg = reg_wins >= 3
vs_uber_clf = clf_wins >= 3
if vs_uber_reg and vs_uber_clf:
    print("  🏆 ENRICHMENT GENUINELY HELPS BOTH REGRESSION AND CLASSIFICATION")
elif vs_uber_reg or vs_uber_clf:
    print("  📊 ENRICHMENT HELPS ONE TASK BUT NOT BOTH")
else:
    print("  ❌ ENRICHMENT DOES NOT MEANINGFULLY HELP")

print(f"""
  vs UBER/LYFT ORIGINAL:
    Uber/Lyft enrichment: +0.0% reg, +0.1% clf (0/10 models improved)
    California enrichment: {reg_delta/abs(avg_br)*100:+.1f}% reg, {clf_delta/abs(avg_bf)*100:+.1f}% clf ({reg_wins+clf_wins}/10 models improved)
""")
