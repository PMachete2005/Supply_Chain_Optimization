"""
Experiment 2 — Full Pipeline
Regression: Predict price WITHOUT service name (forces models to learn actual pricing patterns)
Classification: Predict is_premium from ride context (same as original)

Enrichment: Add REAL unused raw columns (apparentTemp, precipProb, visibility,
pressure, windGust, cloudCover, dewPoint) + interactions — NOT synthetic features.

Pipeline: create_datasets → enrich → run_models → compare base vs enriched
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (r2_score, mean_absolute_error, accuracy_score,
                             f1_score, precision_score, recall_score)
import statsmodels.api as sm
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

RAW_PATH = '/home/kushagarwal/Downloads/archive(1)/rideshare_kaggle.csv'
OUT_DIR = '/home/kushagarwal/CascadeProjects/Supply_Chain_Optimization/UberLyft_Boston_Analysis/experiments/exp2_within_service_price'

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: CREATE BASE DATASETS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("STEP 1: CREATE BASE DATASETS")
print("=" * 80)

df = pd.read_csv(RAW_PATH)
df = df.dropna(subset=['price'])
df = df[df['price'] > 0]
print(f"Cleaned: {len(df):,} records")

# Stratified sample
df['price_cat'] = pd.cut(df['price'], bins=[0, 10, 15, 25, 100], labels=['lo','mid','hi','prem'])
df['strata'] = df['cab_type'].astype(str) + '_' + df['price_cat'].astype(str)
strata_counts = df['strata'].value_counts()
valid = strata_counts[strata_counts >= 10].index
df = df[df['strata'].isin(valid)]
df = df.groupby('strata').apply(
    lambda x: x.sample(n=max(1, int(len(x)/len(df)*100000)), random_state=42)
).reset_index(drop=True)
if len(df) > 100000:
    df = df.sample(100000, random_state=42).reset_index(drop=True)
print(f"Sampled: {len(df):,} records")

# Feature engineering — BASE features (using only the columns create_datasets.py uses)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9) & (df['is_weekend'] == 0)).astype(int)
df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19) & (df['is_weekend'] == 0)).astype(int)
df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

df['is_rainy'] = (df['precipIntensity'] > 0).astype(int)
df['is_cold'] = (df['temperature'] < 40).astype(int)
df['weather_severity'] = df['is_rainy'] + df['is_cold']

df['short_ride'] = (df['distance'] < 2).astype(int)
df['medium_ride'] = ((df['distance'] >= 2) & (df['distance'] < 5)).astype(int)
df['long_ride'] = (df['distance'] >= 5).astype(int)

df['is_premium'] = df['name'].str.contains('Black|Lux|SUV|XL', case=False, na=False).astype(int)
df['distance_surge'] = df['distance'] * df['surge_multiplier']

for col in ['cab_type', 'source', 'destination']:
    df[col + '_enc'] = LabelEncoder().fit_transform(df[col].astype(str))

# === REGRESSION BASE: predict price WITHOUT name ===
reg_base_features = [
    'distance', 'surge_multiplier', 'distance_surge',
    'hour', 'day_of_week', 'is_weekend',
    'is_morning_rush', 'is_evening_rush', 'is_rush_hour', 'is_night',
    'temperature', 'humidity', 'windSpeed',
    'is_rainy', 'is_cold', 'weather_severity',
    'cab_type_enc', 'is_premium',
    'source_enc', 'destination_enc',
    'short_ride', 'medium_ride', 'long_ride'
]

# === CLASSIFICATION BASE: predict is_premium WITHOUT name, surge ===
clf_base_features = [
    'distance',
    'hour', 'day_of_week', 'is_weekend',
    'is_morning_rush', 'is_evening_rush', 'is_rush_hour', 'is_night',
    'temperature', 'humidity', 'windSpeed',
    'is_rainy', 'is_cold', 'weather_severity',
    'cab_type_enc',
    'source_enc', 'destination_enc',
    'short_ride', 'medium_ride', 'long_ride'
]

reg_base = df[reg_base_features + ['price']].copy()
reg_base.columns = [c.replace('_enc', '') for c in reg_base.columns]

clf_base = df[clf_base_features].copy()
clf_base['is_premium'] = df['is_premium'].values
clf_base.columns = [c.replace('_enc', '') for c in clf_base.columns]

print(f"\nRegression base:     {reg_base.shape[0]:,} × {reg_base.shape[1]-1} features → price")
print(f"Classification base: {clf_base.shape[0]:,} × {clf_base.shape[1]-1} features → is_premium")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: ENRICH WITH REAL UNUSED RAW COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 2: ENRICH WITH REAL RAW DATA COLUMNS")
print("=" * 80)

# These are REAL columns from the raw CSV that create_datasets.py ignores
enrichment_cols = {
    'apparentTemperature': 'Feels-like temperature (wind chill / heat index)',
    'precipProbability': 'Probability of precipitation (0-1)',
    'visibility': 'Visibility in miles',
    'pressure': 'Atmospheric pressure (hPa)',
    'windGust': 'Wind gust speed',
    'cloudCover': 'Cloud cover fraction (0-1)',
    'dewPoint': 'Dew point temperature',
}

print("\nNew raw features added:")
for col, desc in enrichment_cols.items():
    print(f"  {col:<25} {desc}")

# Build enriched datasets
reg_enriched = reg_base.copy()
clf_enriched = clf_base.copy()

for col in enrichment_cols:
    reg_enriched[col] = df[col].values
    clf_enriched[col] = df[col].values

# Derived features from the new raw columns (interactions — no leakage)
# Wind chill effect
reg_enriched['wind_chill'] = df['apparentTemperature'].values - df['temperature'].values
clf_enriched['wind_chill'] = df['apparentTemperature'].values - df['temperature'].values

# Low visibility indicator
reg_enriched['low_visibility'] = (df['visibility'] < 3).astype(int).values
clf_enriched['low_visibility'] = (df['visibility'] < 3).astype(int).values

# Storm indicator (low pressure + rain)
reg_enriched['storm_indicator'] = ((df['pressure'] < 1005) & (df['is_rainy'] == 1)).astype(int).values
clf_enriched['storm_indicator'] = ((df['pressure'] < 1005) & (df['is_rainy'] == 1)).astype(int).values

# High wind gust
reg_enriched['high_gust'] = (df['windGust'] > 15).astype(int).values
clf_enriched['high_gust'] = (df['windGust'] > 15).astype(int).values

# Rain probability × rush hour (likely demand spike)
reg_enriched['precip_rush'] = (df['precipProbability'] * df['is_rush_hour']).values
clf_enriched['precip_rush'] = (df['precipProbability'] * df['is_rush_hour']).values

# Low visibility × distance (dangerous long rides)
reg_enriched['visibility_distance'] = (df['visibility'] * df['distance']).values
clf_enriched['visibility_distance'] = (df['visibility'] * df['distance']).values

# Severe composite
reg_enriched['severe_composite'] = (
    (df['visibility'] < 3).astype(int) + 
    (df['pressure'] < 1005).astype(int) + 
    (df['windGust'] > 20).astype(int) +
    (df['precipProbability'] > 0.5).astype(int)
).values
clf_enriched['severe_composite'] = reg_enriched['severe_composite'].values

print(f"\nRegression enriched:     {reg_enriched.shape[0]:,} × {reg_enriched.shape[1]-1} features")
print(f"Classification enriched: {clf_enriched.shape[0]:,} × {clf_enriched.shape[1]-1} features")
print(f"New features: {reg_enriched.shape[1] - reg_base.shape[1]} added")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: RUN 5 REGRESSION + 5 CLASSIFICATION MODELS (BASE & ENRICHED)
# ══════════════════════════════════════════════════════════════════════════════

def run_regression(df_data, label):
    X = df_data.drop('price', axis=1)
    y = df_data['price']
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
    m = Lasso(alpha=0.1, random_state=42, max_iter=10000).fit(Xtrs, ytr)
    pred = m.predict(Xtes)
    results['Lasso'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}
    
    # 4. Decision Tree
    m = DecisionTreeRegressor(max_depth=10, random_state=42).fit(Xtr, ytr)
    pred = m.predict(Xte)
    results['Decision Tree'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}
    
    # 5. Polynomial (top 5 Ridge features)
    top5 = [feat_names[i] for i in np.argsort(np.abs(ridge_coef))[-5:][::-1]]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xtrp = poly.fit_transform(Xtr[top5]); Xtep = poly.transform(Xte[top5])
    m = Ridge(alpha=1.0, random_state=42).fit(Xtrp, ytr)
    pred = m.predict(Xtep)
    results['Polynomial'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}
    
    return results

def run_classification(df_data, label):
    X = df_data.drop('is_premium', axis=1)
    y = df_data['is_premium']
    
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
    
    for name, model, use_scaled in models:
        Xtr_use = Xtrs if use_scaled else Xtr
        Xte_use = Xtes if use_scaled else Xte
        model.fit(Xtr_use, ytr)
        pred = model.predict(Xte_use)
        results[name] = {
            'F1': f1_score(yte, pred),
            'Acc': accuracy_score(yte, pred),
            'Prec': precision_score(yte, pred, zero_division=0),
            'Rec': recall_score(yte, pred),
        }
    
    return results

# Run all
print("\n" + "=" * 80)
print("STEP 3: MODEL RESULTS")
print("=" * 80)

print("\n--- Running regression (base) ---")
reg_base_results = run_regression(reg_base, "Base")
print("--- Running regression (enriched) ---")
reg_enr_results = run_regression(reg_enriched, "Enriched")
print("--- Running classification (base) ---")
clf_base_results = run_classification(clf_base, "Base")
print("--- Running classification (enriched) ---")
clf_enr_results = run_classification(clf_enriched, "Enriched")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: RESULTS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("REGRESSION — Price Prediction (no service name in features)")
print("=" * 80)
print(f"\n{'Model':<18} {'Base R²':>8} {'Enr R²':>8} {'Δ R²':>8} {'Base MAE':>10} {'Enr MAE':>10} {'Δ MAE':>8}")
print("-" * 72)
for name in ['OLS', 'Ridge', 'Lasso', 'Decision Tree', 'Polynomial']:
    b = reg_base_results[name]; e = reg_enr_results[name]
    dr2 = e['R2'] - b['R2']; dmae = e['MAE'] - b['MAE']
    icon = "✅" if dr2 > 0.001 else "➖" if abs(dr2) < 0.001 else "❌"
    print(f"{icon} {name:<16} {b['R2']:>8.4f} {e['R2']:>8.4f} {dr2:>+8.4f} {b['MAE']:>9.2f} {e['MAE']:>9.2f} {dmae:>+7.2f}")

avg_b = np.mean([v['R2'] for v in reg_base_results.values()])
avg_e = np.mean([v['R2'] for v in reg_enr_results.values()])
print(f"\n   Average R²: {avg_b:.4f} → {avg_e:.4f} ({(avg_e-avg_b)/abs(avg_b)*100:+.1f}%)")

print("\n" + "=" * 80)
print("CLASSIFICATION — Premium Vehicle Prediction")
print("=" * 80)
print(f"\n{'Model':<18} {'Base F1':>8} {'Enr F1':>8} {'Δ F1':>8} {'Base Acc':>10} {'Enr Acc':>10}")
print("-" * 62)
for name in ['Logistic Reg', 'Decision Tree', 'Naive Bayes', 'LDA', 'Perceptron']:
    b = clf_base_results[name]; e = clf_enr_results[name]
    df1 = e['F1'] - b['F1']
    icon = "✅" if df1 > 0.001 else "➖" if abs(df1) < 0.001 else "❌"
    print(f"{icon} {name:<16} {b['F1']:>8.4f} {e['F1']:>8.4f} {df1:>+8.4f} {b['Acc']:>9.4f} {e['Acc']:>9.4f}")

avg_b = np.mean([v['F1'] for v in clf_base_results.values()])
avg_e = np.mean([v['F1'] for v in clf_enr_results.values()])
print(f"\n   Average F1: {avg_b:.4f} → {avg_e:.4f} ({(avg_e-avg_b)/abs(avg_b)*100:+.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON WITH ORIGINAL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("COMPARISON: THIS PIPELINE vs ORIGINAL")
print("=" * 80)

print("""
ORIGINAL PIPELINE (with name in regression, leakage-fixed enrichment):
  Regression:     Best R² = 0.9639 (DT), avg = 0.7755, enrichment = +0.0%
  Classification: Best F1 = 0.6892 (LogReg), avg = 0.6488, enrichment = +0.1%

THIS PIPELINE (no name in regression, real raw column enrichment):""")

best_reg = max(reg_enr_results.items(), key=lambda x: x[1]['R2'])
best_clf = max(clf_enr_results.items(), key=lambda x: x[1]['F1'])
avg_reg = np.mean([v['R2'] for v in reg_enr_results.values()])
avg_clf = np.mean([v['F1'] for v in clf_enr_results.values()])
reg_imp = np.mean([reg_enr_results[n]['R2'] - reg_base_results[n]['R2'] for n in reg_base_results])
clf_imp = np.mean([clf_enr_results[n]['F1'] - clf_base_results[n]['F1'] for n in clf_base_results])

print(f"  Regression:     Best R² = {best_reg[1]['R2']:.4f} ({best_reg[0]}), avg = {avg_reg:.4f}, enrichment = {reg_imp:+.4f}")
print(f"  Classification: Best F1 = {best_clf[1]['F1']:.4f} ({best_clf[0]}), avg = {avg_clf:.4f}, enrichment = {clf_imp:+.4f}")

print(f"""
                  Orig         New          Better?
  Reg best R²:    0.9639       {best_reg[1]['R2']:.4f}        {'✅ YES' if best_reg[1]['R2'] > 0.9639 else '❌ NO' if best_reg[1]['R2'] < 0.95 else '➖ ~SAME'}
  Reg avg R²:     0.7755       {avg_reg:.4f}        {'✅ YES' if avg_reg > 0.7755 else '❌ NO'}
  Reg enrichment: +0.0%        {reg_imp:+.4f}       {'✅ YES' if reg_imp > 0.001 else '❌ NO'}
  Clf best F1:    0.6892       {best_clf[1]['F1']:.4f}        {'✅ YES' if best_clf[1]['F1'] > 0.6892 else '❌ NO'}
  Clf avg F1:     0.6488       {avg_clf:.4f}        {'✅ YES' if avg_clf > 0.6488 else '❌ NO'}
  Clf enrichment: +0.1%        {clf_imp:+.4f}       {'✅ YES' if clf_imp > 0.001 else '❌ NO'}
""")

improved = sum([
    best_reg[1]['R2'] > 0.9639,
    avg_reg > 0.7755,
    reg_imp > 0.001,
    best_clf[1]['F1'] > 0.6892,
    avg_clf > 0.6488,
    clf_imp > 0.001
])
print(f"VERDICT: New pipeline is better on {improved}/6 metrics")
