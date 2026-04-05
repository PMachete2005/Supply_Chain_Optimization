"""
HYBRID PIPELINE — Best of both worlds
Regression: WITH name (legitimate feature) + real raw column enrichment
Classification: WITHOUT name (leakage) + real raw column enrichment

Enrichment = 7 real unused raw columns + 7 derived interactions
All honest, zero leakage.
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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("STEP 1: LOAD, CLEAN, SAMPLE")
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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING — BASE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 80)

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

for col in ['cab_type', 'name', 'source', 'destination']:
    df[col + '_enc'] = LabelEncoder().fit_transform(df[col].astype(str))

# BASE feature lists
reg_base_feats = [
    'distance', 'surge_multiplier', 'distance_surge',
    'hour', 'day_of_week', 'is_weekend',
    'is_morning_rush', 'is_evening_rush', 'is_rush_hour', 'is_night',
    'temperature', 'humidity', 'windSpeed', 'precipIntensity',
    'is_rainy', 'is_cold', 'weather_severity',
    'cab_type_enc', 'name_enc', 'is_premium',  # name is legitimate for regression
    'source_enc', 'destination_enc',
    'short_ride', 'medium_ride', 'long_ride'
]

clf_base_feats = [
    'distance',
    'hour', 'day_of_week', 'is_weekend',
    'is_morning_rush', 'is_evening_rush', 'is_rush_hour', 'is_night',
    'temperature', 'humidity', 'windSpeed', 'precipIntensity',
    'is_rainy', 'is_cold', 'weather_severity',
    'cab_type_enc',
    'source_enc', 'destination_enc',
    'short_ride', 'medium_ride', 'long_ride'
]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: ENRICHMENT — Real unused raw columns
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STEP 3: ENRICHMENT — 7 raw columns + 7 derived features")
print("=" * 80)

# 7 real raw columns not used in base
raw_enrichment = ['apparentTemperature', 'precipProbability', 'visibility',
                  'pressure', 'windGust', 'cloudCover', 'dewPoint']

# 7 derived interaction features (no leakage)
df['wind_chill'] = df['apparentTemperature'] - df['temperature']
df['low_visibility'] = (df['visibility'] < 3).astype(int)
df['storm_indicator'] = ((df['pressure'] < 1005) & (df['is_rainy'] == 1)).astype(int)
df['high_gust'] = (df['windGust'] > 15).astype(int)
df['precip_rush'] = df['precipProbability'] * df['is_rush_hour']
df['visibility_distance'] = df['visibility'] * df['distance']
df['severe_composite'] = (
    df['low_visibility'].astype(int) +
    (df['pressure'] < 1005).astype(int) +
    (df['windGust'] > 20).astype(int) +
    (df['precipProbability'] > 0.5).astype(int)
)

derived_enrichment = ['wind_chill', 'low_visibility', 'storm_indicator',
                      'high_gust', 'precip_rush', 'visibility_distance', 'severe_composite']

all_enrichment = raw_enrichment + derived_enrichment

reg_enriched_feats = reg_base_feats + all_enrichment
clf_enriched_feats = clf_base_feats + all_enrichment

print(f"Regression:     {len(reg_base_feats)} base → {len(reg_enriched_feats)} enriched (+{len(all_enrichment)})")
print(f"Classification: {len(clf_base_feats)} base → {len(clf_enriched_feats)} enriched (+{len(all_enrichment)})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: RUN ALL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def run_regression(feature_list, label):
    X = df[feature_list]; y = df['price']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); Xtrs = sc.fit_transform(Xtr); Xtes = sc.transform(Xte)
    feat_names = feature_list
    results = {}

    # 1. OLS
    ols = sm.OLS(ytr, sm.add_constant(Xtrs)).fit()
    pred = ols.predict(sm.add_constant(Xtes))
    results['OLS'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}

    # 2. Ridge
    m = Ridge(alpha=1.0, random_state=42).fit(Xtrs, ytr)
    pred = m.predict(Xtes)
    results['Ridge'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}
    ridge_coef = m.coef_; ridge_feats = feat_names

    # 3. Lasso
    m = Lasso(alpha=0.1, random_state=42, max_iter=10000).fit(Xtrs, ytr)
    pred = m.predict(Xtes)
    results['Lasso'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}
    lasso_selected = sum(m.coef_ != 0)

    # 4. Decision Tree
    m = DecisionTreeRegressor(max_depth=10, random_state=42).fit(Xtr, ytr)
    pred = m.predict(Xte)
    results['Decision Tree'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}
    dt_imp = dict(zip(feat_names, m.feature_importances_))

    # 5. Polynomial (top 5 from Ridge)
    top5_idx = np.argsort(np.abs(ridge_coef))[-5:][::-1]
    top5 = [ridge_feats[i] for i in top5_idx]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xtrp = poly.fit_transform(Xtr[top5]); Xtep = poly.transform(Xte[top5])
    m = Ridge(alpha=1.0, random_state=42).fit(Xtrp, ytr)
    pred = m.predict(Xtep)
    results['Polynomial'] = {'R2': r2_score(yte, pred), 'MAE': mean_absolute_error(yte, pred)}

    return results, ridge_coef, ridge_feats, dt_imp, lasso_selected, top5

def run_classification(feature_list, label):
    X = df[feature_list]; y = df['is_premium']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler(); Xtrs = sc.fit_transform(Xtr); Xtes = sc.transform(Xte)
    feat_names = feature_list
    results = {}; details = {}

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
        if hasattr(model, 'feature_importances_'):
            details[name] = dict(zip(feat_names, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            details[name] = dict(zip(feat_names, coefs))

    return results, details

print("\n" + "=" * 80)
print("STEP 4: RUNNING ALL MODELS")
print("=" * 80)

print("\n[1/4] Regression BASE...")
reg_base_res, rb_coef, rb_feats, rb_dt, rb_lasso, rb_top5 = run_regression(reg_base_feats, "Base")
print("[2/4] Regression ENRICHED...")
reg_enr_res, re_coef, re_feats, re_dt, re_lasso, re_top5 = run_regression(reg_enriched_feats, "Enriched")
print("[3/4] Classification BASE...")
clf_base_res, cb_details = run_classification(clf_base_feats, "Base")
print("[4/4] Classification ENRICHED...")
clf_enr_res, ce_details = run_classification(clf_enriched_feats, "Enriched")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: FULL RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("REGRESSION RESULTS — Price Prediction (with name, no leakage)")
print("=" * 80)

print(f"\n{'Model':<18} {'Base R²':>8} {'Enr R²':>8} {'Δ R²':>8} {'Imp%':>7} {'Base MAE':>10} {'Enr MAE':>10}")
print("-" * 73)
reg_improved = 0
for name in ['OLS', 'Ridge', 'Lasso', 'Decision Tree', 'Polynomial']:
    b = reg_base_res[name]; e = reg_enr_res[name]
    dr2 = e['R2'] - b['R2']
    pct = dr2/abs(b['R2'])*100 if b['R2'] != 0 else 0
    if dr2 > 0.001: reg_improved += 1
    icon = "✅" if dr2 > 0.001 else "➖" if abs(dr2) < 0.001 else "❌"
    print(f"{icon} {name:<16} {b['R2']:>8.4f} {e['R2']:>8.4f} {dr2:>+8.4f} {pct:>+6.1f}% ${b['MAE']:>8.2f} ${e['MAE']:>8.2f}")

avg_b_r2 = np.mean([v['R2'] for v in reg_base_res.values()])
avg_e_r2 = np.mean([v['R2'] for v in reg_enr_res.values()])
avg_b_mae = np.mean([v['MAE'] for v in reg_base_res.values()])
avg_e_mae = np.mean([v['MAE'] for v in reg_enr_res.values()])
print(f"\n   Average R²:  {avg_b_r2:.4f} → {avg_e_r2:.4f} ({(avg_e_r2-avg_b_r2)/abs(avg_b_r2)*100:+.2f}%)")
print(f"   Average MAE: ${avg_b_mae:.2f} → ${avg_e_mae:.2f} (${avg_e_mae-avg_b_mae:+.2f})")
print(f"   Models improved: {reg_improved}/5")

# Top features
print(f"\n   Top 5 Enriched Ridge coefficients:")
for i in np.argsort(np.abs(re_coef))[-5:][::-1]:
    marker = " [NEW]" if re_feats[i] in all_enrichment else ""
    print(f"     {re_feats[i]:<30} {re_coef[i]:>+8.4f}{marker}")

print(f"\n   Top 5 Enriched DT importances:")
for feat, imp in sorted(re_dt.items(), key=lambda x: x[1], reverse=True)[:5]:
    marker = " [NEW]" if feat in all_enrichment else ""
    print(f"     {feat:<30} {imp:.4f}{marker}")

print(f"\n   Lasso selected: {rb_lasso} base → {re_lasso} enriched features")

print("\n" + "=" * 80)
print("CLASSIFICATION RESULTS — Premium Vehicle Prediction (no name, no leakage)")
print("=" * 80)

print(f"\n{'Model':<18} {'Base F1':>8} {'Enr F1':>8} {'Δ F1':>8} {'Imp%':>7} {'Base Acc':>10} {'Enr Acc':>10}")
print("-" * 73)
clf_improved = 0
for name in ['Logistic Reg', 'Decision Tree', 'Naive Bayes', 'LDA', 'Perceptron']:
    b = clf_base_res[name]; e = clf_enr_res[name]
    df1 = e['F1'] - b['F1']
    pct = df1/abs(b['F1'])*100 if b['F1'] != 0 else 0
    if df1 > 0.001: clf_improved += 1
    icon = "✅" if df1 > 0.001 else "➖" if abs(df1) < 0.001 else "❌"
    print(f"{icon} {name:<16} {b['F1']:>8.4f} {e['F1']:>8.4f} {df1:>+8.4f} {pct:>+6.1f}%    {b['Acc']:>8.4f}    {e['Acc']:>8.4f}")

avg_b_f1 = np.mean([v['F1'] for v in clf_base_res.values()])
avg_e_f1 = np.mean([v['F1'] for v in clf_enr_res.values()])
print(f"\n   Average F1:  {avg_b_f1:.4f} → {avg_e_f1:.4f} ({(avg_e_f1-avg_b_f1)/abs(avg_b_f1)*100:+.2f}%)")
print(f"   Models improved: {clf_improved}/5")

# Top features
if 'Decision Tree' in ce_details:
    print(f"\n   Top 5 Enriched DT importances:")
    for feat, imp in sorted(ce_details['Decision Tree'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        marker = " [NEW]" if feat in all_enrichment else ""
        print(f"     {feat:<30} {imp:.4f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON: HYBRID vs ORIGINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL VERDICT: HYBRID vs ORIGINAL PIPELINE")
print("=" * 80)

best_reg = max(reg_enr_res.items(), key=lambda x: x[1]['R2'])
best_clf = max(clf_enr_res.items(), key=lambda x: x[1]['F1'])
reg_enrich_delta = np.mean([reg_enr_res[n]['R2'] - reg_base_res[n]['R2'] for n in reg_base_res])
clf_enrich_delta = np.mean([clf_enr_res[n]['F1'] - clf_base_res[n]['F1'] for n in clf_base_res])

print(f"""
ORIGINAL PIPELINE (leakage-fixed, synthetic enrichment):
  Reg: Best R²=0.9639 (DT), Avg R²=0.7755, Enrichment=+0.0%, MAE=$1.13 best
  Clf: Best F1=0.6892 (LR), Avg F1=0.6488, Enrichment=+0.1%

HYBRID PIPELINE (name kept for reg, real raw column enrichment):
  Reg: Best R²={best_reg[1]['R2']:.4f} ({best_reg[0]}), Avg R²={avg_e_r2:.4f}, Enrichment={reg_enrich_delta:+.4f}, MAE=${min(v['MAE'] for v in reg_enr_res.values()):.2f} best
  Clf: Best F1={best_clf[1]['F1']:.4f} ({best_clf[0]}), Avg F1={avg_e_f1:.4f}, Enrichment={clf_enrich_delta:+.4f}
""")

metrics = [
    ("Reg best R²",    best_reg[1]['R2'], 0.9639),
    ("Reg avg R²",     avg_e_r2, 0.7755),
    ("Reg enrichment", reg_enrich_delta, 0.0000),
    ("Clf best F1",    best_clf[1]['F1'], 0.6892),
    ("Clf avg F1",     avg_e_f1, 0.6488),
    ("Clf enrichment", clf_enrich_delta, 0.001),
]

wins = 0
print(f"  {'Metric':<20} {'Original':>10} {'Hybrid':>10} {'Winner':>10}")
print("  " + "-" * 52)
for name, hybrid, orig in metrics:
    better = hybrid > orig + 0.0005
    winner = "HYBRID ✅" if better else "ORIG" if orig > hybrid + 0.0005 else "TIE"
    if better: wins += 1
    print(f"  {name:<20} {orig:>10.4f} {hybrid:>10.4f} {winner:>10}")

print(f"\n  HYBRID wins: {wins}/6 metrics")
if wins >= 4:
    print("  🏆 HYBRID is clearly better — recommend switching")
elif wins >= 3:
    print("  📊 HYBRID is marginally better — worth switching if enrichment matters")
else:
    print("  📊 ORIGINAL is comparable or better — enrichment story is key differentiator")
