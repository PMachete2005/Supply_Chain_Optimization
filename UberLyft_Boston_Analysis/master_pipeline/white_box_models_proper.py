"""
Uber/Lyft Boston - White Box Models (Proper Pipeline Version)
Follows complete DATA_PIPELINE.md preprocessing steps
5 Regression + 5 Classification models with visible equations
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import statsmodels.api as sm

warnings.filterwarnings('ignore')
np.random.seed(42)

print("="*80)
print("UBER/LYFT BOSTON - WHITE BOX MODELS (FULL PIPELINE)")
print("="*80)

# ============================================
# STEP 1: LOAD RAW DATA
# ============================================
print("\n" + "="*80)
print("STEP 1: LOADING RAW DATA")
print("="*80)

df = pd.read_csv('/home/kushagarwal/Downloads/archive(1)/rideshare_kaggle.csv')
print(f"✓ Loaded: {len(df):,} records, {len(df.columns)} columns")

# ============================================
# STEP 2: DATA CLEANING
# ============================================
print("\n" + "="*80)
print("STEP 2: DATA CLEANING")
print("="*80)

# Remove records with missing prices
null_prices = df['price'].isna().sum()
df_clean = df.dropna(subset=['price'])
print(f"✓ Removed {null_prices:,} records with null prices ({null_prices/len(df)*100:.2f}%)")

# Remove invalid prices (price <= 0)
invalid_prices = (df_clean['price'] <= 0).sum()
df_clean = df_clean[df_clean['price'] > 0]
print(f"✓ Removed {invalid_prices:,} records with invalid prices")

# Remove records with missing critical features
df_clean = df_clean.dropna(subset=['distance', 'surge_multiplier'])

print(f"✓ Clean dataset: {len(df_clean):,} records remaining")

# ============================================
# STEP 3: STRATIFIED SAMPLING (100K)
# ============================================
print("\n" + "="*80)
print("STEP 3: STRATIFIED SAMPLING (100,000 RECORDS)")
print("="*80)

# Create price categories for stratification
df_clean['price_category'] = pd.cut(df_clean['price'], bins=[0, 10, 15, 25, 1000], labels=['low', 'mid', 'high', 'premium'])

# Create strata: cab_type + name + price_category
df_clean['strata'] = df_clean['cab_type'].astype(str) + '_' + df_clean['name'].astype(str) + '_' + df_clean['price_category'].astype(str)

# Stratified sample 100,000 records
# Use simple stratified approach
strata_counts = df_clean['strata'].value_counts()
min_strata_size = 10  # Minimum records per stratum

# Keep only strata with sufficient records
valid_strata = strata_counts[strata_counts >= min_strata_size].index
df_valid = df_clean[df_clean['strata'].isin(valid_strata)]

# Sample proportionally
sample_size = min(100000, len(df_valid))
df_sample = df_valid.groupby('strata').apply(
    lambda x: x.sample(n=max(1, int(len(x) / len(df_valid) * sample_size)), random_state=42)
).reset_index(drop=True)

# If we got fewer than 100K, sample more from the full dataset
if len(df_sample) < 100000:
    remaining = 100000 - len(df_sample)
    additional = df_valid.drop(df_sample.index, errors='ignore').sample(n=min(remaining, len(df_valid) - len(df_sample)), random_state=42)
    df_sample = pd.concat([df_sample, additional]).reset_index(drop=True)

print(f"✓ Stratified sample: {len(df_sample):,} records")
print(f"  Uber/Lyft split: {(df_sample['cab_type']=='Uber').mean()*100:.1f}% / {(df_sample['cab_type']=='Lyft').mean()*100:.1f}%")

# ============================================
# STEP 4: FEATURE ENGINEERING (25+ NEW FEATURES)
# ============================================
print("\n" + "="*80)
print("STEP 4: FEATURE ENGINEERING")
print("="*80)

df = df_sample.copy()

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['week_of_year'] = df['timestamp'].dt.isocalendar().week
df['quarter'] = df['timestamp'].dt.quarter

# Time-based features
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9) & (df['is_weekend'] == 0)).astype(int)
df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19) & (df['is_weekend'] == 0)).astype(int)
df['is_rush_hour'] = df['is_morning_rush'] | df['is_evening_rush']
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
df['is_late_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['is_weekend'] == 0)).astype(int)

# Distance-based features
df['short_ride'] = (df['distance'] < 2).astype(int)
df['medium_ride'] = ((df['distance'] >= 2) & (df['distance'] < 5)).astype(int)
df['long_ride'] = (df['distance'] >= 5).astype(int)

# Weather features
df['is_rainy'] = (df['precipIntensity'] > 0).astype(int) if 'precipIntensity' in df.columns else (df['rain'] > 0).astype(int) if 'rain' in df.columns else 0
df['is_cold'] = (df['temperature'] < 40).astype(int) if 'temperature' in df.columns else 0
df['is_high_humidity'] = (df['humidity'] > 0.7).astype(int) if 'humidity' in df.columns else 0
df['is_high_wind'] = (df['windSpeed'] > 15).astype(int) if 'windSpeed' in df.columns else 0
df['weather_severity'] = df['is_rainy'] + df['is_cold'] + df['is_high_humidity']

# Interaction features (KEY from Boston analysis)
df['distance_surge'] = df['distance'] * df['surge_multiplier']
df['weather_surge'] = df['is_rainy'] * df['surge_multiplier']
df['rush_surge'] = df['is_rush_hour'] * df['surge_multiplier']
df['distance_weather'] = df['distance'] * df['is_rainy']
df['distance_rush'] = df['distance'] * df['is_rush_hour']

# Vehicle features
df['is_premium'] = df['name'].str.contains('Black|Lux|SUV|XL', case=False, na=False).astype(int)
df['uber_premium'] = ((df['cab_type'] == 'Uber') & (df['is_premium'] == 1)).astype(int)
df['lyft_premium'] = ((df['cab_type'] == 'Lyft') & (df['is_premium'] == 1)).astype(int)

print(f"✓ Features after engineering: {df.shape[1]} columns")
feature_count = df.shape[1] - 57  # Assuming 57 original columns
print(f"✓ New features added: ~{feature_count}")

# ============================================
# STEP 5: ENCODING
# ============================================
print("\n" + "="*80)
print("STEP 5: ENCODING CATEGORICAL VARIABLES")
print("="*80)

cat_cols = ['cab_type', 'name', 'source', 'destination']
encoders = {}

for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"✓ Encoded {col}: {len(le.classes_)} categories")

# ============================================
# STEP 6 & 7: FEATURE SELECTION & TARGET PREPARATION
# ============================================
print("\n" + "="*80)
print("STEP 6 & 7: FEATURE SELECTION & TARGET PREPARATION")
print("="*80)

# REGRESSION FEATURES (21 features - include surge)
reg_feature_cols = [
    'distance', 'surge_multiplier', 'hour', 'day_of_week', 'is_weekend',
    'is_morning_rush', 'is_evening_rush', 'is_rush_hour', 'is_night',
    'temperature', 'humidity', 'windSpeed', 'is_rainy', 'is_cold', 'weather_severity',
    'cab_type', 'name', 'is_premium', 'source', 'destination',
    'distance_surge', 'rush_surge', 'short_ride', 'medium_ride', 'long_ride'
]

# Only use columns that exist in the dataset
reg_feature_cols = [col for col in reg_feature_cols if col in df.columns]

# CLASSIFICATION FEATURES (18 features - EXCLUDE surge to prevent leakage)
clf_feature_cols = [col for col in reg_feature_cols if 'surge' not in col]

print(f"✓ Regression features: {len(reg_feature_cols)} (includes surge)")
print(f"✓ Classification features: {len(clf_feature_cols)} (excludes surge)")

# Targets
y_price = df['price']
y_surge = (df['surge_multiplier'] > 1.0).astype(int)

print(f"✓ Target (Regression): price ${y_price.min():.2f} - ${y_price.max():.2f}")
print(f"✓ Target (Classification): surge rate = {y_surge.mean()*100:.1f}%")

X_reg = df[reg_feature_cols]
X_clf = df[clf_feature_cols]

# ============================================
# STEP 8: TRAIN/TEST SPLIT
# ============================================
print("\n" + "="*80)
print("STEP 8: TRAIN/TEST SPLIT (80/20)")
print("="*80)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_price, test_size=0.2, random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_surge, test_size=0.2, random_state=42, stratify=y_surge
)

print(f"✓ Regression split: {len(X_train_r):,} train / {len(X_test_r):,} test")
print(f"✓ Classification split: {len(X_train_c):,} train / {len(X_test_c):,} test")

# ============================================
# WHITE BOX MODELS - REGRESSION
# ============================================
print("\n" + "="*80)
print("WHITE BOX REGRESSION MODELS")
print("="*80)

# Feature names for display
reg_feature_names = [col.replace('_', ' ') for col in reg_feature_cols]

# MODEL 1: OLS Linear Regression
print("\n" + "-"*80)
print("MODEL 1: ORDINARY LEAST SQUARES (OLS)")
print("-"*80)

ols_model = sm.OLS(y_train_r, sm.add_constant(X_train_r)).fit()
ols_preds = ols_model.predict(sm.add_constant(X_test_r))

print(f"R² Score: {ols_model.rsquared:.4f}")
print(f"MAE: ${mean_absolute_error(y_test_r, ols_preds):.2f}")

print("\n📐 EQUATION (top 10 coefficients):")
coefs = ols_model.params.copy()
const_val = coefs.get('const', 0)
if 'const' in coefs.index:
    coefs = coefs.drop('const')

top_coefs = coefs.abs().nlargest(10)

print(f"price = {const_val:.4f}", end="")
for feat in top_coefs.index:
    coef_val = ols_model.params[feat]
    sign = " + " if coef_val >= 0 else " - "
    print(f"{sign}{abs(coef_val):.4f}×{feat}", end="")
print()

print("\n📊 SIGNIFICANT FEATURES (p < 0.05):")
sig_features = ols_model.pvalues[ols_model.pvalues < 0.05].drop('const', errors='ignore')
for feat, pval in sig_features.head(10).items():
    print(f"   {feat}: p={pval:.4f}, coef={coefs[feat]:.4f}")

# MODEL 2: Ridge Regression
print("\n" + "-"*80)
print("MODEL 2: RIDGE REGRESSION (L2 Regularization)")
print("-"*80)

ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_r, y_train_r)
ridge_preds = ridge_model.predict(X_test_r)

print(f"R² Score: {r2_score(y_test_r, ridge_preds):.4f}")
print(f"MAE: ${mean_absolute_error(y_test_r, ridge_preds):.2f}")

print("\n📐 EQUATION:")
print(f"price = {ridge_model.intercept_:.4f}", end="")
for coef, feat in zip(ridge_model.coef_, reg_feature_cols):
    sign = " + " if coef >= 0 else " - "
    print(f"{sign}{abs(coef):.4f}×{feat}", end="")
print()

print("\n📊 TOP 5 FEATURES (by |coefficient|):")
importance = list(zip(reg_feature_cols, np.abs(ridge_model.coef_)))
importance.sort(key=lambda x: x[1], reverse=True)
for feat, imp in importance[:5]:
    print(f"   {feat}: {imp:.4f}")

# MODEL 3: Lasso Regression
print("\n" + "-"*80)
print("MODEL 3: LASSO REGRESSION (L1 - Feature Selection)")
print("-"*80)

lasso_model = Lasso(alpha=0.1, random_state=42, max_iter=10000)
lasso_model.fit(X_train_r, y_train_r)
lasso_preds = lasso_model.predict(X_test_r)

print(f"R² Score: {r2_score(y_test_r, lasso_preds):.4f}")
print(f"MAE: ${mean_absolute_error(y_test_r, lasso_preds):.2f}")

print("\n📐 SPARSE EQUATION:")
print(f"price = {lasso_model.intercept_:.4f}", end="")
selected_features = []
for coef, feat in zip(lasso_model.coef_, reg_feature_cols):
    if abs(coef) > 0.001:
        sign = " + " if coef >= 0 else " - "
        print(f"{sign}{abs(coef):.4f}×{feat}", end="")
        selected_features.append(feat)
print()

print(f"\n📊 SELECTED {len(selected_features)}/{len(reg_feature_cols)} FEATURES:")
print(f"   {', '.join(selected_features[:8])}")

# MODEL 4: Decision Tree (Depth 3)
print("\n" + "-"*80)
print("MODEL 4: DECISION TREE (Max Depth 3 - Human Readable)")
print("-"*80)

tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_train_r, y_train_r)
tree_preds = tree_reg.predict(X_test_r)

print(f"R² Score: {r2_score(y_test_r, tree_preds):.4f}")
print(f"MAE: ${mean_absolute_error(y_test_r, tree_preds):.2f}")

print("\n📋 DECISION RULES:")
tree_rules = export_text(tree_reg, feature_names=reg_feature_cols)
print(tree_rules)

print("\n📊 FEATURE IMPORTANCE:")
importance_tree = list(zip(reg_feature_cols, tree_reg.feature_importances_))
importance_tree.sort(key=lambda x: x[1], reverse=True)
for feat, imp in importance_tree[:5]:
    if imp > 0:
        print(f"   {feat}: {imp:.4f}")

# MODEL 5: Polynomial (Degree 2)
print("\n" + "-"*80)
print("MODEL 5: POLYNOMIAL REGRESSION (Degree 2)")
print("-"*80)

# Use subset of features for polynomial to avoid explosion
poly_features = ['distance', 'surge_multiplier', 'temperature', 'is_premium']
poly_features = [f for f in poly_features if f in reg_feature_cols]

X_train_poly_sub = X_train_r[poly_features]
X_test_poly_sub = X_test_r[poly_features]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_poly_sub)
X_test_poly = poly.transform(X_test_poly_sub)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_r)
poly_preds = poly_model.predict(X_test_poly)

print(f"R² Score: {r2_score(y_test_r, poly_preds):.4f}")
print(f"MAE: ${mean_absolute_error(y_test_r, poly_preds):.2f}")
print(f"Features: {len(poly_features)} → {X_train_poly.shape[1]}")

print("\n📐 TOP 8 POLYNOMIAL TERMS:")
poly_names = poly.get_feature_names_out(input_features=poly_features)
coefs = poly_model.coef_
top_idx = np.argsort(np.abs(coefs))[-8:][::-1]

print(f"price = {poly_model.intercept_:.4f}", end="")
for idx in top_idx:
    coef = coefs[idx]
    name = poly_names[idx]
    sign = " + " if coef >= 0 else " - "
    print(f"{sign}{abs(coef):.4f}×{name}", end="")
print()

# ============================================
# WHITE BOX MODELS - CLASSIFICATION
# ============================================
print("\n" + "="*80)
print("WHITE BOX CLASSIFICATION MODELS (No Surge Features)")
print("="*80)

# Feature names for classification
clf_feature_names = [col.replace('_', ' ') for col in clf_feature_cols]

print(f"Using {len(clf_feature_cols)} features (excludes surge to prevent leakage)")

# MODEL 1: Logistic Regression
print("\n" + "-"*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("-"*80)

log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_c, y_train_c)
log_preds = log_reg.predict(X_test_c)

print(f"Accuracy: {accuracy_score(y_test_c, log_preds):.4f}")
print(f"Precision: {precision_score(y_test_c, log_preds):.4f}")
print(f"Recall: {recall_score(y_test_c, log_preds):.4f}")
print(f"F1 Score: {f1_score(y_test_c, log_preds):.4f}")

print("\n📐 LOG-ODDS EQUATION:")
print(f"log(P/1-P) = {log_reg.intercept_[0]:.4f}", end="")
for coef, feat in zip(log_reg.coef_[0], clf_feature_cols):
    sign = " + " if coef >= 0 else " - "
    print(f"{sign}{abs(coef):.4f}×{feat}", end="")
print()

print("\n📊 ODDS RATIOS (impact on surge probability):")
odds = np.exp(log_reg.coef_[0])
or_list = list(zip(clf_feature_cols, odds))
or_list.sort(key=lambda x: abs(x[1] - 1), reverse=True)
for feat, or_val in or_list[:8]:
    direction = "↑ increases" if or_val > 1 else "↓ decreases"
    print(f"   {feat}: OR={or_val:.4f} ({direction})")

# MODEL 2: Decision Tree Classifier
print("\n" + "-"*80)
print("MODEL 2: DECISION TREE CLASSIFIER (Depth 3)")
print("-"*80)

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
tree_clf.fit(X_train_c, y_train_c)
tree_clf_preds = tree_clf.predict(X_test_c)

print(f"Accuracy: {accuracy_score(y_test_c, tree_clf_preds):.4f}")
print(f"Precision: {precision_score(y_test_c, tree_clf_preds):.4f}")
print(f"Recall: {recall_score(y_test_c, tree_clf_preds):.4f}")
print(f"F1 Score: {f1_score(y_test_c, tree_clf_preds):.4f}")

print("\n📋 DECISION RULES:")
clf_rules = export_text(tree_clf, feature_names=clf_feature_cols)
print(clf_rules)

# MODEL 3: Naive Bayes
print("\n" + "-"*80)
print("MODEL 3: GAUSSIAN NAIVE BAYES")
print("-"*80)

nb_model = GaussianNB()
nb_model.fit(X_train_c, y_train_c)
nb_preds = nb_model.predict(X_test_c)

print(f"Accuracy: {accuracy_score(y_test_c, nb_preds):.4f}")
print(f"Precision: {precision_score(y_test_c, nb_preds):.4f}")
print(f"Recall: {recall_score(y_test_c, nb_preds):.4f}")
print(f"F1 Score: {f1_score(y_test_c, nb_preds):.4f}")

print("\n📐 BAYESIAN RULE:")
print("P(surge | x) ∝ P(x | surge) × P(surge)")
print(f"\n📊 CLASS PRIORS: P(surge)={nb_model.class_prior_[1]:.4f}, P(no surge)={nb_model.class_prior_[0]:.4f}")

print("\n📊 FEATURE STATISTICS (mean ± std):")
for i, class_name in enumerate(['No Surge', 'Surge']):
    print(f"\n{class_name} rides:")
    for j, feat in enumerate(clf_feature_cols[:6]):  # Show first 6
        mean = nb_model.theta_[i][j]
        std = np.sqrt(nb_model.var_[i][j])
        print(f"   {feat}: μ={mean:.3f}, σ={std:.3f}")

# MODEL 4: Linear Discriminant Analysis
print("\n" + "-"*80)
print("MODEL 4: LINEAR DISCRIMINANT ANALYSIS (LDA)")
print("-"*80)

lda_model = LinearDiscriminantAnalysis(priors=[0.5, 0.5])  # Balanced priors
lda_model.fit(X_train_c, y_train_c)
lda_preds = lda_model.predict(X_test_c)

print(f"Accuracy: {accuracy_score(y_test_c, lda_preds):.4f}")
print(f"Precision: {precision_score(y_test_c, lda_preds):.4f}")
print(f"Recall: {recall_score(y_test_c, lda_preds):.4f}")
print(f"F1 Score: {f1_score(y_test_c, lda_preds):.4f}")

print("\n📐 LINEAR DISCRIMINANT EQUATION:")
print(f"{lda_model.intercept_[0]:.4f}", end="")
for coef, feat in zip(lda_model.coef_[0], clf_feature_cols):
    sign = " + " if coef >= 0 else " - "
    print(f"{sign}{abs(coef):.4f}×{feat}", end="")
print(" = 0")

print("\n📊 CLASS CENTROIDS:")
for i, class_name in enumerate(['No Surge', 'Surge']):
    centroid_vals = lda_model.means_[i][:4]  # First 4 features
    print(f"   {class_name}: [{', '.join([f'{v:.2f}' for v in centroid_vals])}...]")

# MODEL 5: Perceptron
print("\n" + "-"*80)
print("MODEL 5: PERCEPTRON (Linear Threshold Classifier)")
print("-"*80)

perceptron = Perceptron(max_iter=1000, random_state=42, class_weight='balanced')
perceptron.fit(X_train_c, y_train_c)
perc_preds = perceptron.predict(X_test_c)

print(f"Accuracy: {accuracy_score(y_test_c, perc_preds):.4f}")
print(f"Precision: {precision_score(y_test_c, perc_preds):.4f}")
print(f"Recall: {recall_score(y_test_c, perc_preds):.4f}")
print(f"F1 Score: {f1_score(y_test_c, perc_preds):.4f}")

print("\n📐 PERCEPTRON EQUATION:")
print("IF (", end="")
print(f"{perceptron.intercept_[0]:.4f}", end="")
for coef, feat in zip(perceptron.coef_[0], clf_feature_cols):
    sign = " + " if coef >= 0 else " - "
    print(f"{sign}{abs(coef):.4f}×{feat}", end="")
print(" ) > 0")
print("   THEN predict SURGE")
print("   ELSE predict NO SURGE")

print("\n📊 WEIGHT INTERPRETATION:")
for feat, weight in zip(clf_feature_cols, perceptron.coef_[0]):
    direction = "↑ increases" if weight > 0 else "↓ decreases"
    print(f"   {feat}: {weight:.4f} ({direction} surge probability)")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*80)
print("FINAL SUMMARY - ALL WHITE BOX MODELS (FULL PIPELINE)")
print("="*80)

print("""
REGRESSION MODELS (Price Prediction):
┌─────────────────────┬────────┬─────────┬──────────────────────────────────┐
│ Model               │ R²     │ MAE     │ Equation Type                    │
├─────────────────────┼────────┼─────────┼──────────────────────────────────┤
│ 1. OLS Linear       │ {:.4f}  │ ${:.2f}  │ Linear coefficients + p-values  │
│ 2. Ridge (L2)       │ {:.4f}  │ ${:.2f}  │ Regularized linear              │
│ 3. Lasso (L1)       │ {:.4f}  │ ${:.2f}  │ Sparse feature selection        │
│ 4. Decision Tree    │ {:.4f}  │ ${:.2f}  │ IF-THEN rules (depth 3)         │
│ 5. Polynomial       │ {:.4f}  │ ${:.2f}  │ Quadratic interactions          │
└─────────────────────┴────────┴─────────┴──────────────────────────────────┘

CLASSIFICATION MODELS (Surge Prediction - No Leakage):
┌─────────────────────┬────────┬─────────┬──────────────────────────────────┐
│ Model               │ Acc    │ F1      │ Decision Logic                   │
├─────────────────────┼────────┼─────────┼──────────────────────────────────┤
│ 1. Logistic Reg     │ {:.4f}  │ {:.4f}   │ Log-odds + sigmoid              │
│ 2. Decision Tree    │ {:.4f}  │ {:.4f}   │ IF-THEN rules                   │
│ 3. Naive Bayes      │ {:.4f}  │ {:.4f}   │ Probabilistic P(x|y)            │
│ 4. LDA              │ {:.4f}  │ {:.4f}   │ Linear discriminant             │
│ 5. Perceptron       │ {:.4f}  │ {:.4f}   │ Linear threshold                │
└─────────────────────┴────────┴─────────┴──────────────────────────────────┘
""".format(
    ols_model.rsquared, mean_absolute_error(y_test_r, ols_preds),
    r2_score(y_test_r, ridge_preds), mean_absolute_error(y_test_r, ridge_preds),
    r2_score(y_test_r, lasso_preds), mean_absolute_error(y_test_r, lasso_preds),
    r2_score(y_test_r, tree_preds), mean_absolute_error(y_test_r, tree_preds),
    r2_score(y_test_r, poly_preds), mean_absolute_error(y_test_r, poly_preds),
    accuracy_score(y_test_c, log_preds), f1_score(y_test_c, log_preds),
    accuracy_score(y_test_c, tree_clf_preds), f1_score(y_test_c, tree_clf_preds),
    accuracy_score(y_test_c, nb_preds), f1_score(y_test_c, nb_preds),
    accuracy_score(y_test_c, lda_preds), f1_score(y_test_c, lda_preds),
    accuracy_score(y_test_c, perc_preds), f1_score(y_test_c, perc_preds)
))

print("="*80)
print("KEY INSIGHTS FROM WHITE BOX MODELS:")
print("="*80)
print("1. All models are INTERPRETABLE - equations/rules are human-readable")
print("2. Classification models EXCLUDE surge features - no data leakage")
print("3. Decision Tree provides the most intuitive IF-THEN rules")
print("4. Linear models show exact coefficient impact on price/surge")
print("5. Full pipeline preprocessing ensures realistic performance estimates")
print("="*80)
