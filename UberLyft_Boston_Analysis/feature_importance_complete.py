"""
Complete Feature Importance Analysis
Ranks all features by importance across multiple models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*90)
print("COMPLETE FEATURE IMPORTANCE ANALYSIS")
print("All Features Ranked by Importance")
print("="*90)

# Load enriched datasets
print("\nLoading enriched datasets...")
reg_df = pd.read_csv('/home/kushagarwal/CascadeProjects/Supply_Chain_Optimization/UberLyft_Boston_Analysis/new_data/processed/regression_dataset_enriched.csv')
clf_df = pd.read_csv('/home/kushagarwal/CascadeProjects/Supply_Chain_Optimization/UberLyft_Boston_Analysis/new_data/processed/classification_dataset_enriched.csv')

print(f"✓ Regression: {reg_df.shape[1]-1} features")
print(f"✓ Classification: {clf_df.shape[1]-1} features")

# ============================================
# REGRESSION FEATURE IMPORTANCE
# ============================================
print("\n" + "="*90)
print("REGRESSION FEATURES (Price Prediction)")
print("="*90)

X_reg = reg_df.drop('price', axis=1)
y_reg = reg_df['price']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# 1. RIDGE REGRESSION (Absolute Coefficients)
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_r, y_train_r)
ridge_importance = pd.DataFrame({
    'feature': X_reg.columns,
    'ridge_coef': np.abs(ridge.coef_),
    'ridge_rank': range(1, len(X_reg.columns)+1)
}).sort_values('ridge_coef', ascending=False).reset_index(drop=True)
ridge_importance['ridge_rank'] = range(1, len(ridge_importance)+1)

# 2. LASSO REGRESSION (Feature Selection)
lasso = Lasso(alpha=0.1, random_state=42, max_iter=10000)
lasso.fit(X_train_r, y_train_r)
lasso_importance = pd.DataFrame({
    'feature': X_reg.columns,
    'lasso_coef': np.abs(lasso.coef_)
}).sort_values('lasso_coef', ascending=False).reset_index(drop=True)
lasso_importance['lasso_rank'] = range(1, len(lasso_importance)+1)

# 3. DECISION TREE (Feature Importance)
tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_reg.fit(X_train_r, y_train_r)
tree_importance = pd.DataFrame({
    'feature': X_reg.columns,
    'tree_importance': tree_reg.feature_importances_
}).sort_values('tree_importance', ascending=False).reset_index(drop=True)
tree_importance['tree_rank'] = range(1, len(tree_importance)+1)

# Combine all regression importances
reg_importance = ridge_importance.merge(lasso_importance[['feature', 'lasso_coef', 'lasso_rank']], on='feature')
reg_importance = reg_importance.merge(tree_importance[['feature', 'tree_importance', 'tree_rank']], on='feature')

# Calculate average rank (lower is better)
reg_importance['avg_rank'] = (reg_importance['ridge_rank'] + reg_importance['lasso_rank'] + reg_importance['tree_rank']) / 3
reg_importance = reg_importance.sort_values('avg_rank')

print(f"\n📊 ALL {len(reg_importance)} REGRESSION FEATURES RANKED:\n")
print(f"{'Rank':<6} {'Feature':<35} {'Ridge':<12} {'Lasso':<12} {'Tree':<12} {'Avg':<8}")
print("-"*90)

for idx, row in reg_importance.iterrows():
    rank = int(row['avg_rank'])
    print(f"{rank:<6} {row['feature']:<35} {row['ridge_coef']:>10.4f}  {row['lasso_coef']:>10.4f}  {row['tree_importance']:>10.4f}  {row['avg_rank']:>6.1f}")

# Top 10 summary
print(f"\n{'='*90}")
print(f"🏆 TOP 10 REGRESSION FEATURES (Consensus across all models):")
print(f"{'='*90}")
for i, (_, row) in enumerate(reg_importance.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['feature']:<35} (Ridge: {row['ridge_coef']:.4f}, Tree: {row['tree_importance']:.4f})")

# ============================================
# CLASSIFICATION FEATURE IMPORTANCE
# ============================================
print("\n" + "="*90)
print("CLASSIFICATION FEATURES (Premium Vehicle Prediction)")
print("="*90)

# Get target column
target_col = 'is_premium_vehicle' if 'is_premium_vehicle' in clf_df.columns else 'is_premium'
X_clf = clf_df.drop(target_col, axis=1)
y_clf = clf_df[target_col]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# 1. LOGISTIC REGRESSION (Absolute Coefficients)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_c, y_train_c)
log_importance = pd.DataFrame({
    'feature': X_clf.columns,
    'logistic_coef': np.abs(log_reg.coef_[0])
}).sort_values('logistic_coef', ascending=False).reset_index(drop=True)
log_importance['logistic_rank'] = range(1, len(log_importance)+1)

# 2. DECISION TREE (Feature Importance)
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train_c, y_train_c)
clf_tree_importance = pd.DataFrame({
    'feature': X_clf.columns,
    'clf_tree_importance': tree_clf.feature_importances_
}).sort_values('clf_tree_importance', ascending=False).reset_index(drop=True)
clf_tree_importance['clf_tree_rank'] = range(1, len(clf_tree_importance)+1)

# Combine classification importances
clf_importance = log_importance.merge(clf_tree_importance[['feature', 'clf_tree_importance', 'clf_tree_rank']], on='feature')
clf_importance['avg_rank'] = (clf_importance['logistic_rank'] + clf_importance['clf_tree_rank']) / 2
clf_importance = clf_importance.sort_values('avg_rank')

print(f"\n📊 ALL {len(clf_importance)} CLASSIFICATION FEATURES RANKED:\n")
print(f"{'Rank':<6} {'Feature':<35} {'Logistic':<12} {'Tree':<12} {'Avg':<8}")
print("-"*90)

for idx, row in clf_importance.iterrows():
    rank = int(row['avg_rank'])
    print(f"{rank:<6} {row['feature']:<35} {row['logistic_coef']:>10.4f}  {row['clf_tree_importance']:>10.4f}  {row['avg_rank']:>6.1f}")

# Top 10 summary
print(f"\n{'='*90}")
print(f"🏆 TOP 10 CLASSIFICATION FEATURES (Consensus across models):")
print(f"{'='*90}")
for i, (_, row) in enumerate(clf_importance.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['feature']:<35} (Logistic: {row['logistic_coef']:.4f}, Tree: {row['clf_tree_importance']:.4f})")

# ============================================
# COMPARATIVE ANALYSIS
# ============================================
print("\n" + "="*90)
print("FEATURE CATEGORY BREAKDOWN")
print("="*90)

# Categorize features
def categorize_feature(name):
    if any(x in name for x in ['surge', 'multiplier']):
        return 'Surge/Demand'
    elif any(x in name for x in ['premium', 'lux', 'black', 'suv', 'xl']):
        return 'Vehicle Type'
    elif any(x in name for x in ['weather', 'rain', 'temp', 'humid', 'wind', 'cold', 'cloud']):
        return 'Weather'
    elif any(x in name for x in ['hour', 'rush', 'night', 'weekend', 'day', 'morning', 'evening']):
        return 'Time'
    elif any(x in name for x in ['distance', 'short', 'medium', 'long', 'ride']):
        return 'Distance'
    elif any(x in name for x in ['source', 'destination', 'zone', 'venue', 'station']):
        return 'Location'
    elif any(x in name for x in ['event', 'transit', 'delay', 'mbta']):
        return 'Events/Transit'
    elif any(x in name for x in ['fuel', 'gas', 'price']):
        return 'Economic'
    elif name in ['cab_type', 'name']:
        return 'Vehicle ID'
    else:
        return 'Other'

reg_importance['category'] = reg_importance['feature'].apply(categorize_feature)
category_summary = reg_importance.groupby('category').agg({
    'ridge_coef': 'mean',
    'feature': 'count'
}).sort_values('ridge_coef', ascending=False).reset_index()

print("\n📊 REGRESSION FEATURES BY CATEGORY (Average Ridge Importance):")
print("-"*70)
for _, row in category_summary.iterrows():
    print(f"{row['category']:<20} {row['feature']:>3} features  (Avg Importance: {row['ridge_coef']:>8.4f})")

# ============================================
# NEW ENRICHED FEATURES IMPACT
# ============================================
print("\n" + "="*90)
print("ENRICHED FEATURES IMPACT ANALYSIS")
print("="*90)

# Identify enriched features
enriched_features = [
    'weather_severity_enhanced', 'is_adverse_weather', 'is_event_time',
    'likely_transit_delay', 'is_high_demand_zone', 'fuel_price_indicator',
    'distance_event_interaction', 'weather_rush_interaction', 'premium_weather_interaction'
]

enriched_df = reg_importance[reg_importance['feature'].isin(enriched_features)].sort_values('ridge_coef', ascending=False)

print("\n🆕 NEW ENRICHED FEATURES (From Web Scraping):")
print("-"*70)
print(f"{'Feature':<35} {'Ridge':<12} {'Tree':<12} {'Combined':<12}")
print("-"*70)

for _, row in enriched_df.iterrows():
    combined = row['ridge_coef'] + row['tree_importance']
    print(f"{row['feature']:<35} {row['ridge_coef']:>10.4f}  {row['tree_importance']:>10.4f}  {combined:>10.4f}")

# Calculate enrichment contribution
top_10_original = set(reg_importance[~reg_importance['feature'].isin(enriched_features)].head(10)['feature'])
top_10_with_enriched = set(reg_importance.head(10)['feature'])
enriched_in_top_10 = [f for f in enriched_features if f in top_10_with_enriched]

print(f"\n✅ ENRICHED FEATURES IN TOP 10: {len(enriched_in_top_10)}/{len(enriched_features)}")
if enriched_in_top_10:
    for f in enriched_in_top_10:
        rank = reg_importance[reg_importance['feature']==f].index[0] + 1
        print(f"   • {f} (Rank #{rank})")

print("\n" + "="*90)
print("ANALYSIS COMPLETE - All features ranked by importance")
print("="*90)
