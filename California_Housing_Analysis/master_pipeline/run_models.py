"""
Step 3: Run Models
5 white-box regression models + 5 white-box classification models
on the enriched California Housing datasets.

Regression models:  OLS, Ridge, Lasso, Decision Tree, Polynomial
Classification models: Logistic Regression, Decision Tree, Naive Bayes, LDA, Perceptron
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (r2_score, mean_absolute_error,
                             accuracy_score, f1_score, precision_score, recall_score)
import statsmodels.api as sm
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
reg_df = pd.read_csv(os.path.join(PROC_DIR, 'regression_dataset_enriched.csv'))
clf_df = pd.read_csv(os.path.join(PROC_DIR, 'classification_dataset_enriched.csv'))

reg_features = [c for c in reg_df.columns if c != 'MedHouseVal']
clf_features = [c for c in clf_df.columns if c != 'is_high_value']

print("=" * 80)
print("WHITE-BOX MODEL ANALYSIS — CALIFORNIA HOUSING (ENRICHED)")
print("=" * 80)
print(f"\nRegression:     {reg_df.shape[0]:,} rows × {len(reg_features)} features → target: MedHouseVal")
print(f"Classification: {clf_df.shape[0]:,} rows × {len(clf_features)} features → target: is_high_value")

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("PART 1: REGRESSION — 5 MODELS")
print("=" * 80)

X_reg = reg_df[reg_features]
y_reg = reg_df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
X_train_s = scaler_reg.fit_transform(X_train)
X_test_s = scaler_reg.transform(X_test)

print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}  |  Features: {len(reg_features)}")
print(f"Price range: ${y_reg.min()*100:.0f}K – ${y_reg.max()*100:.0f}K")

reg_results = {}

# MODEL 1: OLS
print(f"\n{'-'*80}")
print("MODEL 1: OLS (Ordinary Least Squares)")
print("-" * 80)
ols = sm.OLS(y_train, sm.add_constant(X_train_s)).fit()
pred = ols.predict(sm.add_constant(X_test_s))
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
reg_results['OLS'] = {'R2': r2, 'MAE': mae}
print(f"R² = {r2:.4f}  |  MAE = ${mae*100:.0f}K")
# Top 5 significant features
pvals = ols.pvalues.values[1:]  # skip constant
coefs = ols.params.values[1:]
top5_idx = np.argsort(pvals)[:5]
print("\nTop 5 most significant features (by p-value):")
for i in top5_idx:
    print(f"  {reg_features[i]:<30} coef={coefs[i]:>+.4f}  p={pvals[i]:.2e}")

# MODEL 2: Ridge
print(f"\n{'-'*80}")
print("MODEL 2: RIDGE REGRESSION (α=1.0)")
print("-" * 80)
ridge = Ridge(alpha=1.0, random_state=42).fit(X_train_s, y_train)
pred = ridge.predict(X_test_s)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
reg_results['Ridge'] = {'R2': r2, 'MAE': mae}
print(f"R² = {r2:.4f}  |  MAE = ${mae*100:.0f}K")
top5 = np.argsort(np.abs(ridge.coef_))[-5:][::-1]
print("\nTop 5 coefficients:")
for i in top5:
    print(f"  {reg_features[i]:<30} {ridge.coef_[i]:>+.4f}")

# MODEL 3: Lasso
print(f"\n{'-'*80}")
print("MODEL 3: LASSO REGRESSION (α=0.01)")
print("-" * 80)
lasso = Lasso(alpha=0.01, random_state=42, max_iter=10000).fit(X_train_s, y_train)
pred = lasso.predict(X_test_s)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
reg_results['Lasso'] = {'R2': r2, 'MAE': mae}
selected = sum(lasso.coef_ != 0)
print(f"R² = {r2:.4f}  |  MAE = ${mae*100:.0f}K")
print(f"\nSelected {selected}/{len(reg_features)} features:")
for i in np.argsort(np.abs(lasso.coef_))[::-1]:
    if lasso.coef_[i] != 0:
        print(f"  {reg_features[i]:<30} {lasso.coef_[i]:>+.4f}")

# MODEL 4: Decision Tree
print(f"\n{'-'*80}")
print("MODEL 4: DECISION TREE (max_depth=10)")
print("-" * 80)
dt_reg = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_train, y_train)
pred = dt_reg.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
reg_results['Decision Tree'] = {'R2': r2, 'MAE': mae}
print(f"R² = {r2:.4f}  |  MAE = ${mae*100:.0f}K")
top5 = np.argsort(dt_reg.feature_importances_)[-5:][::-1]
print("\nTop 5 feature importances:")
for i in top5:
    print(f"  {reg_features[i]:<30} {dt_reg.feature_importances_[i]:.4f}")

# MODEL 5: Polynomial
print(f"\n{'-'*80}")
print("MODEL 5: POLYNOMIAL REGRESSION (degree=2)")
print("-" * 80)
# Select domain-meaningful features that benefit from polynomial expansion
# Prioritize: income, rooms, occupancy, coast distance, coastal indicator
poly_candidates = ['MedInc', 'AveRooms', 'AveOccup', 'rooms_per_person',
                   'dist_to_coast', 'coastal_income', 'income_coast_interaction',
                   'HouseAge', 'Latitude', 'Longitude']
top5_feats = [f for f in poly_candidates if f in reg_features][:5]
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train[top5_feats])
X_test_poly = poly.transform(X_test[top5_feats])
poly_model = Ridge(alpha=1.0, random_state=42).fit(X_train_poly, y_train)
pred = poly_model.predict(X_test_poly)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
reg_results['Polynomial'] = {'R2': r2, 'MAE': mae}
print(f"R² = {r2:.4f}  |  MAE = ${mae*100:.0f}K")
print(f"Base features: {top5_feats}")
print(f"Expanded: {len(top5_feats)} → {X_train_poly.shape[1]} terms")

# Regression summary
print(f"\n{'='*80}")
print("REGRESSION SUMMARY")
print("=" * 80)
print(f"\n{'Model':<22} {'R²':<10} {'MAE':>10}")
print("-" * 44)
for name in sorted(reg_results, key=lambda x: reg_results[x]['R2'], reverse=True):
    r = reg_results[name]
    print(f"{name:<22} {r['R2']:.4f}    ${r['MAE']*100:.0f}K")
best_reg = max(reg_results.items(), key=lambda x: x[1]['R2'])
print(f"\n✓ Best: {best_reg[0]} (R²={best_reg[1]['R2']:.4f}, MAE=${best_reg[1]['MAE']*100:.0f}K)")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("PART 2: CLASSIFICATION — 5 MODELS")
print("=" * 80)

X_clf = clf_df[clf_features]
y_clf = clf_df['is_high_value']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

scaler_clf = StandardScaler()
X_train_cs = scaler_clf.fit_transform(X_train_c)
X_test_cs = scaler_clf.transform(X_test_c)

print(f"\nTrain: {len(X_train_c):,}  |  Test: {len(X_test_c):,}  |  Features: {len(clf_features)}")
print(f"High-value rate: {y_clf.mean()*100:.1f}%")

clf_results = {}

# MODEL 1: Logistic Regression
print(f"\n{'-'*80}")
print("MODEL 1: LOGISTIC REGRESSION")
print("-" * 80)
lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_cs, y_train_c)
pred = lr.predict(X_test_cs)
acc = accuracy_score(y_test_c, pred)
f1 = f1_score(y_test_c, pred)
prec = precision_score(y_test_c, pred)
rec = recall_score(y_test_c, pred)
clf_results['Logistic Reg'] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
print(f"Acc={acc:.4f}  |  F1={f1:.4f}  |  Prec={prec:.4f}  |  Rec={rec:.4f}")
top5 = np.argsort(np.abs(lr.coef_[0]))[-5:][::-1]
print("\nTop 5 coefficients:")
for i in top5:
    direction = "↑ high" if lr.coef_[0][i] > 0 else "↓ low"
    print(f"  {clf_features[i]:<30} {lr.coef_[0][i]:>+.4f} {direction}")

# MODEL 2: Decision Tree
print(f"\n{'-'*80}")
print("MODEL 2: DECISION TREE (max_depth=10)")
print("-" * 80)
dt_clf = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train_c, y_train_c)
pred = dt_clf.predict(X_test_c)
acc = accuracy_score(y_test_c, pred)
f1 = f1_score(y_test_c, pred)
prec = precision_score(y_test_c, pred)
rec = recall_score(y_test_c, pred)
clf_results['Decision Tree'] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
print(f"Acc={acc:.4f}  |  F1={f1:.4f}  |  Prec={prec:.4f}  |  Rec={rec:.4f}")
top5 = np.argsort(dt_clf.feature_importances_)[-5:][::-1]
print("\nTop 5 feature importances:")
for i in top5:
    print(f"  {clf_features[i]:<30} {dt_clf.feature_importances_[i]:.4f}")

# MODEL 3: Naive Bayes
print(f"\n{'-'*80}")
print("MODEL 3: GAUSSIAN NAIVE BAYES")
print("-" * 80)
nb = GaussianNB().fit(X_train_cs, y_train_c)
pred = nb.predict(X_test_cs)
acc = accuracy_score(y_test_c, pred)
f1 = f1_score(y_test_c, pred)
prec = precision_score(y_test_c, pred)
rec = recall_score(y_test_c, pred)
clf_results['Naive Bayes'] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
print(f"Acc={acc:.4f}  |  F1={f1:.4f}  |  Prec={prec:.4f}  |  Rec={rec:.4f}")

# MODEL 4: LDA
print(f"\n{'-'*80}")
print("MODEL 4: LINEAR DISCRIMINANT ANALYSIS")
print("-" * 80)
lda = LinearDiscriminantAnalysis().fit(X_train_cs, y_train_c)
pred = lda.predict(X_test_cs)
acc = accuracy_score(y_test_c, pred)
f1 = f1_score(y_test_c, pred)
prec = precision_score(y_test_c, pred)
rec = recall_score(y_test_c, pred)
clf_results['LDA'] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
print(f"Acc={acc:.4f}  |  F1={f1:.4f}  |  Prec={prec:.4f}  |  Rec={rec:.4f}")
top5 = np.argsort(np.abs(lda.coef_[0]))[-5:][::-1]
print("\nTop 5 discriminant coefficients:")
for i in top5:
    print(f"  {clf_features[i]:<30} {lda.coef_[0][i]:>+.4f}")

# MODEL 5: Perceptron
print(f"\n{'-'*80}")
print("MODEL 5: PERCEPTRON")
print("-" * 80)
perc = Perceptron(max_iter=1000, random_state=42).fit(X_train_cs, y_train_c)
pred = perc.predict(X_test_cs)
acc = accuracy_score(y_test_c, pred)
f1 = f1_score(y_test_c, pred)
prec = precision_score(y_test_c, pred)
rec = recall_score(y_test_c, pred)
clf_results['Perceptron'] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
print(f"Acc={acc:.4f}  |  F1={f1:.4f}  |  Prec={prec:.4f}  |  Rec={rec:.4f}")

# Classification summary
print(f"\n{'='*80}")
print("CLASSIFICATION SUMMARY")
print("=" * 80)
print(f"\n{'Model':<22} {'Acc':<10} {'F1':<10} {'Prec':<10} {'Rec':<10}")
print("-" * 62)
for name in sorted(clf_results, key=lambda x: clf_results[x]['F1'], reverse=True):
    r = clf_results[name]
    print(f"{name:<22} {r['Acc']:.4f}    {r['F1']:.4f}    {r['Prec']:.4f}    {r['Rec']:.4f}")
best_clf = max(clf_results.items(), key=lambda x: x[1]['F1'])
print(f"\n✓ Best: {best_clf[0]} (F1={best_clf[1]['F1']:.4f}, Acc={best_clf[1]['Acc']:.4f})")

print(f"\n{'='*80}")
print("DONE")
print("=" * 80)
