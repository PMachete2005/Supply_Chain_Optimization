"""
Uber/Lyft Boston — White-Box Model Analysis
Runs 5 regression + 5 classification models on the final enriched datasets.

Usage:
    python run_models.py

Input:
    ../new_data/processed/regression_dataset_enriched.csv
    ../new_data/processed/classification_dataset_enriched.csv
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score)
import statsmodels.api as sm

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────
DATA_DIR = '../new_data/processed'
REG_PATH = f'{DATA_DIR}/regression_dataset_enriched.csv'
CLF_PATH = f'{DATA_DIR}/classification_dataset_enriched.csv'

# ── Load data ────────────────────────────────────────────
print("=" * 80)
print("WHITE-BOX MODEL ANALYSIS — ENRICHED DATASETS")
print("=" * 80)

reg_df = pd.read_csv(REG_PATH)
clf_df = pd.read_csv(CLF_PATH)

print(f"\nRegression:     {reg_df.shape[0]:,} rows × {reg_df.shape[1] - 1} features  →  target: price")
print(f"Classification: {clf_df.shape[0]:,} rows × {clf_df.shape[1] - 1} features  →  target: is_premium")

# ══════════════════════════════════════════════════════════
# PART 1: REGRESSION (Price Prediction)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 1: REGRESSION — 5 MODELS")
print("=" * 80)

X_reg = reg_df.drop('price', axis=1)
y_reg = reg_df['price']
feature_names_reg = X_reg.columns.tolist()

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

scaler_r = StandardScaler()
X_train_rs = scaler_r.fit_transform(X_train_r)
X_test_rs = scaler_r.transform(X_test_r)

print(f"\nTrain: {len(X_train_r):,}  |  Test: {len(X_test_r):,}  |  Features: {len(feature_names_reg)}")
print(f"Price range: ${y_reg.min():.2f} – ${y_reg.max():.2f}")

reg_results = {}

# ── 1. OLS ───────────────────────────────────────────────
print("\n" + "-" * 80)
print("MODEL 1: OLS (Ordinary Least Squares)")
print("-" * 80)

X_ols_train = sm.add_constant(X_train_rs)
X_ols_test = sm.add_constant(X_test_rs)
ols = sm.OLS(y_train_r, X_ols_train).fit()
ols_preds = ols.predict(X_ols_test)

r2 = r2_score(y_test_r, ols_preds)
mae = mean_absolute_error(y_test_r, ols_preds)
reg_results['OLS'] = {'R2': r2, 'MAE': mae}
print(f"R² = {r2:.4f}  |  MAE = ${mae:.2f}")

coefs = ols.params.drop('const', errors='ignore')
top5 = coefs.abs().nlargest(5)
print("\nTop 5 coefficients:")
for feat in top5.index:
    idx = list(coefs.index).index(feat)
    print(f"  {feature_names_reg[idx]:<30} {coefs[feat]:+.4f}")

# ── 2. Ridge ─────────────────────────────────────────────
print("\n" + "-" * 80)
print("MODEL 2: RIDGE REGRESSION (α=1.0)")
print("-" * 80)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_rs, y_train_r)
ridge_preds = ridge.predict(X_test_rs)

r2 = r2_score(y_test_r, ridge_preds)
mae = mean_absolute_error(y_test_r, ridge_preds)
reg_results['Ridge'] = {'R2': r2, 'MAE': mae}
print(f"R² = {r2:.4f}  |  MAE = ${mae:.2f}")

top5 = np.argsort(np.abs(ridge.coef_))[-5:][::-1]
print("\nTop 5 coefficients:")
for i in top5:
    print(f"  {feature_names_reg[i]:<30} {ridge.coef_[i]:+.4f}")

# ── 3. Lasso ─────────────────────────────────────────────
print("\n" + "-" * 80)
print("MODEL 3: LASSO REGRESSION (α=0.1)")
print("-" * 80)

lasso = Lasso(alpha=0.1, random_state=42, max_iter=10000)
lasso.fit(X_train_rs, y_train_r)
lasso_preds = lasso.predict(X_test_rs)

r2 = r2_score(y_test_r, lasso_preds)
mae = mean_absolute_error(y_test_r, lasso_preds)
reg_results['Lasso'] = {'R2': r2, 'MAE': mae}
print(f"R² = {r2:.4f}  |  MAE = ${mae:.2f}")

selected = [(feature_names_reg[i], lasso.coef_[i])
            for i in range(len(lasso.coef_)) if abs(lasso.coef_[i]) > 0.001]
selected.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"\nSelected {len(selected)}/{len(feature_names_reg)} features:")
for feat, coef in selected[:8]:
    print(f"  {feat:<30} {coef:+.4f}")

# ── 4. Decision Tree ─────────────────────────────────────
print("\n" + "-" * 80)
print("MODEL 4: DECISION TREE (max_depth=10)")
print("-" * 80)

tree_r = DecisionTreeRegressor(max_depth=10, random_state=42)
tree_r.fit(X_train_r, y_train_r)
tree_r_preds = tree_r.predict(X_test_r)

r2 = r2_score(y_test_r, tree_r_preds)
mae = mean_absolute_error(y_test_r, tree_r_preds)
reg_results['Decision Tree'] = {'R2': r2, 'MAE': mae}
print(f"R² = {r2:.4f}  |  MAE = ${mae:.2f}")

top5 = np.argsort(tree_r.feature_importances_)[-5:][::-1]
print("\nTop 5 feature importances:")
for i in top5:
    print(f"  {feature_names_reg[i]:<30} {tree_r.feature_importances_[i]:.4f}")

# ── 5. Polynomial Regression (degree=2) ──────────────────
print("\n" + "-" * 80)
print("MODEL 5: POLYNOMIAL REGRESSION (degree=2)")
print("-" * 80)

# Use top 5 features by Ridge importance to avoid combinatorial explosion
top5_feats = [feature_names_reg[i] for i in np.argsort(np.abs(ridge.coef_))[-5:][::-1]]
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_r[top5_feats])
X_test_poly = poly.transform(X_test_r[top5_feats])

poly_model = Ridge(alpha=1.0, random_state=42)
poly_model.fit(X_train_poly, y_train_r)
poly_preds = poly_model.predict(X_test_poly)

r2 = r2_score(y_test_r, poly_preds)
mae = mean_absolute_error(y_test_r, poly_preds)
reg_results['Polynomial'] = {'R2': r2, 'MAE': mae}
print(f"R² = {r2:.4f}  |  MAE = ${mae:.2f}")
print(f"Base features: {', '.join(top5_feats)}")
print(f"Expanded: {len(top5_feats)} → {X_train_poly.shape[1]} terms")

# ── Regression Summary ───────────────────────────────────
print("\n" + "=" * 80)
print("REGRESSION SUMMARY")
print("=" * 80)
print(f"\n{'Model':<22} {'R²':<10} {'MAE':<10}")
print("-" * 42)
for name, m in sorted(reg_results.items(), key=lambda x: x[1]['R2'], reverse=True):
    print(f"{name:<22} {m['R2']:.4f}    ${m['MAE']:.2f}")

best_reg = max(reg_results.items(), key=lambda x: x[1]['R2'])
print(f"\n✓ Best: {best_reg[0]} (R²={best_reg[1]['R2']:.4f}, MAE=${best_reg[1]['MAE']:.2f})")

# ══════════════════════════════════════════════════════════
# PART 2: CLASSIFICATION (Premium Vehicle Prediction)
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PART 2: CLASSIFICATION — 5 MODELS")
print("=" * 80)

X_clf = clf_df.drop('is_premium', axis=1)
y_clf = clf_df['is_premium']
feature_names_clf = X_clf.columns.tolist()

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

scaler_c = StandardScaler()
X_train_cs = scaler_c.fit_transform(X_train_c)
X_test_cs = scaler_c.transform(X_test_c)

print(f"\nTrain: {len(X_train_c):,}  |  Test: {len(X_test_c):,}  |  Features: {len(feature_names_clf)}")
print(f"Premium rate: {y_clf.mean()*100:.1f}%")

clf_results = {}

# ── 1. Logistic Regression ───────────────────────────────
print("\n" + "-" * 80)
print("MODEL 1: LOGISTIC REGRESSION")
print("-" * 80)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_cs, y_train_c)
log_preds = log_reg.predict(X_test_cs)

acc = accuracy_score(y_test_c, log_preds)
f1 = f1_score(y_test_c, log_preds)
prec = precision_score(y_test_c, log_preds)
rec = recall_score(y_test_c, log_preds)
clf_results['Logistic Reg'] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
print(f"Acc={acc:.4f}  |  F1={f1:.4f}  |  Prec={prec:.4f}  |  Rec={rec:.4f}")

top5 = np.argsort(np.abs(log_reg.coef_[0]))[-5:][::-1]
print("\nTop 5 odds ratios:")
for i in top5:
    or_val = np.exp(log_reg.coef_[0][i])
    direction = "↑" if or_val > 1 else "↓"
    print(f"  {feature_names_clf[i]:<30} OR={or_val:.4f} {direction}")

# ── 2. Decision Tree ─────────────────────────────────────
print("\n" + "-" * 80)
print("MODEL 2: DECISION TREE (max_depth=10)")
print("-" * 80)

tree_c = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_c.fit(X_train_c, y_train_c)
tree_c_preds = tree_c.predict(X_test_c)

acc = accuracy_score(y_test_c, tree_c_preds)
f1 = f1_score(y_test_c, tree_c_preds)
prec = precision_score(y_test_c, tree_c_preds)
rec = recall_score(y_test_c, tree_c_preds)
clf_results['Decision Tree'] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
print(f"Acc={acc:.4f}  |  F1={f1:.4f}  |  Prec={prec:.4f}  |  Rec={rec:.4f}")

top5 = np.argsort(tree_c.feature_importances_)[-5:][::-1]
print("\nTop 5 feature importances:")
for i in top5:
    print(f"  {feature_names_clf[i]:<30} {tree_c.feature_importances_[i]:.4f}")

# ── 3. Naive Bayes ───────────────────────────────────────
print("\n" + "-" * 80)
print("MODEL 3: GAUSSIAN NAIVE BAYES")
print("-" * 80)

nb = GaussianNB()
nb.fit(X_train_cs, y_train_c)
nb_preds = nb.predict(X_test_cs)

acc = accuracy_score(y_test_c, nb_preds)
f1 = f1_score(y_test_c, nb_preds)
prec = precision_score(y_test_c, nb_preds)
rec = recall_score(y_test_c, nb_preds)
clf_results['Naive Bayes'] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
print(f"Acc={acc:.4f}  |  F1={f1:.4f}  |  Prec={prec:.4f}  |  Rec={rec:.4f}")
print(f"Class priors: P(standard)={nb.class_prior_[0]:.3f}  P(premium)={nb.class_prior_[1]:.3f}")

# ── 4. LDA ───────────────────────────────────────────────
print("\n" + "-" * 80)
print("MODEL 4: LINEAR DISCRIMINANT ANALYSIS")
print("-" * 80)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_cs, y_train_c)
lda_preds = lda.predict(X_test_cs)

acc = accuracy_score(y_test_c, lda_preds)
f1 = f1_score(y_test_c, lda_preds)
prec = precision_score(y_test_c, lda_preds)
rec = recall_score(y_test_c, lda_preds)
clf_results['LDA'] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
print(f"Acc={acc:.4f}  |  F1={f1:.4f}  |  Prec={prec:.4f}  |  Rec={rec:.4f}")

top5 = np.argsort(np.abs(lda.coef_[0]))[-5:][::-1]
print("\nTop 5 discriminant coefficients:")
for i in top5:
    print(f"  {feature_names_clf[i]:<30} {lda.coef_[0][i]:+.4f}")

# ── 5. Perceptron ────────────────────────────────────────
print("\n" + "-" * 80)
print("MODEL 5: PERCEPTRON")
print("-" * 80)

perc = Perceptron(max_iter=1000, random_state=42)
perc.fit(X_train_cs, y_train_c)
perc_preds = perc.predict(X_test_cs)

acc = accuracy_score(y_test_c, perc_preds)
f1 = f1_score(y_test_c, perc_preds)
prec = precision_score(y_test_c, perc_preds)
rec = recall_score(y_test_c, perc_preds)
clf_results['Perceptron'] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
print(f"Acc={acc:.4f}  |  F1={f1:.4f}  |  Prec={prec:.4f}  |  Rec={rec:.4f}")

# ── Classification Summary ───────────────────────────────
print("\n" + "=" * 80)
print("CLASSIFICATION SUMMARY")
print("=" * 80)
print(f"\n{'Model':<22} {'Acc':<10} {'F1':<10} {'Prec':<10} {'Rec':<10}")
print("-" * 62)
for name, m in sorted(clf_results.items(), key=lambda x: x[1]['F1'], reverse=True):
    print(f"{name:<22} {m['Acc']:.4f}    {m['F1']:.4f}    {m['Prec']:.4f}    {m['Rec']:.4f}")

best_clf = max(clf_results.items(), key=lambda x: x[1]['F1'])
print(f"\n✓ Best: {best_clf[0]} (F1={best_clf[1]['F1']:.4f}, Acc={best_clf[1]['Acc']:.4f})")

# ══════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
