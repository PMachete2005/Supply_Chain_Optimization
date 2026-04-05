"""
Run White Box Models on Prepared Datasets
Regression on regression_dataset.csv
Classification on classification_dataset.csv
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import statsmodels.api as sm

warnings.filterwarnings('ignore')
np.random.seed(42)

print("="*80)
print("WHITE BOX MODELS ON PREPARED DATASETS")
print("="*80)

# ============================================
# PART 1: REGRESSION MODELS
# ============================================
print("\n" + "="*80)
print("PART 1: REGRESSION MODELS (Price Prediction)")
print("Dataset: regression_dataset.csv")
print("="*80)

# Load regression dataset
reg_df = pd.read_csv('/home/kushagarwal/CascadeProjects/UberLyft_Boston_Analysis/regression_dataset.csv')
print(f"\n✓ Loaded: {len(reg_df):,} records, {len(reg_df.columns)} columns")

# Separate features and target
X_reg = reg_df.drop('price', axis=1)
y_reg = reg_df['price']
feature_names_reg = X_reg.columns.tolist()

print(f"✓ Features: {len(feature_names_reg)}")
print(f"✓ Target range: ${y_reg.min():.2f} - ${y_reg.max():.2f}")

# Train/test split
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
print(f"✓ Split: {len(X_train_r):,} train / {len(X_test_r):,} test")

# MODEL 1: OLS
print("\n" + "-"*80)
print("MODEL 1: ORDINARY LEAST SQUARES (OLS)")
print("-"*80)

ols_model = sm.OLS(y_train_r, sm.add_constant(X_train_r)).fit()
ols_preds = ols_model.predict(sm.add_constant(X_test_r))

print(f"R² Score: {ols_model.rsquared:.4f}")
print(f"MAE: ${mean_absolute_error(y_test_r, ols_preds):.2f}")

print("\n📐 TOP 10 COEFFICIENTS:")
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

# MODEL 2: Ridge
print("\n" + "-"*80)
print("MODEL 2: RIDGE REGRESSION")
print("-"*80)

ridge_model = Ridge(alpha=1.0, random_state=42)
ridge_model.fit(X_train_r, y_train_r)
ridge_preds = ridge_model.predict(X_test_r)

print(f"R² Score: {r2_score(y_test_r, ridge_preds):.4f}")
print(f"MAE: ${mean_absolute_error(y_test_r, ridge_preds):.2f}")

print("\n📐 TOP 5 COEFFICIENTS:")
importance = list(zip(feature_names_reg, np.abs(ridge_model.coef_)))
importance.sort(key=lambda x: x[1], reverse=True)
for feat, imp in importance[:5]:
    print(f"   {feat}: {imp:.4f}")

# MODEL 3: Lasso
print("\n" + "-"*80)
print("MODEL 3: LASSO REGRESSION (Feature Selection)")
print("-"*80)

lasso_model = Lasso(alpha=0.1, random_state=42, max_iter=10000)
lasso_model.fit(X_train_r, y_train_r)
lasso_preds = lasso_model.predict(X_test_r)

print(f"R² Score: {r2_score(y_test_r, lasso_preds):.4f}")
print(f"MAE: ${mean_absolute_error(y_test_r, lasso_preds):.2f}")

selected = [feat for feat, coef in zip(feature_names_reg, lasso_model.coef_) if abs(coef) > 0.001]
print(f"\n📐 SELECTED {len(selected)}/{len(feature_names_reg)} FEATURES:")
print(f"   {', '.join(selected[:10])}")

# MODEL 4: Decision Tree
print("\n" + "-"*80)
print("MODEL 4: DECISION TREE (Depth 3)")
print("-"*80)

tree_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg.fit(X_train_r, y_train_r)
tree_preds = tree_reg.predict(X_test_r)

print(f"R² Score: {r2_score(y_test_r, tree_preds):.4f}")
print(f"MAE: ${mean_absolute_error(y_test_r, tree_preds):.2f}")

print("\n📋 DECISION RULES:")
print(export_text(tree_reg, feature_names=feature_names_reg))

# MODEL 5: Polynomial
print("\n" + "-"*80)
print("MODEL 5: POLYNOMIAL REGRESSION (Degree 2)")
print("-"*80)

# Use top 4 features for polynomial
poly_features = [imp[0] for imp in importance[:4]]
X_train_poly = X_train_r[poly_features]
X_test_poly = X_test_r[poly_features]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly_tf = poly.fit_transform(X_train_poly)
X_test_poly_tf = poly.transform(X_test_poly)

poly_model = LinearRegression()
poly_model.fit(X_train_poly_tf, y_train_r)
poly_preds = poly_model.predict(X_test_poly_tf)

print(f"R² Score: {r2_score(y_test_r, poly_preds):.4f}")
print(f"MAE: ${mean_absolute_error(y_test_r, poly_preds):.2f}")
print(f"Features: {len(poly_features)} → {X_train_poly_tf.shape[1]}")

print("\n📐 TOP 5 POLYNOMIAL TERMS:")
poly_names = poly.get_feature_names_out(input_features=poly_features)
coefs = poly_model.coef_
top_idx = np.argsort(np.abs(coefs))[-5:][::-1]
print(f"price = {poly_model.intercept_:.4f}", end="")
for idx in top_idx:
    coef = coefs[idx]
    name = poly_names[idx]
    sign = " + " if coef >= 0 else " - "
    print(f"{sign}{abs(coef):.4f}×{name}", end="")
print()

# REGRESSION SUMMARY
print("\n" + "="*80)
print("REGRESSION SUMMARY")
print("="*80)
print(f"""
┌─────────────────┬────────┬─────────┐
│ Model           │ R²     │ MAE     │
├─────────────────┼────────┼─────────┤
│ OLS Linear      │ {ols_model.rsquared:.4f}  │ ${mean_absolute_error(y_test_r, ols_preds):.2f}  │
│ Ridge           │ {r2_score(y_test_r, ridge_preds):.4f}  │ ${mean_absolute_error(y_test_r, ridge_preds):.2f}  │
│ Lasso           │ {r2_score(y_test_r, lasso_preds):.4f}  │ ${mean_absolute_error(y_test_r, lasso_preds):.2f}  │
│ Decision Tree   │ {r2_score(y_test_r, tree_preds):.4f}  │ ${mean_absolute_error(y_test_r, tree_preds):.2f}  │
│ Polynomial      │ {r2_score(y_test_r, poly_preds):.4f}  │ ${mean_absolute_error(y_test_r, poly_preds):.2f}  │
└─────────────────┴────────┴─────────┘
""")

# ============================================
# PART 2: CLASSIFICATION MODELS
# ============================================
print("\n" + "="*80)
print("PART 2: CLASSIFICATION MODELS (Surge Prediction)")
print("Dataset: classification_dataset.csv")
print("="*80)

# Load classification dataset
clf_df = pd.read_csv('/home/kushagarwal/CascadeProjects/UberLyft_Boston_Analysis/classification_dataset.csv')
print(f"\n✓ Loaded: {len(clf_df):,} records, {len(clf_df.columns)} columns")

# Separate features and target
X_clf = clf_df.drop('is_expensive', axis=1)
y_clf = clf_df['is_expensive']
feature_names_clf = X_clf.columns.tolist()

print(f"✓ Features: {len(feature_names_clf)} (no surge columns)")
print(f"✓ Target: {y_clf.mean()*100:.1f}% surge rate ({y_clf.sum():,} / {len(y_clf):,})")

# Train/test split with stratification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
print(f"✓ Split: {len(X_train_c):,} train / {len(X_test_c):,} test")

# MODEL 1: Logistic Regression
print("\n" + "-"*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("-"*80)

log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_c, y_train_c)
log_preds = log_reg.predict(X_test_c)

acc = accuracy_score(y_test_c, log_preds)
prec = precision_score(y_test_c, log_preds)
rec = recall_score(y_test_c, log_preds)
f1 = f1_score(y_test_c, log_preds)

print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

print("\n📐 LOG-ODDS EQUATION:")
print(f"log(P/1-P) = {log_reg.intercept_[0]:.4f}", end="")
for coef, feat in zip(log_reg.coef_[0], feature_names_clf):
    sign = " + " if coef >= 0 else " - "
    print(f"{sign}{abs(coef):.4f}×{feat}", end="")
print()

print("\n📊 TOP 5 ODDS RATIOS:")
odds = np.exp(log_reg.coef_[0])
or_list = list(zip(feature_names_clf, odds))
or_list.sort(key=lambda x: abs(x[1] - 1), reverse=True)
for feat, or_val in or_list[:5]:
    direction = "↑" if or_val > 1 else "↓"
    print(f"   {feat}: OR={or_val:.4f} ({direction})")

# MODEL 2: Decision Tree
print("\n" + "-"*80)
print("MODEL 2: DECISION TREE CLASSIFIER")
print("-"*80)

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
tree_clf.fit(X_train_c, y_train_c)
tree_clf_preds = tree_clf.predict(X_test_c)

acc = accuracy_score(y_test_c, tree_clf_preds)
prec = precision_score(y_test_c, tree_clf_preds)
rec = recall_score(y_test_c, tree_clf_preds)
f1 = f1_score(y_test_c, tree_clf_preds)

print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

print("\n📋 DECISION RULES:")
print(export_text(tree_clf, feature_names=feature_names_clf))

# MODEL 3: Naive Bayes
print("\n" + "-"*80)
print("MODEL 3: GAUSSIAN NAIVE BAYES")
print("-"*80)

nb_model = GaussianNB()
nb_model.fit(X_train_c, y_train_c)
nb_preds = nb_model.predict(X_test_c)

acc = accuracy_score(y_test_c, nb_preds)
prec = precision_score(y_test_c, nb_preds)
rec = recall_score(y_test_c, nb_preds)
f1 = f1_score(y_test_c, nb_preds)

print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print(f"\n📐 CLASS PRIORS: P(surge)={nb_model.class_prior_[1]:.4f}")

# MODEL 4: LDA
print("\n" + "-"*80)
print("MODEL 4: LINEAR DISCRIMINANT ANALYSIS")
print("-"*80)

lda_model = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
lda_model.fit(X_train_c, y_train_c)
lda_preds = lda_model.predict(X_test_c)

acc = accuracy_score(y_test_c, lda_preds)
prec = precision_score(y_test_c, lda_preds)
rec = recall_score(y_test_c, lda_preds)
f1 = f1_score(y_test_c, lda_preds)

print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

print("\n📐 LINEAR DISCRIMINANT:")
print(f"{lda_model.intercept_[0]:.4f}", end="")
for coef, feat in zip(lda_model.coef_[0], feature_names_clf):
    sign = " + " if coef >= 0 else " - "
    print(f"{sign}{abs(coef):.4f}×{feat}", end="")
print(" = 0")

# MODEL 5: Perceptron
print("\n" + "-"*80)
print("MODEL 5: PERCEPTRON")
print("-"*80)

perceptron = Perceptron(max_iter=1000, random_state=42, class_weight='balanced')
perceptron.fit(X_train_c, y_train_c)
perc_preds = perceptron.predict(X_test_c)

acc = accuracy_score(y_test_c, perc_preds)
prec = precision_score(y_test_c, perc_preds)
rec = recall_score(y_test_c, perc_preds)
f1 = f1_score(y_test_c, perc_preds)

print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

print("\n📐 PERCEPTRON EQUATION:")
print("IF (", end="")
print(f"{perceptron.intercept_[0]:.4f}", end="")
for coef, feat in zip(perceptron.coef_[0], feature_names_clf):
    sign = " + " if coef >= 0 else " - "
    print(f"{sign}{abs(coef):.4f}×{feat}", end="")
print(" ) > 0")
print("   THEN surge")
print("   ELSE no surge")

# CLASSIFICATION SUMMARY
print("\n" + "="*80)
print("CLASSIFICATION SUMMARY")
print("="*80)

# Recalculate all metrics for summary
models = [
    ('Logistic Reg', log_reg),
    ('Decision Tree', tree_clf),
    ('Naive Bayes', nb_model),
    ('LDA', lda_model),
    ('Perceptron', perceptron)
]

print("\n┌─────────────────┬────────┬────────┬────────┬────────┐")
print("│ Model           │ Acc    │ Prec   │ Rec    │ F1     │")
print("├─────────────────┼────────┼────────┼────────┼────────┤")

for name, model in models:
    if name == 'Logistic Reg':
        preds = log_preds
    elif name == 'Decision Tree':
        preds = tree_clf_preds
    elif name == 'Naive Bayes':
        preds = nb_preds
    elif name == 'LDA':
        preds = lda_preds
    else:
        preds = perc_preds
    
    acc = accuracy_score(y_test_c, preds)
    prec = precision_score(y_test_c, preds)
    rec = recall_score(y_test_c, preds)
    f1 = f1_score(y_test_c, preds)
    
    print(f"│ {name:<15} │ {acc:.4f} │ {prec:.4f} │ {rec:.4f} │ {f1:.4f} │")

print("└─────────────────┴────────┴────────┴────────┴────────┘")

print("\n" + "="*80)
print("COMPLETE - All models trained on prepared datasets!")
print("="*80)
