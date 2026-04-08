"""
Run 5 regression and 5 classification models on enriched datasets
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import statsmodels.api as sm

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = '../new_data/processed'
REG_PATH = f'{DATA_DIR}/regression_dataset_enriched.csv'
CLF_PATH = f'{DATA_DIR}/classification_dataset_enriched.csv'


def print_confusion(cm, labels=('standard', 'premium')):
    tn, fp, fn, tp = cm.ravel()
    print(f"  {'':12} Pred {labels[0]:>10}  Pred {labels[1]:>8}")
    print(f"  Act {labels[0]:>8}   {tn:>8}        {fp:>8}")
    print(f"  Act {labels[1]:>8}   {fn:>8}        {tp:>8}")
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  Specificity: {spec:.4f}   TP={tp}  TN={tn}  FP={fp}  FN={fn}")


print("=" * 72)
print("Running models on enriched datasets")
print("=" * 72)

reg_df = pd.read_csv(REG_PATH)
clf_df = pd.read_csv(CLF_PATH)

print(f"\nRegression:     {reg_df.shape[0]:,} rows x {reg_df.shape[1]-1} features   target: price")
print(f"Classification: {clf_df.shape[0]:,} rows x {clf_df.shape[1]-1} features   target: is_premium")

# Regression
print("\n" + "=" * 72)
print("REGRESSION - 5 MODELS")
print("=" * 72)

X_reg = reg_df.drop('price', axis=1)
y_reg = reg_df['price']
feature_names_reg = X_reg.columns.tolist()

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

scaler_r = StandardScaler()
X_train_rs = scaler_r.fit_transform(X_train_r)
X_test_rs = scaler_r.transform(X_test_r)

print(f"\nTrain: {len(X_train_r):,}   Test: {len(X_test_r):,}   Features: {len(feature_names_reg)}")
print(f"Price range: ${y_reg.min():.2f} - ${y_reg.max():.2f}\n")

reg_results = {}


def reg_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return r2, mae, mse, rmse


# 1. OLS
print("-" * 72)
print("MODEL 1: OLS")
X_ols_train = sm.add_constant(X_train_rs)
X_ols_test = sm.add_constant(X_test_rs)
ols = sm.OLS(y_train_r, X_ols_train).fit()
ols_preds = ols.predict(X_ols_test)
r2, mae, mse, rmse = reg_metrics(y_test_r, ols_preds)
reg_results['OLS'] = dict(R2=r2, MAE=mae, MSE=mse, RMSE=rmse)
print(f"R2={r2:.4f}  MAE=${mae:.2f}  MSE={mse:.2f}  RMSE=${rmse:.2f}")
coefs = ols.params.drop('const', errors='ignore')
top5 = coefs.abs().nlargest(5)
print("Top 5 features:")
for feat in top5.index:
    idx = list(coefs.index).index(feat)
    print(f"  {feature_names_reg[idx]:<28} {coefs[feat]:+.4f}")

# 2. Ridge
print("\n" + "-" * 72)
print("MODEL 2: Ridge (alpha=1.0)")
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_rs, y_train_r)
ridge_preds = ridge.predict(X_test_rs)
r2, mae, mse, rmse = reg_metrics(y_test_r, ridge_preds)
reg_results['Ridge'] = dict(R2=r2, MAE=mae, MSE=mse, RMSE=rmse)
print(f"R2={r2:.4f}  MAE=${mae:.2f}  MSE={mse:.2f}  RMSE=${rmse:.2f}")
top5 = np.argsort(np.abs(ridge.coef_))[-5:][::-1]
print("Top 5 features:")
for i in top5:
    print(f"  {feature_names_reg[i]:<28} {ridge.coef_[i]:+.4f}")

# 3. Lasso
print("\n" + "-" * 72)
print("MODEL 3: Lasso (alpha=0.1)")
lasso = Lasso(alpha=0.1, random_state=42, max_iter=10000)
lasso.fit(X_train_rs, y_train_r)
lasso_preds = lasso.predict(X_test_rs)
r2, mae, mse, rmse = reg_metrics(y_test_r, lasso_preds)
reg_results['Lasso'] = dict(R2=r2, MAE=mae, MSE=mse, RMSE=rmse)
print(f"R2={r2:.4f}  MAE=${mae:.2f}  MSE={mse:.2f}  RMSE=${rmse:.2f}")
selected = [(feature_names_reg[i], lasso.coef_[i]) for i in range(len(lasso.coef_)) if abs(lasso.coef_[i]) > 0.001]
selected.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"Selected {len(selected)}/{len(feature_names_reg)} features:")
for feat, coef in selected[:5]:
    print(f"  {feat:<28} {coef:+.4f}")

# 4. Decision Tree
print("\n" + "-" * 72)
print("MODEL 4: Decision Tree (depth=10)")
tree_r = DecisionTreeRegressor(max_depth=10, random_state=42)
tree_r.fit(X_train_r, y_train_r)
tree_r_preds = tree_r.predict(X_test_r)
r2, mae, mse, rmse = reg_metrics(y_test_r, tree_r_preds)
reg_results['Decision Tree'] = dict(R2=r2, MAE=mae, MSE=mse, RMSE=rmse)
print(f"R2={r2:.4f}  MAE=${mae:.2f}  MSE={mse:.2f}  RMSE=${rmse:.2f}")
top5 = np.argsort(tree_r.feature_importances_)[-5:][::-1]
print("Top 5 features:")
for i in top5:
    print(f"  {feature_names_reg[i]:<28} {tree_r.feature_importances_[i]:.4f}")

# 5. Polynomial
print("\n" + "-" * 72)
print("MODEL 5: Polynomial Regression (degree=2)")
top5_feats = [feature_names_reg[i] for i in np.argsort(np.abs(ridge.coef_))[-5:][::-1]]
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_r[top5_feats])
X_test_poly = poly.transform(X_test_r[top5_feats])
poly_model = Ridge(alpha=1.0, random_state=42)
poly_model.fit(X_train_poly, y_train_r)
poly_preds = poly_model.predict(X_test_poly)
r2, mae, mse, rmse = reg_metrics(y_test_r, poly_preds)
reg_results['Polynomial'] = dict(R2=r2, MAE=mae, MSE=mse, RMSE=rmse)
print(f"R2={r2:.4f}  MAE=${mae:.2f}  MSE={mse:.2f}  RMSE=${rmse:.2f}")
print(f"Features: {', '.join(top5_feats)}  ({len(top5_feats)} -> {X_train_poly.shape[1]} terms)")

# Regression Summary
print("\n" + "=" * 72)
print("REGRESSION SUMMARY")
print("=" * 72)
print(f"{'Model':<18} {'R2':>7}  {'MAE':>7}  {'MSE':>8}  {'RMSE':>7}")
print("-" * 62)
for name, m in sorted(reg_results.items(), key=lambda x: x[1]['R2'], reverse=True):
    print(f"{name:<18} {m['R2']:>7.4f}  ${m['MAE']:>6.2f}  {m['MSE']:>8.2f}  ${m['RMSE']:>6.2f}")
best_reg = max(reg_results.items(), key=lambda x: x[1]['R2'])
print(f"\nBest: {best_reg[0]}   R2={best_reg[1]['R2']:.4f}  MAE=${best_reg[1]['MAE']:.2f}")

# Classification
print("\n" + "=" * 72)
print("CLASSIFICATION - 5 MODELS")
print("=" * 72)

X_clf = clf_df.drop('is_premium', axis=1)
y_clf = clf_df['is_premium']
feature_names_clf = X_clf.columns.tolist()

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

scaler_c = StandardScaler()
X_train_cs = scaler_c.fit_transform(X_train_c)
X_test_cs = scaler_c.transform(X_test_c)

print(f"\nTrain: {len(X_train_c):,}   Test: {len(X_test_c):,}   Features: {len(feature_names_clf)}")
print(f"Premium rate: {y_clf.mean()*100:.1f}%\n")

clf_results = {}


def clf_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, prec, rec, cm


# 1. Logistic Regression
print("-" * 72)
print("MODEL 1: Logistic Regression")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_cs, y_train_c)
log_preds = log_reg.predict(X_test_cs)
acc, f1, prec, rec, cm = clf_metrics(y_test_c, log_preds)
clf_results['Logistic Reg'] = dict(Acc=acc, F1=f1, Prec=prec, Rec=rec)
print(f"Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
print_confusion(cm)
top5 = np.argsort(np.abs(log_reg.coef_[0]))[-5:][::-1]
print("Top 5 features (odds ratio):")
for i in top5:
    or_val = np.exp(log_reg.coef_[0][i])
    print(f"  {feature_names_clf[i]:<28} OR={or_val:.4f} {'up' if or_val > 1 else 'dn'}")

# 2. Decision Tree
print("\n" + "-" * 72)
print("MODEL 2: Decision Tree (depth=10)")
tree_c = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_c.fit(X_train_c, y_train_c)
tree_c_preds = tree_c.predict(X_test_c)
acc, f1, prec, rec, cm = clf_metrics(y_test_c, tree_c_preds)
clf_results['Decision Tree'] = dict(Acc=acc, F1=f1, Prec=prec, Rec=rec)
print(f"Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
print_confusion(cm)
top5 = np.argsort(tree_c.feature_importances_)[-5:][::-1]
print("Top 5 features:")
for i in top5:
    print(f"  {feature_names_clf[i]:<28} {tree_c.feature_importances_[i]:.4f}")

# 3. Naive Bayes
print("\n" + "-" * 72)
print("MODEL 3: Naive Bayes")
nb = GaussianNB()
nb.fit(X_train_cs, y_train_c)
nb_preds = nb.predict(X_test_cs)
acc, f1, prec, rec, cm = clf_metrics(y_test_c, nb_preds)
clf_results['Naive Bayes'] = dict(Acc=acc, F1=f1, Prec=prec, Rec=rec)
print(f"Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
print_confusion(cm)
print(f"Priors: P(standard)={nb.class_prior_[0]:.3f}  P(premium)={nb.class_prior_[1]:.3f}")

# 4. LDA
print("\n" + "-" * 72)
print("MODEL 4: LDA")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_cs, y_train_c)
lda_preds = lda.predict(X_test_cs)
acc, f1, prec, rec, cm = clf_metrics(y_test_c, lda_preds)
clf_results['LDA'] = dict(Acc=acc, F1=f1, Prec=prec, Rec=rec)
print(f"Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
print_confusion(cm)
top5 = np.argsort(np.abs(lda.coef_[0]))[-5:][::-1]
print("Top 5 features:")
for i in top5:
    print(f"  {feature_names_clf[i]:<28} {lda.coef_[0][i]:+.4f}")

# 5. Perceptron
print("\n" + "-" * 72)
print("MODEL 5: Perceptron")
perc = Perceptron(max_iter=1000, random_state=42)
perc.fit(X_train_cs, y_train_c)
perc_preds = perc.predict(X_test_cs)
acc, f1, prec, rec, cm = clf_metrics(y_test_c, perc_preds)
clf_results['Perceptron'] = dict(Acc=acc, F1=f1, Prec=prec, Rec=rec)
print(f"Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
print_confusion(cm)

# Classification Summary
print("\n" + "=" * 72)
print("CLASSIFICATION SUMMARY")
print("=" * 72)
print(f"{'Model':<18} {'Acc':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}")
print("-" * 50)
for name, m in sorted(clf_results.items(), key=lambda x: x[1]['F1'], reverse=True):
    print(f"{name:<18} {m['Acc']:>7.4f}  {m['F1']:>7.4f}  {m['Prec']:>7.4f}  {m['Rec']:>7.4f}")
best_clf = max(clf_results.items(), key=lambda x: x[1]['F1'])
print(f"\nBest: {best_clf[0]}   F1={best_clf[1]['F1']:.4f}  Acc={best_clf[1]['Acc']:.4f}")

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
