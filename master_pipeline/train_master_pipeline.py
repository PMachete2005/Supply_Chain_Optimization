#!/usr/bin/env python3
"""
train_master_pipeline.py
========================
ML training pipeline using the preprocessed DataCo datasets from src/data/.
  - Task 1  Classification: Predict Late_delivery_risk
  - Task 2  Regression:     Predict Days for shipping (real)

Data is already cleaned, encoded, and leak-free (see src/data_preprocessing.py).
Results are saved in the same format as outputs/ (CSV tables + joblib models).
"""

import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLF_CSV = os.path.join(BASE_DIR, "src", "data", "classification_dataset.csv")
REG_CSV = os.path.join(BASE_DIR, "src", "data", "regression_dataset.csv")
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PIPELINE_DIR, "models")


def prepare_features(X, *, label_encoders=None, imputer=None, fit=True):
    """LabelEncode any remaining object columns, then median-impute NaNs."""
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if label_encoders is None:
        label_encoders = {}

    for col in cat_cols:
        X[col] = X[col].astype(str)
        if fit:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is not None:
                classes_set = set(le.classes_)
                X[col] = X[col].map(
                    lambda v, _le=le, _cs=classes_set: (
                        _le.transform([v])[0] if v in _cs else -1
                    )
                )
            else:
                X[col] = -1

    if fit:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        arr = imputer.fit_transform(X)
    else:
        arr = imputer.transform(X)

    return pd.DataFrame(arr, columns=X.columns, index=X.index), label_encoders, imputer


def main() -> None:
    print("=" * 70)
    print("  MASTER ML PIPELINE — DataCo Supply Chain (preprocessed data)")
    print("=" * 70)

    # ── 1. Load preprocessed datasets ─────────────────────────────────────
    print("\n[STEP 1] Loading preprocessed datasets …")
    clf_df = pd.read_csv(CLF_CSV)
    reg_df = pd.read_csv(REG_CSV)
    print(f"  Classification dataset: {clf_df.shape}")
    print(f"  Regression dataset:     {reg_df.shape}")

    # ── 2. Classification: Late_delivery_risk ─────────────────────────────
    print("\n" + "=" * 70)
    print("  TASK 1 — Classification: Late_delivery_risk")
    print("=" * 70)

    cls_target = "Late_delivery_risk"
    y_cls = clf_df[cls_target].copy()
    X_cls = clf_df.drop(columns=[cls_target]).copy()
    print(f"  Features: {X_cls.shape[1]}  |  Classes: {sorted(y_cls.unique())}")

    X_cls_enc, cls_encoders, cls_imputer = prepare_features(X_cls, fit=True)
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_cls_enc, y_cls, test_size=0.2, random_state=42, stratify=y_cls,
    )

    clf_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        ),
    }

    clf_rows = []

    for name, model in clf_models.items():
        print(f"\n  Training {name} …")
        t0 = time.time()
        model.fit(X_tr_c, y_tr_c)
        train_sec = round(time.time() - t0, 3)
        y_pred = model.predict(X_te_c)

        acc = accuracy_score(y_te_c, y_pred)
        prec = precision_score(y_te_c, y_pred, average="weighted")
        rec = recall_score(y_te_c, y_pred, average="weighted")
        f1 = f1_score(y_te_c, y_pred, average="macro")

        print(f"    Accuracy : {acc:.4f}  |  Precision : {prec:.4f}  |  Recall : {rec:.4f}  |  F1 : {f1:.4f}")

        print(f"    Running 5-fold CV …")
        cv = cross_val_score(model, X_cls_enc, y_cls, cv=5, scoring="accuracy", n_jobs=-1)
        print(f"    CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

        clf_rows.append({
            "Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1,
            "CV_Acc_mean": cv.mean(), "CV_Acc_std": cv.std(),
            "Train_sec": train_sec, "model_obj": model,
        })

        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=X_cls_enc.columns).nlargest(5)
            print("    Top 5 Feature Importances:")
            for feat, val in imp.items():
                print(f"      {feat:40s} {val:.4f}")

    # Rank by accuracy descending
    clf_rows.sort(key=lambda r: r["Accuracy"], reverse=True)
    for i, row in enumerate(clf_rows, 1):
        row["Rank"] = i

    # Print comparison table
    print(f"\n  ┌─── Classification Comparison ─────────────────────────────────────────────────────────┐")
    print(f"  │ Rank  {'Model':<20s} {'Accuracy':>8s} {'Precision':>9s} {'Recall':>8s} {'F1':>8s} {'CV_Acc':>8s} {'Time':>6s} │")
    print(f"  │ {'─'*4}  {'─'*20} {'─'*8} {'─'*9} {'─'*8} {'─'*8} {'─'*8} {'─'*6} │")
    for row in clf_rows:
        print(f"  │ {row['Rank']:>4d}  {row['Model']:<20s} {row['Accuracy']:>8.4f} {row['Precision']:>9.4f} "
              f"{row['Recall']:>8.4f} {row['F1']:>8.4f} {row['CV_Acc_mean']:>8.4f} {row['Train_sec']:>5.1f}s │")
    print(f"  └─────────────────────────────────────────────────────────────────────────────────────────┘")
    print(f"  Best: {clf_rows[0]['Model']} (Accuracy = {clf_rows[0]['Accuracy']:.4f})")

    # ── 3. Regression: Days for shipping (real) ───────────────────────────
    print("\n" + "=" * 70)
    print("  TASK 2 — Regression: Days for shipping (real)")
    print("=" * 70)

    reg_target = "Days for shipping (real)"
    y_reg = reg_df[reg_target].copy()
    X_reg = reg_df.drop(columns=[reg_target]).copy()
    sched_kept = "Days for shipment (scheduled)" in X_reg.columns
    print(f"  Features: {X_reg.shape[1]}  |  'Days for shipment (scheduled)' kept: {sched_kept}")

    X_reg_enc, reg_encoders, reg_imputer = prepare_features(X_reg, fit=True)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg_enc, y_reg, test_size=0.2, random_state=42,
    )

    reg_models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1,
        ),
    }

    reg_rows = []

    for name, model in reg_models.items():
        print(f"\n  Training {name} …")
        t0 = time.time()
        model.fit(X_tr_r, y_tr_r)
        train_sec = round(time.time() - t0, 3)
        y_pred = model.predict(X_te_r)

        mae = mean_absolute_error(y_te_r, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te_r, y_pred))
        r2 = r2_score(y_te_r, y_pred)

        print(f"    MAE : {mae:.4f}  |  RMSE : {rmse:.4f}  |  R² : {r2:.4f}")

        print(f"    Running 5-fold CV …")
        cv = cross_val_score(model, X_reg_enc, y_reg, cv=5, scoring="r2", n_jobs=-1)
        print(f"    CV R²: {cv.mean():.4f} ± {cv.std():.4f}")

        reg_rows.append({
            "Model": name, "MAE": mae, "RMSE": rmse, "R²": r2,
            "CV_R²_mean": cv.mean(), "CV_R²_std": cv.std(),
            "Train_sec": train_sec, "model_obj": model,
        })

        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=X_reg_enc.columns).nlargest(5)
            print("    Top 5 Feature Importances:")
            for feat, val in imp.items():
                print(f"      {feat:40s} {val:.4f}")

    # Rank by R² descending
    reg_rows.sort(key=lambda r: r["R²"], reverse=True)
    for i, row in enumerate(reg_rows, 1):
        row["Rank"] = i

    # Print comparison table
    print(f"\n  ┌─── Regression Comparison ──────────────────────────────────────────────────┐")
    print(f"  │ Rank  {'Model':<20s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'CV_R²':>8s} {'Time':>6s} │")
    print(f"  │ {'─'*4}  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} │")
    for row in reg_rows:
        print(f"  │ {row['Rank']:>4d}  {row['Model']:<20s} {row['MAE']:>8.4f} {row['RMSE']:>8.4f} "
              f"{row['R²']:>8.4f} {row['CV_R²_mean']:>8.4f} {row['Train_sec']:>5.1f}s │")
    print(f"  └─────────────────────────────────────────────────────────────────────────────┘")
    print(f"  Best: {reg_rows[0]['Model']} (R² = {reg_rows[0]['R²']:.4f})")

    # ── 4. Save Artifacts ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Saving Artifacts")
    print("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save classification results CSV
    clf_csv_rows = []
    for row in clf_rows:
        clf_csv_rows.append({
            "Rank": row["Rank"], "Model": row["Model"],
            "Accuracy": row["Accuracy"], "Precision": row["Precision"],
            "Recall": row["Recall"], "F1": row["F1"],
            "CV_Acc_mean": row["CV_Acc_mean"], "CV_Acc_std": row["CV_Acc_std"],
            "Train_sec": row["Train_sec"],
        })
    clf_results_df = pd.DataFrame(clf_csv_rows)
    clf_results_path = os.path.join(PIPELINE_DIR, "classification_results.csv")
    clf_results_df.to_csv(clf_results_path, index=False)
    print(f"  Saved {clf_results_path}")

    # Save regression results CSV
    reg_csv_rows = []
    for row in reg_rows:
        reg_csv_rows.append({
            "Rank": row["Rank"], "Model": row["Model"],
            "MAE": row["MAE"], "RMSE": row["RMSE"], "R²": row["R²"],
            "CV_R²_mean": row["CV_R²_mean"], "CV_R²_std": row["CV_R²_std"],
            "Train_sec": row["Train_sec"],
        })
    reg_results_df = pd.DataFrame(reg_csv_rows)
    reg_results_path = os.path.join(PIPELINE_DIR, "regression_results.csv")
    reg_results_df.to_csv(reg_results_path, index=False)
    print(f"  Saved {reg_results_path}")

    # Save all classification models
    for row in clf_rows:
        fname = f"clf_{row['Model'].lower().replace(' ', '_')}.joblib"
        joblib.dump(row["model_obj"], os.path.join(MODELS_DIR, fname))
    joblib.dump(
        {"encoders": cls_encoders, "imputer": cls_imputer},
        os.path.join(MODELS_DIR, "classification_preprocessor.joblib"),
    )

    # Save all regression models
    for row in reg_rows:
        fname = f"reg_{row['Model'].lower().replace(' ', '_')}.joblib"
        joblib.dump(row["model_obj"], os.path.join(MODELS_DIR, fname))
    joblib.dump(
        {"encoders": reg_encoders, "imputer": reg_imputer},
        os.path.join(MODELS_DIR, "regression_preprocessor.joblib"),
    )

    print(f"\n  Models saved to {MODELS_DIR}/:")
    for f in sorted(os.listdir(MODELS_DIR)):
        size_mb = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024 / 1024
        print(f"    {f:45s} {size_mb:.1f} MB")

    print(f"\n  Best classification model: {clf_rows[0]['Model']} (Accuracy = {clf_rows[0]['Accuracy']:.4f})")
    print(f"  Best regression model:     {reg_rows[0]['Model']} (R² = {reg_rows[0]['R²']:.4f})")
    print("\n✅  Pipeline complete.")


if __name__ == "__main__":
    main()
