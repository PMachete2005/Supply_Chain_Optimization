import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


try:
    from xgboost import XGBRegressor, XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def get_feature_names_from_pipeline(
    pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    preprocessor = pipeline.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()

    cleaned = []
    for n in names:
        if n.startswith("num__"):
            cleaned.append(n.replace("num__", "", 1))
        elif n.startswith("cat__"):
            cleaned.append(n.replace("cat__", "", 1))
        else:
            cleaned.append(n)
    return np.array(cleaned)


def print_top_feature_importances(
    model_name: str,
    fitted_pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
    top_k: int = 10,
) -> None:
    model = fitted_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        print(f"\n{model_name} does not expose feature_importances_.")
        return

    feature_names = get_feature_names_from_pipeline(
        fitted_pipeline, numeric_features, categorical_features
    )
    importances = model.feature_importances_

    if len(importances) != len(feature_names):
        print(
            f"\nCould not align feature importances for {model_name} "
            f"(importances={len(importances)}, features={len(feature_names)})."
        )
        return

    top_idx = np.argsort(importances)[::-1][:top_k]

    print(f"\nTop {top_k} features from {model_name}:")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"{rank:>2}. {feature_names[idx]}: {importances[idx]:.6f}")


def run_regression_task(regression_csv: Path, random_state: int = 42) -> None:
    print("\n" + "=" * 80)
    print("REGRESSION TASK")
    print("=" * 80)

    df = pd.read_csv(regression_csv)

    target_candidates = ["Arrival_Delay_Days", "Customs_Delay_Days"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise KeyError(
            f"No valid regression target found. Expected one of: {target_candidates}"
        )

    y = pd.to_numeric(df[target_col], errors="coerce")
    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid]

    leakage_cols = [
        "Route_Risk_Level",
        "Route_Risk_Index",
        "Customs_Delay_Days",
        "Actual_Transit_Days",
    ]
    drop_cols = [c for c in target_candidates if c in df.columns]
    drop_cols.extend([c for c in leakage_cols if c in df.columns and c != target_col])
    print(
        f"Dropping leakage-prone columns for regression: "
        f"{[c for c in leakage_cols if c in df.columns and c != target_col]}"
    )
    X = df.drop(columns=drop_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    models: dict[str, Pipeline] = {
        "RandomForestRegressor": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    }

    if XGBOOST_AVAILABLE:
        models["XGBRegressor"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=8,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        print("xgboost is not installed. Skipping XGBRegressor.")

    results = {}
    fitted = {}

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        fitted[name] = pipe

        print(f"\n{name}:")
        print(f"  MAE : {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2  : {r2:.4f}")

    best_name = max(results, key=lambda m: results[m]["R2"])
    print(f"\nBest regression model by R2: {best_name}")

    print_top_feature_importances(
        best_name,
        fitted[best_name],
        numeric_features=num_cols,
        categorical_features=cat_cols,
        top_k=10,
    )


def run_classification_task(classification_csv: Path, random_state: int = 42) -> None:
    print("\n" + "=" * 80)
    print("CLASSIFICATION TASK")
    print("=" * 80)

    df = pd.read_csv(classification_csv)
    delay_source_col = "Customs_Delay_Days"
    if delay_source_col not in df.columns:
        raise KeyError(f"'{delay_source_col}' not found in classification dataset.")

    df["Is_Delayed"] = (pd.to_numeric(df[delay_source_col], errors="coerce").fillna(0) > 0).astype(int)

    target_col = "Is_Delayed"
    y = df[target_col].astype(str)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    leakage_cols = [
        "Route_Risk_Level",
        "Route_Risk_Index",
        "Customs_Delay_Days",
        "Arrival_Delay_Days",
        "Actual_Transit_Days",
    ]
    drop_cols = [target_col] + [c for c in leakage_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    print(f"Using binary target: {target_col} (1 if Customs_Delay_Days > 0 else 0)")
    print(f"Dropping leakage-prone columns for classification: {[c for c in leakage_cols if c in df.columns]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    _, _, y_train_enc, y_test_enc = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    models: dict[str, Pipeline] = {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if XGBOOST_AVAILABLE:
        n_classes = int(len(np.unique(y_train_enc)))
        xgb_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
        if n_classes == 2:
            xgb_params["objective"] = "binary:logistic"
        else:
            xgb_params["objective"] = "multi:softprob"
            xgb_params["num_class"] = n_classes

        models["XGBClassifier"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(**xgb_params),
                ),
            ]
        )
    else:
        print("xgboost is not installed. Skipping XGBClassifier.")

    results = {}
    fitted = {}

    for name, pipe in models.items():
        if name == "XGBClassifier":
            pipe.fit(X_train, y_train_enc)
            preds_encoded = pipe.predict(X_test)
            preds = label_encoder.inverse_transform(preds_encoded.astype(int))
            y_eval = y_test
        else:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            y_eval = y_test

        acc = accuracy_score(y_eval, preds)
        f1_macro = f1_score(y_eval, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(y_eval, preds, average="weighted", zero_division=0)
        results[name] = {"accuracy": acc}
        results[name]["f1_macro"] = f1_macro
        results[name]["f1_weighted"] = f1_weighted
        fitted[name] = pipe

        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print("  Classification report:")
        print(classification_report(y_eval, preds, digits=4, zero_division=0))

    importance_candidates = [
        m for m in ["XGBClassifier", "RandomForestClassifier"] if m in fitted
    ]
    best_for_importance = max(
        importance_candidates,
        key=lambda m: results[m]["f1_weighted"],
    )

    print(
        f"Best tree model for feature importance (by weighted F1): {best_for_importance}"
    )

    print_top_feature_importances(
        best_for_importance,
        fitted[best_for_importance],
        numeric_features=num_cols,
        categorical_features=cat_cols,
        top_k=5,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate regression + classification pipelines for customs data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing final_regression_data.csv and final_classification_data.csv",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    regression_csv = data_dir / "final_regression_data.csv"
    classification_csv = data_dir / "final_classification_data.csv"

    if not regression_csv.exists():
        raise FileNotFoundError(f"Missing file: {regression_csv}")
    if not classification_csv.exists():
        raise FileNotFoundError(f"Missing file: {classification_csv}")

    run_regression_task(regression_csv, random_state=args.random_state)
    run_classification_task(classification_csv, random_state=args.random_state)


if __name__ == "__main__":
    main()
