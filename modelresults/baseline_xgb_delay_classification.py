from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def get_clean_feature_names(pipe: Pipeline) -> np.ndarray:
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    cleaned = []
    for n in names:
        if n.startswith("num__"):
            cleaned.append(n.replace("num__", "", 1))
        elif n.startswith("cat__"):
            cleaned.append(n.replace("cat__", "", 1))
        else:
            cleaned.append(n)
    return np.array(cleaned)


def main() -> None:
    data_path = Path("/home/kushagarwal/Downloads/Datasetsdsci/customs_classification_dataset(3).csv")
    df = pd.read_csv(data_path)

    if "Customs_Delay_Days" not in df.columns:
        raise KeyError("'Customs_Delay_Days' not found in dataset.")

    # Binary target
    df["Is_Delayed"] = (pd.to_numeric(df["Customs_Delay_Days"], errors="coerce").fillna(0) > 0).astype(int)
    y = df["Is_Delayed"]

    # Drop leakage / proxy columns from features
    drop_cols = [
        "Is_Delayed",
        "Route_Risk_Level",
        "Route_Risk_Index",
        "Customs_Delay_Days",
        "Arrival_Delay_Days",
        "Actual_Transit_Days",
    ]
    present_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=present_drop_cols)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    num_cols = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, preds, average="weighted", zero_division=0)

    print("Baseline XGBClassifier (original 25-column classification dataset)")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")

    importances = pipe.named_steps["model"].feature_importances_
    feature_names = get_clean_feature_names(pipe)

    if len(importances) == len(feature_names):
        top_idx = np.argsort(importances)[::-1][:5]
        print("Top 5 feature importances:")
        for i, idx in enumerate(top_idx, start=1):
            print(f"{i}. {feature_names[idx]}: {importances[idx]:.6f}")
    else:
        print("Could not align feature names with importances.")


if __name__ == "__main__":
    main()
