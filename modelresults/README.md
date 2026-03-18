# Model Results Summary

This folder contains merged datasets and training scripts used for leakage-safe modeling experiments.

## Files

- `final_regression_data.csv`
- `final_classification_data.csv`
- `merge_missing_features.py`
- `train_customs_models.py`
- `baseline_xgb_delay_classification.py`

## Data Validity Check (Merged Files)

Both final merged datasets were validated after fixing merge alignment:

- Shape: `10000 x 51` (both)
- Engineered columns added: `26` (`scaled_*` + `delay_tfidf_*`)
- Engineered columns fully non-null: **Yes**
- Duplicate column names: **No**

## Leakage-Safe Modeling Setup

### Regression Target

- Target: `Arrival_Delay_Days` (fallback: `Customs_Delay_Days`)
- Removed from features: `Route_Risk_Index`, `Customs_Delay_Days`, `Actual_Transit_Days`

### Classification Target

- New binary target: `Is_Delayed = 1 if Customs_Delay_Days > 0 else 0`
- Removed from features:
  - `Route_Risk_Level`
  - `Route_Risk_Index`
  - `Customs_Delay_Days`
  - `Arrival_Delay_Days` (if present)
  - `Actual_Transit_Days`

## Latest Metrics (after merge fix + leakage removal)

### Regression

- **RandomForestRegressor**
  - MAE: `1.8546`
  - RMSE: `2.2468`
  - R2: `-0.0055`

- **XGBRegressor**
  - MAE: `1.8748`
  - RMSE: `2.3083`
  - R2: `-0.0613`

### Classification (`Is_Delayed`)

- **LogisticRegression**
  - Accuracy: `0.6305`
  - Macro F1: `0.3867`
  - Weighted F1: `0.4876`

- **RandomForestClassifier**
  - Accuracy: `0.6235`
  - Macro F1: `0.4010`
  - Weighted F1: `0.4963`

- **XGBClassifier**
  - Accuracy: `0.6005`
  - Macro F1: `0.4819`
  - Weighted F1: `0.5466`

## Top Features (Best Classification Model: XGBClassifier)

1. `delay_tfidf_customs`
2. `delay_tfidf_pending`
3. `scaled_Destination_LPI_Overall`
4. `scaled_Route_LPI_Average`
5. `scaled_Route_LPI_Difference`

## Notes

- Perfect/near-perfect earlier scores were caused by leakage proxies.
- Current scores are leakage-safe and more realistic.
