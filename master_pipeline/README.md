# Master ML Pipeline — DataCo Supply Chain

End-to-end machine learning pipeline for the DataCo Supply Chain dataset.  
Predicts **late delivery risk** (classification) and **actual shipping days** (regression) using leak-free feature sets.

## Data

Preprocessed datasets from `src/data/` (produced by `src/data_preprocessing.py`):

| Dataset | Target | Rows | Features |
|---------|--------|------|----------|
| `classification_dataset.csv` | `Late_delivery_risk` (0/1) | 180,519 | 57 |
| `regression_dataset.csv` | `Days for shipping (real)` | 180,519 | 57 |

**Leakage columns removed:** `Order Status`, `Delivery Status`, `shipping date (DateOrders)`, plus cross-task targets.  
**Legitimate feature kept:** `Days for shipment (scheduled)`.

## Results

### Classification — Late Delivery Risk

| Rank | Model | Accuracy | Precision | Recall | F1 | CV Accuracy (5-fold) | Train Time |
|------|-------|----------|-----------|--------|-----|----------------------|------------|
| 1 | **Random Forest** | **0.7491** | 0.7796 | 0.7491 | 0.7485 | 0.6852 ± 0.0428 | 15.4s |
| 2 | Gradient Boosting | 0.7414 | 0.7693 | 0.7414 | 0.7408 | 0.6876 ± 0.0444 | 286.2s |
| 3 | XGBoost | 0.7407 | 0.7682 | 0.7407 | 0.7402 | 0.6836 ± 0.0432 | 7.1s |

**Top 5 features (Random Forest):**
1. Days for shipment (scheduled) — 0.1886
2. Shipping Mode: Standard Class — 0.1465
3. Order Hour — 0.0674
4. Shipping Mode: Same Day — 0.0373
5. Latitude — 0.0364

### Regression — Actual Shipping Days

| Rank | Model | MAE | RMSE | R² | CV R² (5-fold) | Train Time |
|------|-------|-----|------|----|----------------|------------|
| 1 | **Random Forest** | **0.8484** | **1.1213** | **0.5230** | 0.4823 ± 0.0391 | 151.8s |
| 2 | XGBoost | 0.9395 | 1.2123 | 0.4425 | 0.4160 ± 0.0407 | 8.7s |
| 3 | Gradient Boosting | 0.9367 | 1.2127 | 0.4421 | 0.4176 ± 0.0432 | 363.1s |

**Top 5 features (Random Forest):**
1. Days for shipment (scheduled) — 0.6671
2. Latitude — 0.0332
3. Order Hour — 0.0312
4. Shipping Mode: Same Day — 0.0311
5. Order City — 0.0255

## Project Structure

```
master_pipeline/
├── README.md                          # This file
├── train_master_pipeline.py           # Pipeline script
├── classification_results.csv         # Classification metrics (all models)
├── regression_results.csv             # Regression metrics (all models)
└── models/
    ├── clf_random_forest.joblib       # Best classifier (Acc=0.7491)
    ├── clf_gradient_boosting.joblib
    ├── clf_xgboost.joblib
    ├── reg_random_forest.joblib       # Best regressor (R²=0.5230)
    ├── reg_gradient_boosting.joblib
    ├── reg_xgboost.joblib
    ├── classification_preprocessor.joblib
    └── regression_preprocessor.joblib
```

## How to Run

```bash
# From the repo root
python master_pipeline/train_master_pipeline.py
```

**Requirements:** `scikit-learn`, `xgboost`, `pandas`, `numpy`, `joblib`

## Key Design Decisions

- **No data leakage:** `Order Status`, `Delivery Status`, and `shipping date` are excluded from both tasks. Each task also excludes the other task's target.
- **Preprocessing handled upstream:** `src/data_preprocessing.py` handles cleaning, encoding, and feature engineering. This pipeline only trains and evaluates.
- **3 models per task:** Random Forest, Gradient Boosting, and XGBoost — ranked by primary metric with 5-fold cross-validation.
- **Random Forest wins both tasks** — likely due to its ability to handle the one-hot encoded shipping mode features with deeper trees (max_depth=20).
