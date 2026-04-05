# California Housing Price Analysis

White-box machine learning analysis of California housing prices with geographic enrichment from external data sources.

## Dataset

**Source:** [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (sklearn)  
**Records:** 20,640 (20,495 after cleaning)  
**Features:** 8 base → 20 after engineering → 34 after enrichment

## Pipeline

```
python master_pipeline/create_datasets.py   # Step 1: Clean, preprocess, engineer features
python master_pipeline/enrich_data.py       # Step 2: Add geographic enrichment
python master_pipeline/run_models.py        # Step 3: Run 10 white-box models
python master_pipeline/visualization.py     # Step 4: Generate plots
```

## Preprocessing

### Cleaning
- Removed capped values (MedHouseVal ≥ $500.1K — censored data)
- Removed extreme outliers (AveOccup > 10, AveRooms > 15, AveBedrms > 5)
- 145 records removed (0.7%)

### Feature Engineering (12 new features)
- **Ratios:** `bedroom_ratio`, `rooms_per_person`
- **Density:** `pop_density`
- **Log transforms:** `log_population`, `log_income`
- **Bins:** `is_new_house`, `is_old_house`, `is_low_income`, `is_high_income`, `is_crowded`
- **Interactions:** `income_age`, `income_rooms`

### Target Variables
- **Regression:** `MedHouseVal` — median house value ($15K–$500K)
- **Classification:** `is_high_value` — above median price ($180K), 50/50 split

## Enrichment

14 geographic features computed from **3 external data sources:**

| Source | Features |
|--------|----------|
| **NOAA coastline coordinates** (29 points) | `dist_to_coast`, `is_coastal` |
| **US Census Bureau city coordinates** (5 cities) | `dist_SF`, `dist_LA`, `dist_San_Diego`, `dist_Sacramento`, `dist_San_Jose`, `dist_nearest_city` |
| **CEC climate zone boundaries** (4 zones) | `climate_zone`, `is_bay_area`, `is_socal` |
| **Derived interactions** | `income_coast_interaction`, `urban_density`, `coastal_income` |

### Key Enrichment Feature Correlations

| Feature | Corr w/ Price | Corr w/ High-Value |
|---------|:---:|:---:|
| `coastal_income` | **+0.73** | **+0.61** |
| `income_coast_interaction` | +0.54 | +0.40 |
| `is_coastal` | +0.50 | +0.49 |
| `dist_to_coast` | -0.49 | -0.47 |
| `dist_nearest_city` | -0.38 | -0.34 |

## Model Results (Enriched)

### Regression — Median House Value

| Model | R² | MAE |
|-------|:---:|:---:|
| **Decision Tree** | **0.7547** | **$37K** |
| OLS | 0.7267 | $42K |
| Ridge | 0.7254 | $42K |
| Lasso | 0.7124 | $44K |
| Polynomial | 0.6949 | $45K |

### Classification — High-Value Housing

| Model | F1 | Accuracy | Precision | Recall |
|-------|:---:|:---:|:---:|:---:|
| **Decision Tree** | **0.8818** | **0.8819** | 0.8823 | 0.8814 |
| Logistic Regression | 0.8723 | 0.8719 | 0.8695 | 0.8751 |
| LDA | 0.8628 | 0.8607 | 0.8499 | 0.8760 |
| Perceptron | 0.8359 | 0.8317 | 0.8150 | 0.8580 |
| Naive Bayes | 0.8177 | 0.8280 | 0.8696 | 0.7716 |

## Enrichment Impact

**10/10 models improved with geographic enrichment.**

### Regression: Base → Enriched

| Model | Base R² | Enriched R² | Improvement |
|-------|:---:|:---:|:---:|
| Decision Tree | 0.6656 | **0.7547** | **+13.4%** |
| Polynomial | 0.6374 | **0.6949** | **+9.0%** |
| OLS | 0.6828 | **0.7267** | **+6.4%** |
| Ridge | 0.6829 | **0.7254** | **+6.2%** |
| Lasso | 0.6800 | **0.7124** | **+4.8%** |

**Average R²: 0.6697 → 0.7228 (+7.9%)**

### Classification: Base → Enriched

| Model | Base F1 | Enriched F1 | Improvement |
|-------|:---:|:---:|:---:|
| Naive Bayes | 0.7259 | **0.8177** | **+12.6%** |
| Decision Tree | 0.8513 | **0.8818** | **+3.6%** |
| Logistic Regression | 0.8478 | **0.8723** | **+2.9%** |
| Perceptron | 0.8149 | **0.8359** | **+2.6%** |
| LDA | 0.8467 | **0.8628** | **+1.9%** |

**Average F1: 0.8173 → 0.8541 (+4.5%)**

## Key Findings

1. **Geographic enrichment genuinely improves all models** — coastal proximity is the strongest price driver in California after income
2. **`coastal_income`** (income × coastal indicator) is the #1 feature (45% DT importance) — rich coastal neighborhoods are where the premium is
3. **Linear models benefit most from enrichment** — they can't learn geographic patterns from raw lat/long, but can use pre-computed distances
4. **Lasso selected 20/34 features** — confirming enrichment features carry real signal (not noise)
5. **Naive Bayes gained +12.6%** — biggest single improvement, showing geographic features provide genuinely new distributional information

## Project Structure

```
California_Housing_Analysis/
├── README.md
├── requirements.txt
├── master_pipeline/
│   ├── create_datasets.py    # Preprocessing + feature engineering
│   ├── enrich_data.py        # Geographic enrichment
│   ├── run_models.py         # 5 regression + 5 classification models
│   └── visualization.py      # EDA plots
├── data/
│   ├── raw/                  # Original sklearn data
│   └── processed/            # Base + enriched datasets
└── plots/                    # Visualization outputs
```

## Requirements

```
pip install -r requirements.txt
```
