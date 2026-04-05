# Uber/Lyft Boston — Ride Pricing & Premium Vehicle Analysis

White-box ML pipeline analyzing 693K Uber & Lyft rides in Boston, MA.  
Predicts **ride prices** (regression) and **premium vehicle type** (classification)  
using 5 interpretable models each, with web-scraped data enrichment.

---

## Project Structure

```
UberLyft_Boston_Analysis/
├── master_pipeline/
│   ├── create_datasets.py       # Step 1: Clean, sample, engineer features
│   ├── enrich_data.py           # Step 2: Add 9 web-scraped enrichment features
│   ├── run_models.py            # Step 3: 5 regression + 5 classification models
│   └── visualization.py         # Step 4: Generate 6 EDA plots
├── new_data/
│   ├── processed/               # Final datasets (base + enriched)
│   └── plots/                   # 6 visualization PNGs
├── DATA_PIPELINE.md
├── ENRICHED_FEATURES.md
├── FEATURES_THAT_IMPROVED_RESULTS.md
└── WEB_SCRAPING_INSTRUCTIONS.md
```

---

## Data Pipeline

| Step | Description | Output |
|------|-------------|--------|
| 1. Load | Raw `rideshare_kaggle.csv` | 693,071 records, 57 columns |
| 2. Clean | Drop null/invalid prices | 637,976 records |
| 3. Sample | Stratified by cab type + service + price tier | 100,000 records |
| 4. Engineer | Time, distance, weather, interaction features | +25 features |
| 5. Encode | LabelEncoder on cab_type, name, source, destination | Numeric features |
| 6. Split | Regression (with surge) vs Classification (no surge, no name) | 2 datasets |
| 7. Enrich | Web-scraped weather, events, transit, fuel features | +9 features |

### Enrichment Features (Web Scraping)

| Feature | Source | Description |
|---------|--------|-------------|
| `weather_severity_enhanced` | Open-Meteo API | Combined rain + cold + humidity score |
| `is_adverse_weather` | Open-Meteo API | Binary bad weather indicator |
| `is_event_time` | Boston Events | Evening hours during weekday events |
| `likely_transit_delay` | MBTA Patterns | Rush hour + rain (high delay probability) |
| `is_high_demand_zone` | Geographic | Near TD Garden, Fenway, stations |
| `fuel_price_indicator` | EIA Gas Prices | High fuel surcharge period |
| `distance_event_interaction` | Derived | Distance × event time |
| `weather_rush_interaction` | Derived | Rain × rush hour |
| `premium_weather_interaction` | Derived | Premium vehicle × rain |

---

## Model Results

### Regression — Price Prediction

5 white-box models predicting ride price on the **enriched** dataset (100K rows, 35 features).

| Model | R² | MAE | Key Insight |
|-------|-----|------|-------------|
| **Decision Tree** | **0.9696** | **$1.06** | Best performer — splits on fuel indicator, premium, distance×surge |
| Polynomial (deg=2) | 0.9074 | $1.90 | Non-linear interactions improve over linear by +1.1% |
| Ridge | 0.8963 | $2.11 | Regularization prevents overfitting on 35 features |
| OLS | 0.8962 | $2.11 | Near-identical to Ridge — data is well-conditioned |
| Lasso | 0.8957 | $2.10 | Selects 7/35 features — sparse but competitive |

**Top Regression Features** (by Lasso selection):
1. `fuel_price_indicator` — strongest single predictor
2. `is_premium` — premium vehicles cost ~$3.73 more
3. `distance_surge` — distance × surge interaction
4. `name` — specific service type encoding
5. `surge_multiplier` — direct pricing factor

### Classification — Premium Vehicle Prediction

5 white-box models predicting whether a ride uses a premium vehicle (Black, Lux, SUV, XL)  
on the **enriched** dataset (100K rows, 29 features). Target: `is_premium` (58% positive rate).

> **Note:** The `name` column (service type) is excluded from classification features  
> to prevent data leakage, since service name directly encodes the target.

| Model | F1 | Accuracy | Precision | Recall | Key Insight |
|-------|-----|----------|-----------|--------|-------------|
| **LDA** | **0.7797** | 67.2% | 63.9% | 99.9% | Best F1 — near-perfect recall |
| Logistic Reg | 0.7494 | 67.6% | 68.0% | 83.5% | Best accuracy — balanced precision/recall |
| Decision Tree | 0.7473 | 67.0% | 67.2% | 84.2% | Splits on weather severity and premium×weather |
| Perceptron | 0.5882 | 58.5% | 69.3% | 51.1% | Linear boundary struggles with this task |
| Naive Bayes | 0.3620 | 54.8% | 100.0% | 22.1% | Independence assumption violated by correlated features |

**Top Classification Features** (by Decision Tree importance):
1. `weather_severity_enhanced` (45.4%) — strongest discriminator
2. `premium_weather_interaction` (41.7%) — premium demand increases in bad weather
3. `cab_type` (8.9%) — Uber vs Lyft service mix differs
4. `distance` (1.3%) — premium rides tend to be longer
5. `temperature` (1.0%) — weather influences vehicle choice

---

## Enrichment Impact — Base vs Enriched

### Regression (R² improvement)

| Model | Base R² | Enriched R² | Δ R² | Improvement |
|-------|---------|-------------|------|-------------|
| OLS | 0.7013 | 0.8962 | +0.1949 | **+27.8%** |
| Ridge | 0.7013 | 0.8963 | +0.1950 | **+27.8%** |
| Lasso | 0.7005 | 0.8957 | +0.1952 | **+27.9%** |
| Decision Tree | 0.9635 | 0.9696 | +0.0061 | +0.6% |
| Polynomial | 0.8106 | 0.9074 | +0.0969 | **+12.0%** |

**All 5 regression models improved. Average R²: 0.7754 → 0.9130 (+17.7%)**

### Regression MAE improvement

| Model | Base MAE | Enriched MAE | Savings |
|-------|----------|--------------|---------|
| OLS | $3.90 | $2.11 | **-$1.79** |
| Ridge | $3.90 | $2.11 | **-$1.79** |
| Lasso | $3.91 | $2.10 | **-$1.81** |
| Decision Tree | $1.13 | $1.06 | -$0.07 |
| Polynomial | $3.14 | $1.90 | **-$1.24** |

### Classification (F1 improvement)

| Model | Base F1 | Enriched F1 | Δ F1 | Improvement |
|-------|---------|-------------|------|-------------|
| LDA | 0.6817 | 0.7797 | +0.0979 | **+14.4%** |
| Perceptron | 0.5064 | 0.5882 | +0.0818 | **+16.2%** |
| Decision Tree | 0.6829 | 0.7473 | +0.0644 | **+9.4%** |
| Logistic Reg | 0.6892 | 0.7494 | +0.0602 | **+8.7%** |
| Naive Bayes | 0.6840 | 0.3620 | -0.3220 | -47.1% |

**4/5 classification models improved.** Naive Bayes degraded because the enrichment features are correlated, violating its independence assumption.

---

## Usage

```bash
cd master_pipeline/

# Step 1: Create base datasets from raw data
python create_datasets.py

# Step 2: Enrich with web-scraped features
python enrich_data.py

# Step 3: Run all 10 models
python run_models.py

# Step 4: Generate visualizations
python visualization.py
```

## Requirements

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
statsmodels>=0.13.0
requests>=2.28.0
matplotlib>=3.5.0
scipy>=1.9.0
```

## License

Analysis of publicly available Uber/Lyft Boston dataset for educational purposes.
