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

### Data Leakage Prevention

- **`name` excluded from classification** — service name (e.g. "Black", "UberXL") directly encodes the `is_premium` target
- **`surge_multiplier` excluded from classification** — surge is a pricing mechanism, not a predictive feature for vehicle type
- **`fuel_price_indicator`** — based on distance + surge conditions, NOT derived from price target
- **`cab_weather_interaction`** — uses cab_type (Uber/Lyft), NOT the is_premium target

### Enrichment Features (Web Scraping)

| Feature | Source | Description |
|---------|--------|-------------|
| `weather_severity_enhanced` | Open-Meteo API | Combined rain + cold + humidity score |
| `is_adverse_weather` | Open-Meteo API | Binary bad weather indicator |
| `is_event_time` | Boston Events | Evening hours during weekday events |
| `likely_transit_delay` | MBTA Patterns | Rush hour + rain (high delay probability) |
| `is_high_demand_zone` | Geographic | Near TD Garden, Fenway, stations |
| `fuel_price_indicator` | Distance + Surge | Long ride during surge period |
| `distance_event_interaction` | Derived | Distance × event time |
| `weather_rush_interaction` | Derived | Rain × rush hour |
| `cab_weather_interaction` | Derived | Cab type × rain |

---

## Model Results

### Regression — Price Prediction

5 white-box models predicting ride price on the enriched dataset (100K rows, 35 features).

| Model | R² | MAE | Key Insight |
|-------|-----|------|-------------|
| **Decision Tree** | **0.9639** | **$1.13** | Best performer — splits on service name, distance×surge, premium flag |
| Polynomial (deg=2) | 0.8106 | $3.14 | Non-linear interactions between top 5 features |
| Ridge | 0.7012 | $3.90 | Regularized linear model, stable across features |
| OLS | 0.7012 | $3.90 | Near-identical to Ridge — data is well-conditioned |
| Lasso | 0.7005 | $3.91 | Selects 5/35 features — sparse but competitive |

**Top Regression Features** (by Lasso selection):
1. `is_premium` — premium vehicles cost ~$5.03 more
2. `distance_surge` — distance × surge interaction
3. `name` — specific service type encoding
4. `surge_multiplier` — direct pricing factor
5. `cab_type` — Uber vs Lyft base pricing differs

### Classification — Premium Vehicle Prediction

5 white-box models predicting whether a ride uses a premium vehicle (Black, Lux, SUV, XL)  
on the enriched dataset (100K rows, 29 features). Target: `is_premium` (58% positive rate).

| Model | F1 | Accuracy | Precision | Recall | Key Insight |
|-------|-----|----------|-----------|--------|-------------|
| **Naive Bayes** | **0.7125** | 57.9% | 59.0% | 89.9% | Best F1 — high recall, reasonable precision |
| Logistic Reg | 0.6884 | 58.1% | 60.6% | 79.7% | Balanced precision/recall trade-off |
| Decision Tree | 0.6830 | 57.7% | 60.5% | 78.5% | Splits primarily on cab_type and distance |
| LDA | 0.6809 | 58.2% | 61.2% | 76.8% | Linear discriminant on cab_type dominates |
| Perceptron | 0.4816 | 50.1% | 60.6% | 40.0% | Linear boundary insufficient for this task |

**Top Classification Features** (by Decision Tree importance):
1. `cab_type` (52.2%) — strongest discriminator (Uber vs Lyft premium mix differs)
2. `distance` (16.6%) — premium rides tend to be longer
3. `temperature` (8.2%) — weather influences vehicle choice
4. `windSpeed` (7.2%) — adverse conditions correlate with premium preference
5. `source` (5.6%) — pickup location matters (airport, downtown)

---

## Enrichment Impact — Base vs Enriched

### Regression

| Model | Base R² | Enriched R² | Δ R² |
|-------|---------|-------------|------|
| OLS | 0.7013 | 0.7012 | -0.0002 |
| Ridge | 0.7013 | 0.7012 | -0.0001 |
| Lasso | 0.7005 | 0.7005 | ~0.0000 |
| Decision Tree | 0.9635 | 0.9639 | +0.0003 |
| Polynomial | 0.8106 | 0.8106 | ~0.0000 |

**Average R²: 0.7754 → 0.7755 (~0.0% change)**

The base features (distance, surge, service type, time, weather) already capture the pricing signal.  
Enrichment adds no meaningful improvement for regression.

### Classification

| Model | Base F1 | Enriched F1 | Δ F1 |
|-------|---------|-------------|------|
| Naive Bayes | 0.6840 | 0.7125 | **+0.0285 (+4.2%)** |
| Decision Tree | 0.6829 | 0.6830 | +0.0001 |
| Logistic Reg | 0.6892 | 0.6884 | -0.0007 |
| LDA | 0.6817 | 0.6809 | -0.0009 |
| Perceptron | 0.5064 | 0.4816 | -0.0248 |

**Average F1: 0.6488 → 0.6493 (+0.1% change)**

Only Naive Bayes benefits from enrichment (+4.2% F1). The enrichment features  
(weather severity, event time, transit delays, etc.) provide minimal additional signal  
beyond what the base weather and time features already capture.

### Key Takeaway

> **The base engineered features are already strong predictors.** The web-scraped  
> enrichment features are derived from the same underlying signals (weather, time, location)  
> and add redundant information. For enrichment to significantly improve results,  
> it would need truly novel data sources — e.g., real-time MBTA delay feeds,  
> actual event ticket sales, or live traffic congestion data.

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
