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
| 7. Enrich | Web-scraped weather, events, transit features | +8 features |

### Data Leakage Prevention

- **`name` excluded from classification** — service name (e.g. "Black", "UberXL") directly encodes the `is_premium` target
- **`surge_multiplier` excluded from classification** — surge is a pricing mechanism, not a predictive feature for vehicle type
- **`cab_weather_interaction`** — uses cab_type (Uber/Lyft), NOT the is_premium target

### Enrichment Features (Web Scraping)

| Feature | Source | Description |
|---------|--------|-------------|
| `weather_severity_enhanced` | Open-Meteo API | Combined rain + cold + humidity score |
| `is_adverse_weather` | Open-Meteo API | Binary bad weather indicator |
| `is_event_time` | Boston Events | Evening hours during weekday events |
| `likely_transit_delay` | MBTA Patterns | Rush hour + rain (high delay probability) |
| `is_high_demand_zone` | Geographic | Near TD Garden, Fenway, stations |
| `distance_event_interaction` | Derived | Distance × event time |
| `weather_rush_interaction` | Derived | Rain × rush hour |
| `cab_weather_interaction` | Derived | Cab type × rain |

---

## Model Results

### Regression — Price Prediction

5 white-box models predicting ride price on the enriched dataset (100K rows, 35 features).

| Model | R² | MAE | Key Insight |
|-------|-----|------|-------------|
| **Decision Tree** | **0.9631** | **$1.13** | Best performer — splits on service name, distance×surge, premium flag |
| Polynomial (deg=2) | 0.7293 | $3.57 | Non-linear interactions between top 5 features |
| Ridge | 0.7018 | $3.90 | Regularized linear model, stable across features |
| Lasso | 0.7005 | $3.91 | Selects 5/40 features — sparse but competitive |
| OLS | -2.4782 | $16.57 | **Failed** — severe multicollinearity |

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
| **Naive Bayes** | **0.7123** | 57.9% | 59.0% | 89.8% | Best F1 — high recall, reasonable precision |
| Logistic Reg | 0.6868 | 58.3% | 60.9% | 78.8% | Balanced precision/recall trade-off |
| Decision Tree | 0.6794 | 57.7% | 60.7% | 77.2% | Splits primarily on cab_type and distance |
| LDA | 0.6791 | 58.2% | 61.2% | 76.3% | Linear discriminant on cab_type dominates |
| Perceptron | 0.5475 | 52.3% | 61.0% | 49.7% | Linear boundary insufficient for this task |

**Top Classification Features** (by Decision Tree importance):
1. `cab_type` (52.2%) — strongest discriminator (Uber vs Lyft premium mix differs)
2. `distance` (16.6%) — premium rides tend to be longer
3. `temperature` (8.2%) — weather influences vehicle choice
4. `windSpeed` (7.2%) — adverse conditions correlate with premium preference
5. `source` (5.6%) — pickup location matters (airport, downtown)

---

### Enrichment Impact — Base vs Enriched

### Regression

| Model | Base R² | Enriched R² | Δ R² |
|-------|---------|-------------|------|
| Decision Tree | 0.9635 | 0.9631 | -0.0004 |
| Ridge | 0.7013 | 0.7018 | +0.0005 |
| Lasso | 0.7005 | 0.7005 | ~0.0000 |
| Polynomial | 0.8106 | 0.7293 | -0.0813 |
| OLS | 0.7013 | -2.4782 | **Failed** |

**Average R²: 0.7754 → 0.7234 (-6.7% change)**

*The base features (distance, surge, service type) capture most of the pricing signal. The enrichment features provide limited additional value for regression.*

### Classification

| Model | Base F1 | Enriched F1 | Δ F1 |
|-------|---------|-------------|------|
| Naive Bayes | 0.6840 | 0.7123 | **+0.0283 (+4.1%)** |
| Logistic Reg | 0.6892 | 0.6868 | -0.0024 |
| Decision Tree | 0.6829 | 0.6794 | -0.0035 |
| LDA | 0.6817 | 0.6791 | -0.0026 |
| Perceptron | 0.5064 | 0.5475 | **+0.0411 (+8.1%)** |

**Average F1: 0.6488 → 0.6610 (+1.9% improvement)**

The enrichment features provide modest improvement for classification, particularly for Naive Bayes and Perceptron.

### Key Takeaway

> **The base engineered features (distance, surge, service type, time, weather) already capture 96%+ of the pricing signal.** The web-scraped enrichment features add limited predictive value. Decision Tree achieves R² = 0.9631 using only base features. For practical applications, focus on service tier, distance, and surge multiplier as the dominant price drivers.

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
