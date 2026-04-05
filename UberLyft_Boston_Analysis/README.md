# Uber/Lyft Boston Analysis

Complete ML pipeline for Uber & Lyft ride pricing analysis in Boston, MA.

## Project Structure

```
UberLyft_Boston_Analysis/
├── master_pipeline/          # Training scripts and pipelines
│   ├── create_datasets.py    # Data cleaning and feature engineering
│   ├── white_box_models_proper.py  # White box model training
│   ├── run_models_on_datasets.py   # Model evaluation
│   ├── enrich_data.py        # Web scraping and data enrichment
│   └── compare_enriched.py   # Performance comparison
├── modelresults/             # Model outputs and metrics
├── new_data/
│   ├── raw/                  # Original raw data
│   └── processed/            # Cleaned datasets
│       ├── regression_dataset.csv
│       ├── regression_dataset_enriched.csv
│       ├── classification_dataset.csv
│       └── classification_dataset_enriched.csv
└── src/                      # Source modules and utilities
```

## Data Pipeline

1. **Raw Data** (693K records)
   - Source: `rideshare_kaggle.csv`
   - 57 columns, 351 MB

2. **Cleaning** 
   - Removed 55K null prices → 637,976 records

3. **Stratified Sampling**
   - 100,000 records with Uber/Lyft balance (51.8%/48.2%)

4. **Feature Engineering** (+25 features)
   - Time-based: hour, rush_hour, weekend
   - Distance-based: short/medium/long ride
   - Weather: is_rainy, is_cold, severity score
   - Interactions: distance×surge, rain×rush_hour

5. **Web Scraping Enrichment** (+9 features)
   - Weather API (Open-Meteo): temp, precipitation, wind
   - Event patterns: sports games, concerts
   - Transit: MBTA delay patterns
   - Geographic: high-demand zones (TD Garden, Fenway)

## Key Results

### Regression (Price Prediction)
| Model | R² (Original) | R² (Enriched) | Improvement |
|-------|---------------|---------------|-------------|
| Ridge | 0.7013 | 0.8963 | **+27.8%** |
| Decision Tree | 0.7882 | 0.8746 | **+11.0%** |

### Top Features by Impact
1. `distance_surge` (distance × surge_multiplier) - $4.59
2. `is_premium` (Black/SUV/Lux vehicles) - $10.31
3. `weather_rush_interaction` (rain × rush hour) - $4.82
4. `is_high_demand_zone` (near venues) - $3.15
5. `surge_multiplier` - $8.40

## Documentation

- `DATA_PIPELINE.md` - Complete 9-step pipeline documentation
- `ENRICHED_FEATURES.md` - Web scraping sources and features
- `FEATURES_THAT_IMPROVED_RESULTS.md` - Detailed feature impact analysis
- `WEB_SCRAPING_INSTRUCTIONS.md` - Guide for your scraping team

## Usage

```bash
# Create datasets
python master_pipeline/create_datasets.py

# Run white box models
python master_pipeline/white_box_models_proper.py

# Enrich with web scraping
python master_pipeline/enrich_data.py

# Compare performance
python master_pipeline/compare_enriched.py
```

## Requirements

- pandas
- numpy
- scikit-learn
- statsmodels
- requests (for web scraping)

## License

Analysis of publicly available Uber/Lyft Boston dataset for educational purposes.
