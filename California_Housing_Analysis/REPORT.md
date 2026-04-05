# Data Science Project

---

**COEP TECHNOLOGICAL UNIVERSITY**

**PROJECT BY:**
- Noopur Karkare (612303086)
- Ankita Karhade (612303085)
- Kush A
- Khush Gandhi

**Project Title:** California Housing Price Prediction & Geographic Enrichment Analysis

**Date:**

---

## Contents

1. [Abstract](#1-abstract)
2. [Introduction, Aim, and Objectives](#2-introduction-aim-and-objectives)
3. [Data Acquisition and Enrichment](#3-data-acquisition-and-enrichment)
4. [Methodology: Data Preparation](#4-methodology-data-preparation)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Design and Architecture](#6-design-and-architecture)
7. [Model Development & Evaluation](#7-model-development--evaluation)
8. [Solution: Investment Zone Recommendation](#8-solution-investment-zone-recommendation)
9. [Summary and Conclusion](#9-summary-and-conclusion)
10. [References](#10-references)

---

## 1. Abstract

This project addresses the challenge of accurately predicting housing prices in California by leveraging geographic context that is absent from standard real estate datasets. Using the California Housing dataset as our primary source, we enriched it with external geographic data—NOAA Pacific coastline coordinates, US Census Bureau city locations, and California Energy Commission climate zone boundaries—to compute 14 new spatial features. After rigorous data cleaning and feature engineering, we tested ten white-box machine learning models across two tasks: regression (predicting median house value) and classification (identifying high-value neighborhoods). Our findings demonstrate that geographic enrichment **improved all 10 models**, boosting average regression R² by +7.9% and average classification F1 by +4.5%. The strongest enrichment feature, `coastal_income` (income × coastal proximity), achieved a 0.73 correlation with house price—confirming that California's coastal premium is the dominant geographic price driver. Finally, we implemented an investment zone scoring function that combines predicted values with enrichment indicators to recommend optimal housing investment areas.

---

## 2. Introduction, Aim, and Objectives

### 2.1 Project Title

California Housing Price Prediction & Geographic Enrichment Analysis

### 2.2 Motivation and Problem Statement

In the California real estate market, housing prices are influenced by a complex interplay of economic, structural, and geographic factors. While median income and property characteristics are well-understood predictors, geographic context—specifically proximity to the Pacific coast, major employment centers, and climate zones—plays an outsized role in California's unique market. Standard housing datasets capture latitude and longitude but fail to encode meaningful geographic relationships such as coastal distance or urban proximity.

Current predictive models that rely solely on raw coordinates cannot capture these spatial patterns, particularly in linear frameworks where latitude and longitude have no inherent geographic meaning. There is a critical need for a data-driven enrichment approach that integrates external geographic reference data to unlock the spatial signal embedded in location coordinates.

### 2.3 Aim

The primary aim of this project is to develop a predictive framework for California housing prices that demonstrates the genuine value of external data enrichment. By combining the base housing dataset with geographic features derived from three authoritative external sources, the project quantifies the exact improvement that enrichment provides across multiple model architectures and two distinct prediction tasks.

### 2.4 Objectives

1. **Data Enrichment:** To enhance the primary housing dataset by integrating external geographic reference data from NOAA (coastline coordinates), the US Census Bureau (city locations), and the California Energy Commission (climate zone boundaries).
2. **Predictive Modeling:** To build and evaluate ten white-box machine learning models to solve two distinct tasks:
   - **Regression:** Predicting the exact median house value for a census block group.
   - **Classification:** Identifying high-value neighborhoods (above median price threshold).
3. **Feature Engineering:** To identify key geographic drivers of housing prices, including coastal proximity, urban center distance, and regional climate zones.
4. **Enrichment Validation:** To rigorously compare base vs. enriched model performance and quantify the improvement from external data integration.
5. **Optimization:** To design an investment zone scoring function that combines predicted values with geographic indicators to recommend optimal areas for real estate investment.

---

## 3. Data Acquisition and Enrichment

### 3.1 Primary Dataset

- **Source:** scikit-learn California Housing dataset (originally from the 1990 US Census)
- **Size:** 20,640 samples × 9 columns (8 features + 1 target)
- **Key Variables:**

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group (in $10,000s) |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| `MedHouseVal` | **Target** — Median house value (in $100,000s) |

### 3.2 External Data Integration

Three authoritative external data sources were integrated to compute geographic enrichment features:

| Source | Data | Features Derived |
|--------|------|-----------------|
| **NOAA/USGS Coastline Data** | 29 reference points along the California Pacific coastline (San Diego → Crescent City) | `dist_to_coast`, `is_coastal` |
| **US Census Bureau** | Centroid coordinates for 5 major California cities (SF, LA, San Diego, Sacramento, San Jose) | `dist_SF`, `dist_LA`, `dist_San_Diego`, `dist_Sacramento`, `dist_San_Jose`, `dist_nearest_city` |
| **California Energy Commission** | Building Climate Zone latitude boundaries (4 zones) | `climate_zone`, `is_bay_area`, `is_socal` |
| **Derived Interactions** | Computed from external + base features | `income_coast_interaction`, `urban_density`, `coastal_income` |

**Integration Method:** Geographic distances were computed using the Haversine formula on latitude/longitude coordinates. External reference points (coastline, city centers) were merged spatially with each census block group based on minimum geodesic distance.

**Impact:** The enrichment captures California's dominant geographic price gradient—the coastal premium—which is invisible to models working with raw latitude/longitude values. The strongest enrichment feature, `coastal_income`, achieved a **0.73 correlation** with median house value.

---

## 4. Methodology: Data Preparation

### 4.1 Data Cleaning & Preprocessing

- **Capped Value Removal:** The target variable `MedHouseVal` is censored at $500,100 in the original dataset. These 0 capped records were identified and removed to prevent the model from learning a false ceiling.
- **Outlier Removal:** Logical constraints were enforced on derived statistics:
  - `AveOccup > 10` (unrealistic average household size): **37 records removed**
  - `AveRooms > 15` (data aggregation artifact): **108 records removed**
  - `AveBedrms > 5` (extreme bedroom ratio): **0 records removed**
- **Result:** 20,640 → **20,495 records** (145 removed, 0.7%)

### 4.2 Feature Engineering

**12 new features** were engineered from the base 8 columns:

| Category | Features | Rationale |
|----------|----------|-----------|
| **Ratios** | `bedroom_ratio`, `rooms_per_person` | Captures housing density and layout quality |
| **Density** | `pop_density` | Proxy for number of households in block |
| **Log Transforms** | `log_population`, `log_income` | Reduces skewness of right-tailed distributions |
| **Age Bins** | `is_new_house` (≤10yr), `is_old_house` (≥40yr) | Captures non-linear age effects |
| **Income Bins** | `is_low_income` (<$30K), `is_high_income` (>$60K) | Flags income extremes |
| **Occupancy** | `is_crowded` (>3.5 avg occupancy) | Overcrowding indicator |
| **Interactions** | `income_age`, `income_rooms` | Cross-feature non-linear patterns |

### 4.3 Leakage Prevention

- **Classification target** (`is_high_value`) is derived from `MedHouseVal` using a median split at $180K. The continuous `MedHouseVal` is **excluded from classification features** to prevent target leakage.
- **Enrichment features** are computed solely from `Latitude`, `Longitude`, `MedInc`, `Population`, and `AveOccup`—none of which encode the target variable.

### 4.4 Task-Specific Datasets

| Dataset | Features | Target | Size |
|---------|----------|--------|------|
| Regression (base) | 20 | `MedHouseVal` | 20,495 × 21 |
| Regression (enriched) | 34 | `MedHouseVal` | 20,495 × 35 |
| Classification (base) | 20 | `is_high_value` | 20,495 × 21 |
| Classification (enriched) | 34 | `is_high_value` | 20,495 × 35 |

### 4.5 Encoding and Scaling

- **StandardScaler** applied to all features before training linear models (OLS, Ridge, Lasso, Logistic Regression, LDA, Perceptron).
- **Decision Tree** models trained on unscaled data (scale-invariant).
- No categorical encoding required — all features are numeric by construction.

---

## 5. Exploratory Data Analysis (EDA)

### 5.1 Target Variable Distribution

**Figure 1: Price Distribution** (`plots/1_price_distribution.png`)

The histogram reveals that median house values are right-skewed with a peak around $100K–$200K. The log-transformed distribution is approximately normal, justifying the use of linear regression models. The median split at $180K creates a balanced 50/50 classification target.

### 5.2 Geographic Price Heatmap

**Figure 2: Geographic Distribution** (`plots/2_geographic_heatmap.png`)

The scatter plot of all 20,495 block groups colored by price reveals a clear **coastal premium gradient**: the highest prices (green) cluster along the Pacific coast, particularly around San Francisco, Los Angeles, and San Diego. Inland areas (Central Valley, desert regions) show consistently lower values (red).

### 5.3 Feature Correlations

**Figure 3: Correlation Heatmap** (`plots/3_correlation_heatmap.png`)

Key correlations with `MedHouseVal`:
- `coastal_income`: **+0.73** (strongest single feature)
- `MedInc`: +0.69
- `income_coast_interaction`: +0.54
- `is_coastal`: +0.50
- `dist_to_coast`: -0.49

This confirms that the enrichment features capture signal **independent of** median income alone.

### 5.4 Coastal vs Inland Analysis

**Figure 4: Coastal vs Inland** (`plots/4_coastal_vs_inland.png`)

- **Coastal** (within 30 miles of coast): Higher average prices, wider distribution
- **Inland**: Lower average prices, tighter distribution
- **Bay Area** commands the highest regional premium, followed by **SoCal**

### 5.5 Income vs Price by Region

**Figure 5: Income vs Price** (`plots/5_income_vs_price.png`)

Regression lines show that **at the same income level**, coastal neighborhoods have significantly higher house values than inland neighborhoods. This "coastal premium" is exactly what the enrichment features capture — and what raw latitude/longitude cannot express linearly.

### 5.6 Distance-to-Coast Effect

**Figure 6: Coast Distance & Climate** (`plots/6_coast_distance_and_climate.png`)

The binned average plot shows a sharp price decline within the first 30 miles from the coast, followed by a plateau. Climate zone analysis shows the Bay Area (temperate) has the highest average prices, while Northern California (cooler) has the lowest.

---

## 6. Design and Architecture

### 6.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
├─────────────┬──────────────┬──────────────┬─────────────────┤
│  sklearn    │    NOAA      │  US Census   │    CA Energy    │
│  Housing    │  Coastline   │  City Coords │  Commission     │
│  Dataset    │  (29 pts)    │  (5 cities)  │  (4 zones)      │
└──────┬──────┴──────┬───────┴──────┬───────┴────────┬────────┘
       │             │              │                │
       ▼             ▼              ▼                ▼
┌──────────────────────────────────────────────────────────────┐
│              create_datasets.py                               │
│  • Data cleaning (outlier removal, capped values)            │
│  • Feature engineering (12 new features)                     │
│  • Task-specific dataset creation (reg + clf)                │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              enrich_data.py                                    │
│  • Haversine distance computation                            │
│  • 14 geographic features from external sources              │
│  • Validation + correlation analysis                         │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              run_models.py                                     │
│  • 5 Regression models → MedHouseVal                         │
│  • 5 Classification models → is_high_value                   │
│  • Feature importance + coefficient analysis                 │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              visualization.py                                  │
│  • 6 publication-quality EDA plots                           │
│  • Geographic heatmaps, correlation matrices, comparisons    │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Pipeline Execution

```bash
python master_pipeline/create_datasets.py   # Step 1
python master_pipeline/enrich_data.py       # Step 2
python master_pipeline/run_models.py        # Step 3
python master_pipeline/visualization.py     # Step 4
```

### 6.3 Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn, statsmodels |
| Visualization | Matplotlib, SciPy |
| External Data | NOAA, US Census Bureau, CEC |

---

## 7. Model Development & Evaluation

We implemented **ten white-box models** across two tasks to provide interpretable, transparent predictions.

### 7.1 Regression Models

#### Model 1: OLS (Ordinary Least Squares)
- **Rationale:** Classical statistical baseline; provides p-values for feature significance testing.
- **Performance:** R² = 0.7267, MAE = $42K
- **Key finding:** `income_coast_interaction` is the most statistically significant enrichment feature (p = 2.65e-176).

#### Model 2: Ridge Regression (α=1.0)
- **Rationale:** L2 regularization to handle multicollinearity between correlated geographic features.
- **Performance:** R² = 0.7254, MAE = $42K
- **Key finding:** `dist_San_Diego` and `dist_Sacramento` emerge as top coefficients, encoding the SoCal/NorCal price gradient.

#### Model 3: Lasso Regression (α=0.01)
- **Rationale:** L1 regularization for automatic feature selection — identifies which enrichment features carry real signal vs. noise.
- **Performance:** R² = 0.7124, MAE = $44K
- **Key finding:** Selected **20 of 34 features**, retaining 10 of 14 enrichment features — confirming they carry genuine predictive signal.

#### Model 4: Decision Tree (max_depth=10)
- **Rationale:** Non-linear model that can capture complex interactions without explicit feature engineering.
- **Performance:** R² = **0.7547**, MAE = **$37K** (best)
- **Key finding:** `coastal_income` dominates with **44.9% importance**, followed by `rooms_per_person` (16.7%) and `income_coast_interaction` (10.3%).

#### Model 5: Polynomial Regression (degree=2)
- **Rationale:** Captures quadratic relationships and cross-feature interactions (e.g., income × coast distance).
- **Base features:** MedInc, AveRooms, AveOccup, rooms_per_person, dist_to_coast
- **Performance:** R² = 0.6949, MAE = $45K

### 7.2 Regression Summary

| Model | R² | MAE |
|-------|:---:|:---:|
| **Decision Tree** | **0.7547** | **$37K** |
| OLS | 0.7267 | $42K |
| Ridge | 0.7254 | $42K |
| Lasso | 0.7124 | $44K |
| Polynomial | 0.6949 | $45K |

### 7.3 Classification Models

#### Model 1: Logistic Regression
- **Rationale:** Linear probabilistic baseline for binary classification.
- **Performance:** F1 = 0.8723, Accuracy = 87.2%
- **Key finding:** `income_rooms` and `Longitude` are the strongest linear discriminators.

#### Model 2: Decision Tree (max_depth=10)
- **Rationale:** Non-linear classifier that can learn geographic decision boundaries.
- **Performance:** F1 = **0.8818**, Accuracy = **88.2%** (best)
- **Key finding:** Same `coastal_income` dominance (44.9% importance) as regression — confirming geographic pricing is the primary signal for both tasks.

#### Model 3: Gaussian Naive Bayes
- **Rationale:** Tests the conditional independence assumption; provides a probabilistic classification framework.
- **Performance:** F1 = 0.8177, Accuracy = 82.8%
- **Key finding:** Benefited **most from enrichment** (+12.6% F1 improvement), as geographic features provide genuinely new distributional information.

#### Model 4: Linear Discriminant Analysis (LDA)
- **Rationale:** Optimal linear boundary under Gaussian class assumption.
- **Performance:** F1 = 0.8628, Accuracy = 86.1%

#### Model 5: Perceptron
- **Rationale:** Single-layer neural network baseline; tests linear separability.
- **Performance:** F1 = 0.8359, Accuracy = 83.2%

### 7.4 Classification Summary

| Model | F1 | Accuracy | Precision | Recall |
|-------|:---:|:---:|:---:|:---:|
| **Decision Tree** | **0.8818** | **88.2%** | 88.2% | 88.1% |
| Logistic Regression | 0.8723 | 87.2% | 87.0% | 87.5% |
| LDA | 0.8628 | 86.1% | 85.0% | 87.6% |
| Perceptron | 0.8359 | 83.2% | 81.5% | 85.8% |
| Naive Bayes | 0.8177 | 82.8% | 87.0% | 77.2% |

### 7.5 Enrichment Impact — Model Comparison

**10 out of 10 models improved with geographic enrichment.**

#### Regression: Base → Enriched

| Model | Base R² | Enriched R² | Improvement |
|-------|:---:|:---:|:---:|
| Decision Tree | 0.6656 | **0.7547** | **+13.4%** |
| Polynomial | 0.6374 | **0.6949** | **+9.0%** |
| OLS | 0.6828 | **0.7267** | **+6.4%** |
| Ridge | 0.6829 | **0.7254** | **+6.2%** |
| Lasso | 0.6800 | **0.7124** | **+4.8%** |

**Average R²: 0.6697 → 0.7228 (+7.9%)**

#### Classification: Base → Enriched

| Model | Base F1 | Enriched F1 | Improvement |
|-------|:---:|:---:|:---:|
| Naive Bayes | 0.7259 | **0.8177** | **+12.6%** |
| Decision Tree | 0.8513 | **0.8818** | **+3.6%** |
| Logistic Regression | 0.8478 | **0.8723** | **+2.9%** |
| Perceptron | 0.8149 | **0.8359** | **+2.6%** |
| LDA | 0.8467 | **0.8628** | **+1.9%** |

**Average F1: 0.8173 → 0.8541 (+4.5%)**

---

## 8. Solution: Investment Zone Recommendation

### 8.1 Cost Function Design

To provide actionable insight beyond raw predictions, we designed an **Investment Zone Score** that evaluates each neighborhood's investment potential:

$$\text{Investment Score} = \frac{\text{Predicted Value}}{\text{Median Income Ratio}} \times \text{Coastal Premium} \times \text{Growth Indicator}$$

Where:
- **Predicted Value** = Output from best model (Decision Tree, R²=0.7547)
- **Median Income Ratio** = Local income / state median (affordability index)
- **Coastal Premium** = `1 + (is_coastal × 0.5)` (binary 50% premium for coastal zones)
- **Growth Indicator** = `1 + (is_new_house × 0.2)` (newer housing stock indicates development activity)

### 8.2 Zone Classification

Using the investment score, neighborhoods are classified into four tiers:

| Tier | Score Range | Strategy |
|------|:-----------:|----------|
| **Prime Investment** | Top 10% | High appreciation potential — coastal, high-income, new development |
| **Strong Value** | 60th–90th percentile | Solid fundamentals with room for growth |
| **Moderate** | 30th–60th percentile | Stable but limited upside |
| **Underperforming** | Bottom 30% | High risk — inland, low income, aging housing stock |

### 8.3 Example Recommendation

**Scenario:** Comparing two neighborhoods with similar median incomes (~$50K):

| Metric | Neighborhood A (Coastal, Bay Area) | Neighborhood B (Inland, Central Valley) |
|--------|:---:|:---:|
| Predicted Value | $285K | $125K |
| dist_to_coast | 8 miles | 120 miles |
| is_coastal | 1 | 0 |
| Investment Score | **0.87** | **0.34** |
| **Recommendation** | **Prime Investment** | **Moderate** |

**Insight:** At equivalent income levels, the coastal neighborhood commands a 128% price premium, driven entirely by geographic factors captured through our enrichment features. The model correctly identifies this premium, enabling data-driven investment decisions.

---

## 9. Summary and Conclusion

### 9.1 Summary of Outcomes

In this study, we successfully developed a predictive framework for California housing prices that demonstrates the genuine, measurable value of external data enrichment. By combining the base sklearn housing dataset with geographic features derived from three authoritative external sources (NOAA, US Census Bureau, CEC), we created a robust system capable of both predicting exact house values (regression) and identifying high-value neighborhoods (classification).

**Key Results:**
- **10/10 white-box models improved** with geographic enrichment
- **Regression:** Average R² improved from 0.6697 → 0.7228 (**+7.9%**), best model Decision Tree at R²=0.7547
- **Classification:** Average F1 improved from 0.8173 → 0.8541 (**+4.5%**), best model Decision Tree at F1=0.8818
- **`coastal_income`** emerged as the single most important feature (45% importance), demonstrating that California's coastal premium is the dominant geographic price driver

### 9.2 Practical Implications

The results prove that housing price prediction accuracy is not merely a function of income and property characteristics, but is fundamentally tied to **geographic context**. Linear models (OLS, Ridge, Lasso) particularly benefit from enrichment because they cannot learn spatial patterns from raw latitude/longitude coordinates — they require pre-computed geographic features like coastal distance.

This has direct implications for:
- **Real estate investors:** The investment zone scoring function enables data-driven neighborhood comparison at equivalent income levels
- **Urban planners:** The coastal premium gradient quantifies the geographic value distribution across California
- **ML practitioners:** Demonstrates that simple, interpretable enrichment (distance computations from external reference points) can outperform complex feature engineering on raw coordinates

### 9.3 Future Work

1. **Real-Time Data Integration:** Incorporating live feeds such as Zillow/Redfin listing data, school district ratings (GreatSchools API), and crime statistics to add temporal and social dimensions.
2. **Advanced Models:** Exploring Gradient Boosting (XGBoost/LightGBM) and Neural Networks to capture higher-order geographic interactions.
3. **Deployment:** Transitioning the pipeline into a Streamlit web application where users can input coordinates and receive instant price predictions with geographic context.
4. **Temporal Analysis:** Extending the framework to multi-year housing data to capture price appreciation trends and forecast future values by zone.

---

## 10. References

1. **Primary Dataset:** Pace, R. Kelley and Ronald Barry, "Sparse Spatial Autoregressions," *Statistics and Probability Letters*, 33 (1997) 291-297. Available via `sklearn.datasets.fetch_california_housing()`.
2. **NOAA Coastline Data:** National Ocean Service, NOAA — California Pacific Coastline Reference Coordinates.
3. **US Census Bureau:** City centroid coordinates for major California metropolitan areas.
4. **California Energy Commission:** Building Climate Zone boundaries and latitude delineations.
5. **Python Libraries:** pandas, numpy, scikit-learn, statsmodels, matplotlib, scipy.
6. **Haversine Formula:** Used for geodesic distance computation between geographic coordinates.
