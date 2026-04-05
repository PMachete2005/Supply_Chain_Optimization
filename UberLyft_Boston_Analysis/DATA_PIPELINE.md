# Uber/Lyft Boston Dataset - Complete Data Pipeline

## 1. RAW DATA STARTING POINT

**File:** `rideshare_kaggle.csv`
- **Records:** 693,071
- **Columns:** 57
- **Size:** 351 MB

**Key Columns:**
- `price` - Ride fare in USD
- `distance` - Trip distance in miles
- `cab_type` - 'Uber' or 'Lyft'
- `name` - Vehicle tier (UberX, Black, Lyft XL, etc.)
- `source` - Pickup location
- `destination` - Dropoff location
- `temp` - Temperature in Fahrenheit
- `humidity` - Humidity percentage
- `wind` - Wind speed
- `rain` - Rainfall intensity
- `surge_multiplier` - Price multiplier during high demand
- `timestamp` - Ride timestamp

---

## 2. DATA CLEANING

```python
# Step 1: Remove records with missing prices
df_clean = df.dropna(subset=['price'])  # Removed 55,095 records (7.95%)

# Step 2: Remove invalid prices (price <= 0)
df_clean = df_clean[df_clean['price'] > 0]
```

**Result:** 637,976 records remaining

---

## 3. STRATIFIED SAMPLING

```python
# Create price categories for stratification
df['price_category'] = pd.cut(df['price'], bins=[0, 10, 15, 25, 100])

# Stratified by cab_type + ride_name + price_category
df['strata'] = df['cab_type'] + '_' + df['name'] + '_' + df['price_category'].astype(str)

# Sample 100,000 records maintaining distributions
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=100000, random_state=42)
for _, sample_idx in split.split(df, df['strata']):
    df_sample = df.iloc[sample_idx]
```

**Result:** 100K records with preserved Uber/Lyft balance (51.8% / 48.2%)

---

## 4. FEATURE ENGINEERING (25 new features)

### Time-Based Features (6 features)
| Feature | Logic |
|---------|-------|
| `hour` | Extract from timestamp |
| `day_of_week` | Monday=0, Sunday=6 |
| `is_weekend` | 1 if Saturday/Sunday |
| `is_morning_rush` | 7am-9am on weekdays |
| `is_evening_rush` | 5pm-7pm on weekdays |
| `is_night` | 10pm-5am |
| `is_rush_hour` | Morning OR evening rush |

### Distance-Based Features (3 features)
| Feature | Logic |
|---------|-------|
| `short_ride` | distance < 2 miles |
| `medium_ride` | 2 ≤ distance < 5 miles |
| `long_ride` | distance ≥ 5 miles |

### Interaction Features (3 features)
| Feature | Logic |
|---------|-------|
| `distance_surge` | distance × surge_multiplier |
| `weather_surge` | is_rainy × surge_multiplier |
| `rush_surge` | is_rush_hour × surge_multiplier |

### Weather Features (4 features)
| Feature | Logic |
|---------|-------|
| `is_rainy` | rain > 0 |
| `is_cold` | temp < 40°F |
| `is_high_humidity` | humidity > 70% |
| `weather_severity` | Sum of rain + cold + high_humidity |

### Vehicle Features (3 features)
| Feature | Logic |
|---------|-------|
| `is_premium` | Black, SUV, Lux tiers |
| `uber_premium` | Uber Black/SUV only |
| `lyft_premium` | Lyft Lux/Plus only |

### Time Features (6 features)
| Feature | Logic |
|---------|-------|
| `is_business_hours` | 9am-5pm weekdays |
| `is_late_night` | 12am-5am |
| `month` | Extract from timestamp |
| `day` | Extract from timestamp |
| `week_of_year` | Week number |
| `quarter` | Q1, Q2, Q3, Q4 |

**Total:** 57 → 82 columns (25 new features)

---

## 5. FEATURE SELECTION

### For Regression (21 features)
Include ALL features including surge-related:
- `distance`, `surge_multiplier`, `distance_surge`, `weather_surge`, `rush_surge`
- `hour`, `day_of_week`, `is_weekend`, `is_morning_rush`, `is_evening_rush`, `is_rush_hour`
- `temp`, `humidity`, `wind`, `rain`, `is_rainy`, `is_cold`, `weather_severity`
- `cab_type`, `name`, `is_premium`, `source`, `destination`

**Target:** `y_price = df['price']`

### For Classification (18 features)
**EXCLUDED surge columns** to prevent data leakage:
- ❌ `surge_multiplier`
- ❌ `distance_surge`
- ❌ `weather_surge`
- ❌ `rush_surge`

**Included:**
- `distance`, `hour`, `day_of_week`, `is_weekend`, `is_morning_rush`, `is_evening_rush`
- `temp`, `humidity`, `wind`, `rain`, `is_rainy`, `is_cold`, `weather_severity`
- `cab_type`, `name`, `is_premium`, `source`, `destination`

**Target:** `y_surge = (df['surge_multiplier'] > 1.0).astype(int)`
- **Surge rate:** 3.3% of rides have surge

---

## 6. ENCODING

```python
from sklearn.preprocessing import LabelEncoder

# Label encode categorical columns
cat_cols = ['cab_type', 'name', 'source', 'destination']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Save encoders for inference
joblib.dump(encoders, 'models/encoders.joblib')
```

---

## 7. TARGET PREPARATION

### Regression Target
```python
y_price = df['price']
```
- Range: $2.00 - $97.50
- Mean: $16.50
- Std: $10.23

### Classification Target
```python
y_surge = (df['surge_multiplier'] > 1.0).astype(int)
```
- Class distribution: 96.7% (no surge) / 3.3% (surge)
- Imbalance ratio: ~30:1

---

## 8. TRAIN/TEST SPLIT

```python
from sklearn.model_selection import train_test_split

# Regression split (include surge features)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_price, test_size=0.2, random_state=42
)

# Classification split (exclude surge features)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_surge, test_size=0.2, random_state=42, stratify=y_surge
)
```

**Result:** 80,000 train / 20,000 test (80/20 split)

---

## 9. CLASS IMBALANCE HANDLING

### Random Forest
```python
RandomForestClassifier(
    class_weight='balanced',  # Auto-adjust for imbalance
    random_state=42
)
```

### XGBoost
```python
XGBClassifier(
    scale_pos_weight=10,  # Weight positive class 10x
    random_state=42
)
```

### Why scale_pos_weight=10?
- Negatives: 96.7% → weight = 1
- Positives: 3.3% → weight = 10
- Balances 30:1 ratio to ~3:1 effective ratio

---

## 10. FINAL PIPELINE SUMMARY

| Stage | Input | Output |
|-------|-------|--------|
| Raw Data | 693,071 records | 57 columns |
| Cleaning | Drop nulls, invalid | 637,976 records |
| Sampling | Stratified 100K | 100,000 records |
| Feature Engineering | 57 columns | 82 columns (+25) |
| Encoding | Categorical strings | Numeric encoded |
| Split | 100K records | 80K train / 20K test |

**Final Features:**
- Regression: 21 features (includes surge)
- Classification: 18 features (excludes surge - no data leakage)

---

## Files Generated

1. `rideshare_sample_stratified.csv` - 100K stratified sample
2. `models/best_classifier.joblib` - Trained classifier
3. `models/best_regressor.joblib` - Trained regressor
4. `models/classification_encoders.joblib` - Label encoders
5. `models/regression_encoders.joblib` - Label encoders

---

## Key Insights from Pipeline

1. **Data Quality:** 7.95% of records had missing prices (removed)
2. **Class Imbalance:** Only 3.3% rides have surge (handled with class weights)
3. **Feature Importance:** `distance_surge` is strongest predictor (77% importance)
4. **Surge Prevention:** Removed surge features from classification to prevent leakage
5. **Stratification:** Maintained Uber/Lyft balance (51.8% / 48.2%) in sample

---

*Generated: April 2026*
*Dataset: Uber & Lyft Rideshare Data - Boston, MA*
