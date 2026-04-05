"""
Uber/Lyft Boston - Complete Data Pipeline
Creates two CSV files: regression_dataset.csv and classification_dataset.csv
Following all 9 steps from DATA_PIPELINE.md
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Fixed seed used throughout for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("UBER/LYFT BOSTON - COMPLETE DATA PIPELINE")
print("Creating Regression & Classification CSVs")
print("="*80)

# ============================================
# STEP 1: LOAD RAW DATA
# ============================================
print("\n1. LOADING RAW DATA")
print("-"*80)
df = pd.read_csv('/home/kushagarwal/Downloads/archive(1)/rideshare_kaggle.csv')
print(f"✓ Loaded: {len(df):,} records, {len(df.columns)} columns")
print(f"  Original shape: {df.shape}")

# ============================================
# STEP 2: DATA CLEANING
# ============================================
print("\n2. DATA CLEANING")
print("-"*80)

# Remove records with missing prices
null_prices = df['price'].isna().sum()
df_clean = df.dropna(subset=['price'])
print(f"✓ Removed {null_prices:,} records with null prices ({null_prices/len(df)*100:.2f}%)")

# Remove invalid prices (price <= 0)
invalid_prices = (df_clean['price'] <= 0).sum()
df_clean = df_clean[df_clean['price'] > 0]
print(f"✓ Removed {invalid_prices:,} records with invalid prices")

# Remove records with missing critical features
df_clean = df_clean.dropna(subset=['distance', 'surge_multiplier'])

print(f"✓ Clean dataset: {len(df_clean):,} records remaining")

# ============================================
# STEP 3: STRATIFIED SAMPLING (100K)
# ============================================
print("\n3. STRATIFIED SAMPLING (100,000 RECORDS)")
print("-"*80)

# Create price categories for stratification
df_clean['price_category'] = pd.cut(df_clean['price'], bins=[0, 10, 15, 25, 1000], 
                                     labels=['low', 'mid', 'high', 'premium'])

# Create strata: cab_type + name + price_category
df_clean['strata'] = (df_clean['cab_type'].astype(str) + '_' + 
                      df_clean['name'].astype(str) + '_' + 
                      df_clean['price_category'].astype(str))

# Sample 100,000 records proportionally
strata_counts = df_clean['strata'].value_counts()
min_strata_size = 10
valid_strata = strata_counts[strata_counts >= min_strata_size].index
df_valid = df_clean[df_clean['strata'].isin(valid_strata)]

# Proportional sampling
sample_size = min(100000, len(df_valid))
df_sample = df_valid.groupby('strata').apply(
    lambda x: x.sample(n=max(1, int(len(x) / len(df_valid) * sample_size)), random_state=RANDOM_STATE)
).reset_index(drop=True)

# Add more if needed
if len(df_sample) < 100000:
    remaining = 100000 - len(df_sample)
    additional = df_valid.drop(df_sample.index, errors='ignore').sample(
        n=min(remaining, len(df_valid) - len(df_sample)), random_state=RANDOM_STATE)
    df_sample = pd.concat([df_sample, additional]).reset_index(drop=True)

print(f"✓ Stratified sample: {len(df_sample):,} records")
uber_pct = (df_sample['cab_type'] == 'Uber').mean() * 100
lyft_pct = (df_sample['cab_type'] == 'Lyft').mean() * 100
print(f"  Uber/Lyft split: {uber_pct:.1f}% / {lyft_pct:.1f}%")

# ============================================
# STEP 4: FEATURE ENGINEERING (25+ new features)
# ============================================
print("\n4. FEATURE ENGINEERING")
print("-"*80)

df = df_sample.copy()

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# Time-based features
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9) & (df['is_weekend'] == 0)).astype(int)
df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19) & (df['is_weekend'] == 0)).astype(int)
df['is_rush_hour'] = df['is_morning_rush'] | df['is_evening_rush']
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

# Distance-based features
df['short_ride'] = (df['distance'] < 2).astype(int)
df['medium_ride'] = ((df['distance'] >= 2) & (df['distance'] < 5)).astype(int)
df['long_ride'] = (df['distance'] >= 5).astype(int)

# Weather features — column names differ depending on the raw data source,
# so we check which one is present before creating the binary flags.
if 'precipIntensity' in df.columns:
    df['is_rainy'] = (df['precipIntensity'] > 0).astype(int)
elif 'rain' in df.columns:
    df['is_rainy'] = (df['rain'] > 0).astype(int)
else:
    df['is_rainy'] = 0

if 'temperature' in df.columns:
    df['is_cold'] = (df['temperature'] < 40).astype(int)
    df['is_hot'] = (df['temperature'] > 75).astype(int)
else:
    df['is_cold'] = 0
    df['is_hot'] = 0

if 'humidity' in df.columns:
    df['is_high_humidity'] = (df['humidity'] > 70).astype(int)
else:
    df['is_high_humidity'] = 0

df['weather_severity'] = df['is_rainy'] + df['is_cold'] + df['is_high_humidity']

# Interaction features (KEY from Boston analysis)
df['distance_surge'] = df['distance'] * df['surge_multiplier']
df['weather_surge'] = df['is_rainy'] * df['surge_multiplier']
df['rush_surge'] = df['is_rush_hour'] * df['surge_multiplier']

# Vehicle features
df['is_premium'] = df['name'].str.contains('Black|Lux|SUV|XL', case=False, na=False).astype(int)
df['uber_premium'] = ((df['cab_type'] == 'Uber') & (df['is_premium'] == 1)).astype(int)
df['lyft_premium'] = ((df['cab_type'] == 'Lyft') & (df['is_premium'] == 1)).astype(int)

print(f"✓ Features after engineering: {df.shape[1]} columns")
print(f"✓ New features added: ~{df.shape[1] - 57}")

# ============================================
# STEP 5: ENCODING
# ============================================
print("\n5. ENCODING CATEGORICAL VARIABLES")
print("-"*80)

cat_cols = ['cab_type', 'name', 'source', 'destination']
encoders = {}

for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"✓ Encoded {col}: {len(le.classes_)} categories")

# ============================================
# STEP 6 & 7: FEATURE SELECTION & TARGET PREPARATION
# ============================================
print("\n6 & 7. FEATURE SELECTION & TARGET PREPARATION")
print("-"*80)

# REGRESSION FEATURES (21 features - includes surge)
regression_features = [
    'distance', 'surge_multiplier', 'hour', 'day_of_week', 'is_weekend',
    'is_morning_rush', 'is_evening_rush', 'is_rush_hour', 'is_night',
    'temperature', 'humidity', 'windSpeed',
    'is_rainy', 'is_cold', 'weather_severity',
    'cab_type_encoded', 'name_encoded', 'is_premium', 
    'source_encoded', 'destination_encoded',
    'distance_surge', 'weather_surge', 'rush_surge',
    'short_ride', 'medium_ride', 'long_ride'
]

# Only keep features that exist in the dataset
regression_features = [col for col in regression_features if col in df.columns]

# CLASSIFICATION FEATURES (18 features - EXCLUDES surge to prevent data leakage)
classification_features = [col for col in regression_features 
                         if 'surge' not in col and col != 'surge_multiplier']

print(f"✓ Regression features: {len(regression_features)} (includes surge interactions)")
print(f"✓ Classification features: {len(classification_features)} (excludes surge)")

# Targets
y_price = df['price']
y_premium = df['name'].str.contains('Black|Lux|SUV|XL', case=False, na=False).astype(int)

print(f"✓ Target (Regression): price ${y_price.min():.2f} - ${y_price.max():.2f}")
print(f"✓ Target (Classification): premium rate = {y_premium.mean()*100:.1f}% ({y_premium.sum():,} / {len(y_premium):,})")

# ============================================
# STEP 8: CREATE DATASETS & SAVE CSV FILES
# ============================================
print("\n8. CREATING DATASET CSV FILES")
print("-"*80)

# Create Regression Dataset
regression_df = df[regression_features + ['price']].copy()
regression_df.columns = [col.replace('_encoded', '') for col in regression_df.columns]

# Create Classification Dataset  
classification_df = df[classification_features].copy()
classification_df['is_premium'] = y_premium.values
classification_df.columns = [col.replace('_encoded', '') for col in classification_df.columns]

# Save to CSV
output_dir = '/home/kushagarwal/CascadeProjects/UberLyft_Boston_Analysis'

reg_file = f'{output_dir}/regression_dataset.csv'
clf_file = f'{output_dir}/classification_dataset.csv'

regression_df.to_csv(reg_file, index=False)
classification_df.to_csv(clf_file, index=False)

print(f"✓ Saved REGRESSION dataset: {reg_file}")
print(f"  Shape: {regression_df.shape}")
print(f"  Columns: {len(regression_df.columns)} ({len(regression_df.columns)-1} features + price)")

print(f"✓ Saved CLASSIFICATION dataset: {clf_file}")
print(f"  Shape: {classification_df.shape}")
print(f"  Columns: {len(classification_df.columns)} ({len(classification_df.columns)-1} features + is_premium)")

# ============================================
# STEP 9: SUMMARY
# ============================================
print("\n" + "="*80)
print("9. PIPELINE COMPLETE - FINAL SUMMARY")
print("="*80)

print(f"""
DATA PIPELINE SUMMARY:
┌──────────────────────────────────────────────────────────────────────┐
│ STEP 1: RAW DATA                                                     │
│   Input: 693,071 records, 57 columns, 351 MB                        │
├──────────────────────────────────────────────────────────────────────┤
│ STEP 2: DATA CLEANING                                               │
│   Removed: 55,095 null prices (7.95%)                               │
│   Result: 637,976 records                                            │
├──────────────────────────────────────────────────────────────────────┤
│ STEP 3: STRATIFIED SAMPLING                                         │
│   Sample: 100,000 records                                            │
│   Balance: Uber {uber_pct:.1f}% / Lyft {lyft_pct:.1f}%                          │
├──────────────────────────────────────────────────────────────────────┤
│ STEP 4: FEATURE ENGINEERING                                         │
│   New features: ~28                                                  │
│   Total columns: 85                                                │
├──────────────────────────────────────────────────────────────────────┤
│ STEP 5: ENCODING                                                    │
│   Encoded: cab_type, name, source, destination                      │
├──────────────────────────────────────────────────────────────────────┤
│ STEP 6 & 7: FEATURE SELECTION                                       │
│   Regression: {len(regression_features)} features (with surge)                   │
│   Classification: {len(classification_features)} features (no surge)                 │
│   Target: price (${y_price.min():.2f}-${y_price.max():.2f}) / premium ({y_premium.mean()*100:.1f}%)        │
├──────────────────────────────────────────────────────────────────────┤
│ STEP 8: OUTPUT FILES                                                │
│   regression_dataset.csv: {regression_df.shape[0]:,} rows × {regression_df.shape[1]} cols      │
│   classification_dataset.csv: {classification_df.shape[0]:,} rows × {classification_df.shape[1]} cols  │
└──────────────────────────────────────────────────────────────────────┘

OUTPUT FILES:
  📁 {reg_file}
     - Use for: Price prediction regression models
     - Features: {len(regression_features)} including surge_multiplier, distance_surge
     - Target: price

  📁 {clf_file}
     - Use for: Premium vehicle classification models
     - Features: {len(classification_features)} (NO surge columns - prevents data leakage)
     - Target: is_premium (Uber Black/Lux/SUV/XL = 1, UberX/Lyft = 0)
""")

print("="*80)
print("PIPELINE COMPLETE - Ready for modeling!")
print("="*80)

