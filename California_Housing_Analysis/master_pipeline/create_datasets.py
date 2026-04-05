"""
Step 1: Create Datasets
Load California Housing data, clean, preprocess, engineer features,
and produce separate regression and classification datasets.

Regression target:  MedHouseVal (continuous, $100K units)
Classification target: is_high_value (binary, above median price)

Preprocessing steps:
  1. Load raw data from sklearn
  2. Remove outliers (capped values, extreme occupancy)
  3. Feature engineering (ratios, bins, log transforms)
  4. Separate feature sets for regression vs classification
  5. Save base datasets to data/processed/
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

np.random.seed(42)

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJ_DIR, 'data', 'raw')
PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: LOAD RAW DATA")
print("=" * 70)

data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

# Save raw data for reference
raw_path = os.path.join(RAW_DIR, 'california_housing_raw.csv')
df.to_csv(raw_path, index=False)
print(f"Raw data saved: {raw_path}")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nTarget (MedHouseVal) stats:")
print(f"  Range: ${df['MedHouseVal'].min()*100:.0f}K – ${df['MedHouseVal'].max()*100:.0f}K")
print(f"  Mean:  ${df['MedHouseVal'].mean()*100:.0f}K")
print(f"  Median: ${df['MedHouseVal'].median()*100:.0f}K")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: DATA CLEANING")
print("=" * 70)

n_before = len(df)

# 2a. Remove capped values
# MedHouseVal is capped at 5.0001 ($500K) — these are censored, not real prices
capped = (df['MedHouseVal'] >= 5.0001).sum()
df = df[df['MedHouseVal'] < 5.0001]
print(f"Removed {capped} capped price records (MedHouseVal >= $500.1K)")

# 2b. Remove extreme outliers in AveOccup (likely data errors)
# Average occupancy > 10 people per household is unrealistic
extreme_occup = (df['AveOccup'] > 10).sum()
df = df[df['AveOccup'] <= 10]
print(f"Removed {extreme_occup} extreme occupancy records (AveOccup > 10)")

# 2c. Remove extreme AveRooms (> 15 rooms avg is likely data error)
extreme_rooms = (df['AveRooms'] > 15).sum()
df = df[df['AveRooms'] <= 15]
print(f"Removed {extreme_rooms} extreme room count records (AveRooms > 15)")

# 2d. Remove extreme AveBedrms
extreme_bed = (df['AveBedrms'] > 5).sum()
df = df[df['AveBedrms'] <= 5]
print(f"Removed {extreme_bed} extreme bedroom count records (AveBedrms > 5)")

print(f"\nCleaned: {n_before:,} → {len(df):,} records ({n_before - len(df)} removed)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 70)

# 3a. Room ratios
df['bedroom_ratio'] = df['AveBedrms'] / (df['AveRooms'] + 0.001)
df['rooms_per_person'] = df['AveRooms'] / (df['AveOccup'] + 0.001)
print("Added: bedroom_ratio, rooms_per_person")

# 3b. Population density
df['pop_density'] = df['Population'] / (df['AveOccup'] + 0.001)
print("Added: pop_density (households proxy)")

# 3c. Log transforms for skewed features
df['log_population'] = np.log1p(df['Population'])
df['log_income'] = np.log1p(df['MedInc'])
print("Added: log_population, log_income")

# 3d. House age bins
df['is_new_house'] = (df['HouseAge'] <= 10).astype(int)
df['is_old_house'] = (df['HouseAge'] >= 40).astype(int)
print("Added: is_new_house (<= 10yr), is_old_house (>= 40yr)")

# 3e. Income bins
df['is_low_income'] = (df['MedInc'] < 3.0).astype(int)
df['is_high_income'] = (df['MedInc'] > 6.0).astype(int)
print("Added: is_low_income (< $30K), is_high_income (> $60K)")

# 3f. Occupancy categories
df['is_crowded'] = (df['AveOccup'] > 3.5).astype(int)
print("Added: is_crowded (AveOccup > 3.5)")

# 3g. Income × age interaction
df['income_age'] = df['MedInc'] * df['HouseAge']
print("Added: income_age interaction")

# 3h. Income × rooms interaction
df['income_rooms'] = df['MedInc'] * df['AveRooms']
print("Added: income_rooms interaction")

n_engineered = df.shape[1] - 9  # 9 = 8 original features + target
print(f"\nTotal engineered features: {n_engineered}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CREATE SEPARATE DATASETS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: CREATE REGRESSION & CLASSIFICATION DATASETS")
print("=" * 70)

# --- REGRESSION DATASET ---
# Target: MedHouseVal (continuous)
# Features: all engineered features
reg_features = [c for c in df.columns if c != 'MedHouseVal']
reg_df = df[reg_features + ['MedHouseVal']].copy()

print(f"\nRegression dataset:")
print(f"  Shape: {reg_df.shape[0]:,} × {len(reg_features)} features + target")
print(f"  Target: MedHouseVal (${reg_df['MedHouseVal'].min()*100:.0f}K – ${reg_df['MedHouseVal'].max()*100:.0f}K)")
print(f"  Features: {reg_features}")

# --- CLASSIFICATION DATASET ---
# Target: is_high_value (binary: above median price)
# Features: same as regression BUT exclude MedHouseVal (the target source)
# Also exclude MedInc for a harder/more interesting classification task?
# No — keep MedInc, it's a legitimate feature (income is known before buying)

median_val = df['MedHouseVal'].median()
df['is_high_value'] = (df['MedHouseVal'] > median_val).astype(int)

# For classification, we must NOT include MedHouseVal (it IS the target)
clf_features = [c for c in df.columns if c not in ['MedHouseVal', 'is_high_value']]
clf_df = df[clf_features + ['is_high_value']].copy()

print(f"\nClassification dataset:")
print(f"  Shape: {clf_df.shape[0]:,} × {len(clf_features)} features + target")
print(f"  Target: is_high_value (median split at ${median_val*100:.0f}K)")
print(f"  Class balance: {clf_df['is_high_value'].mean()*100:.1f}% high value")
print(f"  Features: {clf_features}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: SAVE DATASETS")
print("=" * 70)

reg_path = os.path.join(PROC_DIR, 'regression_dataset.csv')
clf_path = os.path.join(PROC_DIR, 'classification_dataset.csv')

reg_df.to_csv(reg_path, index=False)
clf_df.to_csv(clf_path, index=False)

print(f"Saved: {reg_path}")
print(f"  {reg_df.shape[0]:,} rows × {reg_df.shape[1]} columns")
print(f"Saved: {clf_path}")
print(f"  {clf_df.shape[0]:,} rows × {clf_df.shape[1]} columns")

print(f"""
{'='*70}
PREPROCESSING COMPLETE
{'='*70}

Summary:
  Raw records:      20,640
  After cleaning:   {len(df):,}
  Engineered feats: {n_engineered}
  Regression:       {reg_df.shape[1]-1} features → MedHouseVal
  Classification:   {clf_df.shape[1]-1} features → is_high_value

Next step: python enrich_data.py
""")
