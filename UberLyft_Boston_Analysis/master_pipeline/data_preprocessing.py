import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 1. LOAD DATA
reg_df = pd.read_csv("../new_data/processed/regression_dataset_enriched.csv")
clf_df = pd.read_csv("../new_data/processed/classification_dataset_enriched.csv")
print("\n--- 1. LOADED DATA ---")
print(reg_df[['distance', 'price', 'cab_type']].head(3))

# 2. CLEANING & TARGET DEFINITION
reg_df = reg_df.dropna().query('price > 0')
clf_df = clf_df.dropna()
if 'is_expensive' in clf_df.columns:
    clf_df.drop(columns=['is_expensive'], inplace=True)
print("\n--- 2. CLEANED ---")

# 3. ENCODING
cat_cols = ['cab_type', 'name', 'source', 'destination']
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([reg_df[col], clf_df[col] if col in clf_df.columns else pd.Series()]).astype(str)
    le.fit(combined)
    reg_df[col] = le.transform(reg_df[col].astype(str))
    if col in clf_df.columns:
        clf_df[col] = le.transform(clf_df[col].astype(str))
print("\n--- 3. ENCODED CATEGORIES ---")
print(reg_df[cat_cols].head(3))

# 4. REMOVE REDUNDANCY & LEAKAGE
clf_df.drop(columns=['name', 'cab_type'], inplace=True, errors='ignore')
drop_logic = ['short_ride', 'medium_ride', 'long_ride']
reg_df.drop(columns=drop_logic, inplace=True, errors='ignore')
clf_df.drop(columns=drop_logic, inplace=True, errors='ignore')
print("\n--- 4. REMOVED LEAKAGE/REDUNDANCY ---")

# 5. OUTLIER HANDLING (REGRESSION)
q_low, q_high = reg_df['price'].quantile([0.01, 0.99])
reg_df = reg_df[(reg_df['price'] >= q_low) & (reg_df['price'] <= q_high)]
print("\n--- 5. OUTLIERS REMOVED ---")
print(f"Price Range: {reg_df['price'].min()} - {reg_df['price'].max()}")

# 6. SCALING & CONSTANT FEATURE REMOVAL
num_cols = reg_df.select_dtypes(include=['float64', 'int64']).columns.drop(['price', 'is_premium'], errors='ignore')
scaler = StandardScaler()

reg_df[num_cols] = scaler.fit_transform(reg_df[num_cols])
clf_num_cols = clf_df.select_dtypes(include=['float64', 'int64']).columns.drop(['is_premium'], errors='ignore')
clf_df[clf_num_cols] = scaler.fit_transform(clf_df[clf_num_cols])

# Drop columns that have 0 variance (constant values)
reg_df = reg_df.loc[:, reg_df.std() > 0]
clf_df = clf_df.loc[:, clf_df.std() > 0]
print("\n--- 6. SCALING & CONSTANT REMOVAL COMPLETE ---")

# 7. INTEGRITY CHECKS
print("\n--- 7. INTEGRITY CHECKS ---")
# Classification Balance
counts = clf_df['is_premium'].value_counts(normalize=True) * 100
print(f"Target Distribution (is_premium):\n{counts.to_string(header=False)} %")

# Regression Correlations
top_corr = reg_df.corr()['price'].abs().sort_values(ascending=False)[1:4]
print(f"\nTop 3 Price Drivers (Absolute Corr):\n{top_corr.to_string()}")

# 8. FINAL FORMAT (Target Last)
reg_target = reg_df.pop('price')
reg_df['price'] = reg_target
clf_target = clf_df.pop('is_premium')
clf_df['is_premium'] = clf_target

# 9. SAVE
output_path = "../new_data/processed/"
reg_df.to_csv(f"{output_path}regression_dataset_final.csv", index=False)
clf_df.to_csv(f"{output_path}classification_dataset_final.csv", index=False)

print(f"\n--- 9. SAVED ---")
print(f"Final Shapes: Reg {reg_df.shape}, Clf {clf_df.shape}")
print("PIPELINE COMPLETE")