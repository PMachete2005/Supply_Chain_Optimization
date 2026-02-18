# =====================================================
# DATA CLEANING & FEATURE ENGINEERING
# Cross-Border Trade & Customs Dataset (NEW VERSION)
# =====================================================

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import json

# =====================================================
# 1. LOAD DATA
# =====================================================

RAW_PATH = "../data/raw/trade_customs_dataset_backup.csv"
OUTPUT_PATH = "processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

df = pd.read_csv(RAW_PATH)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)
print("----------------------------------")


# =====================================================
# 2. DATE CONVERSIONS
# =====================================================

date_cols = [
    'Shipment_Date',
    'Estimated_Arrival_Date',
    'Actual_Arrival_Date'
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col])

print("Date columns converted")
print("----------------------------------")


# =====================================================
# 3. TRANSIT & CALENDAR FEATURES
# =====================================================

df['Planned_Transit_Days'] = (
    df['Estimated_Arrival_Date'] - df['Shipment_Date']
).dt.days

df['Actual_Transit_Days'] = (
    df['Actual_Arrival_Date'] - df['Shipment_Date']
).dt.days

df['Arrival_Delay_Days'] = (
    df['Actual_Arrival_Date'] - df['Estimated_Arrival_Date']
).dt.days

df['Shipment_Month'] = df['Shipment_Date'].dt.month
df['Shipment_Weekday'] = df['Shipment_Date'].dt.weekday

print("Transit features created")
print("----------------------------------")


# =====================================================
# 4. COMPLIANCE & RISK FEATURES
# =====================================================

df['Has_Prior_Offense'] = (df['Prior_Offense_Count'] > 0).astype(int)

df['Compliance_Risk_Score'] = (
    (1 - df['Compliance_Score']) +
    (df['Prior_Offense_Count'] * 0.3)
)

df['Document_Issue'] = df['Document_Status'].isin(
    ['Missing', 'Error']
).astype(int)

df['Route_Risk_Level'] = pd.cut(
    df['Route_Risk_Index'],
    bins=[0, 0.33, 0.66, 1.0],
    labels=['Low', 'Medium', 'High']
)

print("Risk features created")
print("----------------------------------")


# =====================================================
# 5. TEXT FEATURE (TF-IDF)
# =====================================================

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=300
)

delay_tfidf = tfidf.fit_transform(df['Delay_Reason'])
print("TF-IDF shape:", delay_tfidf.shape)
print("----------------------------------")


# =====================================================
# 6. LABEL ENCODING CATEGORICAL FEATURES
# =====================================================

categorical_cols = [
    'Origin_Country',
    'Destination_Country',
    'Transport_Mode',
    'Carrier_Name',
    'Route_Code',
    'Commodity_Type',
    'Tariff_Category',
    'Inspection_Type',
    'Document_Status',
    'Route_Risk_Level'
]

le = LabelEncoder()

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("Categorical encoding complete")
print("----------------------------------")


# =====================================================
# 7. SCALING NUMERIC FEATURES
# =====================================================

numeric_cols = [
    'Declared_Value_USD',
    'Weight_kg',
    'Compliance_Score',
    'Prior_Offense_Count',
    'Route_Risk_Index',
    'Planned_Transit_Days',
    'Actual_Transit_Days',
    'Compliance_Risk_Score'
]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("Numeric scaling complete")
print("----------------------------------")


# =====================================================
# 8. DROP UNUSED COLUMNS
# =====================================================

drop_cols = [
    'Shipment_ID',
    'Shipment_Date',
    'Estimated_Arrival_Date',
    'Actual_Arrival_Date',
    'Delay_Reason'
]

df.drop(columns=drop_cols, inplace=True)

print("Columns dropped")
print("Current Shape:", df.shape)
print("----------------------------------")


# =====================================================
# 9. DEFINE TARGETS & FEATURES
# =====================================================

# Regression target
TARGET_DELAY = 'Arrival_Delay_Days'

# Classification target
TARGET_RISK = 'Route_Risk_Level'

exclude_cols = [
    TARGET_DELAY,
    TARGET_RISK,
    'Risk_Flag'  # if exists
]

X = df.drop(columns=exclude_cols, errors='ignore')

y_reg = df[TARGET_DELAY]
y_clf = df[TARGET_RISK]

print("Feature matrix shape:", X.shape)
print("----------------------------------")


# =====================================================
# 10. CREATE FINAL DATASETS
# =====================================================

regression_df = X.copy()
regression_df[TARGET_DELAY] = y_reg

classification_df = X.copy()
classification_df[TARGET_RISK] = y_clf


# =====================================================
# 11. SAVE FILES
# =====================================================

regression_path = os.path.join(
    OUTPUT_PATH,
    "new_customs_regression_dataset.csv"
)

classification_path = os.path.join(
    OUTPUT_PATH,
    "new_customs_classification_dataset.csv"
)

metadata_path = os.path.join(
    OUTPUT_PATH,
    "new_feature_metadata.json"
)

regression_df.to_csv(regression_path, index=False)
classification_df.to_csv(classification_path, index=False)

feature_metadata = {
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "regression_target": TARGET_DELAY,
    "classification_target": TARGET_RISK
}

with open(metadata_path, "w") as f:
    json.dump(feature_metadata, f, indent=4)

print("Files saved successfully.")
print("Regression shape:", regression_df.shape)
print("Classification shape:", classification_df.shape)
print("----------------------------------")
