import pandas as pd 
import numpy as np

#steps done 
#1. Clean data
#2. Handle missing values
#3. Remove leakage
#4. Feature engineering (dates)
#5. Encoding
#6. Save datasets

df = pd.read_csv("../new_data/processed/DataCoSupplyChainDataset_enriched.csv")

# General info
print(df.head())    
print(df.shape)
print(df.info())

#------------------------------------------------------------

# check and remove duplicate rows
print("Duplicate rows before:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)

#------------------------------------------------------------

# basic inconsistency checks

# enforce non-negative where logically required
non_negative_cols = [
    'Sales',
    'Order Item Quantity',
    'Product Price',
    'Order Item Total'
]

for col in non_negative_cols:
    if col in df.columns:
        print(f"{col} negative values:", (df[col] < 0).sum())
        df[col] = df[col].clip(lower=0)

# allow negative profit (real-world losses)
print("Benefit per order negative values:", (df['Benefit per order'] < 0).sum())

#------------------------------------------------------------

# Check missing values
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

# Drop useless columns
df.drop(columns=[
    'Product Description',  
    'Customer Email',
    'Customer Password',
    'Product Image',
    'Order Zipcode'  
], inplace=True)

# Fill small missing values
df['Customer Lname'] = df['Customer Lname'].fillna("Unknown")
df['Customer Zipcode'] = df['Customer Zipcode'].fillna(df['Customer Zipcode'].median())

# Handle LPI columns
lpi_cols = [
    'Destination_LPI_Overall',
    'Destination_LPI_Customs',
    'Destination_LPI_Infrastructure',
    'Destination_LPI_Logistics',
    'Destination_LPI_Tracking',
    'Destination_LPI_Timeliness',
    'Route_LPI_Average',
    'Route_LPI_Difference'
]

for col in lpi_cols:
    df[col] = df[col].fillna(df[col].median())

# Final  -> dataset is now clean and stable
print("Total missing values:", df.isnull().sum().sum())

#------------------------------------------------------------

#delete leakage columns 

date_cols = [
    'order date (DateOrders)',
    'shipping date (DateOrders)'
]

for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month
    df[col + '_day'] = df[col].dt.day

# drop original date columns
df.drop(columns=date_cols, inplace=True)

#---------------------------------------------

leakage_cols = [
    'Days for shipping (real)',
    'Delivery Status'
]

id_cols = [
    'Order Id',
    'Customer Id',
    'Order Customer Id',
    'Order Item Id',
    'Product Card Id',
    'Order Item Cardprod Id'
]

text_cols = [
    'Customer Fname',
    'Customer Lname',
    'Customer Street',
    'Product Name'
]

df.drop(columns=leakage_cols + id_cols + text_cols, inplace=True)
print("New shape after column dropping:", df.shape)

#------------------------------------------------------------

#encoding categorical variables-> results in higher number of columns due to one-hot encoding... low-cardinality columns expanded and high cardinality columns label encoded
cat_cols = df.select_dtypes(include='object').columns
print("Categorical columns:", cat_cols)
print("Count: ", len(cat_cols))

low_card_cols = [col for col in cat_cols if df[col].nunique() < 10]
high_card_cols = [col for col in cat_cols if df[col].nunique() >= 10]

# one-hot encoding
df = pd.get_dummies(df, columns=low_card_cols, drop_first=True)

from sklearn.preprocessing import LabelEncoder 
label_encoders = {}

# label encoding
for col in high_card_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Remaining object columns:",
      df.select_dtypes(include='object').columns)

print("New shape after encoding:", df.shape)

#------------------------------------------------------------

# classification -> predict if an order will be delivered late 0 = ontime and 1 = late 
classification_df = df.copy()

# target already exists
classification_target = 'Late_delivery_risk'

# drop regression target to avoid leakage
classification_df = classification_df.drop(columns=['Days for shipment (scheduled)'])
classification_df.to_csv("../src/data/classification_dataset.csv", index=False)
print("Classification dataset shape:", classification_df.shape)

#------------------------------------------------------------

# regression -> predict expected shipping duration (days)
regression_df = df.copy()

regression_target = 'Days for shipment (scheduled)'

# drop classification target
regression_df = regression_df.drop(columns=['Late_delivery_risk'])
regression_df.to_csv("../src/data/regression_dataset.csv", index=False)
print("Regression dataset shape:", regression_df.shape)