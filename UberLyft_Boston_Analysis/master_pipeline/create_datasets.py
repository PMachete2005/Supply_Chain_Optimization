import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# STEP 1: LOAD DATA
df = pd.read_csv('/home/kushagarwal/Downloads/archive(1)/rideshare_kaggle.csv')

# STEP 2: CLEAN DATA
df_clean = df.dropna(subset=['price'])
df_clean = df_clean[df_clean['price'] > 0]
df_clean = df_clean.dropna(subset=['distance', 'surge_multiplier'])

# STEP 3: STRATIFIED SAMPLING
df_clean['price_category'] = pd.cut(df_clean['price'], bins=[0, 10, 15, 25, 1000], 
                                   labels=['low', 'mid', 'high', 'premium'])

df_clean['strata'] = (
    df_clean['cab_type'].astype(str) + '_' +
    df_clean['name'].astype(str) + '_' +
    df_clean['price_category'].astype(str)
)

strata_counts = df_clean['strata'].value_counts()
valid_strata = strata_counts[strata_counts >= 10].index
df_valid = df_clean[df_clean['strata'].isin(valid_strata)]

sample_size = min(100000, len(df_valid))

df_sample = df_valid.groupby('strata').apply(
    lambda x: x.sample(n=max(1, int(len(x) / len(df_valid) * sample_size)), random_state=RANDOM_STATE)
).reset_index(drop=True)

if len(df_sample) < 100000:
    remaining = 100000 - len(df_sample)
    additional = df_valid.drop(df_sample.index, errors='ignore').sample(
        n=min(remaining, len(df_valid) - len(df_sample)), random_state=RANDOM_STATE)
    df_sample = pd.concat([df_sample, additional]).reset_index(drop=True)

# STEP 4: FEATURE ENGINEERING
df = df_sample.copy()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9) & (df['is_weekend'] == 0)).astype(int)
df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19) & (df['is_weekend'] == 0)).astype(int)
df['is_rush_hour'] = df['is_morning_rush'] | df['is_evening_rush']
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

df['short_ride'] = (df['distance'] < 2).astype(int)
df['medium_ride'] = ((df['distance'] >= 2) & (df['distance'] < 5)).astype(int)
df['long_ride'] = (df['distance'] >= 5).astype(int)

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

df['distance_surge'] = df['distance'] * df['surge_multiplier']
df['weather_surge'] = df['is_rainy'] * df['surge_multiplier']
df['rush_surge'] = df['is_rush_hour'] * df['surge_multiplier']

df['is_premium'] = df['name'].str.contains('Black|Lux|SUV|XL', case=False, na=False).astype(int)
df['uber_premium'] = ((df['cab_type'] == 'Uber') & (df['is_premium'] == 1)).astype(int)
df['lyft_premium'] = ((df['cab_type'] == 'Lyft') & (df['is_premium'] == 1)).astype(int)

# STEP 5: ENCODING
cat_cols = ['cab_type', 'name', 'source', 'destination']
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))

# STEP 6 & 7: FEATURES + TARGETS
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

regression_features = [col for col in regression_features if col in df.columns]

classification_features = [
    col for col in regression_features
    if 'surge' not in col and col != 'surge_multiplier'
]

y_price = df['price']
y_premium = df['name'].str.contains('Black|Lux|SUV|XL', case=False, na=False).astype(int)

# STEP 8: CREATE & SAVE DATASETS
regression_df = df[regression_features + ['price']].copy()
regression_df.columns = [col.replace('_encoded', '') for col in regression_df.columns]

classification_df = df[classification_features].copy()
classification_df['is_premium'] = y_premium.values
classification_df.columns = [col.replace('_encoded', '') for col in classification_df.columns]

output_dir = '/home/kushagarwal/CascadeProjects/UberLyft_Boston_Analysis'

reg_file = f'{output_dir}/regression_dataset.csv'
clf_file = f'{output_dir}/classification_dataset.csv'

regression_df.to_csv(reg_file, index=False)
classification_df.to_csv(clf_file, index=False)
