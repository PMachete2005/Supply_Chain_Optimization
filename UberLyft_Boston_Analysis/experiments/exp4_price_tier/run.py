"""
Experiment 4: Price Tier Classification (Multi-class)
Predict which price tier a ride falls into from contextual + ride features.

Target: price_tier (4 classes: budget, economy, premium, luxury)
Why interesting: Multi-class classification with meaningful price boundaries.
Uses ALL features including service type — tests full model capability.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv('/home/kushagarwal/Downloads/archive(1)/rideshare_kaggle.csv')
df = df.dropna(subset=['price'])
print(f"Loaded {len(df):,} records")

# === TARGET: Price tiers based on quantiles ===
q25, q50, q75 = df['price'].quantile([0.25, 0.5, 0.75])
print(f"Price quartiles: Q1=${q25:.2f}, Q2=${q50:.2f}, Q3=${q75:.2f}")

def assign_tier(price):
    if price <= q25: return 0  # budget
    elif price <= q50: return 1  # economy
    elif price <= q75: return 2  # premium
    else: return 3  # luxury

df['price_tier'] = df['price'].apply(assign_tier)
tier_names = {0: 'Budget', 1: 'Economy', 2: 'Premium', 3: 'Luxury'}

print("\nTier distribution:")
for tier, name in tier_names.items():
    count = (df['price_tier'] == tier).sum()
    pct = count / len(df) * 100
    price_range = df[df['price_tier'] == tier]['price']
    print(f"  {name:<10} n={count:>7,} ({pct:.1f}%)  ${price_range.min():.2f} – ${price_range.max():.2f}")

# === FEATURES (everything except price itself) ===
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_rush_hour'] = (((df['hour']>=7)&(df['hour']<=9))|((df['hour']>=17)&(df['hour']<=19))).astype(int) * (1 - df['is_weekend'])
df['is_night'] = ((df['hour']>=22)|(df['hour']<=5)).astype(int)
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

df['is_rainy'] = (df['precipIntensity'] > 0).astype(int)
df['is_cold'] = (df['temperature'] < 40).astype(int)
df['wind_chill'] = df['apparentTemperature'] - df['temperature']
df['low_visibility'] = (df['visibility'] < 3).astype(int)

for col in ['cab_type', 'name', 'source', 'destination']:
    df[col+'_enc'] = LabelEncoder().fit_transform(df[col].astype(str))

df['distance_surge'] = df['distance'] * df['surge_multiplier']
df['short_ride'] = (df['distance'] < 2).astype(int)
df['long_ride'] = (df['distance'] >= 5).astype(int)
df['rain_rush'] = df['is_rainy'] * df['is_rush_hour']

features = [
    'distance', 'distance_surge', 'surge_multiplier',
    'short_ride', 'long_ride',
    'hour', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend',
    'is_rush_hour', 'is_night',
    'temperature', 'humidity', 'windSpeed', 'precipIntensity',
    'visibility', 'pressure', 'cloudCover', 'windGust',
    'is_rainy', 'is_cold', 'wind_chill', 'low_visibility',
    'rain_rush',
    'cab_type_enc', 'name_enc', 'source_enc', 'destination_enc'
]

df_sample = df.sample(100000, random_state=42).reset_index(drop=True)
X = df_sample[features]
y = df_sample['price_tier']

print(f"\nDataset: {len(X):,} rows × {len(features)} features → 4 classes")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# === MODELS ===
print("\n" + "=" * 80)
print("PRICE TIER PREDICTION — 5 WHITE-BOX MODELS (4-class)")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Naive Bayes': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'Perceptron': Perceptron(max_iter=1000, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"\n--- {name} ---")
    use_scaled = name not in ['Decision Tree']
    Xtr = X_train_s if use_scaled else X_train
    Xte = X_test_s if use_scaled else X_test
    
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average='macro')
    f1_weighted = f1_score(y_test, preds, average='weighted')
    results[name] = {'Acc': acc, 'F1_macro': f1_macro, 'F1_weighted': f1_weighted}
    print(f"Acc={acc:.4f}  F1(macro)={f1_macro:.4f}  F1(weighted)={f1_weighted:.4f}")
    
    if hasattr(model, 'feature_importances_'):
        top5 = np.argsort(model.feature_importances_)[-5:][::-1]
        print("Top features:")
        for i in top5:
            print(f"  {features[i]:<30} {model.feature_importances_[i]:.4f}")

# Best model — full report
best_name = max(results, key=lambda k: results[k]['F1_macro'])
print(f"\n--- Best Model: {best_name} — Full Classification Report ---")
best_model = models[best_name]
use_scaled = best_name not in ['Decision Tree']
Xte = X_test_s if use_scaled else X_test
preds = best_model.predict(Xte)
print(classification_report(y_test, preds, target_names=[tier_names[i] for i in range(4)]))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n{'Model':<22} {'Acc':<10} {'F1(macro)':<12} {'F1(weighted)':<12}")
print("-" * 56)
for name, m in sorted(results.items(), key=lambda x: x[1]['F1_macro'], reverse=True):
    print(f"{name:<22} {m['Acc']:.4f}    {m['F1_macro']:.4f}      {m['F1_weighted']:.4f}")
