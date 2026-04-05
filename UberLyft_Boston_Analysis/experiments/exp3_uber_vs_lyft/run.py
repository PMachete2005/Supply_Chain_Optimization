"""
Experiment 3: Uber vs Lyft Choice
What conditions make riders choose Uber over Lyft (or vice versa)?

Target: cab_type (binary: Uber=1, Lyft=0)
Features: distance, time, weather, location (NOT price or service name)
Why interesting: Tests whether contextual factors predict platform choice.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv('/home/kushagarwal/Downloads/archive(1)/rideshare_kaggle.csv')
df = df.dropna(subset=['price'])
print(f"Loaded {len(df):,} records")

# === TARGET ===
df['is_uber'] = (df['cab_type'] == 'Uber').astype(int)
print(f"Uber rate: {df['is_uber'].mean()*100:.1f}%")

# === FEATURES ===
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
df['high_wind'] = (df['windGust'] > 20).astype(int)
df['overcast'] = (df['cloudCover'] > 0.8).astype(int)
df['severe_weather'] = df['is_rainy'] + df['is_cold'] + df['low_visibility'] + df['high_wind']

for col in ['source', 'destination']:
    df[col+'_enc'] = LabelEncoder().fit_transform(df[col].astype(str))

df['rain_rush'] = df['is_rainy'] * df['is_rush_hour']
df['short_ride'] = (df['distance'] < 2).astype(int)
df['long_ride'] = (df['distance'] >= 5).astype(int)

features = [
    'distance', 'short_ride', 'long_ride',
    'hour', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend',
    'is_rush_hour', 'is_night',
    'temperature', 'humidity', 'windSpeed', 'precipIntensity', 'precipProbability',
    'visibility', 'pressure', 'cloudCover', 'windGust', 'dewPoint',
    'is_rainy', 'is_cold', 'wind_chill', 'low_visibility', 'high_wind',
    'overcast', 'severe_weather', 'rain_rush',
    'source_enc', 'destination_enc',
    'surge_multiplier'
]

df_sample = df.sample(100000, random_state=42).reset_index(drop=True)
X = df_sample[features]
y = df_sample['is_uber']

print(f"\nDataset: {len(X):,} rows × {len(features)} features")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# === MODELS ===
print("\n" + "=" * 80)
print("UBER vs LYFT PREDICTION — 5 WHITE-BOX MODELS")
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
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds)
    results[name] = {'Acc': acc, 'F1': f1, 'Prec': prec, 'Rec': rec}
    print(f"Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        top5 = np.argsort(np.abs(coefs))[-5:][::-1]
        print("Top features:")
        for i in top5:
            print(f"  {features[i]:<30} {coefs[i]:+.4f}")
    elif hasattr(model, 'feature_importances_'):
        top5 = np.argsort(model.feature_importances_)[-5:][::-1]
        print("Top features:")
        for i in top5:
            print(f"  {features[i]:<30} {model.feature_importances_[i]:.4f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\n{'Model':<22} {'Acc':<10} {'F1':<10} {'Prec':<10} {'Rec':<10}")
print("-" * 62)
for name, m in sorted(results.items(), key=lambda x: x[1]['F1'], reverse=True):
    print(f"{name:<22} {m['Acc']:.4f}    {m['F1']:.4f}    {m['Prec']:.4f}    {m['Rec']:.4f}")
