"""
Experiment 2: Within-Service Price Prediction
Predict price for a SINGLE service type (e.g., UberX) from contextual features.

Why: Full dataset price is trivially determined by service_name.
     Within a single service, price variation comes from distance, surge, and context.
     This tests whether weather/time/location actually affect pricing.

Target: price (regression)
Dataset: UberX rides only (~55K rides)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import statsmodels.api as sm
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv('/home/kushagarwal/Downloads/archive(1)/rideshare_kaggle.csv')
df = df.dropna(subset=['price'])

print("Service types and counts:")
for name, group in df.groupby('name'):
    print(f"  {name:<20} n={len(group):>6,}  avg_price=${group['price'].mean():.2f}  std=${group['price'].std():.2f}")

# Filter to single service types with most variation
services_to_test = {
    'UberX': df[df['name'] == 'UberX'].copy(),
    'Lyft': df[df['name'] == 'Lyft'].copy(),
    'Black SUV': df[df['name'] == 'Black SUV'].copy(),
}

for service_name, sdf in services_to_test.items():
    print(f"\n{'='*80}")
    print(f"SERVICE: {service_name} ({len(sdf):,} rides)")
    print(f"Price range: ${sdf['price'].min():.2f} – ${sdf['price'].max():.2f}, mean=${sdf['price'].mean():.2f}")
    print(f"{'='*80}")
    
    # Features
    sdf['timestamp'] = pd.to_datetime(sdf['timestamp'])
    sdf['hour'] = sdf['timestamp'].dt.hour
    sdf['day_of_week'] = sdf['timestamp'].dt.dayofweek
    sdf['is_weekend'] = (sdf['day_of_week'] >= 5).astype(int)
    sdf['is_rush_hour'] = (((sdf['hour']>=7)&(sdf['hour']<=9))|((sdf['hour']>=17)&(sdf['hour']<=19))).astype(int) * (1 - sdf['is_weekend'])
    sdf['is_night'] = ((sdf['hour']>=22)|(sdf['hour']<=5)).astype(int)
    sdf['hour_sin'] = np.sin(2*np.pi*sdf['hour']/24)
    sdf['hour_cos'] = np.cos(2*np.pi*sdf['hour']/24)
    
    sdf['is_rainy'] = (sdf['precipIntensity'] > 0).astype(int)
    sdf['is_cold'] = (sdf['temperature'] < 40).astype(int)
    sdf['wind_chill'] = sdf['apparentTemperature'] - sdf['temperature']
    sdf['low_visibility'] = (sdf['visibility'] < 3).astype(int)
    
    for col in ['source', 'destination']:
        sdf[col+'_enc'] = LabelEncoder().fit_transform(sdf[col].astype(str))
    
    sdf['rain_rush'] = sdf['is_rainy'] * sdf['is_rush_hour']
    sdf['distance_sq'] = sdf['distance'] ** 2
    
    # Two feature sets: with and without surge
    feats_no_surge = [
        'distance', 'distance_sq', 'hour', 'hour_sin', 'hour_cos',
        'day_of_week', 'is_weekend', 'is_rush_hour', 'is_night',
        'temperature', 'humidity', 'windSpeed', 'precipIntensity',
        'is_rainy', 'is_cold', 'wind_chill', 'low_visibility',
        'rain_rush', 'source_enc', 'destination_enc'
    ]
    
    feats_with_surge = feats_no_surge + ['surge_multiplier']
    
    for label, feats in [("Without surge", feats_no_surge), ("With surge", feats_with_surge)]:
        print(f"\n--- {label} ({len(feats)} features) ---")
        
        X = sdf[feats].values
        y = sdf['price'].values
        
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler(); Xtrs = sc.fit_transform(Xtr); Xtes = sc.transform(Xte)
        
        # OLS
        ols = sm.OLS(ytr, sm.add_constant(Xtrs)).fit()
        pred = ols.predict(sm.add_constant(Xtes))
        print(f"  OLS:    R²={r2_score(yte,pred):.4f}  MAE=${mean_absolute_error(yte,pred):.2f}")
        
        # Ridge
        m = Ridge(alpha=1.0, random_state=42).fit(Xtrs, ytr)
        pred = m.predict(Xtes)
        print(f"  Ridge:  R²={r2_score(yte,pred):.4f}  MAE=${mean_absolute_error(yte,pred):.2f}")
        
        # Lasso
        m = Lasso(alpha=0.1, random_state=42, max_iter=10000).fit(Xtrs, ytr)
        pred = m.predict(Xtes)
        print(f"  Lasso:  R²={r2_score(yte,pred):.4f}  MAE=${mean_absolute_error(yte,pred):.2f}")
        
        # Decision Tree
        m = DecisionTreeRegressor(max_depth=10, random_state=42).fit(Xtr, ytr)
        pred = m.predict(Xte)
        print(f"  Tree:   R²={r2_score(yte,pred):.4f}  MAE=${mean_absolute_error(yte,pred):.2f}")
        
        # Polynomial
        Xtr_df = pd.DataFrame(Xtr, columns=feats)
        Xte_df = pd.DataFrame(Xte, columns=feats)
        top3 = ['distance', 'distance_sq', 'surge_multiplier'] if 'surge_multiplier' in feats else ['distance', 'distance_sq', 'hour']
        top3 = [f for f in top3 if f in feats]
        poly = PolynomialFeatures(degree=2, include_bias=False)
        Xtrp = poly.fit_transform(Xtr_df[top3])
        Xtep = poly.transform(Xte_df[top3])
        m = Ridge(alpha=1.0, random_state=42).fit(Xtrp, ytr)
        pred = m.predict(Xtep)
        print(f"  Poly:   R²={r2_score(yte,pred):.4f}  MAE=${mean_absolute_error(yte,pred):.2f}")
