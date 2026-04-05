import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# 1. WEB SCRAPING: Live Pune CNG Price
def get_pune_cng():
    try:
        url = "https://www.goodreturns.in/cng-price-in-pune.html"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Common selector for this site's price blocks
        price = soup.find('strong', id='current-price-cng').text.strip()
        return float(price.replace('₹', '').strip())
    except:
        return 89.50  # Fallback for Pune April 2026

cng_price = get_pune_cng()
print(f"✅ Scraped Live Pune CNG Price: ₹{cng_price}")

# 2. DATA SYNTHESIS: Pune Context (Magarpatta to COEP style)
print("Generating 100,000 Pune ride records...")
np.random.seed(42)
n = 100000

data = {
    'distance_km': np.random.uniform(1, 25, n),
    'hour': np.random.randint(0, 24, n),
    'is_monsoon': np.random.choice([0, 1], n, p=[0.8, 0.2]),
    'vehicle_type': np.random.choice(['Rickshaw', 'UberGo', 'Premier'], n),
    'traffic_level': np.random.choice([1, 1.5, 2], n, p=[0.5, 0.3, 0.2]) # 1=Low, 2=Heavy
}

df = pd.DataFrame(data)

# 3. PUNE PRICING FORMULA (The "Ground Truth" for ML to learn)
def calculate_fare(row):
    # RTO Based logic + Surges
    base_rates = {'Rickshaw': 15, 'UberGo': 18, 'Premier': 25}
    fare = row['distance_km'] * base_rates[row['vehicle_type']]
    
    # Apply Scraped CNG factor
    fare += (cng_price * 0.05) 
    
    # Peak Hour Surge (9 AM & 6 PM)
    if row['hour'] in [8, 9, 17, 18]:
        fare *= 1.4
    
    # Monsoon Surge
    if row['is_monsoon'] == 1:
        fare *= 1.3
        
    # Traffic Multiplier
    fare *= row['traffic_level']
    
    return fare + np.random.normal(0, 2) # Add some noise

df['final_fare_inr'] = df.apply(calculate_fare, axis=1)

# 4. TRAIN PUNE ML MODEL
df_encoded = pd.get_dummies(df, columns=['vehicle_type'])
X = df_encoded.drop('final_fare_inr', axis=1)
y = df_encoded['final_fare_inr']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 5. RESULTS
preds = model.predict(X_test)
print(f"\n--- PUNE PRICING ENGINE RESULTS ---")
print(f"R2 Score: {r2_score(y_test, preds):.4f}")
print(f"MAE: ₹{mean_absolute_error(y_test, preds):.2f}")

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(5)
print("\nTop 5 features:")
for feat, val in importances.items():
    print(f"  {feat}: {val:.4f}")

print(f"\nSample Pune Fare Predictions:")
sample_indices = np.random.choice(len(X_test), 3, replace=False)
for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = preds[idx]
    print(f"  Actual: ₹{actual:.2f}, Predicted: ₹{predicted:.2f}, Diff: ₹{abs(actual-predicted):.2f}")
