import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Paths
FLIGHT_CSV = "/tmp/flight_sample_stratified.csv"
MODELS_DIR = "/home/kushagarwal/CascadeProjects/Supply_Chain_Optimization/master_pipeline/models"


def add_flight_features(df):
    """Add engineered features for flight delay prediction"""
    df = df.copy()
    
    # Time-based features
    df['is_early_morning'] = ((df['hour'] >= 5) & (df['hour'] < 8)).astype(int)
    df['is_morning'] = ((df['hour'] >= 8) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
    df['is_evening'] = ((df['hour'] >= 16) & (df['hour'] < 20)).astype(int)
    df['is_night'] = ((df['hour'] >= 20) | (df['hour'] < 5)).astype(int)
    
    # Weekend indicator
    df['is_weekend'] = 0  # Need day of week, not available
    
    # Season features
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
    
    # Distance categories
    df['short_haul'] = (df['distance'] < 500).astype(int)
    df['medium_haul'] = ((df['distance'] >= 500) & (df['distance'] < 1500)).astype(int)
    df['long_haul'] = (df['distance'] >= 1500).astype(int)
    
    # Flight load proxy (if available)
    if 'totflight' in df.columns:
        df['high_traffic_route'] = (df['totflight'] > df['totflight'].median()).astype(int)
    
    # Interaction features
    df['morning_busy_route'] = df['is_morning'] * df.get('high_traffic_route', 0)
    df['winter_long_haul'] = df['is_winter'] * df['long_haul']
    
    # Previous delay patterns by route (if avgdelay available)
    if 'avgdelay' in df.columns:
        df['high_avg_delay_route'] = (df['avgdelay'] > df['avgdelay'].median()).astype(int)
    
    return df


def encode_features(X, encoders=None, fit=True):
    """Encode categorical features"""
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if encoders is None:
        encoders = {}
    
    for col in cat_cols:
        X[col] = X[col].astype(str)
        if fit:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is not None:
                X[col] = X[col].map(lambda v: le.transform([v])[0] if v in set(le.classes_) else -1)
            else:
                X[col] = -1
    
    return X, encoders


def main():
    print("="*70)
    print("FLIGHT DELAY PREDICTION - ML PIPELINE")
    print("="*70)
    
    # Load data
    print("\nLoading flight data...")
    df = pd.read_csv(FLIGHT_CSV)
    print(f"Dataset: {df.shape[0]:,} records, {df.shape[1]} features")
    
    # Add engineered features
    print("\nAdding engineered features...")
    df = add_flight_features(df)
    print(f"Features after engineering: {df.shape[1]}")
    
    # Define features to use
    feature_cols = [
        'month', 'day', 'hour', 'minute', 'distance',
        'is_early_morning', 'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        'is_winter', 'is_spring', 'is_summer', 'is_fall',
        'short_haul', 'medium_haul', 'long_haul',
        'carrier', 'origin', 'dest'
    ]
    
    # Add optional features if available
    if 'avgdelay' in df.columns:
        feature_cols.append('avgdelay')
    if 'high_traffic_route' in df.columns:
        feature_cols.extend(['high_traffic_route', 'morning_busy_route'])
    if 'high_avg_delay_route' in df.columns:
        feature_cols.append('high_avg_delay_route')
    
    # Ensure all feature columns exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"\nUsing {len(feature_cols)} features for modeling")
    
    # Prepare data
    X = df[feature_cols].copy()
    y_dep = df['is_delayed_dep']  # Classification target
    y_arr_delay = df['arr_delay']  # Regression target
    
    # Encode features
    X_enc, encoders = encode_features(X, fit=True)
    
    # Split data
    X_train, X_test, y_train_c, y_test_c = train_test_split(
        X_enc, y_dep, test_size=0.2, random_state=42, stratify=y_dep
    )
    _, _, y_train_r, y_test_r = train_test_split(
        X_enc, y_arr_delay, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train):,}, Test set: {len(X_test):,}")
    
    # ==================== CLASSIFICATION ====================
    print("\n" + "="*70)
    print("CLASSIFICATION: Departure Delay > 15 minutes")
    print("="*70)
    
    clf_models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "Random Forest (50)": RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
    }
    
    clf_results = []
    
    for name, model in clf_models.items():
        print(f"\n{name}:")
        start = time.time()
        model.fit(X_train, y_train_c)
        train_time = time.time() - start
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test_c, preds)
        prec = precision_score(y_test_c, preds, average="weighted")
        rec = recall_score(y_test_c, preds, average="weighted")
        f1 = f1_score(y_test_c, preds, average="macro")
        
        print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"  Time: {train_time:.2f}s")
        
        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X_enc.columns).nlargest(5)
            print("  Top 5 features:")
            for feat, val in importances.items():
                print(f"    {feat}: {val:.4f}")
        
        clf_results.append({
            "model_name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "time": train_time
        })
    
    clf_results.sort(key=lambda x: x["accuracy"], reverse=True)
    print(f"\nBest classifier: {clf_results[0]['model_name']} ({clf_results[0]['accuracy']:.4f})")
    
    # ==================== REGRESSION ====================
    print("\n" + "="*70)
    print("REGRESSION: Arrival Delay (minutes)")
    print("="*70)
    
    reg_models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=20, random_state=42),
        "Random Forest (50)": RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
    }
    
    reg_results = []
    
    for name, model in reg_models.items():
        print(f"\n{name}:")
        start = time.time()
        model.fit(X_train, y_train_r)
        train_time = time.time() - start
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test_r, preds)
        rmse = np.sqrt(mean_squared_error(y_test_r, preds))
        r2 = r2_score(y_test_r, preds)
        
        print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        print(f"  Time: {train_time:.2f}s")
        
        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X_enc.columns).nlargest(5)
            print("  Top 5 features:")
            for feat, val in importances.items():
                print(f"    {feat}: {val:.4f}")
        
        reg_results.append({
            "model_name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "time": train_time
        })
    
    reg_results.sort(key=lambda x: x["r2"], reverse=True)
    print(f"\nBest regressor: {reg_results[0]['model_name']} ({reg_results[0]['r2']:.4f})")
    
    # Error analysis
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    
    best_reg = reg_results[0]
    best_model = reg_models[best_reg['model_name']]
    preds_r = best_model.predict(X_test)
    residuals = y_test_r - preds_r
    
    print(f"\nAnalyzing {best_reg['model_name']} residuals:")
    print(f"  Mean residual: {residuals.mean():.3f}")
    print(f"  Std of residuals: {residuals.std():.3f}")
    print(f"  Largest under-predictions: {residuals.nlargest(3).values}")
    print(f"  Largest over-predictions: {residuals.nsmallest(3).values}")
    
    # ==================== COMPARISON SUMMARY ====================
    print("\n" + "="*70)
    print("COMPARISON: Flight vs Supply Chain Dataset")
    print("="*70)
    
    print("\nFlight Delay Dataset Results:")
    print(f"  Classification Accuracy: {clf_results[0]['accuracy']:.4f}")
    print(f"  Regression R²: {reg_results[0]['r2']:.4f}")
    print(f"  Dataset size: 30,000 flights")
    print(f"  Target: 19.1% delayed departures")
    
    print("\nSupply Chain Dataset Results (for comparison):")
    print(f"  Classification Accuracy: ~0.77")
    print(f"  Regression R²: ~0.54")
    print(f"  Dataset size: 180,519 orders")
    print(f"  Target: Variable delivery risk")
    
    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save best models
    joblib.dump(clf_models[clf_results[0]['model_name']], 
                os.path.join(MODELS_DIR, "flight_delay_classifier.joblib"))
    joblib.dump(reg_models[reg_results[0]['model_name']], 
                os.path.join(MODELS_DIR, "flight_delay_regressor.joblib"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "flight_encoders.joblib"))
    
    print("Models saved to:", MODELS_DIR)
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
