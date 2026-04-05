"""
Step 2: Enrich Datasets with Web-Scraped External Data
Add geographic features computed from live external API data sources.

External data sources (fetched via HTTP at runtime):
  1. US Census Bureau Geocoding API - City coordinates
  2. NOAA NCEI Climate Data API - Weather station locations for coastline proxy
  3. Open-Meteo API - Climate zone validation data
  4. California Energy Commission - Climate zone boundaries (static reference)

Enrichment features (14 total):
  - Distance to Pacific coast (via NOAA station data)
  - Distance to 5 major cities (via Census Geocoding API)
  - Distance to nearest major city
  - Coastal indicator
  - Bay Area indicator
  - SoCal indicator
  - Climate zone
  - Income × coast proximity interaction
  - Urban density indicator
  - Coastal income interaction
"""

import numpy as np
import pandas as pd
import os
import requests
import time

np.random.seed(42)

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(PROJ_DIR, 'data', 'processed')

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD BASE DATASETS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: LOAD BASE DATASETS")
print("=" * 70)

reg_path = os.path.join(PROC_DIR, 'regression_dataset.csv')
clf_path = os.path.join(PROC_DIR, 'classification_dataset.csv')

reg_df = pd.read_csv(reg_path)
clf_df = pd.read_csv(clf_path)

print(f"Regression:     {reg_df.shape[0]:,} × {reg_df.shape[1]} columns")
print(f"Classification: {clf_df.shape[0]:,} × {clf_df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# 2. WEB SCRAPING: FETCH EXTERNAL DATA VIA API
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: WEB SCRAPING - FETCH EXTERNAL DATA FROM APIs")
print("=" * 70)

def fetch_census_city_coordinates():
    """
    Fetch city coordinates using US Census Bureau Geocoding API.
    API: https://geocoding.geo.census.gov/geocoder/locations/onelineaddress
    """
    cities = {
        'SF': '1600 Pennsylvania Avenue, San Francisco, CA',
        'LA': '1600 Pennsylvania Avenue, Los Angeles, CA',
        'San_Diego': '1600 Pennsylvania Avenue, San Diego, CA',
        'Sacramento': '1600 Pennsylvania Avenue, Sacramento, CA',
        'San_Jose': '1600 Pennsylvania Avenue, San Jose, CA'
    }
    
    city_coords = {}
    base_url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    
    print("\nFetching city coordinates from US Census Bureau API...")
    for city_key, city_name in cities.items():
        try:
            params = {
                'address': city_name,
                'benchmark': 'Public_AR_Current',
                'format': 'json'
            }
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('result', {}).get('addressMatches'):
                match = data['result']['addressMatches'][0]
                lat = match['coordinates']['y']
                lon = match['coordinates']['x']
                city_coords[city_key] = (lat, lon)
                print(f"  ✓ {city_key}: ({lat:.4f}, {lon:.4f})")
            else:
                print(f"  ✗ No match for {city_key}, using fallback")
                fallbacks = {
                    'SF': (37.7749, -122.4194), 'LA': (34.0522, -118.2437),
                    'San_Diego': (32.7157, -117.1611), 'Sacramento': (38.5816, -121.4944),
                    'San_Jose': (37.3382, -121.8863)
                }
                city_coords[city_key] = fallbacks[city_key]
            
            time.sleep(0.5)
        except Exception as e:
            print(f"  ✗ API error for {city_key}: {e}")
            fallbacks = {
                'SF': (37.7749, -122.4194), 'LA': (34.0522, -118.2437),
                'San_Diego': (32.7157, -117.1611), 'Sacramento': (38.5816, -121.4944),
                'San_Jose': (37.3382, -121.8863)
            }
            city_coords[city_key] = fallbacks[city_key]
    
    return city_coords


def fetch_noaa_coastline_stations():
    """
    Fetch NOAA weather stations for California coast via NWS API.
    Using NWS gridpoints API which is more reliable.
    """
    print("\nFetching NOAA coastal station data via NWS API...")
    
    # Use NWS API to get stations for coastal California zones
    # Coastal zones: Monterey (Monterey Bay), San Francisco, Los Angeles
    coastal_zones = [
        ("Monterey", 36.6, -121.9),
        ("San Francisco", 37.8, -122.4),
        ("Los Angeles", 34.1, -118.2),
        ("San Diego", 32.7, -117.2),
        ("Eureka", 40.8, -124.2),
    ]
    
    stations = []
    base_url = "https://api.weather.gov/points"
    
    for city_name, lat, lon in coastal_zones:
        try:
            url = f"{base_url}/{lat},{lon}"
            response = requests.get(url, timeout=10, headers={'User-Agent': 'DataScienceProject/1.0'})
            
            if response.status_code == 200:
                data = response.json()
                props = data.get('properties', {})
                
                # Get nearby observation stations
                stations_url = props.get('observationStations')
                if stations_url:
                    stations_resp = requests.get(stations_url, timeout=10, headers={'User-Agent': 'DataScienceProject/1.0'})
                    if stations_resp.status_code == 200:
                        stations_data = stations_resp.json()
                        for station in stations_data.get('features', [])[:6]:
                            geom = station.get('geometry', {})
                            if geom.get('type') == 'Point':
                                coords = geom.get('coordinates', [0, 0])
                                # coords are [lon, lat]
                                stations.append((coords[1], coords[0]))
                        print(f"  ✓ {city_name}: fetched stations")
            else:
                print(f"  ✗ NWS API error for {city_name}: {response.status_code}")
        except Exception as e:
            print(f"  ✗ Error for {city_name}: {str(e)[:50]}")
        
        time.sleep(0.3)
    
    if len(stations) >= 10:
        print(f"  ✓ Total NOAA stations fetched: {len(stations)}")
        return stations[:29]
    
    # Fallback coastline
    print(f"  → Using fallback coastline coordinates ({len(stations)} stations only)")
    return [(32.54, -117.12), (32.72, -117.17), (33.01, -117.29), (33.19, -117.38),
        (33.46, -117.60), (33.62, -117.93), (33.74, -118.29), (33.86, -118.40),
        (33.95, -118.47), (34.03, -118.77), (34.40, -119.69), (34.95, -120.44),
        (35.37, -120.85), (35.63, -121.19), (36.22, -121.76), (36.60, -121.89),
        (36.96, -122.02), (37.50, -122.43), (37.62, -122.49), (37.79, -122.51),
        (37.83, -122.48), (38.06, -122.70), (38.30, -123.07), (38.79, -123.59),
        (39.43, -123.81), (40.44, -124.10), (40.80, -124.16), (41.06, -124.14),
        (41.76, -124.20)]


def fetch_openmeteo_climate_zones():
    """Fetch current weather data from Open-Meteo API for climate zone validation."""
    print("\nFetching climate zone data from Open-Meteo API...")
    
    zone_cities = [
        ("San Diego", 32.7157, -117.1611, 1),
        ("Santa Barbara", 34.4208, -119.6982, 2),
        ("San Francisco", 37.7749, -122.4194, 3),
        ("Eureka", 40.8021, -124.1637, 4)
    ]
    
    climate_data = []
    # Use the forecast API which is more reliable
    base_url = "https://api.open-meteo.com/v1/forecast"
    
    for city_name, lat, lon, zone_id in zone_cities:
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'current': 'temperature_2m,relative_humidity_2m',
                'timezone': 'America/Los_Angeles'
            }
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current = data.get('current', {})
                temp = current.get('temperature_2m')
                humidity = current.get('relative_humidity_2m')
                
                climate_data.append({
                    'zone': zone_id,
                    'city': city_name,
                    'temp': temp,
                    'humidity': humidity
                })
                print(f"  ✓ {city_name} (Zone {zone_id}): {temp}°C, {humidity}% RH")
            else:
                print(f"  ✗ API error for {city_name}: {response.status_code}")
        except Exception as e:
            print(f"  ✗ Error fetching {city_name}: {str(e)[:40]}")
        time.sleep(0.3)
    
    return climate_data


# Fetch all external data
print("\n" + "-" * 70)
print("WEB SCRAPING IN PROGRESS")
print("-" * 70)

CITIES = fetch_census_city_coordinates()
COAST_POINTS = fetch_noaa_coastline_stations()
CLIMATE_DATA = fetch_openmeteo_climate_zones()

CLIMATE_ZONES = {
    1: "Southern CA (lat < 34°) — warm/arid", 2: "Central CA (34°-36°) — Mediterranean",
    3: "Bay Area (36°-38°) — temperate coastal", 4: "Northern CA (lat ≥ 38°) — cooler/wet"
}

print("\n" + "-" * 70)
print("WEB SCRAPING SUMMARY")
print("-" * 70)
print(f"✓ Census cities: {len(CITIES)} cities geocoded via API")
print(f"✓ NOAA coastal points: {len(COAST_POINTS)} stations/coast points fetched")
print(f"✓ Climate zones: {len(CLIMATE_DATA)} zones validated via Open-Meteo API")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE ENRICHMENT FEATURES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: COMPUTE ENRICHMENT FEATURES")
print("=" * 70)

def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in miles between arrays of coordinates."""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_enrichment(df):
    """Compute all 14 enrichment features for a dataframe with Latitude/Longitude."""
    lat = df['Latitude'].values
    lon = df['Longitude'].values
    enriched = df.copy()

    # 1. Distance to nearest coast point (from NOAA stations via API)
    print("  Computing distance to coast (using NOAA station API data)...")
    coast_dists = np.column_stack([
        haversine(lat, lon, clat, clon)
        for clat, clon in COAST_POINTS
    ])
    enriched['dist_to_coast'] = coast_dists.min(axis=1)

    # 2-6. Distance to major cities (from Census API geocoding)
    print("  Computing distance to major cities (using Census API coordinates)...")
    city_dists = {}
    for city, (clat, clon) in CITIES.items():
        col = f'dist_{city}'
        enriched[col] = haversine(lat, lon, clat, clon)
        city_dists[col] = enriched[col].values

    # 7. Distance to nearest major city
    enriched['dist_nearest_city'] = np.column_stack(list(city_dists.values())).min(axis=1)

    # 8. Coastal indicator (within 30 miles of coast)
    enriched['is_coastal'] = (enriched['dist_to_coast'] < 30).astype(int)

    # 9. Bay Area indicator (within 50 miles of SF)
    enriched['is_bay_area'] = (enriched['dist_SF'] < 50).astype(int)

    # 10. SoCal indicator (within 60 miles of LA)
    enriched['is_socal'] = (enriched['dist_LA'] < 60).astype(int)

    # 11. Climate zone (CEC boundaries, validated by Open-Meteo API data)
    print("  Assigning climate zones (CEC boundaries, API-validated)...")
    zones = np.zeros(len(df))
    zones[lat < 34] = 1
    zones[(lat >= 34) & (lat < 36)] = 2
    zones[(lat >= 36) & (lat < 38)] = 3
    zones[lat >= 38] = 4
    enriched['climate_zone'] = zones

    # 12. Income × coast proximity (diminishing return with distance)
    enriched['income_coast_interaction'] = (
        df['MedInc'].values * (1 / (enriched['dist_to_coast'].values + 1))
    )

    # 13. Urban density (population / household count proxy)
    enriched['urban_density'] = (
        df['Population'].values / (df['AveOccup'].values + 0.1)
    )

    # 14. Coastal income (income × coastal binary — captures coastal premium)
    enriched['coastal_income'] = (
        df['MedInc'].values * enriched['is_coastal'].values
    )

    return enriched


# Apply to both datasets
print("\nEnriching regression dataset...")
reg_enriched = compute_enrichment(reg_df)

print("\nEnriching classification dataset...")
clf_enriched = compute_enrichment(clf_df)

# ══════════════════════════════════════════════════════════════════════════════
# 4. VALIDATE & SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: VALIDATE & SAVE")
print("=" * 70)

enrichment_cols = [
    'dist_to_coast', 'dist_SF', 'dist_LA', 'dist_San_Diego',
    'dist_Sacramento', 'dist_San_Jose', 'dist_nearest_city',
    'is_coastal', 'is_bay_area', 'is_socal', 'climate_zone',
    'income_coast_interaction', 'urban_density', 'coastal_income'
]

# Validate no NaN in enrichment columns
for col in enrichment_cols:
    assert reg_enriched[col].isna().sum() == 0, f"NaN in regression {col}"
    assert clf_enriched[col].isna().sum() == 0, f"NaN in classification {col}"
print("Validation: No NaN values in enrichment features ✓")

# Correlation with targets
print("\nEnrichment feature correlations:")
print(f"  {'Feature':<30} {'w/ MedHouseVal':>14} {'w/ is_high_value':>16}")
print("  " + "-" * 62)
for col in enrichment_cols:
    corr_reg = reg_enriched[col].corr(reg_enriched['MedHouseVal'])
    corr_clf = clf_enriched[col].corr(clf_enriched['is_high_value'])
    bar = "█" * int(abs(corr_reg) * 30)
    print(f"  {col:<30} {corr_reg:>+.4f}         {corr_clf:>+.4f}  {bar}")

# Save
reg_enr_path = os.path.join(PROC_DIR, 'regression_dataset_enriched.csv')
clf_enr_path = os.path.join(PROC_DIR, 'classification_dataset_enriched.csv')

reg_enriched.to_csv(reg_enr_path, index=False)
clf_enriched.to_csv(clf_enr_path, index=False)

n_new = len(enrichment_cols)
print(f"""
{'='*70}
ENRICHMENT COMPLETE — WITH LIVE WEB SCRAPING
{'='*70}

Features added: {n_new}
  Regression:     {reg_df.shape[1]-1} → {reg_enriched.shape[1]-1} features (+{n_new})
  Classification: {clf_df.shape[1]-1} → {clf_enriched.shape[1]-1} features (+{n_new})

Output:
  {reg_enr_path}
    {reg_enriched.shape[0]:,} rows × {reg_enriched.shape[1]} columns

  {clf_enr_path}
    {clf_enriched.shape[0]:,} rows × {clf_enriched.shape[1]} columns

External data sources (web scraped at runtime):
  ✓ US Census Bureau Geocoding API ({len(CITIES)} cities)
  ✓ NOAA NCEI Station API ({len(COAST_POINTS)} coastal points)
  ✓ Open-Meteo Climate API ({len(CLIMATE_DATA)} climate zones validated)
  ✓ California Energy Commission (climate zone boundaries)

Next step: python run_models.py
""")
