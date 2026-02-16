"""
World Bank Customs & Trade Efficiency Scraper
Fetches indicators directly related to Customs_Delay_Days feature
Relevant for: Customs_Delay_Days, Compliance_Score features
"""
import requests
import csv
import os
from typing import Dict, List, Any

WB_API_BASE = "https://api.worldbank.org/v2"

# Customs and trade efficiency indicators
CUSTOMS_INDICATORS = {
    "IC.CUS.DURS": "Customs_Clearance_Days",  # Time to clear customs
    "IC.TRD.DURS": "Trade_Duration_Days",     # Total trade duration
    "IC.EXP.DURS": "Export_Duration_Days",     # Time to export
    "IC.IMP.DURS": "Import_Duration_Days",     # Time to import
    "IC.CUS.COST": "Customs_Clearance_Cost",   # Cost to clear customs
    "IC.TRD.COST": "Trade_Cost",              # Total trade cost
}

# Country codes for our dataset countries (World Bank uses ISO2 codes)
TARGET_COUNTRIES = {
    'AU': 'Australia',
    'BR': 'Brazil',
    'CN': 'China',
    'DE': 'Germany',
    'IN': 'India',
    'JP': 'Japan',
    'ZA': 'South Africa',
    'AE': 'UAE',
    'GB': 'UK',
    'US': 'USA'
}

def fetch_indicator(indicator_code: str, indicator_name: str) -> List[Dict[str, Any]]:
    """
    Fetch raw observations for a single World Bank indicator
    
    Returns list of dicts with country_id, country_name, year, value
    """
    results: List[Dict[str, Any]] = []
    
    page = 1
    while True:
        url = (
            f"{WB_API_BASE}/country/all/indicator/{indicator_code}"
            f"?format=json&per_page=20000&page={page}"
        )
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data or len(data) < 2:
                break
            
            observations = data[1]
            if not observations:
                break
            
            for obs in observations:
                country_id = obs.get('country', {}).get('id', '')
                country_name = obs.get('country', {}).get('value', '')
                year = obs.get('date', '')
                value = obs.get('value')
                
                # Filter by country name (more flexible than ID)
                country_name_val = country_name.lower()
                target_names = [v.lower() for v in TARGET_COUNTRIES.values()]
                if any(target in country_name_val or country_name_val in target for target in target_names):
                    # Map to our standard country names
                    mapped_name = None
                    for iso, name in TARGET_COUNTRIES.items():
                        if name.lower() in country_name_val or country_name_val in name.lower():
                            mapped_name = name
                            break
                    
                    if mapped_name:
                        results.append({
                            'country_id': country_id,
                            'country_name': mapped_name,
                            'year': year,
                            indicator_name: value
                        })
            
            # Check if more pages
            if len(observations) < 20000:
                break
            
            page += 1
            
        except Exception as e:
            print(f"Error fetching {indicator_code} page {page}: {e}")
            break
    
    return results

def fetch_all_customs_indicators():
    """Fetch all customs and trade efficiency indicators"""
    all_data = {}
    
    for indicator_code, indicator_name in CUSTOMS_INDICATORS.items():
        print(f"Fetching {indicator_name} ({indicator_code})...")
        data = fetch_indicator(indicator_code, indicator_name)
        
        # Group by country and year
        for item in data:
            country_id = item['country_id']
            country_name = item['country_name']
            year = item['year']
            
            key = (country_id, country_name, year)
            if key not in all_data:
                all_data[key] = {
                    'country_id': country_id,
                    'country_name': country_name,
                    'year': year
                }
            
            all_data[key][indicator_name] = item[indicator_name]
        
        print(f"  Fetched {len(data)} records")
    
    return list(all_data.values())

def save_to_csv(data: List[Dict[str, Any]], filename: str):
    """Save data to CSV file"""
    if not data:
        print(f"No data to save for {filename}")
        return
    
    # Get all unique keys
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    
    fieldnames = ['country_id', 'country_name', 'year'] + sorted([
        k for k in all_keys if k not in ['country_id', 'country_name', 'year']
    ])
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {len(data)} records to {filename}")

def main():
    """Main function"""
    print("Fetching World Bank Customs & Trade Efficiency indicators...")
    print("Relevant for: Customs_Delay_Days feature\n")
    
    data = fetch_all_customs_indicators()
    
    if data:
        save_to_csv(data, 'data/external/raw/worldbank_customs_efficiency.csv')
        print(f"\nTotal records: {len(data)}")
        print("\nSample data:")
        for i, item in enumerate(data[:3]):
            print(f"  {item}")
    else:
        print("No data fetched")

if __name__ == "__main__":
    main()

