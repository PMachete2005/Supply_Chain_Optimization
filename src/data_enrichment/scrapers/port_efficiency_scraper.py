"""
Port & Airport Efficiency Scraper
Fetches infrastructure efficiency data for Sea and Air transport modes
Relevant for: Transport_Mode (Sea/Air), Route_Risk_Index features

Note: This uses World Bank infrastructure indicators as a proxy for port/airport efficiency
"""
import requests
import csv
import os
from typing import Dict, List, Any

WB_API_BASE = "https://api.worldbank.org/v2"

# Infrastructure indicators relevant for Sea/Air transport
INFRASTRUCTURE_INDICATORS = {
    # Port infrastructure
    "IS.SHP.GOOD.TU": "Port_Container_Traffic",  # Container port traffic
    "IS.SHP.GOOD.TU.TC": "Port_Container_Traffic_TEU",  # Container traffic in TEU
    
    # Air transport
    "IS.AIR.DPRT": "Air_Departures",  # Air transport, registered carrier departures
    "IS.AIR.PSGR": "Air_Passengers",  # Air transport, passengers carried
    
    # General infrastructure quality (from LPI)
    "LP.LPI.INFR.XQ": "Infrastructure_Quality",  # Infrastructure quality score
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
                
                # Only keep our target countries
                if country_id in TARGET_COUNTRIES:
                    results.append({
                        'country_id': country_id,
                        'country_name': TARGET_COUNTRIES[country_id],
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

def fetch_all_infrastructure_indicators():
    """Fetch all infrastructure indicators"""
    all_data = {}
    
    for indicator_code, indicator_name in INFRASTRUCTURE_INDICATORS.items():
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
    print("Fetching Port & Airport Efficiency indicators...")
    print("Relevant for: Transport_Mode (Sea/Air), Route_Risk_Index features\n")
    
    data = fetch_all_infrastructure_indicators()
    
    if data:
        save_to_csv(data, 'data/external/raw/port_airport_efficiency.csv')
        print(f"\nTotal records: {len(data)}")
        print("\nSample data:")
        for i, item in enumerate(data[:3]):
            print(f"  {item}")
    else:
        print("No data fetched")

if __name__ == "__main__":
    main()

