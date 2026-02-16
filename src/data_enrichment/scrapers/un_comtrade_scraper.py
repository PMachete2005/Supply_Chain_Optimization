"""
UN Comtrade Scraper
Fetches bilateral trade volumes between origin-destination country pairs
Relevant for: Origin_Country, Destination_Country, Route_Code features
"""
import requests
import csv
import time
import json
import os
from typing import List, Dict, Any

# Country ISO codes mapping (UN Comtrade uses ISO3 codes)
COUNTRY_ISO3_MAP = {
    'Australia': 'AUS',
    'Brazil': 'BRA',
    'China': 'CHN',
    'Germany': 'DEU',
    'India': 'IND',
    'Japan': 'JPN',
    'South Africa': 'ZAF',
    'UAE': 'ARE',
    'UK': 'GBR',
    'USA': 'USA'
}

# Reverse mapping for display
ISO3_COUNTRY_MAP = {v: k for k, v in COUNTRY_ISO3_MAP.items()}

def fetch_bilateral_trade(partner_code, reporter_code, year=2022):
    """
    Fetch bilateral trade data from UN Comtrade API
    
    Args:
        partner_code: ISO3 code of partner country (destination)
        reporter_code: ISO3 code of reporter country (origin)
        year: Year of trade data
    
    Returns:
        List of dicts with trade volume data
    """
    base_url = "https://comtradeapi.un.org/data/v1/get"
    
    params = {
        'type': 'C',  # Commodities
        'freq': 'A',  # Annual
        'px': 'HS',   # Harmonized System classification
        'ps': year,   # Period (year)
        'r': reporter_code,  # Reporter (origin)
        'p': partner_code,   # Partner (destination)
        'rg': 'all',  # Trade flow (all = imports + exports)
        'cc': 'TOTAL',  # Total trade (all commodities)
        'fmt': 'json'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and len(data['data']) > 0:
            return data['data']
        else:
            return []
    except Exception as e:
        print(f"Error fetching trade data for {reporter_code} -> {partner_code}: {e}")
        return []

def fetch_all_bilateral_trade(years=[2020, 2021, 2022]):
    """
    Fetch bilateral trade data for all country pairs in dataset
    
    Args:
        years: List of years to fetch
    
    Returns:
        List of dicts with all bilateral trade data
    """
    all_data = []
    countries = list(COUNTRY_ISO3_MAP.values())
    
    print(f"Fetching bilateral trade data for {len(countries)} countries...")
    print(f"Total pairs: {len(countries) * (len(countries) - 1)}")
    print("Note: UN Comtrade API may require registration for full access")
    
    pair_count = 0
    for origin_iso in countries:
        for dest_iso in countries:
            if origin_iso == dest_iso:
                continue
            
            pair_count += 1
            print(f"Fetching {ISO3_COUNTRY_MAP[origin_iso]} -> {ISO3_COUNTRY_MAP[dest_iso]} ({pair_count}/{len(countries) * (len(countries) - 1)})...")
            
            for year in years:
                data_list = fetch_bilateral_trade(dest_iso, origin_iso, year)
                if data_list:
                    for item in data_list:
                        item['origin_iso'] = origin_iso
                        item['destination_iso'] = dest_iso
                        item['origin_country'] = ISO3_COUNTRY_MAP[origin_iso]
                        item['destination_country'] = ISO3_COUNTRY_MAP[dest_iso]
                        item['year'] = year
                        all_data.append(item)
                
                # Rate limiting - be respectful to API
                time.sleep(0.5)
    
    return all_data

def process_trade_data(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process raw Comtrade data into clean format
    
    Args:
        raw_data: List of dicts from Comtrade API
    
    Returns:
        List of processed dicts with key trade metrics
    """
    if not raw_data:
        return []
    
    # Aggregate by origin-destination-year
    aggregated = {}
    
    for item in raw_data:
        origin = item.get('origin_country', '')
        dest = item.get('destination_country', '')
        origin_iso = item.get('origin_iso', '')
        dest_iso = item.get('destination_iso', '')
        year = item.get('year', '')
        
        key = (origin, dest, origin_iso, dest_iso, year)
        
        if key not in aggregated:
            aggregated[key] = {
                'origin_country': origin,
                'destination_country': dest,
                'origin_iso': origin_iso,
                'destination_iso': dest_iso,
                'year': year,
                'bilateral_trade_volume_usd': 0,
                'bilateral_trade_quantity': 0
            }
        
        # Sum trade values
        trade_value = item.get('TradeValue', 0) or 0
        quantity = item.get('qty', 0) or 0
        
        aggregated[key]['bilateral_trade_volume_usd'] += float(trade_value)
        aggregated[key]['bilateral_trade_quantity'] += float(quantity)
    
    return list(aggregated.values())

def save_to_csv(data: List[Dict[str, Any]], filename: str):
    """Save data to CSV file"""
    if not data:
        print(f"No data to save for {filename}")
        return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fieldnames = list(data[0].keys())
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved {len(data)} records to {filename}")

def main():
    """Main function to fetch and save bilateral trade data"""
    print("Starting UN Comtrade data collection...")
    print("Relevant for: Origin_Country, Destination_Country, Route_Code features\n")
    
    # Fetch data for recent years
    raw_data = fetch_all_bilateral_trade(years=[2020, 2021, 2022])
    
    if not raw_data:
        print("\nNo data fetched. This might be due to API limitations.")
        print("UN Comtrade API requires registration for full access.")
        print("Creating sample structure for manual data entry...")
        
        # Create empty structure
        sample_data = [{
            'origin_country': '',
            'destination_country': '',
            'origin_iso': '',
            'destination_iso': '',
            'year': '',
            'bilateral_trade_volume_usd': 0,
            'bilateral_trade_quantity': 0
        }]
        save_to_csv(sample_data, 'data/external/raw/un_comtrade_bilateral_trade.csv')
        print("Created empty template at data/external/raw/un_comtrade_bilateral_trade.csv")
        return
    
    # Save raw data
    save_to_csv(raw_data, 'data/external/raw/un_comtrade_bilateral_trade_raw.csv')
    
    # Process the data
    processed_data = process_trade_data(raw_data)
    
    # Save processed data
    if processed_data:
        save_to_csv(processed_data, 'data/external/raw/un_comtrade_bilateral_trade.csv')
        print(f"\nSample processed data:")
        for i, item in enumerate(processed_data[:3]):
            print(f"  {item}")
    else:
        print("No processed data to save")

if __name__ == "__main__":
    main()

