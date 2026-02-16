"""
Add LPI features (Groups 1, 2, 3) to the raw dataset using CSV module
This avoids pandas import issues
"""
import csv
import os

def load_lpi_data():
    """Load LPI data from CSV"""
    lpi_path = 'data/external/raw/worldbank_lpi_simple.csv'
    
    if not os.path.exists(lpi_path):
        raise FileNotFoundError(f"LPI data not found at {lpi_path}")
    
    lpi_data = {}
    
    with open(lpi_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            country = row['country_name']
            year = row['year']
            
            # Only process 2022 data for our target countries
            if year == '2022' and country in [
                'Australia', 'Brazil', 'China', 'Germany', 'India',
                'Japan', 'South Africa', 'United Arab Emirates',
                'United Kingdom', 'United States'
            ]:
                # Map World Bank names to dataset names
                dataset_country = {
                    'Australia': 'Australia',
                    'Brazil': 'Brazil',
                    'China': 'China',
                    'Germany': 'Germany',
                    'India': 'India',
                    'Japan': 'Japan',
                    'South Africa': 'South Africa',
                    'United Arab Emirates': 'UAE',
                    'United Kingdom': 'UK',
                    'United States': 'USA'
                }.get(country)
                
                if dataset_country:
                    lpi_data[dataset_country] = {
                        'LPI_Overall': float(row['LPI_Overall']) if row['LPI_Overall'] else None,
                        'LPI_Customs': float(row['LPI_Customs']) if row['LPI_Customs'] else None,
                        'LPI_Infrastructure': float(row['LPI_Infrastructure']) if row['LPI_Infrastructure'] else None,
                        'LPI_Logistics': float(row['LPI_Logistics']) if row['LPI_Logistics'] else None,
                        'LPI_Tracking': float(row['LPI_Tracking']) if row['LPI_Tracking'] else None,
                        'LPI_Timeliness': float(row['LPI_Timeliness']) if row['LPI_Timeliness'] else None,
                    }
    
    return lpi_data

def enrich_dataset():
    """Enrich raw dataset with LPI features"""
    
    raw_path = 'data/raw/trade_customs_dataset.csv'
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
    
    print("Loading LPI data...")
    lpi_data = load_lpi_data()
    print(f"Loaded LPI data for {len(lpi_data)} countries")
    
    print("Loading raw dataset...")
    rows = []
    fieldnames = None
    
    with open(raw_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            origin = row['Origin_Country']
            dest = row['Destination_Country']
            
            # Get LPI data for origin and destination
            origin_lpi = lpi_data.get(origin, {})
            dest_lpi = lpi_data.get(dest, {})
            
            # Group 1: Origin Country LPI Features (6 columns)
            row['Origin_LPI_Overall'] = str(origin_lpi.get('LPI_Overall', ''))
            row['Origin_LPI_Customs'] = str(origin_lpi.get('LPI_Customs', ''))
            row['Origin_LPI_Infrastructure'] = str(origin_lpi.get('LPI_Infrastructure', ''))
            row['Origin_LPI_Logistics'] = str(origin_lpi.get('LPI_Logistics', ''))
            row['Origin_LPI_Tracking'] = str(origin_lpi.get('LPI_Tracking', ''))
            row['Origin_LPI_Timeliness'] = str(origin_lpi.get('LPI_Timeliness', ''))
            
            # Group 2: Destination Country LPI Features (6 columns)
            row['Destination_LPI_Overall'] = str(dest_lpi.get('LPI_Overall', ''))
            row['Destination_LPI_Customs'] = str(dest_lpi.get('LPI_Customs', ''))
            row['Destination_LPI_Infrastructure'] = str(dest_lpi.get('LPI_Infrastructure', ''))
            row['Destination_LPI_Logistics'] = str(dest_lpi.get('LPI_Logistics', ''))
            row['Destination_LPI_Tracking'] = str(dest_lpi.get('LPI_Tracking', ''))
            row['Destination_LPI_Timeliness'] = str(dest_lpi.get('LPI_Timeliness', ''))
            
            # Group 3: Route-Level LPI Features (4 columns)
            origin_overall = origin_lpi.get('LPI_Overall')
            dest_overall = dest_lpi.get('LPI_Overall')
            origin_customs = origin_lpi.get('LPI_Customs')
            dest_customs = dest_lpi.get('LPI_Customs')
            origin_infra = origin_lpi.get('LPI_Infrastructure')
            dest_infra = dest_lpi.get('LPI_Infrastructure')
            
            if origin_overall is not None and dest_overall is not None:
                row['Route_LPI_Average'] = str((origin_overall + dest_overall) / 2)
                row['Route_LPI_Difference'] = str(dest_overall - origin_overall)
            else:
                row['Route_LPI_Average'] = ''
                row['Route_LPI_Difference'] = ''
            
            if origin_customs is not None and dest_customs is not None:
                row['Route_Customs_LPI_Average'] = str((origin_customs + dest_customs) / 2)
            else:
                row['Route_Customs_LPI_Average'] = ''
            
            if origin_infra is not None and dest_infra is not None:
                row['Route_Infrastructure_Gap'] = str(dest_infra - origin_infra)
            else:
                row['Route_Infrastructure_Gap'] = ''
            
            rows.append(row)
    
    # Check for missing values
    lpi_columns = [
        'Origin_LPI_Overall', 'Origin_LPI_Customs', 'Origin_LPI_Infrastructure',
        'Origin_LPI_Logistics', 'Origin_LPI_Tracking', 'Origin_LPI_Timeliness',
        'Destination_LPI_Overall', 'Destination_LPI_Customs', 'Destination_LPI_Infrastructure',
        'Destination_LPI_Logistics', 'Destination_LPI_Tracking', 'Destination_LPI_Timeliness',
        'Route_LPI_Average', 'Route_LPI_Difference', 'Route_Customs_LPI_Average',
        'Route_Infrastructure_Gap'
    ]
    
    missing_count = 0
    for row in rows:
        for col in lpi_columns:
            if not row.get(col) or row[col] == '':
                missing_count += 1
                print(f"WARNING: Missing value in row {len(rows)} for {col}")
                break
    
    if missing_count > 0:
        raise ValueError(f"Found {missing_count} rows with missing LPI values. All values must be filled.")
    
    print(f"✓ Validated: All {len(rows)} rows have complete LPI data")
    
    # Update fieldnames
    new_fieldnames = list(fieldnames) + lpi_columns
    
    # Create backup first
    backup_path = 'data/raw/trade_customs_dataset_backup.csv'
    print(f"\nCreating backup at {backup_path}...")
    with open(backup_path, 'w', encoding='utf-8', newline='') as f:
        with open(raw_path, 'r', encoding='utf-8') as orig:
            f.write(orig.read())
    
    # Write enriched dataset
    print(f"Writing enriched dataset to {raw_path}...")
    with open(raw_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Successfully added {len(lpi_columns)} LPI columns")
    print(f"✓ Total rows: {len(rows)}")
    print(f"✓ All rows have complete data")
    
    return len(rows), len(lpi_columns)

if __name__ == "__main__":
    try:
        rows, cols = enrich_dataset()
        print(f"\n✓ Enrichment complete! Added {cols} columns to {rows} rows")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise

