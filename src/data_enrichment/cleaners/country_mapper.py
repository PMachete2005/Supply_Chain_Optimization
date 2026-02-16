"""
Map country names from various sources to dataset standard
"""
COUNTRY_MAPPING = {
    # Dataset uses these names
    'India': 'India',
    'China': 'China',
    'Germany': 'Germany',
    'Brazil': 'Brazil',
    'Japan': 'Japan',
    'USA': 'United States',
    'UK': 'United Kingdom',
    'UAE': 'United Arab Emirates',
    'South Africa': 'South Africa',
    'Australia': 'Australia',
    
    # Common variations
    'United States of America': 'United States',
    'US': 'United States',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'U.A.E.': 'United Arab Emirates',
    # Add more mappings as needed
}

def standardize_country_name(country_name):
    """Standardize country name to dataset format"""
    return COUNTRY_MAPPING.get(country_name, country_name)
