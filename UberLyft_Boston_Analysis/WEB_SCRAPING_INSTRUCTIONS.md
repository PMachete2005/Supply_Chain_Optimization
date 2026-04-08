# Web Scraping Instructions for ML Data Enrichment
## For: Your Friend (Web Scraping Lead)
## Goal: Improve model performance through data enrichment

---

## OVERVIEW

**Current State:** You have a raw dataset (rideshare/transportation data) with basic features (distance, time, simple weather)

**Goal:** Add external data sources that capture WHY prices surge

**Expected Impact:** +20-30% R² improvement, -40% MAE reduction (based on Boston results)

---

## PRIORITY 1: WEATHER DATA (HIGHEST IMPACT)

### Source: Open-Meteo API (FREE - No API Key Required)
**URL:** `https://api.open-meteo.com/v1/forecast`  
**Documentation:** `https://open-meteo.com/en/docs`

### What to Scrape:
```
Endpoint: https://archive-api.open-meteo.com/v1/archive
Parameters:
  - latitude: [YOUR_CITY_LAT]
  - longitude: [YOUR_CITY_LON]
  - start_date: [MATCH_YOUR_DATASET_DATE_RANGE]
  - end_date: [MATCH_YOUR_DATASET_DATE_RANGE]
  - hourly: temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,cloud_cover,wind_speed_10m,pressure_msl
  - timezone: [YOUR_TIMEZONE]
```

### Data to Extract (8 fields):
| Field | Type | Why It Matters |
|-------|------|----------------|
| `temperature_2m` | float | Celsius (more precise than Fahrenheit) |
| `relative_humidity_2m` | int | % humidity (0-100) |
| `precipitation` | float | mm of rain/snow (quantifies intensity) |
| `rain` | float | Instant rain rate (detects downpours) |
| `weather_code` | int | WMO code (standardized conditions) |
| `cloud_cover` | int | % clouds (0-100) |
| `wind_speed_10m` | float | km/h (storm detection) |
| `pressure_msl` | float | hPa (weather change indicator) |

### Python Code for Your Friend:
```python
import requests
import pandas as pd

def scrape_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", 
                   "rain", "weather_code", "cloud_cover", "wind_speed_10m", "pressure_msl"],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Convert to DataFrame
    hourly = data['hourly']
    df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly['time']),
        'temp_c': hourly['temperature_2m'],
        'humidity': hourly['relative_humidity_2m'],
        'precipitation_mm': hourly['precipitation'],
        'rain_rate': hourly['rain'],
        'weather_code': hourly['weather_code'],
        'cloud_cover': hourly['cloud_cover'],
        'wind_speed': hourly['wind_speed_10m'],
        'pressure': hourly['pressure_msl']
    })
    
    # Create derived features
    df['is_raining'] = (df['precipitation_mm'] > 0).astype(int)
    df['is_heavy_rain'] = (df['precipitation_mm'] > 5).astype(int)
    df['is_cold'] = (df['temp_c'] < 5).astype(int)  # < 41°F
    df['is_hot'] = (df['temp_c'] > 30).astype(int)  # > 86°F
    df['is_windy'] = (df['wind_speed'] > 20).astype(int)
    df['is_storm'] = ((df['is_raining'] == 1) & (df['is_windy'] == 1)).astype(int)
    df['weather_severity_score'] = df['is_raining'] + df['is_heavy_rain'] + df['is_cold'] + df['is_hot'] + df['is_windy']
    
    return df
```

### Output Format:
- Save as: `weather_data.csv`
- Columns: datetime, temp_c, humidity, precipitation_mm, rain_rate, weather_code, cloud_cover, wind_speed, pressure, is_raining, is_heavy_rain, is_cold, is_hot, is_windy, is_storm, weather_severity_score

---

## PRIORITY 2: EVENTS DATA (HIGH IMPACT)

### Source A: Eventbrite API (Free tier available)
**URL:** `https://www.eventbriteapi.com/v3/events/search`  
**API Key:** Required (get from eventbrite.com/developer)

### Source B: Ticketmaster API (Free tier available)
**URL:** `https://developer.ticketmaster.com/`  
**API Key:** Required

### Source C: Sports Schedules (Web Scraping)
**URLs:**
- NBA: `https://www.nba.com/celtics/schedule` (adjust for your city)
- NHL: `https://www.nhl.com/bruins/schedule`
- MLB: `https://www.mlb.com/redsox/schedule`
- NFL: `https://www.nfl.com/schedules/`

### What to Scrape:
| Field | Type | Example |
|-------|------|---------|
| `event_date` | date | 2024-04-15 |
| `event_time` | time | 19:00 |
| `event_type` | string | 'sports', 'concert', 'festival' |
| `venue_name` | string | 'TD Garden', 'Fenway Park' |
| `venue_lat` | float | 42.3662 |
| `venue_lon` | float | -71.0621 |
| `expected_attendance` | int | 18000 |

### Python Code (Eventbrite):
```python
import requests

def scrape_events(city, start_date, end_date, api_key):
    url = "https://www.eventbriteapi.com/v3/events/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    params = {
        "location.address": city,
        "start_date.range_start": start_date,
        "start_date.range_end": end_date,
        "categories": "110,103,109"  # Sports, Music, Travel/Outdoor
    }
    
    response = requests.get(url, headers=headers, params=params)
    events = response.json()['events']
    
    event_data = []
    for event in events:
        event_data.append({
            'event_id': event['id'],
            'name': event['name']['text'],
            'start_time': event['start']['local'],
            'venue': event['venue']['name'],
            'venue_lat': event['venue']['latitude'],
            'venue_lon': event['venue']['longitude'],
            'category': event['category']['short_name'] if event.get('category') else 'other'
        })
    
    return pd.DataFrame(event_data)
```

### Sports Scraping (BeautifulSoup):
```python
from bs4 import BeautifulSoup
import requests

def scrape_sports_schedule(team_url):
    """Example for scraping team schedule"""
    response = requests.get(team_url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Adjust selectors based on specific site
    games = soup.find_all('div', class_='game-item')
    
    schedule = []
    for game in games:
        schedule.append({
            'date': game.find('span', class_='date').text,
            'time': game.find('span', class_='time').text,
            'home_team': game.find('span', class_='home').text,
            'venue': game.find('span', class_='venue').text
        })
    
    return pd.DataFrame(schedule)
```

### Output Format:
- Save as: `events_data.csv`
- Columns: event_id, name, start_time, venue, venue_lat, venue_lon, category

---

## PRIORITY 3: TRAFFIC DATA (MEDIUM-HIGH IMPACT)

### Source A: TomTom Traffic API (Free tier: 2,500 calls/day)
**URL:** `https://developer.tomtom.com/traffic-api`  
**API Key:** Required

### Source B: Google Maps Distance Matrix API (Paid, but accurate)
**URL:** `https://developers.google.com/maps/documentation/distance-matrix`  
**API Key:** Required

### What to Scrape:
```
For each major route in your city:
  - Origin lat/lon (major pickup zones)
  - Destination lat/lon (major dropoff zones)
  - Query every hour during your dataset timeframe
```

| Field | Type | Why It Matters |
|-------|------|----------------|
| `route_id` | string | Identifier for origin-destination pair |
| `timestamp` | datetime | When traffic was measured |
| `duration_in_traffic` | int | Seconds (vs normal duration) |
| `traffic_delay` | int | Extra seconds due to traffic |
| `congestion_level` | string | 'low', 'moderate', 'heavy', 'severe' |

### Python Code (TomTom):
```python
def scrape_traffic(origin_lat, origin_lon, dest_lat, dest_lon, api_key):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin_lat},{origin_lon}:{dest_lat},{dest_lon}/json"
    params = {
        "key": api_key,
        "traffic": "true",
        "departAt": "[TIMESTAMP]"  # Your ride timestamp
    }
    
    response = requests.get(url, params=params)
    route = response.json()['routes'][0]
    
    return {
        'duration_no_traffic': route['summary']['noTrafficTravelTimeInSeconds'],
        'duration_with_traffic': route['summary']['travelTimeInSeconds'],
        'traffic_delay': route['summary']['trafficDelayInSeconds'],
        'congestion': route['summary']['trafficCongestion']
    }
```

### Output Format:
- Save as: `traffic_data.csv`
- Columns: route_id, timestamp, origin_lat, origin_lon, dest_lat, dest_lon, duration_no_traffic, duration_with_traffic, traffic_delay, congestion

---

## PRIORITY 4: PUBLIC TRANSIT DATA (MEDIUM IMPACT)

### Source: Transit Agency API
**Examples:**
- MBTA (Boston): `https://www.mbta.com/developers/v3-api`
- MTA (NYC): `https://new.mta.info/developers`
- BART (SF): `https://www.bart.gov/schedules/developers`

### What to Scrape:
| Field | Type | Why It Matters |
|-------|------|----------------|
| `transit_line` | string | 'Red Line', 'Subway A', etc. |
| `timestamp` | datetime | When delay occurred |
| `delay_minutes` | int | How long the delay was |
| `delay_reason` | string | 'mechanical', 'weather', 'crowding' |
| `affected_stations` | list | Which stations were impacted |
| `station_lat/lon` | float | Location of affected stations |

### Python Code (MBTA Example):
```python
def scrape_transit_delays(api_key):
    url = "https://api-v3.mbta.com/predictions"
    headers = {"x-api-key": api_key}
    
    response = requests.get(url, headers=headers)
    predictions = response.json()['data']
    
    delays = []
    for pred in predictions:
        if pred['attributes']['delay']:
            delays.append({
                'line': pred['relationships']['route']['data']['id'],
                'station': pred['relationships']['stop']['data']['id'],
                'delay_seconds': pred['attributes']['delay'],
                'timestamp': pred['attributes']['departure_time']
            })
    
    return pd.DataFrame(delays)
```

### Output Format:
- Save as: `transit_delays.csv`
- Columns: transit_line, timestamp, delay_minutes, delay_reason, affected_stations, station_lat, station_lon

---

## PRIORITY 5: FUEL PRICES (LOW-MEDIUM IMPACT)

### Source: EIA API (U.S. Energy Information Administration)
**URL:** `https://www.eia.gov/opendata/`  
**API Key:** Required (free registration)

### What to Scrape:
```
Endpoint: https://api.eia.gov/v2/seriesid/PET.EMM_EPM0_PTE_NUS_DPG.W
Parameters:
  - frequency: weekly
  - start: [YOUR_DATASET_START]
  - end: [YOUR_DATASET_END]
```

| Field | Type | Why It Matters |
|-------|------|----------------|
| `date` | date | Week of measurement |
| `gas_price` | float | USD per gallon |
| `price_change` | float | Week-over-week change |
| `is_high_price` | int | 1 if > $3.00/gallon |

### Python Code:
```python
def scrape_gas_prices(api_key, start_date, end_date):
    url = "https://api.eia.gov/v2/petroleum/pri/gnd/data"
    params = {
        "frequency": "weekly",
        "data": ["value"],
        "facets": {"product": ["EPD2D"]},  # Regular gasoline
        "start": start_date,
        "end": end_date,
        "api_key": api_key
    }
    
    response = requests.get(url, params=params)
    data = response.json()['response']['data']
    
    df = pd.DataFrame(data)
    df['value'] = pd.to_numeric(df['value'])
    df['is_high_price'] = (df['value'] > 3.0).astype(int)
    
    return df[['period', 'value', 'is_high_price']].rename(
        columns={'period': 'date', 'value': 'gas_price'}
    )
```

### Output Format:
- Save as: `fuel_prices.csv`
- Columns: date, gas_price, is_high_price

---

## MERGING INSTRUCTIONS

Once your friend has scraped all data, merge it with your raw dataset:

```python
# 1. Load all data
raw_df = pd.read_csv('raw_dataset.csv')
weather_df = pd.read_csv('weather_data.csv')
events_df = pd.read_csv('events_data.csv')
traffic_df = pd.read_csv('traffic_data.csv')
transit_df = pd.read_csv('transit_delays.csv')
fuel_df = pd.read_csv('fuel_prices.csv')

# 2. Merge on timestamp (nearest hour)
raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

# Round both to nearest hour for merging
raw_df['merge_time'] = raw_df['timestamp'].dt.round('H')
weather_df['merge_time'] = weather_df['datetime'].dt.round('H')

enriched_df = raw_df.merge(weather_df, on='merge_time', how='left')

# 3. Create interaction features (THESE ARE CRITICAL!)
enriched_df['weather_rush_interaction'] = (
    enriched_df['is_raining'] * enriched_df['is_rush_hour']
)
enriched_df['distance_event_interaction'] = (
    enriched_df['distance'] * enriched_df['is_event_time']
)
enriched_df['premium_weather_interaction'] = (
    enriched_df['is_premium'] * enriched_df['is_raining']
)

# 4. Save enriched dataset
enriched_df.to_csv('enriched_dataset.csv', index=False)
```

---

## DELIVERABLES CHECKLIST

Give this to your friend:

- [ ] **weather_data.csv** - 8 raw + 7 derived features from Open-Meteo
- [ ] **events_data.csv** - Sports games, concerts, festivals with venues
- [ ] **traffic_data.csv** - Route-level traffic delays
- [ ] **transit_delays.csv** - Public transit issues (line, station, delay)
- [ ] **fuel_prices.csv** - Weekly gas prices
- [ ] **merging_script.py** - Code to combine all with your raw dataset

---

## EXPECTED TIMELINE

| Task | Time | Priority |
|------|------|----------|
| Weather scraping | 2-3 hours | **DO FIRST** |
| Events scraping | 4-6 hours | Do second |
| Traffic scraping | 3-4 hours | Do third |
| Transit scraping | 2-3 hours | Do fourth |
| Fuel prices | 1 hour | Do last |
| Merging + testing | 2-3 hours | Final step |

**Total: 2-3 days of work**

---

## EXPECTED RESULTS

Based on Boston results with similar enrichment:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| R² Score | ~0.70 | ~0.90 | **+20-30%** |
| MAE | ~$4.00 | ~$2.10 | **-45%** |
| F1 (surge) | ~0.14 | ~0.15 | Minimal (surge is hard!) |

---

## CONTACT FOR QUESTIONS

**Open-Meteo:** Free, works immediately, no key needed  
**Eventbrite:** Need API key, 1000 calls/hour free tier  
**TomTom:** Need API key, 2500 calls/day free tier  
**MBTA:** Need API key, free for developers  
**EIA:** Need API key, free registration

**Troubleshooting:**
- Rate limiting: Add `time.sleep(1)` between calls
- Missing data: Use forward-fill or interpolation
- API errors: Wrap in try/except with retry logic

---

## QUICK START FOR YOUR FRIEND

1. **Start with Open-Meteo** (no key, works immediately)
2. **Get Eventbrite key** (most events)
3. **Scrape sports schedules** (beautifulsoup)
4. **Get TomTom key** (traffic data)
5. **Merge everything** (use the merging script)

**The weather + events combination will give you 80% of the improvement!**
