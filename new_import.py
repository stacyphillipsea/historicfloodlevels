#### New way of importing the data making use of the typical ranges and additional info in the
# flood monitoring API, which is linked to in the Hydrology API sameAs field
# Has been optimised to work with the other code that is the same as the historic_level_analysis

import pandas as pd
import numpy as np      # For numerical operations
from datetime import datetime
from datetime import timedelta
import requests         # For API call
import json             # For API call
import dash             # For app
from dash import dcc, html, dash_table      # For app layout
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go           # For Plotly charts
import os                                   # For filepath operations
from pptx import Presentation       # For Powerpoint
from pptx.util import Pt, Inches
import folium               # For map
from io import StringIO     # To loadd JSON to dataframe
import logging


# Initialize a set for unique catchments and a dictionary for catchment-based stations
catchments = set()
stations_by_catchment = {}

# Iterate through the data_dict
for station, details in data_dict.items():
    catchment = details['catchment']
    catchments.add(catchment)
    
    if catchment not in stations_by_catchment:
        stations_by_catchment[catchment] = []
    stations_by_catchment[catchment].append(station)

# Convert catchments set to a sorted list
catchments_list = sorted(list(catchments))

print("Unique Catchments:", catchments_list)
print("Stations by Catchment:", stations_by_catchment)



### Create lists of stations in each catchment
# Could then use theses as lists for getting data

def get_wiski_ids_for_catchment(catchment_name):
    # Construct the URL using the catchment name
    url = f"https://environment.data.gov.uk/flood-monitoring/id/stations?catchmentName={catchment_name}"
    
    # Fetch the JSON response from the API
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    data = response.json()

    # Extract the list of wiskiIDs
    wiski_ids = [station['wiskiID'] for station in data['items']]
    
    return wiski_ids

# List of catchment names
catchment_names = [
    "Teme",
    "Severn Uplands",
    "Wye",
    "Worcestershire Middle Severn",
    "Shropshire Middle Severn",
    "Warwickshire Avon",
    "Severn Vale"
]

# Dictionary to store lists of wiskiIDs for each catchment
catchment_wiski_ids = {}

# Iterate through the list of catchments and get the wiskiIDs
for catchment_name in catchment_names:
    # Retrieve wiskiIDs for the current catchment
    wiski_ids = get_wiski_ids_for_catchment(catchment_name)
    
    # Create a key in the dictionary with the format '<catchment_name>_list'
    key = f"{catchment_name.replace(' ', '_').lower()}_list"
    
    # Store the wiskiIDs in the dictionary
    catchment_wiski_ids[key] = wiski_ids

# Combine all lists into one big list
all_severn_wiski_ids = [wiski_id for ids in catchment_wiski_ids.values() for wiski_id in ids]

# Print the combined list and its length
print(f"Total number of stations in the Severn catchment: {len(all_severn_wiski_ids)} stations")

# Print individual lists and their lengths
for key, ids in catchment_wiski_ids.items():
    print(f"{key}: {len(ids)} stations")

catchment_wiski_ids['severn_uplands_list']



### GET YOUR DATA BITS
# Define key constants
BASE_URL = "http://environment.data.gov.uk/hydrology/id"
BASE_STATIONS_URL = "http://environment.data.gov.uk/hydrology/id/stations"

#MIN_DATE_STR = "2015-10-01"
MAX_DATE_STR = "2024-02-29"
#MIN_DATE = datetime.strptime(MIN_DATE_STR, '%Y-%m-%d')
MAX_DATE = datetime.strptime(MAX_DATE_STR, '%Y-%m-%d')


## Load data
# Metadata spreadsheet
sites_of_interest_merge = pd.read_csv('sites_of_interest_merge.csv')

# Historic records
gaugeboard_data = pd.read_csv('gaugeboard_data.csv')

# WMD gauge list
wmd_gauges = pd.read_csv('All_WMD_gauges_FETA.csv')
WISKI_IDS = wmd_gauges['Site number'].dropna().tolist()
WISKI_IDS = [f"{name}" for name in WISKI_IDS]
### SUBSETTING FOR TESTING
WISKI_IDS = WISKI_IDS[:3]
WISKI_IDS = catchment_wiski_ids['worcestershire_middle_severn_list'] 

# MET Office storms
storms = pd.read_excel('Met Office named storms.xlsx')
storms['startdate'] = pd.to_datetime(storms['startdate']).dt.date
storms['enddate'] = pd.to_datetime(storms['enddate']).dt.date
# Adjusting the filter length
storms['startdate'] = storms['startdate'] - timedelta(days=2)
storms['enddate'] = storms['enddate'] + timedelta(days=2)

DATE_FILTERS = {row['Name']: (str(row['startdate']), str(row['enddate']), 'blue') for _, row in storms.iterrows()}


# Isolate threshold/max values from metadata spreadsheet
threshold_values = sites_of_interest_merge[sites_of_interest_merge['Threshold'].notnull()]
threshold_values.loc[:, 'Threshold'] = threshold_values['Threshold'].astype(float) # Ensure original is modified, removing SettingWithCopyWarning
threshold_dict = threshold_values.set_index('Gauge')['Threshold'].to_dict()

#### SETUP LOGGING
# Ensure the logs directory exists
log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Setup logging if not configured already
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_directory, 'station_data.log'))

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)




### MAKE YOUR FUNCTIONS
# Fetch data for a single station
BASE_URL = "http://environment.data.gov.uk/hydrology/id"
BASE_STATIONS_URL = "http://environment.data.gov.uk/hydrology/id/stations"

successful_stations = {}  # Dictionary to store successful WISKI IDs and station names

import requests
import json
import pandas as pd
from datetime import datetime

# Define key constants
BASE_URL = "http://environment.data.gov.uk/hydrology/id"
BASE_STATIONS_URL = "http://environment.data.gov.uk/hydrology/id/stations"
MAX_DATE_STR = "2024-02-29"
MAX_DATE = datetime.strptime(MAX_DATE_STR, '%Y-%m-%d')

# Initialize a dictionary to track successful stations
successful_stations = {}

def convert_eaAreaName(ea_area_name):
    if ea_area_name == "Midlands - Staffordshire Warwickshire and West Midlands":
        return "SWWM"
    elif ea_area_name == "Midlands - Shropshire Herefordshire Worcestershire and Gloucestershire":
        return "SHWG"
    else:
        return "Not in WMD"

def fetch_json(url):
    """Helper function to fetch and load JSON data from a URL."""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def extract_same_as_url(item_data):
    """Extract 'sameAs' URL from item_data."""
    items = item_data.get('items', {})
    if isinstance(items, dict):
        return items.get('sameAs', {}).get('@id')
    elif isinstance(items, list) and items:
        return items[0].get('sameAs', {}).get('@id')
    return None

def extract_stage_scale_data(same_as_data):
    """Extract stage scale data including 'typicalRangeHigh' and 'typicalRangeLow'."""
    stage_scale = same_as_data.get('items', {}).get('stageScale', {})
    ea_area_name = same_as_data.get('items', {}).get('eaAreaName')

    return {
        'typical_range_high': stage_scale.get('typicalRangeHigh'),
        'typical_range_low': stage_scale.get('typicalRangeLow'),
        'catchment': same_as_data.get('items', {}).get('catchmentName'),
        'eaAreaName': convert_eaAreaName(ea_area_name),
        'eaRegionName': same_as_data.get('items', {}).get('eaRegionName')
    }

def fetch_station_data(wiski_id):
    try:
        # Fetch the initial station data
        url_endpoint = f"{BASE_STATIONS_URL}?wiskiID={wiski_id}"
        data = fetch_json(url_endpoint)

        if not data.get('items'):
            print(f"No station items found for WISKI ID {wiski_id}")
            return None

        item = data['items'][0]
        item_id_url = item.get('@id')

        # Fetch data from item_id URL to get 'sameAs'
        item_data = fetch_json(item_id_url)
        same_as_url = extract_same_as_url(item_data)
        
        if same_as_url:
            same_as_data = fetch_json(same_as_url)
            stage_scale_data = extract_stage_scale_data(same_as_data)
            typical_range_high = stage_scale_data['typical_range_high']
            typical_range_low = stage_scale_data['typical_range_low']
            print(f"Stage Scale Data: {stage_scale_data}")
        else:
            print("No 'sameAs' URL found")
            typical_range_high = typical_range_low = None
            stage_scale_data = {}

        # Extract basic details
        label_field = item.get('label')
        name = str(label_field[1] if isinstance(label_field, list) else label_field)
        river_name = item.get('riverName')
        latitude = item.get('lat')
        longitude = item.get('long')
        dateOpened_str = item.get('dateOpened') or "01/01/2000"

        # Fetch measures and readings data
        measure_url = f"{BASE_URL}/measures?station.wiskiID={wiski_id}&observedProperty=waterLevel&periodName=daily&valueType=max"
        measure = fetch_json(measure_url)

        if not measure.get('items'):
            print(f"No measure items found for WISKI ID {wiski_id}")
            return None

        measure_id = measure['items'][0]['@id']
        readings_url = f"{measure_id}/readings?mineq-date={dateOpened_str}&maxeq-date={MAX_DATE_STR}"
        readings = fetch_json(readings_url)
        readings_items = readings.get('items', [])

        if readings_items:
            df = pd.DataFrame.from_dict(readings_items)
            df['dateTime'] = pd.to_datetime(df['dateTime'])
            successful_stations[wiski_id] = name  # Add successful WISKI ID and name to the dictionary
            print(f"Successful station fetched: {name} ({wiski_id})")
            return {
                'name': name,
                'date_values': df[['dateTime', 'value']],
                'river_name': river_name,
                'lat': latitude,
                'long': longitude,
                **stage_scale_data,
                'dateOpened_str': dateOpened_str,
                'threshold': typical_range_high
            }
        else:
            print(f"No readings found for {name} ({wiski_id})")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for WISKI ID {wiski_id}: {e}")
    finally:
        # Print the dictionary of successful WISKI IDs and station names at the end of the function
        print(f"Successful stations: {successful_stations}")
    return None


# Fetch data for all stations
def fetch_all_station_data():
    data_dict = {}
    for wiski_id in WISKI_IDS:
        station_data = fetch_station_data(wiski_id)
        if station_data:
            data_dict[station_data['name']] = station_data
    return data_dict

# Find maximum values for each filter
def find_max_values(df, filters):
    max_values = {}
    # Ensure 'dateTime' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['dateTime']):
        df['dateTime'] = pd.to_datetime(df['dateTime'])
    for filter_name, date_range in filters.items():
        min_date, max_date, color = date_range
        condition = (df['dateTime'] >= min_date) & (df['dateTime'] <= max_date)
        filtered_df = df[condition].dropna()  # Drop rows with NaN values
        if not filtered_df.empty:
            max_value_row = filtered_df.loc[filtered_df['value'].idxmax(), ['dateTime', 'value']]
            max_value_row['value'] = round(max_value_row['value'], 2)  # Round the maximum value to 2 decimal places
            max_values[filter_name] = max_value_row
    return max_values

# Find and store maximum values for all stations
def find_and_store_max_values(data_dict):
    max_values = {}
    for station_name, station_data in data_dict.items():
        df = station_data.get('date_values')
        if df is not None:
            max_values[station_name] = find_max_values(df, DATE_FILTERS)
    return max_values

#Generate storm info for the peak table
def generate_storm_info():
    storm_info = []
    for storm, (start_date, end_date, _) in DATE_FILTERS.items():
        formatted_start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%d %b %Y')
        formatted_end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%d %b %Y')
        storm_info.append(html.Div([
            html.P(f"{storm}: {formatted_start_date} to {formatted_end_date}", style={'font-size': '14px'}),
        ]))
    return storm_info


# Function to load station data from JSON file
def load_station_data_from_json(file_path):
    try:
        # Load data from JSON file
        with open(file_path, "r") as json_file:
            data_dict = json.load(json_file)
        
        # Convert date_values from JSON strings to DataFrames
        for station_data in data_dict.values():
            station_data['date_values'] = pd.read_json(StringIO(station_data['date_values']))

        return data_dict
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None


# #### FUNCTION TO MAKE DICTIONARY OFFLINE AND THEN LOAD

# # # Fetch and save data for all stations
# data_dict = fetch_all_station_data()

# def fetch_and_save_all_station_data():
#     data_dict = {}
#     for wiski_id in WISKI_IDS:
#         station_data = fetch_station_data(wiski_id)
#         if station_data:
#             # Convert DataFrame to JSON-serializable format
#             date_values_json = station_data['date_values'].to_json(orient='records')
#             # Replace DataFrame with JSON string in station_data dictionary
#             station_data['date_values'] = date_values_json
#             data_dict[station_data['name']] = station_data

#     # Specify the file path where you want to save the JSON file
#     file_path = "C:\\Users\\SPHILLIPS03\\Documents\\repos\\historicfloodlevels\\historic_nested_dict.json"

#     # Save the dictionary containing station data to a JSON file
#     with open(file_path, "w") as json_file:
#         json.dump(data_dict, json_file)

#     print("JSON file saved successfully.")

# fetch_and_save_all_station_data()

### CALL YOUR FUNCTIONS 
# Load station data from JSON file
file_path = "historic_nested_dict.json"
data_dict = load_station_data_from_json(file_path)

if data_dict:
    print("Data loaded successfully.")
else:
    print("Error loading data.")

# Find and store maximum values for all stations
max_values = find_and_store_max_values(data_dict)

# Create peak table DataFrame
df, peak_table_all = process_peak_table_all(max_values, sites_of_interest_merge)

#####################################################
#####################################################

## Categorising by SHWG and SWWM
# Initialize lists for SHWG and SWWM stations
shwg_stations = []
swwm_stations = []

# Iterate through the data_dict and filter stations
for station, details in data_dict.items():
    if details['eaAreaName'] == 'SHWG':
        shwg_stations.append(station)
    elif details['eaAreaName'] == 'SWWM':
        swwm_stations.append(station)

print("SHWG Stations:", shwg_stations)
print("SWWM Stations:", swwm_stations)

