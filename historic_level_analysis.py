################################
## Analysing historic Winters ##
################################

from scipy.signal import find_peaks, find_peaks_cwt
import pandas as pd 
import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import json
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import interact, FloatSlider, IntSlider, Button, VBox
import ipywidgets as widgets
from IPython.display import display


# Load data
sites_of_interest_merge = pd.read_csv('sites_of_interest_merge.csv')
file_path = "historic_nested_dict.json"

#### Changed the approach and have now nested the thresholds within the data_dict
### DON'T NEED THIS NOW I HAVE IT ALL IN ONE DICTIONARY ALREADY
# # Get thresholds and create a combined dictionary
# def load_combined_data(file_path, sites_of_interest_merge):
#     try:
#         # Load data from JSON file
#         with open(file_path, "r") as json_file:
#             data_dict = json.load(json_file)
        
#         # Convert date_values from JSON strings to DataFrames
#         for station_name, station_data in data_dict.items():
#             station_data['date_values'] = pd.read_json(StringIO(station_data['date_values']), convert_dates=['dateTime'], date_unit='ms')
        
#         # Get thresholds and add to the data_dict
#         threshold_values = sites_of_interest_merge[sites_of_interest_merge['Threshold'].notnull()]
#         threshold_values.loc[:, 'Threshold'] = threshold_values['Threshold'].astype(float)
#         threshold_dict = threshold_values.set_index('Gauge')['Threshold'].to_dict()
        
#         for station_name in data_dict.keys():
#             data_dict[station_name]['threshold'] = threshold_dict.get(station_name, None)
        
#         return data_dict
#     except FileNotFoundError:
#         print(f"File {file_path} not found.")
#         return None

# Load combined data
data_dict = load_combined_data(file_path, sites_of_interest_merge)

## Setup winter periods
# Function to calculate the number of days above threshold for each winter period
# Returns a dataframe and the average

def calculate_days_above_threshold(site_name, data_dict):
    # Retrieve the relevant data and threshold
    df = data_dict[site_name]['date_values']
    threshold = data_dict[site_name]['threshold']
    
    if threshold is None:
        print(f"No threshold defined for site {site_name}.")
        return pd.DataFrame(), None  # Return an empty DataFrame and None for the average if no threshold is defined
    
    # Convert dateTime to just a date
    df['date'] = df['dateTime'].dt.normalize()
    
    # Determine the start year (earliest year in the dataset) and end year (current year)
    start_year = df['date'].dt.year.min() + 1
    end_year = datetime.now().year - 1
    
    results = {}
    for year in range(start_year, end_year + 1):
        start_date = pd.Timestamp(year, 10, 1)  # October 1st of the current year
        end_date = pd.Timestamp(year + 1, 3, 1)  # 1st March of the next year
        
        # Filter data for the current winter period
        winter_period_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Calculate the number of days the value is greater than the threshold
        total_days = len(winter_period_data)
        above_threshold = (winter_period_data['value'] > threshold).sum()
        percent = round((above_threshold / total_days * 100), 1) if total_days > 0 else 0.0
        
        # Store results for the current winter period
        results[f"Winter {year}-{year + 1}"] = {
            'start_date': start_date,
            'end_date': end_date,
            'above_threshold': above_threshold,
            'percent': percent
        }
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    
    # Ensure start_date and end_date are datetime objects
    results_df['start_date'] = pd.to_datetime(results_df['start_date'])
    results_df['end_date'] = pd.to_datetime(results_df['end_date'])
    
    # Format each of the columns
    results_df['start_date'] = results_df['start_date'].dt.date
    results_df['end_date'] = results_df['end_date'].dt.date
    results_df['above_threshold'] = pd.to_numeric(results_df['above_threshold'])
    results_df['percent'] = pd.to_numeric(results_df['percent'])
    
    # Calculate the average number of days above threshold
    average_days = results_df['above_threshold'].mean() if not results_df.empty else None
    
    return results_df, average_days

## Make a plot with that dataframe, colour the top 5, and add an average line
def plot_days_above_threshold_graph_objects(results_df, site_name, average_days):
    # Format the index to show just the year range
    results_df.index = results_df.index.str.replace('Winter ', '')
    
    # Ensure the index is sorted in chronological order
    results_df = results_df.sort_index()

    # Identify the top 5 years with the highest number of days above threshold
    top_5_years = results_df.nlargest(5, 'above_threshold').index

    # Assign colors based on whether the year is in the top 5
    colors = results_df.index.map(lambda x: 'red' if x in top_5_years else 'blue')

    # Prepare text for data labels, excluding 0 values
    text_labels = results_df['above_threshold'].apply(lambda x: str(x) if x > 0 else '')

    # Create the bar chart with specified colors
    fig = go.Figure()

    # Add bars
    fig.add_trace(
        go.Bar(
            x=results_df.index,
            y=results_df['above_threshold'],
            text=text_labels,
            textposition='outside',  # Place text outside the bars
            marker_color=colors,  # Color by top 5 years
            showlegend=False  # Hide legend for bars
        )
    )
    
    # Add average line
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=[average_days] * len(results_df),
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name=f'Average: {average_days:.2f}',  # Set legend name for average line
            text=[f'{average_days:.2f}'] * len(results_df),
            textposition='top center'  # Valid positions for Scatter
        )
    )
    
    # Update layout and text properties
    fig.update_layout(
        title=f'Number of Days Above Threshold for Each Winter Period at {site_name}',
        xaxis_title='Winter Period',
        yaxis_title='Days Above Threshold',
        xaxis_tickangle=-90
    )
    fig.update_traces(
        texttemplate='%{text}',
        textfont_size=10
    )

    # Show the plot
    fig.show()

# Iterate through each site in data_dict
for site_name in data_dict.keys():
    print(f"Processing site: {site_name}")
    
    # Calculate results and average
    results_df, average_days = calculate_days_above_threshold(site_name, data_dict)
    
    if not results_df.empty:
        # Print the results DataFrame
        # print(results_df)
        
        # Plot the results with average line
        plot_days_above_threshold_graph_objects(results_df, site_name, average_days)
    else:
        print(f"No data available for site {site_name}.")



### Making a heatmap to show this in matrix format for all sites
### This doesn't show all the sites, the normalised one below does
def prepare_heatmap_data(data_dict):
    rows = []

    for site_name, site_data in data_dict.items():
        df = site_data['date_values']
        threshold = site_data['threshold']

        if threshold is None:
            continue

        df['date'] = df['dateTime'].dt.normalize()
        start_year = df['date'].dt.year.min() + 1
        end_year = datetime.now().year - 1

        for year in range(start_year, end_year + 1):
            start_date = pd.Timestamp(year, 10, 1)
            end_date = pd.Timestamp(year + 1, 3, 1)
            winter_period_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            above_threshold = (winter_period_data['value'] > threshold).sum()

            rows.append({
                'site_name': site_name,
                'year': f"{year}-{year + 1}",
                'days_above_threshold': above_threshold
            })

    return pd.DataFrame(rows)

# Example usage with data_dict
heatmap_df = prepare_heatmap_data(data_dict)

print(heatmap_df.head())

def plot_heatmap(df):
    heatmap_data = df.pivot(index='site_name', columns='year', values='days_above_threshold')
    
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale='Viridis',
        labels={'color': 'Days Above Threshold'},
        title='Heatmap of Days Above Threshold by Site and Year'
    )
    
    fig.update_layout(
        xaxis_title='Winter Period',
        yaxis_title='Site Name'
    )
    
    fig.show()

# Plot the heatmap
plot_heatmap(heatmap_df)




### Normalised values with all sites

# Prepare data and normalize
def prepare_heatmap_data(data_dict):
    rows = []

    for site_name, site_data in data_dict.items():
        df = site_data['date_values']
        threshold = site_data['threshold']

        if threshold is None:
            continue

        df['date'] = df['dateTime'].dt.normalize()
        start_year = df['date'].dt.year.min() + 1
        end_year = datetime.now().year - 1

        for year in range(start_year, end_year + 1):
            start_date = pd.Timestamp(year, 10, 1)
            end_date = pd.Timestamp(year + 1, 3, 1)
            winter_period_data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            above_threshold = (winter_period_data['value'] > threshold).sum()

            rows.append({
                'site_name': site_name,
                'year': f"{year}-{year + 1}",
                'days_above_threshold': above_threshold
            })

    df = pd.DataFrame(rows)
    
    # Normalize the data
    max_days = df['days_above_threshold'].max()
    if max_days > 0:
        df['days_above_threshold_normalized'] = df['days_above_threshold'] / max_days
    else:
        df['days_above_threshold_normalized'] = df['days_above_threshold']  # Handle case where max_days is 0
    
    print("Prepared DataFrame:")
    print(df.head())  # Debug: print the first few rows of the DataFrame
    print(f"Max days for normalization: {max_days}")
    
    return df

# Plot the heatmap with continuous color scheme
def plot_heatmap(df):
    # Create pivot table for normalized days above threshold
    heatmap_data_normalized = df.pivot(index='site_name', columns='year', values='days_above_threshold_normalized')
    
    # Debug: Check the pivoted data
    print("Pivoted Heatmap Data:")
    print(heatmap_data_normalized)
    print("Site names:", heatmap_data_normalized.index)
    print("Years:", heatmap_data_normalized.columns)

    # Create the heatmap for normalized days above threshold
    fig = px.imshow(
        heatmap_data_normalized,
        color_continuous_scale='sunsetdark',
        labels={'color': 'Normalized Days Above Threshold'},
        title='Heatmap of Normalized Days Above Threshold by Site and Year'
    )

    # Ensure the layout accommodates all site names
    fig.update_layout(
        xaxis_title='Winter Period',
        yaxis_title='Site Name',
        yaxis=dict(
            tickmode='array',
            tickvals=list(heatmap_data_normalized.index),
            ticktext=list(heatmap_data_normalized.index)
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=list(heatmap_data_normalized.columns),
            ticktext=list(heatmap_data_normalized.columns)
        ),
        height=800,  # Adjust overall height of the plot
        width=1200,  # Adjust overall width of the plot
        coloraxis_colorbar=dict(
            title='Normalized Days Above Threshold',
            tickvals=[0, 1],  # Set specific tick values for colorbar
            ticktext=['0', '1'],  # Set tick labels for colorbar
            lenmode='fraction',
            len=0.5,  # Set colorbar length as fraction of the plot height
            thickness=20  # Adjust thickness of the colorbar
        )
    )
    fig.show()

# Prepare the data
heatmap_df = prepare_heatmap_data(data_dict)

# Plot the heatmap
plot_heatmap(heatmap_df)

