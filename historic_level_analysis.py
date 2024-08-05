################################
## Analysing historic Winters ##
################################

from scipy.signal import find_peaks, find_peaks_cwt
import pandas as pd 
import numpy as np
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
import json
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import interact, FloatSlider, IntSlider, Button, VBox
import ipywidgets as widgets
from IPython.display import display
import os



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


### CALL YOUR FUNCTIONS 
# Load station data from JSON file
file_path = "historic_nested_dict.json"
data_dict = load_station_data_from_json(file_path)

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



################
# Prepare and plot heatmaps (normalised and regular)
################

def prepare_heatmap_data(data_dict, normalize=False):
    rows = []

    for site_name, site_data in data_dict.items():
        df = site_data['date_values']
        threshold = site_data['threshold']

        if threshold is None:
            continue

        # Ensure dateTime is in datetime format
        df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')
        if df['dateTime'].isnull().all():
            print(f"No valid dateTime data for site {site_name}. Skipping.")
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
    
    if normalize:
        # Normalize data by site
        df['days_above_threshold_normalized'] = df.groupby('site_name')['days_above_threshold'].transform(lambda x: x / x.max())
        value_column = 'days_above_threshold_normalized'
        title = 'Heatmap of Normalized Days Above Threshold by Site and Year'
    else:
        value_column = 'days_above_threshold'
        title = 'Heatmap of Days Above Threshold by Site and Year'
    
    return df, value_column, title
    
def plot_heatmap(df, value_column, title):
    heatmap_data = df.pivot(index='site_name', columns='year', values=value_column)
    
    # Debugging: Print the heatmap data
    print("Heatmap Data (Pivot Table):")

    
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale='Viridis',
        labels={'color': value_column.replace('_', ' ').title()},
        title=title
    )
    
    fig.update_layout(
        xaxis_title='Winter Period',
        yaxis_title='Site Name',
        yaxis=dict(
            tickmode='array',
            tickvals=list(heatmap_data.index),
            ticktext=list(heatmap_data.index)
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=list(heatmap_data.columns),
            ticktext=list(heatmap_data.columns)
        ),
        height=800,  # Adjust overall height of the plot
        width=1200,  # Adjust overall width of the plot
        coloraxis_colorbar=dict(
            title=value_column.replace('_', ' ').title(),
            lenmode='fraction',
            len=0.5,  # Set colorbar length as fraction of the plot height
            thickness=20  # Adjust thickness of the colorbar
        )
    )
    fig.show()

# Example usage with data_dict
heatmap_df, value_column, title = prepare_heatmap_data(data_dict, normalize=False)
plot_heatmap(heatmap_df, value_column, title)

# For normalized data
normalized_heatmap_df, normalized_value_column, normalized_title = prepare_heatmap_data(data_dict, normalize=True)
plot_heatmap(normalized_heatmap_df, normalized_value_column, normalized_title)



####################################
### Grouping by catchment ##
####################################

# Ensure the directory exists
output_dir = "html_heatmaps"
os.makedirs(output_dir, exist_ok=True)

def prepare_heatmap_data(data_dict, normalize=False):
    rows = []

    for site_name, site_data in data_dict.items():
        df = site_data['date_values']
        threshold = site_data['threshold']
        catchment_name = site_data.get('catchment', 'Unknown Catchment')  # Assuming catchment info is available

        if threshold is None:
            continue

        # Ensure dateTime is in datetime format
        df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')
        if df['dateTime'].isnull().all():
            print(f"No valid dateTime data for site {site_name}. Skipping.")
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
                'catchment_name': catchment_name,
                'site_name': site_name,
                'year': f"{year}-{year + 1}",
                'days_above_threshold': above_threshold
            })

    df = pd.DataFrame(rows)

    # Filter out rows where 'days_above_threshold' is NaN
    df = df.dropna(subset=['days_above_threshold'])

    # Filter out sites with fewer than 10 years of data
    site_years_count = df.groupby('site_name')['year'].nunique()
    valid_sites = site_years_count[site_years_count >= 10].index
    df = df[df['site_name'].isin(valid_sites)]

    # Remove sites with all values as 0
    pivot_df = df.pivot_table(index='site_name', values='days_above_threshold', aggfunc='sum')
    valid_sites = pivot_df[pivot_df.sum(axis=1) > 0].index
    df = df[df['site_name'].isin(valid_sites)]
    
    if normalize:
        # Normalize data by site within each catchment
        df['days_above_threshold_normalized'] = df.groupby(['catchment_name', 'site_name'])['days_above_threshold'].transform(lambda x: x / x.max())
        value_column = 'days_above_threshold_normalized'
    else:
        value_column = 'days_above_threshold'
    
    return df, value_column

def plot_heatmap(df, value_column, title, filename):
    heatmap_data = df.pivot(index='site_name', columns='year', values=value_column)
    
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale='Plasma',
        labels={'color': value_column.replace('_', ' ').title()},
        title=title
    )
    
    fig.update_layout(
        xaxis_title='Winter Period',
        yaxis_title='Site Name',
        yaxis=dict(
            tickmode='array',
            tickvals=list(heatmap_data.index),
            ticktext=list(heatmap_data.index)
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=list(heatmap_data.columns),
            ticktext=list(heatmap_data.columns)
        ),
        height=800,  # Adjust overall height of the plot
        width=1200,  # Adjust overall width of the plot
        coloraxis_colorbar=dict(
            title=value_column.replace('_', ' ').title(),
            lenmode='fraction',
            len=0.5,  # Set colorbar length as fraction of the plot height
            thickness=20  # Adjust thickness of the colorbar
        )
    )

    # Save the plot as an HTML file
    fig.write_html(filename)
    print(f"Saved heatmap to {filename}")

def plot_heatmaps_by_catchment(data_dict, normalize=False):
    heatmap_df, value_column = prepare_heatmap_data(data_dict, normalize)
    
    catchments = heatmap_df['catchment_name'].unique()
    for catchment in catchments:
        catchment_df = heatmap_df[heatmap_df['catchment_name'] == catchment]
        title = f"Heatmap of {'Normalized ' if normalize else ''}Days Above Threshold for Catchment: {catchment}"
        filename = f"{catchment}_heatmap_{'normalized' if normalize else 'raw'}.html"
        file_path = os.path.join(output_dir, filename)
        plot_heatmap(catchment_df, value_column, title, file_path)

def plot_aggregated_heatmap(data_dict, normalize=False):
    heatmap_df, value_column = prepare_heatmap_data(data_dict, normalize)
    title = f"Heatmap of {'Normalized ' if normalize else ''}Days Above Threshold for All Sites"
    filename = f"aggregated_heatmap_{'normalized' if normalize else 'raw'}.html"
    file_path = os.path.join(output_dir, filename)
    plot_heatmap(heatmap_df, value_column, title, file_path)

# Example usage
#plot_heatmaps_by_catchment(data_dict, normalize=False)
plot_heatmaps_by_catchment(data_dict, normalize=True)
#plot_aggregated_heatmap(data_dict, normalize=False)
plot_aggregated_heatmap(data_dict, normalize=True)


###############################################
######## Playing about with rolling means #####
###############################################


import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def calculate_days_above_threshold(site_name, data_dict, rolling_window=5):
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
            'above_threshold': above_threshold,
            'percent': percent
        }
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    
    # Ensure 'above_threshold' column is present
    if 'above_threshold' not in results_df.columns:
        print(f"'above_threshold' column is missing in results_df for site {site_name}.")
    
    # Calculate the rolling mean of the number of days above threshold
    if 'above_threshold' in results_df.columns:
        results_df['rolling_mean'] = results_df['above_threshold'].rolling(window=rolling_window, min_periods=1).mean()
    
    # Calculate the average number of days above threshold
    average_days = results_df['above_threshold'].mean() if not results_df.empty else None
    
    return results_df, average_days

def plot_rolling_mean(results_df, site_name):
    # Format the index to show just the year range
    results_df.index = results_df.index.str.replace('Winter ', '')
    
    # Ensure the index is sorted in chronological order
    results_df = results_df.sort_index()

    # Create the line chart for rolling mean
    fig = go.Figure()

    # Add line for rolling mean
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df['rolling_mean'],
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='blue'),
            name='Rolling Mean'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f'Rolling Mean of Days Above Threshold for Each Winter Period at {site_name}',
        xaxis_title='Winter Period',
        yaxis_title='Rolling Mean of Days Above Threshold',
        xaxis_tickangle=-90
    )
    
    # Show the plot
    fig.show()



# Iterate through each site in data_dict
for site_name in data_dict.keys():
    print(f"Processing site: {site_name}")
    
    # Calculate results and average
    results_df, average_days = calculate_days_above_threshold(site_name, data_dict)
    
    if not results_df.empty:
        # Print the results DataFrame for debugging
        print(f"Results DataFrame for {site_name}:\n{results_df}")
        
        # Plot the rolling mean
        plot_rolling_mean(results_df, site_name)
    else:
        print(f"No data available for site {site_name}.")


import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def calculate_days_above_threshold(site_name, data_dict, rolling_window=5):
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
            'above_threshold': above_threshold,
            'percent': percent
        }
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T
    
    # Calculate the rolling mean of the number of days above threshold
    if 'above_threshold' in results_df.columns:
        results_df['rolling_mean'] = results_df['above_threshold'].rolling(window=rolling_window, min_periods=1).mean()
    
        # Normalize the rolling mean by the maximum days experienced at the site
        max_days = results_df['above_threshold'].max() if not results_df.empty else 1
        results_df['normalized_rolling_mean'] = results_df['rolling_mean'] / max_days
    
    # Calculate the average number of days above threshold
    average_days = results_df['above_threshold'].mean() if not results_df.empty else None
    
    return results_df, average_days

def plot_catchment_rolling_mean(data_dict):
    # Group sites by catchment
    catchment_sites = {}
    for site_name, site_data in data_dict.items():
        catchment_name = site_data.get('catchment', 'Unknown Catchment')
        if catchment_name not in catchment_sites:
            catchment_sites[catchment_name] = []
        catchment_sites[catchment_name].append(site_name)
    
    for catchment_name, sites in catchment_sites.items():
        fig = go.Figure()
        
        # Collect all winter periods to ensure x-axis consistency
        all_winter_periods = set()
        
        # First pass to collect all winter periods
        for site_name in sites:
            results_df, average_days = calculate_days_above_threshold(site_name, data_dict)
            if not results_df.empty:
                all_winter_periods.update(results_df.index)
        
        # Create a DataFrame to sort winter periods
        all_winter_periods_df = pd.DataFrame({
            'winter_period': list(all_winter_periods)
        })
        all_winter_periods_df['start_year'] = all_winter_periods_df['winter_period'].apply(
            lambda x: int(x.split('-')[0].split(' ')[1])
        )
        all_winter_periods_df.sort_values(by='start_year', inplace=True)
        sorted_winter_periods = all_winter_periods_df['winter_period'].tolist()
        
        for site_name in sites:
            print(f"Processing site: {site_name} for catchment: {catchment_name}")
            
            # Calculate results and average
            results_df, average_days = calculate_days_above_threshold(site_name, data_dict)
            
            if not results_df.empty:
                # Reindex results_df to ensure all periods are included
                results_df = results_df.reindex(sorted_winter_periods, fill_value=0)
                
                # Normalize the rolling mean by the maximum value across all sites
                if 'normalized_rolling_mean' in results_df.columns:
                    max_normalized_value = results_df['normalized_rolling_mean'].max()
                    if max_normalized_value > 0:
                        results_df['normalized_rolling_mean'] = results_df['normalized_rolling_mean'] / max_normalized_value
                    
                    # Add line for normalized rolling mean
                    fig.add_trace(
                        go.Scatter(
                            x=results_df.index,
                            y=results_df['normalized_rolling_mean'],
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=8),
                            name=f'{site_name} Normalized Rolling Mean'
                        )
                    )
        
        # Update layout
        fig.update_layout(
            title=f'Normalized Rolling Mean of Days Above Threshold for Each Winter Period in {catchment_name}',
            xaxis_title='Winter Period',
            yaxis_title='Normalized Rolling Mean of Days Above Threshold',
            xaxis_tickangle=-90
        )
        
        # Show the plot
        fig.show()

# Call the function to plot normalized rolling means for each catchment
plot_catchment_rolling_mean(data_dict)
