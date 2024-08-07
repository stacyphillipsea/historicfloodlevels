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
    
    # Ensure dateTime is in datetime format
    df['dateTime'] = pd.to_datetime(df['dateTime'], errors='coerce')
    
    # Convert dateTime to just a date
    df['date'] = df['dateTime'].dt.normalize()
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
    
    # Debug: Print intermediate results dictionary
    print(f"Results created for site {site_name}")
    
    if not results:
        print(f"No results found for site {site_name}.")
        return pd.DataFrame(), None
    
    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    
    # Ensure start_date and end_date are datetime objects
    try:
        results_df['start_date'] = pd.to_datetime(results_df['start_date'], errors='coerce')
        results_df['end_date'] = pd.to_datetime(results_df['end_date'], errors='coerce')
    except KeyError as e:
        print(f"KeyError: {e}. Available columns: {results_df.columns}")
        return pd.DataFrame(), None
    
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

    # Return the figure object instead of showing it
    return fig

# Store plots in a dictionary
plots = {}

# Iterate through each site in data_dict
for site_name in data_dict.keys():
    print(f"Processing site: {site_name}")
    
    # Calculate results and average
    results_df, average_days = calculate_days_above_threshold(site_name, data_dict)
    
    if not results_df.empty:
        # Print the results DataFrame
        # print(results_df)
        
        # Plot the results with average line
        fig = plot_days_above_threshold_graph_objects(results_df, site_name, average_days)
        
        # Store the figure object in the plots dictionary
        plots[site_name] = fig

        print("Plot has made and stored in plots dictionary")
        print(f"Access using plots['{site_name}'].show()")
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

# Dictionary to store plot objects
plot_objects = {}

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

    # Store the figure object in the dictionary
    plot_objects[title] = fig
    print(f"Plot saved: access via plot_objects['{title}'].show()")


def plot_heatmaps_by_catchment(data_dict, normalize=False):
    heatmap_df, value_column = prepare_heatmap_data(data_dict, normalize)
    
    catchments = heatmap_df['catchment_name'].unique()
    for catchment in catchments:
        catchment_df = heatmap_df[heatmap_df['catchment_name'] == catchment]
        title = f"{'Normalized ' if normalize else ''} Heatmap for Days Above Threshold for Catchment: {catchment}"
        filename = f"{catchment}_heatmap_{'normalized' if normalize else 'raw'}.html"
        file_path = os.path.join(output_dir, filename)
        plot_heatmap(catchment_df, value_column, title, file_path)

def plot_aggregated_heatmap(data_dict, normalize=False):
    heatmap_df, value_column = prepare_heatmap_data(data_dict, normalize)
    title = f"{'Normalized ' if normalize else ''} Heatmap for Days Above Threshold for All Sites"
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

 
################################################
##### Catchment averages ########
################################################

##### Line charts

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def calculate_days_above_threshold(site_name, data_dict):
    # Retrieve the relevant data and threshold
    df = data_dict[site_name]['date_values']
    threshold = data_dict[site_name]['threshold']
    
    if threshold is None:
        print(f"No threshold defined for site {site_name}.")
        return pd.DataFrame()  # Return an empty DataFrame if no threshold is defined
    
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
    return results_df

def calculate_catchment_averages(data_dict):
    # Group sites by catchment
    catchment_sites = {}
    for site_name, site_data in data_dict.items():
        catchment_name = site_data.get('catchment', 'Unknown Catchment')
        if catchment_name not in catchment_sites:
            catchment_sites[catchment_name] = []
        catchment_sites[catchment_name].append(site_name)
    
    # Data structure to hold average days above threshold per catchment and year
    catchment_averages = {}
    
    for catchment_name, sites in catchment_sites.items():
        print(f"Processing catchment: {catchment_name}")
        
        yearly_totals = {}
        for site_name in sites:
            results_df = calculate_days_above_threshold(site_name, data_dict)
            for index, row in results_df.iterrows():
                year = index.split('-')[0].split(' ')[1]  # Extract the year from index
                if year not in yearly_totals:
                    yearly_totals[year] = []
                yearly_totals[year].append(row['above_threshold'])
        
        # Calculate average for each year in the catchment
        avg_per_year = {year: sum(days) / len(days) if days else 0 for year, days in yearly_totals.items()}
        catchment_averages[catchment_name] = avg_per_year
    
    return catchment_averages

def plot_catchment_averages(catchment_averages):
    fig = go.Figure()
    
    # Prepare data for plotting
    for catchment_name, avg_days in catchment_averages.items():
        years = sorted(avg_days.keys())
        averages = [avg_days[year] for year in years]
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=averages,
                mode='lines+markers',
                line=dict(width=2),
                marker=dict(size=8),
                name=f'{catchment_name}'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Average Days Above Threshold by Catchment and Year',
        xaxis_title='Year',
        yaxis_title='Average Days Above Threshold',
        xaxis=dict(tickvals=[str(year) for year in range(min(map(int, years)), max(map(int, years)) + 1)]),
        yaxis=dict(range=[0, max(max(avg_days.values(), default=0) for avg_days in catchment_averages.values()) * 1.1])
    )
    
    # Show the plot
    fig.show()

# Call the function to calculate and plot average days above threshold by catchment
catchment_averages = calculate_catchment_averages(data_dict)
plot_catchment_averages(catchment_averages)


#### Stacked bar chart
# Added total values for the highest 5 years but this is inappropriate for the dataset

def plot_catchment_averages_stacked_bar(catchment_averages, color_mapping):
    # Exclude the 'Unknown Catchment'
    catchment_averages = {k: v for k, v in catchment_averages.items() if k != 'Unknown Catchment'}
    
    years = sorted({year for avg_days in catchment_averages.values() for year in avg_days.keys()})
    fig = go.Figure()
    
    # Add traces for each catchment with specified colors
    for catchment_name, avg_days in catchment_averages.items():
        values = [avg_days.get(year, 0) for year in years]
        fig.add_trace(
            go.Bar(
                x=years,
                y=values,
                name=catchment_name,
                marker_color=color_mapping.get(catchment_name, 'gray')  # Use the color specified in the mapping or default to gray
            )
        )
    
    # Create bar chart
    fig.update_layout(
        title='Average Days Above Threshold by Catchment and Year (Stacked Bar Chart)',
        xaxis_title='Year',
        yaxis_title='Average Days Above Threshold',
        barmode='stack'
    )

    fig.show()

# Define a color palette
color_mapping = {
    'Severn Vale': '#1b9e77',  
    'Warwickshire Avon': '#d95f02',  
    'Shropshire Middle Severn': '#7570b3',  
    'Worcestershire Middle Severn': '#e7298a',  
    'Wye': '#66a61e',  
    'Severn Uplands': '#e6ab02',  
    'Teme': '#a6761d',    
}

plot_catchment_averages_stacked_bar(catchment_averages, color_mapping)

###########################################
############ Comparing with issue criteria
###########################################

# Initialize a list to hold the extracted data
extracted_data = []

# Loop through the dictionary and extract the 'name' and 'threshold'
for key, value in data_dict.items():
    extracted_data.append({'name': value['name'], 'threshold': value['threshold']})

# Convert the list of dictionaries to a DataFrame
thresholds = pd.DataFrame(extracted_data)

# Display the DataFrame
print(thresholds)


issue_criteria = pd.read_excel('issue_criteria.xlsx')
issue_criteria = issue_criteria.rename(columns={'RES': 'flood_alert_threshold'})


# Filter the issue_criteria DataFrame where Level is "FA"
issue_criteria_filtered = issue_criteria[issue_criteria['Level'] == 'FA']

# Merge the DataFrames on the matching columns ('name' and 'Gauge')
merged_df = pd.merge(thresholds, issue_criteria_filtered, left_on='name', right_on='Gauge')

# Calculate the difference between the thresholds and round to 1 decimal place
merged_df['threshold_difference'] = (merged_df['threshold'] - merged_df['flood_alert_threshold']).round(1)

# Filter out rows where threshold_difference is 0
filtered_df = merged_df[merged_df['threshold_difference'] != 0]

# Sort the DataFrame by threshold_difference in ascending order
sorted_df = filtered_df.sort_values(by='threshold_difference', ascending=True)

# Optionally, select only the columns of interest
result_df = sorted_df[['name', 'threshold', 'flood_alert_threshold', 'threshold_difference']]

# Display the result DataFrame
result_df

######################################################
## Getting tables out of docx (converted from doc)
######################################################


#### Used to have the Mean water and datum copying across but it looks like it has broken when i was playing with the sentence case
#### Can't get it back at the moment, try again tomorrow!

import os
import pandas as pd
from docx import Document
import win32com.client
from dateutil.parser import parse, ParserError

def convert_doc_to_docx(doc_path):
    print(f"Converting {doc_path} to .docx")
    if not os.path.exists(doc_path):
        print(f"File not found: {doc_path}")
        return None
    
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(doc_path))
        docx_path = os.path.splitext(doc_path)[0] + ".docx"
        doc.SaveAs(docx_path, FileFormat=16)  # 16 corresponds to the .docx format
        doc.Close()
        word.Quit()
        return docx_path
    except Exception as e:
        print(f"Failed to convert {doc_path} to .docx: {e}")
        return None

def extract_relevant_table_from_docx(docx_path):
    print(f"Processing document: {docx_path}")
    try:
        doc = Document(docx_path)
    except Exception as e:
        print(f"Failed to read document {docx_path}: {e}")
        return None
    
    tables = doc.tables
    if len(tables) == 0:
        print(f"No tables found in document: {docx_path}")
        return None
    elif len(tables) == 1:
        print(f"Found one table in document, extracting it")
        return extract_table_data(tables[0])
    elif len(tables) >= 2:
        print(f"Found two or more tables in document, extracting the second one")
        return extract_table_data(tables[1])
    else:
        print(f"Unexpected number of tables in document: {docx_path}")
        return None

def remove_duplicate_columns(df):
    if df.columns.duplicated().any():
        print("Duplicate columns found. Removing duplicates.")
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def replace_blank_headers(df):
    df.columns = [col if col.strip() else "Date" for col in df.columns]
    return df

def is_date(string):
    try:
        parse(string, dayfirst=True)
        return True
    except (ValueError, ParserError):
        return False

def process_date_column(df):
    if "Date" in df.columns:
        df["Info"] = df["Date"].apply(lambda x: x if not is_date(x) else pd.NA)
        df["Date"] = df["Date"].apply(lambda x: x if is_date(x) else pd.NA)
    return df

def update_mts_threshold_name(df):
    if "Info" in df.columns:
        # Check for specific words in the "Info" column
        for i in range(len(df)):
            if pd.notna(df.at[i, "Info"]):
                if "Mean Water Level" in df.at[i, "Info"]:
                    df.at[i, "MTS Threshold Name"] = "Mean Water Level"
                    df.at[i, "Info"] = pd.NA  # Set Info to null
                elif "STATION DATUM" in df.at[i, "Info"]:
                    df.at[i, "MTS Threshold Name"] = "STATION DATUM"
                    df.at[i, "Info"] = pd.NA  # Set Info to null
    return df

def convert_to_sentence_case(text):
    if pd.isna(text):
        return text
    return text.capitalize()

def rename_columns(df):
    # Strip any leading or trailing whitespace from the column names
    df.columns = [col.strip() for col in df.columns]
    
    # Define column mapping
    column_mapping = {
        "Warning to issue\n(ta name & code)": "Warning",
        "Mts threshold name": "Threshold",
        "Flooded areas & actions": "Actions",
        "M.a.l.d": "MALD",
        "M.a.o.d": "MAOD",
        "Date": "Date",
        "Info": "Info",
        "Station": "Station"
    }
    
    # Print columns for debugging
    print("Original columns:", df.columns.tolist())
    
    # Rename columns using the mapping
    df.rename(columns={key: column_mapping.get(key, key) for key in df.columns}, inplace=True)
    
    # Print renamed columns for debugging
    print("Renamed columns:", df.columns.tolist())
    
    return df

def extract_table_data(table):
    data = []
    for row in table.rows:
        text = [convert_to_sentence_case(cell.text.strip()) for cell in row.cells]
        data.append(text)
    
    if len(data) < 2:
        print("Not enough rows to create headers.")
        return pd.DataFrame(data)  # Return data as is if less than 2 rows
    
    # Use the first row as headers
    headers = data[0]
    rows = data[1:]
    
    df = pd.DataFrame(rows, columns=headers)
    
    # Remove duplicate columns and replace blank headers
    df = remove_duplicate_columns(df)
    df = replace_blank_headers(df)
    
    # Process the "Date" column and move non-date values to "Info"
    df = process_date_column(df)
    
    # Update "MTS Threshold Name" based on the "Info" column
    df = update_mts_threshold_name(df)
    
    # Rename columns to new names
    df = rename_columns(df)
    
    return df

def write_tables_to_excel(tables, doc_names, excel_path):
    if not tables:
        print("No tables found to write to Excel.")
        return
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for doc_name, table in zip(doc_names, tables):
            if not table.empty:
                sheet_name = f"{doc_name}_Table"
                print(f"Writing table from {doc_name} to sheet {sheet_name}")
                table.to_excel(writer, sheet_name=sheet_name, index=False)

def write_all_to_excel(tables, doc_names, all_path):
    if not tables:
        print("No tables found to write to all.")
        return
    
    all_data = []
    for doc_name, table in zip(doc_names, tables):
        if not table.empty:
            table['Station'] = doc_name  # Add a column with the document name
            all_data.append(table)
    
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        # Ensure the sixth column is labeled "Date"
        if len(all_df.columns) >= 6 and all_df.columns[5] == "Date":
            all_df.rename(columns={all_df.columns[5]: "Date"}, inplace=True)
        # Rename columns to new names in the summary file
        all_df = rename_columns(all_df)
        print(f"Writing all to {all_path}")
        all_df.to_excel(all_path, index=False)

def convert_docx_to_excel(docx_dir, output_excel_path, all_excel_path):
    if not os.path.exists(docx_dir):
        print(f"Directory does not exist: {docx_dir}")
        return
    
    doc_names = []
    tables = []
    
    for file_name in os.listdir(docx_dir):
        file_path = os.path.join(docx_dir, file_name)
        if file_name.endswith('.doc'):
            docx_file = convert_doc_to_docx(file_path)
            if docx_file:
                doc_name = os.path.splitext(os.path.basename(docx_file))[0]
                doc_names.append(doc_name)
                table = extract_relevant_table_from_docx(docx_file)
                tables.append(table)
        elif file_name.endswith('.docx'):
            doc_name = os.path.splitext(os.path.basename(file_path))[0]
            doc_names.append(doc_name)
            table = extract_relevant_table_from_docx(file_path)
            tables.append(table)
    
    if not tables:
        print(f"No tables found in directory: {docx_dir}")
        return
    
    print(f"Output file will be saved to: {output_excel_path}")
    write_tables_to_excel(tables, doc_names, output_excel_path)
    
    # Write all
    print(f"Summary file will be saved to: {all_excel_path}")
    write_all_to_excel(tables, doc_names, all_excel_path)

# Example usage:
docx_directory = 'C:\\Users\\SPHILLIPS03\\Documents\\repos\\historicfloodlevels\\thermometer_sheets'
output_excel_file = 'C:\\Users\\SPHILLIPS03\\Documents\\repos\\historicfloodlevels\\thermometer_by_gauge.xlsx'
all_excel_file = 'C:\\Users\\SPHILLIPS03\\Documents\\repos\\historicfloodlevels\\thermometer_all.xlsx'
convert_docx_to_excel(docx_directory, output_excel_file, all_excel_file)
