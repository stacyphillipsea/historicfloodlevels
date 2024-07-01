######
# Trying to find a way to identify peak automatically
# Original code from https://medium.com/@chrisjpulliam/quickly-finding-peaks-in-mass-spectrometry-data-using-scipy-fcf3999c5057
######
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import json
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import interact, FloatSlider, IntSlider, Button, VBox
import ipywidgets as widgets
from IPython.display import display

# Function to load station data from JSON file
def load_station_data_from_json(file_path):
    try:
        # Load data from JSON file
        with open(file_path, "r") as json_file:
            data_dict = json.load(json_file)
        
        # Convert date_values from JSON strings to DataFrames
        for station_data in data_dict.values():
            station_data['date_values'] = pd.read_json(StringIO(station_data['date_values']), convert_dates=['dateTime'], date_unit='ms')

        return data_dict
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def identify_peaks_for_site(site_data, site_name):
    # Initial plot setup
    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(x=site_data['dateTime'], y=site_data['value'], mode='lines', name='River levels'))

    # Add markers for the peaks
    peak_scatter = fig.add_trace(go.Scatter(mode='markers', name='Peaks', marker=dict(color='red', size=8)))

    # Update layout
    fig.update_layout(
        title=f'River Levels over time with peaks - {site_name}',
        xaxis_title='Date',
        yaxis_title='Level (m)'
    )

    # Function to update peaks
    def update_peaks(prominence, height, distance):
        peak_idx, _ = find_peaks(site_data['value'], prominence=prominence, height=height, distance=distance)
        peak_dates = site_data['dateTime'].iloc[peak_idx]
        peak_values = site_data['value'].iloc[peak_idx]
        with fig.batch_update():
            fig.data[1].x = peak_dates
            fig.data[1].y = peak_values

    # Create sliders
    prominence_slider = FloatSlider(value=0.4, min=0.1, max=2.0, step=0.1, description='Prominence')
    height_slider = FloatSlider(value=1, min=0, max=5, step=0.1, description='Height')
    distance_slider = IntSlider(value=1, min=1, max=10, step=1, description='Distance')

    # Link sliders to the update function
    interact(update_peaks, prominence=prominence_slider, height=height_slider, distance=distance_slider)

    # Function to extract peaks and display DataFrame
    def extract_peaks(b, site_name):
        peak_idx, _ = find_peaks(site_data['value'], prominence=prominence_slider.value, height=height_slider.value, distance=distance_slider.value)
        peak_dates = site_data['dateTime'].iloc[peak_idx]
        peak_values = site_data['value'].iloc[peak_idx]
        peaks_df_name = f"peak_df_{site_name.replace(' ', '')}"
        peak_df = pd.DataFrame({'Peak Date': peak_dates, 'Peak Value': peak_values})
        print(f"Peaks for {site_name}:")
        globals()[peaks_df_name] = peak_df
        display(peak_df)

    # Create button to extract peaks
    extract_button = Button(description="Extract Peaks")
    extract_button.on_click(lambda b: extract_peaks(b, site_name))

    # Display the plot and button
    display(VBox([fig, extract_button]))

# Load station data
file_path = "historic_nested_dict.json"
data_dict = load_station_data_from_json(file_path)


# Choose site name
site_name = 'Diglis'
# Call the function to identify peaks for the selected site
identify_peaks_for_site(data_dict[site_name]['date_values'], site_name)



# Choose site name
site_name = 'Hereford Bridge'
# Call the function to identify peaks for the selected site
identify_peaks_for_site(data_dict[site_name]['date_values'], site_name)

# Choose site name
site_name = 'Welsh Bridge'
# Call the function to identify peaks for the selected site
identify_peaks_for_site(data_dict[site_name]['date_values'], site_name)















# # Plot the peak river levels
# fig = px.scatter(all_peaks_df, x='Peak Date', y='Peak Value', color='Site', 
#                  title='Peak River Levels by Site')
# fig.update_traces(marker=dict(size=8))
# fig.show()

# # Group by two-week intervals and count the number of peaks
# all_peaks_df['Peak Date'] = pd.to_datetime(all_peaks_df['Peak Date'])
# all_peaks_df['Period'] = all_peaks_df['Peak Date'].dt.to_period('2W').astype(str)

# # Convert 'Period' back to datetime for sorting
# all_peaks_df['Period_Start'] = pd.PeriodIndex(all_peaks_df['Period'], freq='2W').start_time

# peak_counts = all_peaks_df.groupby(['Period_Start', 'Site']).size().reset_index(name='Count')

# # Plot the bar chart
# fig = px.bar(peak_counts, x='Period_Start', y='Count', color='Site', barmode='group',
#              title='Peak River Levels by Site and Two-Week Intervals')
# fig.update_layout(xaxis_title='Date (2-Week Periods)', yaxis_title='Number of Peaks')
# fig.show()



###### Assigning peaks to given storm events

# Concatenate the dataframes
all_peaks_df = pd.concat([peak_df_HerefordBridge.assign(Site='Hereford Bridge'),
                          peak_df_Diglis.assign(Site='Diglis'),
                          peak_df_WelshBridge.assign(Site='Welsh Bridge'),
                          peak_df_Tamworth.assign(Site='Tamworth')])

all_peaks_df['Peak Date'] = pd.to_datetime(all_peaks_df['Peak Date'])

# Creating storm lists
storms = pd.read_excel('Met Office named storms.xlsx')
storms['startdate'] = pd.to_datetime(storms['startdate'])
storms['enddate'] = pd.to_datetime(storms['enddate'])
# Adjusting the filter length
storms['startdate'] = storms['startdate'] - timedelta(days=2)
storms['enddate'] = storms['enddate'] + timedelta(days=2)
storms.dtypes #should be datetime not object

# Create function to match date to storm
def match_storm(date):
    for _, storm in storms.iterrows():
        if storm['startdate'] <= date <= storm['enddate']:
            return storm['Name']
    return None

# Apply the function to create a new column 'Storm_Name'
all_peaks_df['Storm_Name'] = all_peaks_df['Peak Date'].apply(match_storm)
all_peaks_df


### making a plot to show the peak dates and storms
all_peaks_df['Peak Date'] = pd.to_datetime(all_peaks_df['Peak Date'])

# Separate data with and without storm names
labeled_data = all_peaks_df.dropna(subset=['Storm_Name'])
unlabeled_data = all_peaks_df[all_peaks_df['Storm_Name'].isnull()]

# Plotting with Plotly
fig = go.Figure()

# Scatter plot for unlabeled data (without storm names)
fig.add_trace(go.Scatter(
    x=unlabeled_data['Peak Date'],
    y=unlabeled_data['Peak Value'],
    mode='markers',
    marker=dict(color='blue'),
    name='No Storm Name',
    text=unlabeled_data['Site'],  # Display site name on hover
))

# Scatter plot for labeled data (with storm names)
fig.add_trace(go.Scatter(
    x=labeled_data['Peak Date'],
    y=labeled_data['Peak Value'],
    mode='markers',
    marker=dict(color='red'),
    name='Storm Name',
    text=labeled_data.apply(lambda row: f"Site: {row['Site']}<br>Storm: {row['Storm_Name']}", axis=1),  # Display site and storm names on hover
))

# Customize layout
fig.update_layout(
    title='Peak Value vs Peak Date',
    xaxis_title='Peak Date',
    yaxis_title='Peak Value',
    legend_title='Legend',
)

# Show plot
fig.show()


from plotly.subplots import make_subplots

# Number of unique sites
unique_sites = all_peaks_df['Site'].unique()

# Create subplot grid
fig = make_subplots(rows=len(unique_sites), cols=1, 
                    subplot_titles=unique_sites)

# Iterate over unique sites and add scatter plots
for i, site in enumerate(unique_sites):
    site_data = all_peaks_df[all_peaks_df['Site'] == site]
    labeled_data = site_data.dropna(subset=['Storm_Name'])
    unlabeled_data = site_data[site_data['Storm_Name'].isnull()]
    
    # Scatter plot for unlabeled data (without storm names)
    fig.add_trace(go.Scatter(
        x=unlabeled_data['Peak Date'],
        y=unlabeled_data['Peak Value'],
        mode='markers',
        marker=dict(color='blue'),
        name='No Storm Name',
        text=unlabeled_data.apply(lambda row: f"Site: {row['Site']}", axis=1),  # Display site name on hover
    ), row=i+1, col=1)
    
    # Scatter plot for labeled data (with storm names)
    fig.add_trace(go.Scatter(
        x=labeled_data['Peak Date'],
        y=labeled_data['Peak Value'],
        mode='markers',
        marker=dict(color='red'),
        name='Storm Name',
        text=labeled_data.apply(lambda row: f"Site: {row['Site']}<br>Storm: {row['Storm_Name']}", axis=1),  # Display site and storm names on hover
    ), row=i+1, col=1)

    fig.update_xaxes(range=[datetime(2015, 1, 1), datetime.now()], row=i+1, col=1)

# Update layout
fig.update_layout(
    title='Peak Value vs Peak Date by Site',
    xaxis_title='Peak Date',
    yaxis_title='Peak Value',
    showlegend=False,  # Hide legend as subplots have individual legends
    height=1000,  # Adjust height as needed
    width=800,  # Adjust width as needed
)

# Show plot
fig.show()
