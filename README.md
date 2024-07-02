# Welcome to my Read Me doc!

## This is the code for calculating historic peak information!

I have branched my levels-data app, moved it to a new repo and cloned it. From here I am going to tweak the code. I think it is set up in VS code correctly.

The aim of this work is to utlise my previous code to look at previous storm events and get peak information.

Harry D has given me information on all historic storms and all sites in WMD with WISKI IDs.

## Functionality that is currently working
* Makes a WIsKI IDs list from the csv file. Currently using a slice of 3 stations to test code.
* Makes a date_filters dictionary for each of the storms, I have extended the date ranges to try and capture the storm event itself.
* Extracting 15minute data only gets me as far back as 2018 (might be a data load problem). * Daily max data gives me datetimes of peaks so should work
* Created a historic_nest_dict of the daily max data for 3 sites to work with
* Can find the peaks fine but the storms don't necessarily match with the events on the gaugeboard and vice versa. Dates may need further twiddling

# Peak analysis work
* Have produced code that allows you to identify peaks over a time series.
* Interactive sliders allow the user to define prominence, height, and distance parameters to identify the peaks wanted
* These can then be assigned to a dataframe when the user is happy
* Height could probably be defined by the threshold value
* Some code afterwards to try and identify time periods with peaks i.e. storm events
* Data_dict only gets 3 sites at the moment, and this wouldn't work iteratively as user input is needed
* Have used the 'stationOpened' parameter in the API to extend to full historical record


## Things that could be improved
* Haven't touched any of the visualisations yet
* Haven't deployed as an app yet
* Peak Table All uses sites of interest so because Bidford isn't on there it doesn't return all the results so just looking at df for now
* Started playing about with peak finding algorithms. Current one is too sensitive
* Added Tamworth to the list
* Can assign storm name based on if the peak date is within the start and end date of the storm.
* Made a plot to see how many peaks are found during storms or not - could use this to optimise peak settings? Data is currently just using threshold value and defaults
* Using the threshold dataframe as the height values 


## Things to remember to run in VSCODE
* rsconnect deploy dash . -n LevelsApp --entrypoint levels_app:app 
* pip freeze > requirements.txt

* have used pd.set_option('display.max_rows', None) to show full dataframe
* can reset with pd.reset_option('display.max_rows')

