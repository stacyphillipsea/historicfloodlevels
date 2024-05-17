# Welcome to my Read Me doc!

## This is the code for calculating historic peak information!

I have branched my levels-data app, moved it to a new repo and cloned it. From here I am going to tweak the code. I think it is set up in VS code correctly.

The aim of this work is to utlise my previous code to look at previous storm events and get peak information.

Harry D has given me information on all historic storms and all sites in WMD with WISKI IDs.

## Functionality that is currently working
* Makes a WSIKI IDs list from the csv file. Currently using a slice of 3 stations to test code.
* Makes a date_filters dictionary for each of the storms, I have extended the date ranges to try and capture the storm event itself.
* Extracting 15minute data only gets me as far back as 2018 (might be a data load problem). Daily max data gives me datetimes of peaks so should work
* Create a historic_nest_dict of the daily max data to work with
* Can find the peaks fine but the storms don't necessarily match with the events on the gaugeboard and vice versa. Dates may need further twiddling


## Things that could be improved
* Haven't touched any of the visualisations yet
* Haven't deployed as an app yet
* Peak Table All uses sites of interest so because Bidford isn't on there it doesn't return all the results so just looking at df for now
* Started playing about with peak finding algorithms. Current one is too sensitive


## Things to remember to run in VSCODE
* rsconnect deploy dash . -n LevelsApp --entrypoint levels_app:app 
* pip freeze > requirements.txt

* have used pd.set_option('display.max_rows', None) to show full dataframe
* can reset with pd.reset_option('display.max_rows')

