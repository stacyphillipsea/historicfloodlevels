# Welcome to my Read Me doc!

## This is the code for calculating historic peak information!

I have branched my levels-data app, moved it to a new repo and cloned it. From here I am going to tweak the code. I think it is set up in VS code correctly.

The aim of this work is to utlise my previous code to look at previous storm events and get peak information.

Harry D has given me information on all historic storms and all sites in WMD with WISKI IDs.

## Functionality that is currently working
* Makes a WISKI IDs list from the csv file. Currently using a subset of stations to test code.
* Daily max data works fine for peaks
* Created a historic_nest_dict of the daily max data for subset of sites to work with
* Peaks now identified using find_peaks instead of feeding it data ranges
* Have removed the sliders for data exploration and have started looking at assigning the parameters, but function is still there for when that is appropriate
* Have used the 'stationOpened' parameter in the API to extend to full historical record
* Added Tamworth, Gloucester, Bidford to the list
* Can assign storm name based on if the peak date is within the start and end date of the storm.
* Made a plot to see how many peaks are found during storms or not - could use this to optimise peak settings? Data is currently just using threshold value and defaults
* Using the threshold dataframe as the height values 
* Settings for prominence and distance will need to be different for SHWG and SWWM because of how the hydrographs vary

## Things that could be improved
* Haven't touched any of the visualisations yet
* Haven't deployed as an app yet
* Peak Table All uses sites of interest so because Bidford isn't on there it doesn't return all the results so just looking at df for now

## Things to remember to run in VSCODE
* rsconnect deploy dash . -n LevelsApp --entrypoint levels_app:app 
* pip freeze > requirements.txt

* have used pd.set_option('display.max_rows', None) to show full dataframe
* can reset with pd.reset_option('display.max_rows')

