# Welcome to my Read Me doc!

## This is the code for calculating historic peak information!
I have started a new Read Me for my analysis, where I am trying to see if there has been any changes over time in how our rivers are responding

## Functionality that is currently working
* I now have a data_dict that uses the historic data_dict and combines the manual threshold list so  it is all in one plcae (not how I have it for FETA or elsewhere)
* Calculates the number of days in each Winter period that the river has been above the threshold
* It does it for the whoelrecord available, adding 1 year to the start date to ensure a full year of data is used
* Calculates average number of days it is above the threshold for teach site
* This will output a dataframe and make a plot
* Plot has all years, with the top 5 values highlighted, and teh average line added with the average value in the legend
* Heatmap that pivots and shows all sites together with a continuous colour scale
* Normalised heatmap by the maximum number of days above threshold to allow comparison of sites

## Things that could be improved
* Non-normalised heatmap is not showing all sites, normalised one is
* I now know how to get at the typical high values from the API - TRY THIS!!


