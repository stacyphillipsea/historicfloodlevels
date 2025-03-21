# Welcome to my Read Me doc!

## This is the code for my Apprenticeship End Point Assessment Project!

This project aims to make a Dash App that analyses EA Gauging Station data across the West Midlands, making use of the EA Hydrology API
It calculates peak levels for given storm events over the winter of 2023-2024, visualises them in charts and tables, and compares them to historic peak levels to provide context.

I have already done loads of work on this before using Git Hub, so earlier versions of the code are not included.

## Functionality that is currently working
* Double dropdown menu for River name then Station name
* Peak graph and peak table
* All dates throughout are formatted as I want them to be
* The download data button currently works well
* Peak plot is being loaded into a Powerpoint that can be downloaded locally
* Have downloaded and stored the data_dict as a JSON that is then loaded into the file, instead of calling the API everytime
* Difference from and Years to peak date column added
* Map showing position of selected station and all stations
* Navbar formatted with items on left and right
* Apprenticeships logo added
* Map shows formatted popup and colour coded symbols based on river name
* Basic table border formatting, sort and filter added for both top-10 and all peak table
* Intro added at the top
* Hyperlinks throughout to help navigation
* Popup modal added to show the data quality messaging
* Photo aded next to info
* Email address added to top intro
* Data sources section updated
* Link at bottom to send back to top
* Import libraries tidied
* App log warnings fixed
* Table widths fixed and headers fixed
* Added hyperlink for full screen image
* External hyperlinks now are forced to open in new tab
* Used "app.get_asset_url()" to get correct path for local images so they are pulled in correctly, instead of having them as urls
* y-axis chart labels go across two lines
* Cambridge Spark logo addeds

## Things that could be improved
* Would like to colour code storms in the top 10 table
* Cannot get the download button for the Powerpoint to work. Created separate branch (Powerpoint-Download-Attempt). Tried again using the new source url path and it just wouldn't work!! 
* Navbar is not pulling the images from the project folder, had to give it urls instead
* Attempted to tidy up the callback and update functions but didn't work (didn't save)
* Would be nice to include more icons, couldn't get it to work with a "To top" button
* Should include data quality tag information in here somewhere
* WISKI IDs list should be derived from the sites of interest merge using unique, currently it is a static list. Can compare this list with existing list to check info is correct

### THINK I HAVE GOT THIS BACK TO WHERE IT SHOULD BE AND HAVE SINCE BRANCHED TO EXTEND THE SITE LIST
* New nested_data_dict it too big to upload to git so after faffing around with large file storage, i reset and just tell git to ignore the nested_dict_extended file

## Updates 29/05/2024
* Updated the nested_dict to run through the whole like of WISKI sites 
* This returned some sites with multiple river names so this now just takes the first one in the list, and also None in the River name
* Had to update the map and component code to be able to deal with Nones in the river list
* Performed an outer join to make the peak table all, so that it isn't relying on the sites being in the sites of interest merge
* There are stray stations at the bottom of the dataframe that aren't matched - this is due to the merge happening on name not on WISKI ID and there are errors in the names in the system
* There are now 231 sites in the dataset
* Not enough colours for river pins to be unique but still works, it just reuses colours
* Updated zoom to zoom in more now there are so many pins

## Extending-site-list-2 branch
## Updates 30/05/2024
* Tried updating the merge so it worked on WISKI IDS but it broke the visualisations
* Need to not change the df, max_values and process_peak_table_all because the dictionary structure doesn't work with the extra WISKI ID stored in there
* Discarded changes!
* Need to edit the WYE IDs so that they start 055

## Updates 05/06/2024
* Increased the timeout time in Posit from 60 to 120
* Hashed out the Powerpoint functions to improve efficiency
* Edited the WISKI IDs and data_dict to include River Wye stations
* Created a processed and unprocessed ids list to help with data loading
* Tidied up print statements to help identify sites that don't have data

## Updates 10/07/2024
* Changed deployment code to work with Python 10 (updated Python 12 won't deploy to Posit currently)
* Made a local environment in which to run 310
* Corrected date of Elin & Fergus peak from 9th Nov to 9th Dec: 158 records change their peak times and dates as a result
* Updated the data_dict with the Ross on Wye data from MTS
* Turned this into a function so could be used elsewhere
* Added spatial search to see if in SHWG or SWWM - now that is there, how do I get it into the peak table all

## Updates 08/10/2024 (SEPT2024FloodLevels branch)
* Copied files to make a new FETA to look at September 2024 analysis
* Have managed to publish to a new app in sept24_levels_app folder (broke the multipage one...)
* Want to try the peak ID because different rivers had differnt events and all responded differently
* Realised that some river names are blank and so can't be accessed via the dropdown - tried to get it to fix at source when doing the JSON call but ran into difficulties. If riverName isn't available it won't be in the API response, rather than it just being blank. Instead I fixed it by iterating throughthe data_dict after it has been created and replacing the value 
* Lengthened the period of time I was looking at. NEED TO DOUBLE CHECK THAT THE CODE IS SAVING THE JSON IN THE RIGHT PLACE, I DON'T THINK i KEPT THIUS WHEN I REVERTED THE CHANGES
* Almost got last years data in there as a comparison but I broke it so I reverted the changes


## Updates 20/03/2025
* Fixed all the top part of the code to be able to get the data from September 24 to Feb 25
* Used Harry's filters and used Hex codes to change the colours to what he wants.
* API seemed to change how you get the readings, and can no longer construict it from the WIKSI ID
* Figured out how to fix the URL and it is now able to go and get the readings and creat a data_dict in the same way as before
* All the functions above the app appear to work and do not cause errors
* Deals in 2 places with the river list returning multiple rivers (takes the first one)
* Works running it to the app!


## Things to remember to run in VSCODE
* rsconnect deploy dash . -n LevelsApp --entrypoint levels_app:app 
* pip freeze > requirements.txt

