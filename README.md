# Welcome to my Read Me doc!

## This is the code for my Apprenticeship End Point Assessment Project!

This project aims to make a Dash App that analyses EA Gauging Station data across the West Midlands, making use of the EA Hydrology API
It calculates peak levels for given storm events over the winter of 2023-2024, visualises them in charts and tables, and compares them to historic peak levels to provide context.

I have already done loads of work on this before using Git Hub, so earlier versions of the code ar enot included.

## Functionality that is currently working
* Double dropdown menu  for River name then Station name
* Peak graph and peak table
* All dates throughout are formatted as I want them to be
* The download data button currently works well
* Peak plot is being loaded into a Powerpoint that can be downloaded locally
* Have downloaded and stored the data_dict as a JSON that is then loaded into the file, instead of calling the API everytime
* Difference from and Years to peak date column added
* Map showing position of selected station and all stations
* Navbar formatted with items on left and right
* Map shows formatted popup and colour coded symbols based on river name
* Basic table border formatting, sort and filter added for both top-10 and all peak table
* Intro added at the top
* Hyperlinks added to intro to help navigation
* Popup modal added to show the data quality messaging
* Photo aded next to info
* Email address added to top intro
* Data sources section updated

## Things that could be improved
* Would like to colour code storms in the top 10 table
* The Historic versus winter chart is fine, but it is a different size when it default loads and then is the intended size when the callback update code is triggered
* The exceptional levels table and the all peak data table are not formatted
* Cannot get the download button for the Powerpoint to work. Created separate branch (Powerpoint-Download-Attempt)
* Navbar is not pulling the images from the project folder, had to give it urls instead
* Attempted to tidy up the callback and update functions but didn't work (didn't save)


## Things to remember to run in VSCODE
* rsconnect deploy dash . -n LevelsApp --entrypoint levels_app:app 
* pip freeze > requirements.txt

