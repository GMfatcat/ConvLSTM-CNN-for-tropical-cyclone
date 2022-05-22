# download.py
# Written By Connor Cozad
#
# Purpose of this file:
# This file downloads the satellite images which will then be processed in assemble.py
#
# Outline of this file:
# - Loops through each year and downloads .tar.gz files containing satellite images for Atlantic and Pacific hurricanes
# - Only extracts files from the .tar.gz files that contain images of hurricanes that we know the wind speed of
# - When script is finished, the Satellite Imagery folder contains all netcdf files, which hold the satellite images


import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import tarfile


def download_hursat(years):
    # Reads in the Best Track dataset, which contain records of the location and maximum wind speed of every recorded
    # hurricane in the Atlantic and Eastern/Central Pacific basins
    best_track_data = pd.read_csv('besttrack.csv')

    for year in years:
        # Scrapes a webpage to get list of all .tar.gz files. Each file contains all the satellite images associated
        # with a particular hurricane.
        year_directory_url = 'https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/' + year
        year_directory_page = requests.get(year_directory_url).text
        year_directory_soup = BeautifulSoup(year_directory_page, 'html.parser')
        year_directory_file_urls = [year_directory_url + '/' + node.get('href') for node in
                                    year_directory_soup.find_all('a') if node.get('href').endswith('tar.gz')]
        print('\n' + year + ' file loaded.')

        files_processed = 0
        for storm_file_url in year_directory_file_urls:
            # Determine whether the best track dataset has information about this particular hurricane. This filters
            # out storms in basins other than the Atlantic or Pacific, since the best track dataset doesn't have
            # information for those storms.
            storm_name = storm_file_url.split('_')[-2]
            year = int(storm_file_url.split('_')[3][:4])
            file_has_match_in_best_track = not best_track_data.loc[
                (best_track_data['year'] == year) & (best_track_data['storm_name'] == storm_name)
            ].empty

            if file_has_match_in_best_track:
                # Build a string, which will be file path where we save the .tar.gz when downloaded
                file_name = storm_file_url.split('/')[-1]
                storm_file_path = 'Satellite Imagery/' + file_name

                # Create the Satellite Imagery folder if it doesn't already exist
                if not os.path.exists('Satellite Imagery'):
                    os.makedirs('Satellite Imagery')

                # Open the .tar.gz and copy it's contents from the web, onto our computer
                request = requests.get(storm_file_url, allow_redirects=True)
                open(storm_file_path, 'wb').write(request.content)
                request.close()

                # Open the .tar.gz file and loop through each file inside. Each of these netcdf files contains a
                # satellite image of a hurricane at a moment in time
                tar = tarfile.open(storm_file_path)
                file_prefixes_in_directory = []
                for file_name in tar.getnames():
                    # Get the date and time of the satellite image, and the name of the satellite that took the image
                    fulldate = file_name.split(".")[2] + file_name.split(".")[3] + file_name.split(".")[4]
                    time = file_name.split(".")[5]
                    satellite = file_name.split(".")[7][:3]

                    # Determine whether the best track dataset has a record for the date and time of this storm.
                    file_has_match_in_best_track = not best_track_data.loc[
                        (best_track_data['fulldate'] == int(fulldate)) & (best_track_data['time'] == int(time))].empty

                    # Determine whether another image of this hurricane at this exact time has already been extracted
                    # from the .tar.gz
                    is_redundant = '.'.join(file_name.split('.')[:6]) in file_prefixes_in_directory

                    # If the requirements are met, extract the netcdf file from this .tar.gz and save it locally
                    if file_has_match_in_best_track and not is_redundant and satellite == "GOE":
                        f = tar.extractfile(file_name)
                        open('Satellite Imagery/' + file_name, 'wb').write(f.read())
                        file_prefixes_in_directory.append('.'.join(file_name.split('.')[:6]))

                tar.close()
                os.remove(storm_file_path)

            files_processed += 1
            print_progress('Processing Files for ' + str(year), files_processed, len(year_directory_file_urls))


def print_progress(action, progress, total):
    percent_progress = round((progress / total) * 100, 1)
    print('\r' + action + '... ' + str(percent_progress) + '% (' + str(progress) + ' of ' + str(total) + ')', end='')


if __name__ == "__main__":
    # Specify a list of years. Satellite images of hurricanes from those years will be downloaded. More years will
    # provide more data for the neural network to work with in model.py, but will take longer to download.
    #YEARS_TO_DOWNLOAD = ['2016', '2015', '2014', '2013', '2012']
    YEARS_TO_DOWNLOAD = ['2012']

    download_hursat(YEARS_TO_DOWNLOAD)
