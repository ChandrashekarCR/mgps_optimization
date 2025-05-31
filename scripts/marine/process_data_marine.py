# Importing libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np


# Function to process the data to feed it into the neural network
def process_data(in_data):

    # Initialize label and scalers
    le_sea = LabelEncoder()
    stdscaler_lat = StandardScaler() # I can try MinMaxScaler as well
    stdscaler_long = StandardScaler() # I can try MinMaxScaler as well
    coordinate_scaler = StandardScaler()
    
    # Convert all the categorical variables into numbers
    in_data['sea_encoding'] = in_data[['Sea']].apply(le_sea.fit_transform)
    in_data['lat_scaled'] = stdscaler_lat.fit_transform(in_data[['latitude']])
    in_data['long_scaled'] = stdscaler_long.fit_transform(in_data[['longitude']])
    
    # Another way of scaling latitiude and longitude data. 
    # https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature 
    # Convert latitude and longitutde into radians
    in_data['latitude_rad'] = np.deg2rad(in_data['latitude'])
    in_data['longitude_rad'] = np.deg2rad(in_data['longitude'])

    # Calculate x, y, z coordinates
    in_data['x'] = np.cos(in_data['latitude_rad']) * np.cos(in_data['longitude_rad'])
    in_data['y'] = np.cos(in_data['latitude_rad']) * np.sin(in_data['longitude_rad'])
    in_data['z'] = np.sin(in_data['latitude_rad'])

    # Scale the x, y, z coordinates together
    in_data[['scaled_x','scaled_y','scaled_z']] = coordinate_scaler.fit_transform (in_data[['x','y','z']])

    # Encoding dictionary for simpler plotting and understanding the results
    sea_encoding_map = dict(zip(le_sea.transform(le_sea.classes_), le_sea.classes_))

    return in_data, le_sea, stdscaler_lat, stdscaler_long, coordinate_scaler , sea_encoding_map
