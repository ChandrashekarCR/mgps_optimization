# This script is used to modify the target variables such that, the values are suitable for training the neural network.

# Importing libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Function to process the data to feed it into the neural network
def process_data(data_path):

    try:
        in_data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")


    # Initialize label and scalers
    le_continent = LabelEncoder()
    le_city = LabelEncoder()
    stdscaler_lat = StandardScaler() 
    stdscaler_long = StandardScaler() 
    coordinate_scaler = StandardScaler()

    
    # Convert all the categorical variables into numbers
    in_data['city_encoding'] = in_data[['city']].apply(le_city.fit_transform)
    in_data['continent_encoding'] = in_data[['continent']].apply(le_continent.fit_transform)
    in_data['lat_scaled'] = stdscaler_lat.fit_transform(in_data[['latitude']])
    in_data['long_scaled'] = stdscaler_long.fit_transform(in_data[['longitude']])

    
    # Another way of scaling latitiude and longitude data. 
    # https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature 
    # Convert latitude and longitutde into radians
    in_data['latitude_rad'] = np.deg2rad(in_data['latitude'])
    in_data['longitude_rad'] = np.deg2rad(in_data['longitude'])

    # Calculate x, y, z coordinates -  Converting polar co-ordinates into cartesian co-ordinates
    in_data['x'] = np.cos(in_data['latitude_rad']) * np.cos(in_data['longitude_rad'])
    in_data['y'] = np.cos(in_data['latitude_rad']) * np.sin(in_data['longitude_rad'])
    in_data['z'] = np.sin(in_data['latitude_rad'])

    # Scale the x, y, z coordinates together
    in_data[['scaled_x','scaled_y','scaled_z']] = coordinate_scaler.fit_transform (in_data[['x','y','z']])

    # Encoding dictionary for simpler plotting and understanding the results
    continent_encoding_map = dict(zip(le_continent.transform(le_continent.classes_), le_continent.classes_))
    city_encoding_map = dict(zip(le_city.transform(le_city.classes_),le_city.classes_))

    # Define all non-feature columns
    non_feature_columns = [
        'city', 'continent', 'latitude', 'longitude', # Original identifier/target columns
        'city_encoding', 'continent_encoding', # Encoded categorical targets
        'lat_scaled', 'long_scaled', # Old scaled lat/long (if not used as features)
        'latitude_rad', 'longitude_rad', # Intermediate radian values
        'x', 'y', 'z', # Intermediate cartesian coordinates
        'scaled_x', 'scaled_y', 'scaled_z','Unnamed: 0' # Final XYZ targets
    ]

    # Select X by dropping non-feature columns
    # Use errors='ignore' in case some columns don't exist (e.g., if you only keep one scaling method)
    X = in_data.drop(columns=non_feature_columns, errors='ignore').values.astype(np.float32)

    # Define target columns explicitly
    y_columns = ['continent_encoding', 'city_encoding', 'scaled_x','scaled_y','scaled_z']
    y = in_data[y_columns].values.astype(np.float32)

    # --- Changes end here ---

    return in_data, X, y, le_continent, le_city, coordinate_scaler, continent_encoding_map, city_encoding_map



def split_data(X,y, test_size=0.2, random_state=123):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


#in_data, X, y, le_continent, le_city, coordinate_scaler ,continent_encoding_map, city_encoding_map = process_data("/home/chandru/binp37/results/metasub/tax_metasub_data.csv")

#print(X.shape)
