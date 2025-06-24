import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from geopy.distance import geodesic 

def calculate_mae_km(df, predicted_lat_col, predicted_lon_col, true_lat_col, true_lon_col):
    """
    Calculates the Mean Absolute Error (MAE) in kilometers for predicted
    latitude and longitude compared to true values.

    Args:
        df (pd.DataFrame): DataFrame containing predicted and true latitude/longitude columns.
        predicted_lat_col (str): Name of the predicted latitude column.
        predicted_lon_col (str): Name of the predicted longitude column.
        true_lat_col (str): Name of the true latitude column.
        true_lon_col (str): Name of the true longitude column.

    Returns:
        tuple: A tuple containing the Mean Absolute Error for latitude (km)
               and the Mean Absolute Error for longitude (km). Returns (np.nan, np.nan)
               if the true latitude or longitude columns are not found.
    """
    lat_distances_km = []
    lon_distances_km = []

    if true_lat_col in df.columns and true_lon_col in df.columns:
        for idx, row in df.iterrows():
            pred_lat = row[predicted_lat_col]
            pred_lon = row[predicted_lon_col]
            true_lat = row[true_lat_col]
            true_lon = row[true_lon_col]

            if np.isfinite(pred_lat) and np.isfinite(pred_lon) and np.isfinite(true_lat) and np.isfinite(true_lon):
                # Approximate kilometers per degree
                lat_km_per_degree = geodesic((true_lat, pred_lon), (true_lat + 1, pred_lon)).kilometers
                lon_km_per_degree = geodesic((pred_lat, true_lon), (pred_lat, true_lon + 1)).kilometers

                lat_diff_km = abs(pred_lat - true_lat) * lat_km_per_degree
                lon_diff_km = abs(pred_lon - true_lon) * lon_km_per_degree

                lat_distances_km.append(lat_diff_km)
                lon_distances_km.append(lon_diff_km)
            else:
                lat_distances_km.append(np.nan)
                lon_distances_km.append(np.nan)

        mae_lat_km = np.nanmean(np.abs(lat_distances_km))
        mae_lon_km = np.nanmean(np.abs(lon_distances_km))

        return mae_lat_km, mae_lon_km
    else:
        print(f"Warning: '{true_lat_col}' or '{true_lon_col}' columns not found. Cannot calculate MAE in km.")
        return np.nan, np.nan

# Inverse transform xyz cordinates into latitude and longitude values
def inverse_transform_spherical(scaled_xyz, coordinate_scaler):
    """Inverse transforms scaled x, y, z back to latitude and longitude (degrees)."""
    xyz = coordinate_scaler.inverse_transform(scaled_xyz)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    latitude_rad = np.arcsin(np.clip(z, -1, 1))
    longitude_rad = np.arctan2(y, x)
    latitude_deg = np.degrees(latitude_rad)
    longitude_deg = np.degrees(longitude_rad)
    return latitude_deg, longitude_deg


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

    return in_data, X, y, le_continent, le_city, coordinate_scaler, continent_encoding_map, city_encoding_map



in_data, X, y, le_continent, le_city, coordinate_scaler, continent_encoding_map, city_encoding_map = process_data("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")



# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X[:],y[:,2:],random_state=123,test_size=0.2)
# Split train into train and validation as well
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=123, test_size=0.2)



print('Training, Validation and Testing matrices shapes')
print("\nTraining\n")
print(X_train.shape, y_train.shape)
print("\nValidation\n")
print(X_val.shape, y_val.shape)
print("\nTesting\n")
print(X_test.shape, y_test.shape)



# Instantiate a base regressor
base_model = XGBRegressor(
    objective='reg:squarederror',  # for regression
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=123,
    verbosity=1
)

# Wrap in a multi-output regressor
model = MultiOutputRegressor(base_model)

# Fit the model on training data
model.fit(X_train, y_train)

# Predict on validation and test sets
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Evaluate performance on validation and test data
def evaluate_regression(true, pred, dataset_name=""):
    print(f"\nPerformance on {dataset_name} set:")
    print("RMSE:", np.sqrt(mean_squared_error(true, pred)))
    print("R² Score:", r2_score(true, pred))

evaluate_regression(y_val, y_val_pred, "Validation")
evaluate_regression(y_test, y_test_pred, "Test")

true_lat, true_long = inverse_transform_spherical(y_test,coordinate_scaler)
predicted_lat, predicted_long = inverse_transform_spherical(y_test_pred,coordinate_scaler)

# Create a DataFrame to hold the coordinates
results_df = pd.DataFrame({
    'true_lat': true_lat,
    'true_long': true_long,
    'pred_lat': predicted_lat,
    'pred_long': predicted_long
})

# Call the MAE in km function
mae_lat_km, mae_long_km = calculate_mae_km(
    results_df,
    predicted_lat_col='pred_lat',
    predicted_lon_col='pred_long',
    true_lat_col='true_lat',
    true_lon_col='true_long'
)

print("\nMean Absolute Error (in kilometers):")
print(f"Latitude MAE: {mae_lat_km:.2f} km")
print(f"Longitude MAE: {mae_long_km:.2f} km")




# Read the data 
df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
df = pd.concat([df.iloc[:,:-4],df['city']],axis=1)
x_data = df[df.columns[:-1]][:]
print(x_data.shape)
y_data = df[df.columns[-1]][:]
le = LabelEncoder()
y_data = le.fit_transform(y_data)
print(le.classes_)


# Train-test-split
X_train, X_test, y_train, y_test = train_test_split(x_data,y_data,random_state=123,test_size=0.2, stratify=y_data)
# Split train into train and validation as well
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=123, test_size=0.2,stratify=y_train)

print('Training, Validation and Testing matrices shapes')
print("\nTraining\n")
print(X_train.shape, y_train.shape)
print("\nValidation\n")
print(X_val.shape, y_val.shape)
print("\nTesting\n")
print(X_test.shape, y_test.shape)


# Set the model XGB Classifier

xgb_classifier = XGBClassifier(objective="multi:softmax",
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42, n_estimators = 100, max_depth = 3,
)
xgb_classifier.fit(X_train, y_train)

# Train on the training dataset
xgb_classifier.fit(X_train,y_train)

# Validate on the validation dataset
y_pred = xgb_classifier.predict(X_test)

test_accuracy = accuracy_score(y_test,y_pred)
print(f"The test accuracy on the validation dataset is {test_accuracy:.4f}")

# Print classification report
print("\nClassfication Report:\n",classification_report(y_test,y_pred))

# Print Confusion Matrix
print("\nConfusion Matrix\n", confusion_matrix(y_test,y_pred))



