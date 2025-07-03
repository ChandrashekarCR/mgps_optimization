# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from geopy.distance import geodesic
from sklearn.multioutput import MultiOutputRegressor


def calculate_mae_km(df, predicted_lat_col, predicted_lon_col, true_lat_col, true_lon_col):
    lat_distances_km = []
    lon_distances_km = []

    if true_lat_col in df.columns and true_lon_col in df.columns:
        for idx, row in df.iterrows():
            pred_lat = row[predicted_lat_col]
            pred_lon = row[predicted_lon_col]
            true_lat = row[true_lat_col]
            true_lon = row[true_lon_col]

            if np.isfinite(pred_lat) and np.isfinite(pred_lon) and np.isfinite(true_lat) and np.isfinite(true_lon):
                lat_km_per_degree = geodesic((true_lat, pred_lon), (true_lat + 1, pred_lon)).kilometers
                lon_km_per_degree = geodesic((pred_lat, true_lon), (pred_lat, true_lon + 1)).kilometers

                lat_diff_km = abs(pred_lat - true_lat) * lat_km_per_degree
                lon_diff_km = abs(pred_lon - true_lon) * lon_km_per_degree

                lat_distances_km.append(lat_diff_km)
                lon_distances_km.append(lon_diff_km)
            else:
                lat_distances_km.append(np.nan)
                lon_distances_km.append(np.nan)

        mae_lat_km = np.nanmean(lat_distances_km)
        mae_lon_km = np.nanmean(lon_distances_km)

        return mae_lat_km, mae_lon_km
    else:
        print(f"Warning: '{true_lat_col}' or '{true_lon_col}' columns not found. Cannot calculate MAE in km.")
        return np.nan, np.nan

def inverse_transform_spherical(scaled_xyz, coordinate_scaler):
    xyz = coordinate_scaler.inverse_transform(scaled_xyz)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    latitude_rad = np.arcsin(np.clip(z, -1, 1))
    longitude_rad = np.arctan2(y, x)
    latitude_deg = np.degrees(latitude_rad)
    longitude_deg = np.degrees(longitude_rad)
    return latitude_deg, longitude_deg

def process_data(data_path):
    in_data = pd.read_csv(data_path)

    # Initialize label and scalers
    le_continent = LabelEncoder()
    le_city = LabelEncoder()
    coordinate_scaler = StandardScaler()

    # Encode categorical targets
    in_data['continent_encoding'] = le_continent.fit_transform(in_data['continent'])
    in_data['city_encoding'] = le_city.fit_transform(in_data['city'])

    # Convert latitude and longitude to radians
    in_data['latitude_rad'] = np.deg2rad(in_data['latitude'])
    in_data['longitude_rad'] = np.deg2rad(in_data['longitude'])

    # Calculate x, y, z coordinates
    in_data['x'] = np.cos(in_data['latitude_rad']) * np.cos(in_data['longitude_rad'])
    in_data['y'] = np.cos(in_data['latitude_rad']) * np.sin(in_data['longitude_rad'])
    in_data['z'] = np.sin(in_data['latitude_rad'])

    # Scale x,y,z coordinates together
    in_data[['scaled_x', 'scaled_y', 'scaled_z']] = coordinate_scaler.fit_transform(in_data[['x', 'y', 'z']])

    # Define non-feature columns to drop from features
    non_feature_cols = [
        'city', 'continent', 'latitude', 'longitude',
        'continent_encoding', 'city_encoding',
        'latitude_rad', 'longitude_rad',
        'x', 'y', 'z',
        'scaled_x', 'scaled_y', 'scaled_z'
    ]

    # Features: all columns except these
    X = in_data.drop(columns=non_feature_cols)
    # Targets separately
    y_continent = in_data['continent_encoding']
    y_city = in_data['city_encoding']
    y_cord = in_data[['scaled_x', 'scaled_y', 'scaled_z']]  # scaled xyz as lat proxy
    # We will inverse transform later for latitude and longitude

    return in_data, X, y_continent, y_city, y_cord, le_continent, le_city, coordinate_scaler

# Load and process data
data_path = "/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv"
in_data, X, y_continent, y_city, y_cord, le_continent, le_city, coordinate_scaler = process_data(data_path)

# Split datasets for hierarchical training
# First split train/test for all data
X_train, X_test, y_cont_train, y_cont_test, y_city_train, y_city_test, y_lat_train, y_lat_test = train_test_split(
    X, y_continent, y_city, y_cord, test_size=0.2, random_state=123, stratify=y_continent
)
# Then split train into train/val
X_train, X_val, y_cont_train, y_cont_val, y_city_train, y_city_val, y_lat_train, y_lat_val = train_test_split(
    X_train, y_cont_train, y_city_train, y_lat_train, test_size=0.2, random_state=123, stratify=y_cont_train
)

# 1) Train continent classifier
xgb_continent = XGBClassifier(
    objective="multi:softmax",
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_estimators=100,
    max_depth=3,
)
xgb_continent.fit(X_train, y_cont_train)

# Predict continent probabilities for train, val, test
cont_train_probs = pd.DataFrame(xgb_continent.predict_proba(X_train), index=X_train.index)
cont_val_probs = pd.DataFrame(xgb_continent.predict_proba(X_val), index=X_val.index)
cont_test_probs = pd.DataFrame(xgb_continent.predict_proba(X_test), index=X_test.index)

# Append predicted continent probabilities to original features to predict city
X_train_city = pd.concat([X_train, cont_train_probs], axis=1)
X_val_city = pd.concat([X_val, cont_val_probs], axis=1)
X_test_city = pd.concat([X_test, cont_test_probs], axis=1)

# 2) Train city classifier
xgb_city = XGBClassifier(
    objective="multi:softmax",
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_estimators=100,
    max_depth=3,
)
xgb_city.fit(X_train_city, y_city_train)

# Predict city probabilities for train, val, test
city_train_probs = pd.DataFrame(xgb_city.predict_proba(X_train_city), index=X_train_city.index)
city_val_probs = pd.DataFrame(xgb_city.predict_proba(X_val_city), index=X_val_city.index)
city_test_probs = pd.DataFrame(xgb_city.predict_proba(X_test_city), index=X_test_city.index)

# Append predicted city probabilities + original features to predict latitude (regression)
X_train_lat = pd.concat([X_train, city_train_probs], axis=1)
X_val_lat = pd.concat([X_val, city_val_probs], axis=1)
X_test_lat = pd.concat([X_test, city_test_probs], axis=1)

# 3) Train latitude regressor (multi-output regression for scaled_x, scaled_y, scaled_z)

xgb_lat = MultiOutputRegressor(
    XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
)
xgb_lat.fit(X_train_lat, y_lat_train)

# Predict latitude scaled xyz for train, val, test
lat_train_pred = pd.DataFrame(xgb_lat.predict(X_train_lat), index=X_train_lat.index, columns=y_cord.columns)
lat_val_pred = pd.DataFrame(xgb_lat.predict(X_val_lat), index=X_val_lat.index, columns=y_cord.columns)
lat_test_pred = pd.DataFrame(xgb_lat.predict(X_test_lat), index=X_test_lat.index, columns=y_cord.columns)


# Evaluate continent classification
y_cont_test_pred = xgb_continent.predict(X_test)
print("Continent Classification Report:\n", classification_report(y_cont_test, y_cont_test_pred))
print("Continent Test Accuracy:", accuracy_score(y_cont_test, y_cont_test_pred))

# Evaluate city classification
y_city_test_pred = xgb_city.predict(X_test_city)
print("\nCity Classification Report:\n", classification_report(y_city_test, y_city_test_pred))
print("City Test Accuracy:", accuracy_score(y_city_test, y_city_test_pred))

# Inverse transform predicted and true xyz to latitude and longitude degrees for MAE calculation
true_lat_test, true_long_test = inverse_transform_spherical(y_lat_test.values, coordinate_scaler)
pred_lat_test, pred_long_test = inverse_transform_spherical(lat_test_pred.values, coordinate_scaler)

results_df = pd.DataFrame({
    'true_lat': true_lat_test,
    'true_long': true_long_test,
    'pred_lat': pred_lat_test,
    'pred_long': pred_long_test
})

mae_lat_km, mae_long_km = calculate_mae_km(
    results_df,
    predicted_lat_col='pred_lat',
    predicted_lon_col='pred_long',
    true_lat_col='true_lat',
    true_lon_col='true_long'
)

print(f"\nMean Absolute Error (Latitude) in km: {mae_lat_km:.3f}")
print(f"Mean Absolute Error (Longitude) in km: {mae_long_km:.3f}")
