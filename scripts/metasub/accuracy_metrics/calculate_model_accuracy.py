# Importing libraries
import torch
import numpy as np
import pandas as pd
from geopy.distance import geodesic 
from sklearn.metrics import precision_score, recall_score, f1_score

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


def check_combined_accuracy(loader, model, coordinate_scaler=None, device="cpu",
                            has_continent=True, has_city=True, has_xyz=True):
    model.to(device)
    model.eval()

    total = 0
    correct_continent = 0
    correct_cities = 0

    all_prediction_continents, all_target_continents = [], []
    all_prediction_cities, all_target_cities = [], []
    all_prediction_xyz, all_target_xyz = [], []

    # Use the initial dropout rate for evaluation
    evaluation_dropout_rate = model.initial_dropout_rate

    with torch.no_grad():
        for batch_idx, (data, continent_city, lat_long_rad) in enumerate(loader):
            batch_size = data.size(0)
            data = data.to(device)

            # Prepare targets based on what's needed
            if has_continent:
                cont_targ = continent_city[:, 0].long().to(device)
            if has_city:
                city_targ = continent_city[:, 1].long().to(device)
            if has_xyz:
                xyz_targ_scaled = lat_long_rad.float().to(device)

            # Model output: unpack based on expected outputs
            outputs = model(data,evaluation_dropout_rate)
            idx = 0

            if has_continent:
                continent_logits = outputs[idx]
                idx += 1
                _, predictions_continent = continent_logits.max(1)
                correct_continent += (predictions_continent == cont_targ).sum().item()
                all_prediction_continents.extend(predictions_continent.detach().cpu().numpy())
                all_target_continents.extend(cont_targ.detach().cpu().numpy())

            if has_city:
                city_logits = outputs[idx]
                idx += 1
                _, predictions_cities = city_logits.max(1)
                correct_cities += (predictions_cities == city_targ).sum().item()
                all_prediction_cities.extend(predictions_cities.detach().cpu().numpy())
                all_target_cities.extend(city_targ.detach().cpu().numpy())

            if has_xyz:
                xyz_pred_scaled = outputs[idx]
                all_prediction_xyz.append(xyz_pred_scaled.detach().cpu().numpy())
                all_target_xyz.append(xyz_targ_scaled.detach().cpu().numpy())

            total += batch_size

    # ===== Metrics =====
    if has_continent:
        accuracy_continent = correct_continent / total * 100
        precision_continent = precision_score(all_target_continents, all_prediction_continents, average='weighted', zero_division=0)
        recall_continent = recall_score(all_target_continents, all_prediction_continents, average='weighted', zero_division=0)
        f1_continent = f1_score(all_target_continents, all_prediction_continents, average='weighted', zero_division=0)
        print(f'Continent Accuracy: {accuracy_continent:.2f}%')
        print(f'Continent Precision: {precision_continent:.4f}')
        print(f'Continent Recall: {recall_continent:.4f}')
        print(f'Continent F1-Score: {f1_continent:.4f}')

    if has_city:
        accuracy_cities = correct_cities / total * 100
        precision_city = precision_score(all_target_cities, all_prediction_cities, average='weighted', zero_division=0)
        recall_city = recall_score(all_target_cities, all_prediction_cities, average='weighted', zero_division=0)
        f1_city = f1_score(all_target_cities, all_prediction_cities, average='weighted', zero_division=0)
        print(f'City Accuracy: {accuracy_cities:.2f}%')
        print(f'City Precision: {precision_city:.4f}')
        print(f'City Recall: {recall_city:.4f}')
        print(f'City F1-Score: {f1_city:.4f}')

    if has_xyz:
        all_prediction_xyz = np.concatenate(all_prediction_xyz, axis=0)
        all_target_xyz = np.concatenate(all_target_xyz, axis=0)

        predicted_lat_deg, predicted_long_deg = inverse_transform_spherical(all_prediction_xyz, coordinate_scaler)
        target_lat_deg, target_long_deg = inverse_transform_spherical(all_target_xyz, coordinate_scaler)

        predictions_df = pd.DataFrame({
            'predicted_lat': predicted_lat_deg,
            'predicted_lon': predicted_long_deg,
            'true_latitude': target_lat_deg,
            'true_longitude': target_long_deg
        })

        mae_lat_km, mae_lon_km = calculate_mae_km(
            predictions_df,
            predicted_lat_col='predicted_lat',
            predicted_lon_col='predicted_lon',
            true_lat_col='true_latitude',
            true_lon_col='true_longitude'
        )

        print(f'Mean Absolute Error (km) - Latitude: {mae_lat_km:.4f}')
        print(f'Mean Absolute Error (km) - Longitude: {mae_lon_km:.4f}')

    # ===== Return only what's needed =====
    results = {}

    if has_continent:
        results.update({
            "accuracy_continent": accuracy_continent,
            "precision_continent": precision_continent,
            "recall_continent": recall_continent,
            "f1_continent": f1_continent,
            "predicted_continent": all_prediction_continents,
            "target_continent": all_target_continents
        })

    if has_city:
        results.update({
            "accuracy_city": accuracy_cities,
            "precision_city": precision_city,
            "recall_city": recall_city,
            "f1_city": f1_city,
            "predicted_city": all_prediction_cities,
            "target_city": all_target_cities
        })

    if has_xyz:
        results.update({
            "mae_lat_km": mae_lat_km,
            "mae_lon_km": mae_lon_km,
            "predicted_coords": np.array([predicted_lat_deg, predicted_long_deg]).T,
            "target_coords": np.array([target_lat_deg, target_long_deg]).T
        })

    return results

