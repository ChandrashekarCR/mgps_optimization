import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from geopy.distance import geodesic 
from shapely.geometry import Point
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd



# Confusion matrix to visualize the labels and predictions that are correct
def plot_confusion_matrix(labels,predictions,label_map,filename):
    cm = confusion_matrix(labels,predictions)
    # Visualize the confusion matrix using seaborn
    plt.figure(figsize=((len(label_map) + 2), (len(label_map)+ 2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(label_map.values()), yticklabels=list(label_map.values()))
    plt.xlabel('Predicted Continent')
    plt.ylabel('True Continent')
    plt.xticks(rotation=60)
    plt.title('Confusion Matrix for Continent Predictions')
    plt.tight_layout()
    plt.savefig(filename) # Save the plot as an image
    plt.show()

# Plot both training and validation losses
def plot_losses(train_losses, val_losses, filename):
    epochs = range(1, len(next(iter(train_losses.values()))) + 1)  # Get number of epochs from the length of the first list

    plt.figure(figsize=(12, 8))
    
    for loss_name, losses in train_losses.items():
        plt.plot(epochs, losses, label=f'Train {loss_name.capitalize()} Loss', linestyle='-', marker='o')
    
    for loss_name, losses in val_losses.items():
        plt.plot(epochs, losses, label=f'Val {loss_name.capitalize()} Loss', linestyle='--', marker='x')
    
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)  # Save the plot as an image
    plt.show()



# Plot the points on the world map for visualization
def plot_points_on_world_map(true_lat, true_long, predicted_lat, predicted_long, filename):
    """Plots true and predicted latitude and longitude on a world map."""
    world = gpd.read_file("/home/chandru/binp37/data/geopandas/ne_110m_admin_0_countries.shp")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    world.plot(ax=ax, color='lightgray', edgecolor='black')

    # Plot true locations
    geometry_true = [Point(xy) for xy in zip(true_long, true_lat)]
    geo_df_true = gpd.GeoDataFrame(geometry_true, crs=world.crs, geometry=geometry_true)  # Specify geometry
    geo_df_true.plot(ax=ax, marker='o', color='blue', markersize=15, label='True Locations')

    # Plot predicted locations
    geometry_predicted = [Point(xy) for xy in zip(predicted_long, predicted_lat)]
    geo_df_predicted = gpd.GeoDataFrame(geometry_predicted, crs=world.crs, geometry=geometry_predicted)  # Specify geometry
    geo_df_predicted.plot(ax=ax, marker='x', color='red', markersize=15, label='Predicted Locations')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('True vs. Predicted Locations on World Map')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename) # Save the plot as an image
    plt.show()




def pull_land(df, coastline_path, countries_path, lat_col='predicted_lat', lon_col='predicted_lon', true_lat_col='true_latitude', true_lon_col='true_longitude'):
    """
    Adjusts points that fall in water by moving them to the nearest coastline
    and calculates the distance of these adjusted points from the true locations.

    Args:
        df (pd.DataFrame): DataFrame containing predicted and true latitude/longitude columns.
        coastline_path (str): Path to the coastline shapefile.
        countries_path (str): Path to the world countries shapefile.
        lat_col (str, optional): Name of the predicted latitude column. Defaults to 'predicted_lat'.
        lon_col (str, optional): Name of the predicted longitude column. Defaults to 'predicted_lon'.
        true_lat_col (str, optional): Name of the true latitude column. Defaults to 'true_latitude'.
        true_lon_col (str, optional): Name of the true longitude column. Defaults to 'true_longitude'.

    Returns:
        pd.DataFrame: DataFrame with 'updated_lat' and 'updated_lon' columns
                      (adjusted for water points, original for land),
                      and 'distance_from_true_km' column.
    """

    # Load coastline shapefile
    try:
        coastline = gpd.read_file(coastline_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Coastline shapefile not found at: {coastline_path}")

    # Load countries shapefile
    try:
        world = gpd.read_file(countries_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Countries shapefile not found at: {countries_path}")

    # Convert DataFrame to GeoDataFrame
    df['geometry'] = df.apply(lambda row: Point(row[lon_col], row[lat_col]) if np.isfinite(row[lon_col]) and np.isfinite(row[lat_col]) else None, axis=1)
    gdf = gpd.GeoDataFrame(df.dropna(subset=['geometry']).copy(), geometry='geometry', crs="EPSG:4326")

    # Initialize updated lat/lon with original predicted values
    gdf['updated_lat'] = gdf[lat_col]
    gdf['updated_lon'] = gdf[lon_col]

    # Identify points in water using the world countries shapefile
    in_water_mask = ~gdf.geometry.within(world.geometry.unary_union)
    water_points = gdf[in_water_mask].copy()
    original_water_geometry = water_points['geometry'].copy() # Keep track of original water geometries

    # Extract coastline points
    coast_points = []
    for geom in coastline.geometry:
        if geom.geom_type == 'LineString':
            coast_points.extend(list(geom.coords))
        elif geom.geom_type == 'MultiLineString':
            for line in geom:
                coast_points.extend(list(line.coords))

    # Filter out NaN or infinite values
    coast_points = [point for point in coast_points if np.isfinite(point[0]) and np.isfinite(point[1])]

    if not coast_points:
        raise ValueError("No valid coastline points found!")

    coast_tree = cKDTree(coast_points)  # KDTree for nearest neighbor search

    # Find nearest coastline point for water points
    new_coords = []
    for idx, row in water_points.iterrows():
        if not np.isfinite(row.geometry.x) or not np.isfinite(row.geometry.y):
            new_coords.append((np.nan, np.nan))
            continue  # Skip invalid points

        _, nearest_idx = coast_tree.query([row.geometry.x, row.geometry.y])
        nearest_point = coast_points[nearest_idx]
        new_coords.append((nearest_point[0], nearest_point[1]))

    # Update water points with new coordinates
    water_points['updated_lon'] = [coord[0] if isinstance(coord, tuple) else np.nan for coord in new_coords]
    water_points['updated_lat'] = [coord[1] if isinstance(coord, tuple) else np.nan for coord in new_coords]

    # Update the 'updated_lat' and 'updated_lon' in the main gdf for water points
    gdf.loc[water_points.index, 'updated_lon'] = water_points['updated_lon'].astype(np.float32)
    gdf.loc[water_points.index, 'updated_lat'] = water_points['updated_lat'].astype(np.float32)


    return gdf.drop(columns=['geometry'])


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

def plot_predictions_with_coastline(predictions_df, coastline_path, lat_col='predicted_lat', lon_col='predicted_lon', updated_lat_col='updated_lat', updated_lon_col='updated_lon', true_lat_col='true_latitude', true_lon_col='true_longitude', filename=None):
    """
    Plots the original predicted points and the adjusted (pulled to coastline) points
    on a world map with coastlines. Also plots the true locations if available.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing predicted and updated
                                       latitude and longitude columns.
        coastline_path (str): Path to the coastline shapefile.
        lat_col (str, optional): Name of the original predicted latitude column.
                                   Defaults to 'predicted_lat'.
        lon_col (str, optional): Name of the original predicted longitude column.
                                   Defaults to 'predicted_lon'.
        updated_lat_col (str, optional): Name of the updated latitude column.
                                         Defaults to 'updated_lat'.
        updated_lon_col (str, optional): Name of the updated longitude column.
                                         Defaults to 'updated_lon'.
        true_lat_col (str, optional): Name of the true latitude column.
                                      Defaults to 'true_latitude'.
        true_lon_col (str, optional): Name of the true longitude column.
                                       Defaults to 'true_longitude'.
        filename (str, optional): Path to save the plot. If None, the plot is shown.
                                  Defaults to None.
    """
    try:
        coastline = gpd.read_file(coastline_path)
    except FileNotFoundError:
        print(f"Error: Coastline shapefile not found at {coastline_path}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    coastline.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)

    # Plot original predicted points
    geometry_predicted = [Point(xy) for xy in zip(predictions_df[lon_col], predictions_df[lat_col])]
    predictions_gdf = gpd.GeoDataFrame(predictions_df, geometry=geometry_predicted, crs="EPSG:4326")
    predictions_gdf.plot(ax=ax, marker='o', color='red', markersize=10, label='Original Predictions')

    # Plot adjusted predicted points
    geometry_updated = [Point(xy) for xy in zip(predictions_df[updated_lon_col], predictions_df[updated_lat_col])]
    updated_predictions_gdf = gpd.GeoDataFrame(predictions_df, geometry=geometry_updated, crs="EPSG:4326")
    updated_predictions_gdf.plot(ax=ax, marker='x', color='blue', markersize=10, label='Adjusted Predictions (to Coastline)')

    # Plot true points if available
    if true_lat_col in predictions_df.columns and true_lon_col in predictions_df.columns:
        geometry_true = [Point(xy) for xy in zip(predictions_df[true_lon_col], predictions_df[true_lat_col])]
        true_gdf = gpd.GeoDataFrame(predictions_df, geometry=geometry_true, crs="EPSG:4326")
        true_gdf.plot(ax=ax, marker='.', color='green', markersize=15, label='True Locations')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Original vs. Adjusted Predicted Locations on Coastline")
    ax.legend(loc='upper right')
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    plt.grid(True, linestyle='--', alpha=0.5)

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

