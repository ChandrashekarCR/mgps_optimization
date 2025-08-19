"""
Main Ensemble Model Script

This script orchestrates the ensemble learning pipeline for hierarchical geographical prediction.
It imports and utilizes various model wrappers (XGBoost, CatBoost, LightGBM, TabPFN, GrowNet, Neural Networks)
for both classification (continent/city) and regression (coordinates) tasks. The ensemble combines predictions
from multiple models using stacking/meta-learning, and provides error analysis and visualization.

Usage:
- Imports model wrappers from submodules (e.g., catboost_classification.py).
- Runs multi-stage ensemble for continent, city, and coordinate prediction.
- Performs error analysis and saves results.

This script is the entry point for the ensemble workflow.
"""

# The main ensemble model

# Import libraries 
import os

# Import data processing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import other models
# 1. XGBoost
from xgboost_ensemble.xgboost_classification import run_xgboost_classifier
from xgboost_ensemble.xgboost_regression import run_xgboost_regressor
# 2. TabPFN
from tab_pfn.tab_pfn_classificaiton import run_tabpfn_classifier
from tab_pfn.tab_pfn_regression import run_tabpfn_regressor
# 3. LightGBM
from lightgbm_ensemble.lightgbm_classification import run_lightgbm_classifier
from lightgbm_ensemble.lightgbm_regression import run_lightgbm_regressor
# 4. CatBoost
from catboost_ensemble.catboost_classification import run_catboost_classifier
from catboost_ensemble.catboost_regression import run_catboost_regressor
# 5. GrowNet
from grownet.grownet_classification import run_grownet_classifier 
from grownet.grownet_regressor import run_grownet_regressor
# 6. Neural Networks
from simple_nn.nn_classification import run_nn_classifier
from simple_nn.nn_regression import run_nn_regressor

# Geo pandas for plotting
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

from geopy.distance import geodesic 
from shapely.geometry import Point

# Import deep learning libraries
import torch

# Logging
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and process the dataset

# Data processing function for hierarchical model
def process_data_hierarchical(df):
    """Process data for hierarchical prediction"""
    # Process continuous features
    cont_cols = [col for col in df.columns if col not in [
        'latitude', 'longitude',
        'latitude_rad', 'longitude_rad', 'x', 'y', 'z',
        'scaled_x', 'scaled_y', 'scaled_z', 'continent', 'city'
    ]]
    
    # Get the features
    x_cont = df[cont_cols].values
    
    # Encode continent labels
    continent_encoder = LabelEncoder()
    y_continent = continent_encoder.fit_transform(df['continent'].values)
    
    # Encode city labels
    city_encoder = LabelEncoder()
    y_city = city_encoder.fit_transform(df['city'].values)
    
    # Calculate coordinates if not already present
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        df['latitude_rad'] = np.deg2rad(df['latitude'])
        df['longitude_rad'] = np.deg2rad(df['longitude'])
        df['x'] = np.cos(df['latitude_rad']) * np.cos(df['longitude_rad'])
        df['y'] = np.cos(df['latitude_rad']) * np.sin(df['longitude_rad'])
        df['z'] = np.sin(df['latitude_rad'])
    
    # Scale coordinates
    coord_scaler = StandardScaler()
    y_coords = coord_scaler.fit_transform(df[['x', 'y', 'z']].values)
    
    continents = continent_encoder.classes_
    cities = city_encoder.classes_
    
    print(f"Continents: {len(continents)} ({continents})")
    print(f"Cities: {len(cities)}")
    print(f"Continuous features: {len(cont_cols)}")
    
    return {
        'x_cont': x_cont,
        'y_continent': y_continent,
        'y_city': y_city,
        'y_coords': y_coords, # This is for neural networks. Scaling is required
        'y_latitude': df['latitude'].values, # This is for XGBoost, we don't need to scale this
        'y_longitude':df['longitude'].values, # This is for XGBoost, we don't need to scale this
        'encoders': {
            'continent': continent_encoder,
            'city': city_encoder,
            'coord': coord_scaler
        },
        'continents': continents,
        'cities': cities
    }

# Hierarchial split to keep track of the indices
def hierarchical_split(X_cont, y_continent, y_city, y_coords, y_lat, y_lon, test_size=0.2, random_state=42):
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X_cont, y_continent))

    return {
        'X_train': X_cont[train_idx],
        'X_test': X_cont[test_idx],
        'y_cont_train': y_continent[train_idx],
        'y_cont_test': y_continent[test_idx],
        'y_city_train': y_city[train_idx],
        'y_city_test': y_city[test_idx],
        'y_coords_train': y_coords[train_idx],
        'y_coords_test': y_coords[test_idx],
        'y_lat_train': y_lat[train_idx],
        'y_lat_test': y_lat[test_idx],
        'y_lon_train': y_lon[train_idx],
        'y_lon_test': y_lon[test_idx],
        'train_idx': train_idx,
        'test_idx': test_idx
    }

# Distance between two points on the earth
def haversine_distance(lat1,lon1,lat2,lon2):
    """
    Calculate the great circle distance between two points on the earth
    """
    # Radius of the earth
    R = 6371.0

    # Convert from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2) **2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c # in kilometers

# Converting cartesian co-ordinates values to latitude and longitude
def xyz_to_latlon(xyz_coords):
    """
    Convert the XYZ coordinates to latitude and longitude
    """
    x,y,z = xyz_coords[:,0],xyz_coords[:,1],xyz_coords[:,2]

    # Convert to latitude and longitude
    lat_rad = np.arcsin(np.clip(z,-1,1)) # Clip to avoid numerical issues
    lon_rad = np.arctan2(y,x)

    # Convert to degrees
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)

    return np.stack([lat_deg,lon_deg],axis=1)

# Plot the points on the world map for visualization
def plot_points_on_world_map(true_lat, true_long, predicted_lat, predicted_long, filename):
    """
    Plots true and predicted latitude and longitude on a world map.
    Args:
        true_lat: True latitude value
        true_long: True longitude value
        predicted_lat: Prediction by the neural netwrok
        predicted_long: Prediction by the neural network
        filename: Path and the name of the file to save the plot.
    Returns:
        A figure is saved in the correct directory.
    """
    # A file that is required to load the world map with proper countries
    world = gpd.read_file("/home/chandru/binp37/data/geopandas/ne_110m_admin_0_countries.shp") 
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    world.plot(ax=ax, color='lightgray', edgecolor='black')
    # Plot true locations
    geometry_true = [Point(xy) for xy in zip(true_long, true_lat)]
    geo_df_true = gpd.GeoDataFrame(geometry_true, crs=world.crs, geometry=geometry_true) 
    geo_df_true.plot(ax=ax, marker='o', color='blue', markersize=15, label='True Locations')
    # Plot predicted locations
    geometry_predicted = [Point(xy) for xy in zip(predicted_long, predicted_lat)]
    geo_df_predicted = gpd.GeoDataFrame(geometry_predicted, crs=world.crs, geometry=geometry_predicted) 
    geo_df_predicted.plot(ax=ax, marker='x', color='red', markersize=15, label='Predicted Locations')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('True vs. Predicted Locations on World Map')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename) # Save the plot as an image
    plt.show()

# Train the ensemble models on classification tasks -> Continent and city classification
def train_hierarchical_layer(
        X_train,
        X_test,
        y_train,
        y_test,
        run_xgboost_classifier=None,
        run_grownet_classifier=None,
        run_nn_classifier=None,
        run_tabpfn_classifier=None,
        run_lightgbm_classifier=None,
        run_catboost_classifier=None,
        tune_hyperparams=False,
        apply_smote=False,
        random_state=42,
        n_splits=3,
        accuracy_threshold=0.8):
    """
    Efficient single-stage hierarchical layer:
    1. Run all models with default params in CV to identify models that meet accuracy threshold
    2. Tune hyperparameters only for filtered models (except TabPFN, which uses max time setting)
    3. Generate OOF predictions using tuned models
    4. Train final ensemble on OOF predictions from tuned models
    """
    
    # Define all possible models with their configurations
    model_configs = {
        'xgb': {
            'name': 'XGBoost',
            'function': run_xgboost_classifier,
            'enabled': run_xgboost_classifier is not None,
            'tune_params': {'n_trials': 50, 'timeout': 1800}
        },
        'grownet': {
            'name': 'GrowNet',
            'function': run_grownet_classifier,
            'enabled': run_grownet_classifier is not None,
            'tune_params': {'n_trials': 50, 'timeout': 1800}
        },
        'nn': {
            'name': 'Neural Network',
            'function': run_nn_classifier,
            'enabled': run_nn_classifier is not None,
            'tune_params': {'n_trials': 50, 'timeout': 1800}
        },
        'tabpfn': {
            'name': 'TabPFN',
            'function': run_tabpfn_classifier,
            'enabled': run_tabpfn_classifier is not None,
            'max_time_options': [30, 60, 120, 180, 300]  # TabPFN uses max_time instead of normal tuning
        },
        'lightgbm': {
            'name': 'LightGBM',
            'function': run_lightgbm_classifier,
            'enabled': run_lightgbm_classifier is not None,
            'tune_params': {'n_trials': 50, 'timeout': 1800}
        },
        'catboost': {
            'name': 'CatBoost',
            'function': run_catboost_classifier,
            'enabled': run_catboost_classifier is not None,
            'tune_params': {'n_trials': 50, 'timeout': 1800}
        }
    }
    
    # Filter to only enabled models
    enabled_models = {k: v for k, v in model_configs.items() if v['enabled']}
    
    if not enabled_models:
        raise ValueError("At least one model function must be provided (not None)")
    
    logging.info(f"Enabled models: {list(enabled_models.keys())}")
    
    # STAGE 1: Initial CV loop to identify models that meet threshold
    logging.info("STAGE 1: Running initial CV to identify models that meet the threshold...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_train_samples = X_train.shape[0]
    n_classes = len(np.unique(y_train))

    # Track accuracies for model filtering
    model_fold_accuracies = {model_key: [] for model_key in enabled_models.keys()}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        logging.info(f"Processing Fold {fold+1}/{n_splits} for initial evaluation")
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        if apply_smote:
            X_fold_train, y_fold_train = SMOTE(random_state=random_state).fit_resample(X_fold_train, y_fold_train)
        
        for model_key, config in enabled_models.items():
            logging.info(f"  Initial evaluation of {config['name']} on fold {fold+1}...")
            try:
                fold_result = config['function'](
                    X_fold_train, y_fold_train, X_fold_val, y_fold_val,
                    tune_hyperparams=False, params=None, verbose=False
                )
                
                if fold_result.get('skipped', False):
                    logging.info(f"  {config['name']} was skipped on fold {fold+1}")
                    model_fold_accuracies[model_key].append(0.0)
                else:
                    accuracy = fold_result['accuracy']
                    logging.info(f"  {config['name']} fold {fold+1} accuracy: {accuracy:.4f}")
                    model_fold_accuracies[model_key].append(accuracy)
                    
            except Exception as e:
                logging.error(f"Error running {config['name']} on fold {fold+1}: {e}")
                model_fold_accuracies[model_key].append(0.0)

    # Calculate average accuracies and filter models
    model_avg_accuracies = {k: np.mean(v) for k, v in model_fold_accuracies.items()}
    passed_models = [k for k, acc in model_avg_accuracies.items() if acc >= accuracy_threshold]
    
    logging.info(f"Model average accuracies: {model_avg_accuracies}")
    logging.info(f"Models passing threshold ({accuracy_threshold*100:.1f}%): {passed_models}")
    
    if not passed_models:
        raise ValueError(f"No models met the accuracy threshold of {accuracy_threshold*100:.1f}%.")

    # STAGE 2: Hyperparameter tuning or configuration for passed models
    best_params = {}
    if tune_hyperparams:
        logging.info("STAGE 2: Tuning hyperparameters for filtered models...")
        
        X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
            X_train, y_train, test_size=0.2, random_state=101, stratify=y_train
        )
        
        for model_key in passed_models:
            config = enabled_models[model_key]
            
            # Special handling for TabPFN - no tuning, just use the highest max_time
            if model_key == 'tabpfn':
                max_time = max(config['max_time_options'])
                best_params[model_key] = {'max_time': max_time}
                logging.info(f"TabPFN will use max_time = {max_time} (no tuning needed)")
                continue
                
            # Standard tuning for other models
            logging.info(f"Tuning {config['name']} hyperparameters...")
            try:
                tune_result = config['function'](
                    X_train_tune, y_train_tune, X_val_tune, y_val_tune,
                    tune_hyperparams=True, verbose=True, **config['tune_params']
                )
                best_params[model_key] = tune_result['params']
                logging.info(f"Best {config['name']} params: {best_params[model_key]}")
                
            except Exception as e:
                logging.error(f"Error tuning {config['name']}: {e}")
                best_params[model_key] = None
    else:
        # If not tuning, TabPFN still gets highest max_time
        best_params = {model_key: None for model_key in passed_models}
        if 'tabpfn' in passed_models:
            max_time = max(enabled_models['tabpfn']['max_time_options'])
            best_params['tabpfn'] = {'max_time': max_time}
            logging.info(f"TabPFN will use max_time = {max_time}")

    # STAGE 3: Generate OOF predictions using tuned models
    logging.info("STAGE 3: Generating OOF predictions using tuned models...")
    
    oof_probs = {}
    for model_key in passed_models:
        oof_probs[model_key] = np.zeros((n_train_samples, n_classes))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        logging.info(f"Processing Fold {fold+1}/{n_splits} for OOF generation")
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        if apply_smote:
            X_fold_train, y_fold_train = SMOTE(random_state=random_state).fit_resample(X_fold_train, y_fold_train)
        
        for model_key in passed_models:
            config = enabled_models[model_key]
            logging.info(f"  Running tuned {config['name']} on fold {fold+1} for OOF...")
            
            try:
                fold_result = config['function'](
                    X_fold_train, y_fold_train, X_fold_val, y_fold_val,
                    tune_hyperparams=False, params=best_params[model_key], verbose=False
                )
                
                if fold_result.get('skipped', False):
                    logging.info(f"  Tuned {config['name']} was skipped on fold {fold+1}")
                    # Fill with uniform probabilities as fallback
                    oof_probs[model_key][val_idx] = np.full((len(val_idx), n_classes), 1.0/n_classes)
                else:
                    logging.info(f"  Storing {config['name']} OOF predictions for fold {fold+1}")
                    oof_probs[model_key][val_idx] = fold_result['predicted_probabilities']
                    
            except Exception as e:
                logging.error(f"Error generating OOF for {config['name']} on fold {fold+1}: {e}")
                oof_probs[model_key][val_idx] = np.full((len(val_idx), n_classes), 1.0/n_classes)

    # Create meta training features from OOF predictions of tuned models
    meta_feature_list = [oof_probs[model_key] for model_key in passed_models]
    meta_X_train = np.hstack(meta_feature_list)
    logging.info(f"Meta training features shape: {meta_X_train.shape}")

    # STAGE 4: Train final models on full training data with tuned parameters
    logging.info("STAGE 4: Training final models on full training data...")
    test_results = {}
    
    for model_key in passed_models:
        config = enabled_models[model_key]
        logging.info(f"Training final {config['name']} model on full training data...")
        
        try:
            result = config['function'](
                X_train, y_train, X_test, y_test,
                tune_hyperparams=False,
                params=best_params[model_key],
                verbose=True
            )
            test_results[model_key] = result['predicted_probabilities']
            logging.info(f"Successfully trained final {config['name']} model")
            
        except Exception as e:
            logging.error(f"Error training final {config['name']} model: {e}")
            test_results[model_key] = np.full((X_test.shape[0], n_classes), 1.0/n_classes)

    # Create meta test features from tuned models
    meta_test_feature_list = [test_results[model_key] for model_key in passed_models]
    meta_X_test = np.hstack(meta_test_feature_list)
    logging.info(f"Meta test features shape: {meta_X_test.shape}")

    # Train meta model
    logging.info("Training meta model...")
    meta_model = xgb.XGBClassifier(objective='multi:softprob', random_state=random_state)
    meta_model.fit(meta_X_train, y_train)

    # Make predictions
    train_preds = meta_model.predict(meta_X_train)
    test_preds = meta_model.predict(meta_X_test)

    # Print summary
    logging.info(f"\nSummary:")
    logging.info(f"- Used models: {passed_models}")
    logging.info(f"- Meta features: {meta_X_train.shape[1]}")
    logging.info(f"- Meta model train accuracy: {accuracy_score(y_train, train_preds):.4f}")
    logging.info(f"- Meta model test accuracy: {accuracy_score(y_test, test_preds):.4f}")

    return meta_model, meta_X_train, meta_X_test, train_preds, test_preds


# Train the ensemble models on regression tasks -> Co-ordinates predictions
def train_hierarchical_coordinate_layer(
        X_train, X_test, y_train_lat, y_train_lon,
        y_test_lat, y_test_lon, y_train_coords,
        y_test_coords, coord_scaler,
        run_xgboost_regressor = None,
        run_grownet_regressor = None,
        run_nn_regressor = None,
        run_tabpfn_regressor = None,
        run_lightgbm_regressor = None,
        run_catboost_regressor = None,
        tune_hyperparams = False,
        random_state = 42,
        n_splits = 3
    ):
    """
    Two-stage hierarchical coordinate prediction:
    1. Run all models with default params, select best by average median distance
    2. Tune hyperparameters only for the best model
    """

    model_configs = {
        'xgb':{
            'name':'XGBoost',
            'function':run_xgboost_regressor,
            'enabled': run_xgboost_regressor is not None,
            'prediction_type':'sequential',
            'tune_params': {'n_trials': 50, 'timeout': 1800}
        },
        'grownet':{
            'name':'GrowNet',
            'function':run_grownet_regressor,
            'enabled':run_grownet_regressor is not None,
            'prediction_type':'xyz',
            'tune_params': {'n_trials': 50, 'timeout': 1800}
        },
        'nn':{
            'name':'Neural Network',
            'function': run_nn_regressor,
            'enabled':run_nn_regressor is not None,
            'prediction_type':'xyz',
            'tune_params': {'n_trials': 50, 'timeout': 1800}
        },
        'tabpfn':{
            'name':'TabPFN',
            'function':run_tabpfn_regressor,
            'enabled':run_tabpfn_regressor is not None,
            'prediction_type':'xyz',
            'tune_params': {'n_trials': 20, 'max_time_options': [30, 60, 120]}
        },
        'lightgbm':{
            'name':'LightGBM',
            'function':run_lightgbm_regressor,
            'enabled':run_lightgbm_regressor is not None,
            'prediction_type':'sequential',
            'tune_params': {'n_trials': 50, 'timeout': 1800}
        },
        'catboost':{
            'name':'CatBoost',
            'function':run_catboost_regressor,
            'enabled':run_catboost_regressor is not None,
            'prediction_type':'sequential',
            'tune_params': {'n_trials': 50, 'timeout': 1800}
        }
    }

    enabled_models = {k: v for k,v in model_configs.items() if v['enabled']}

    if not enabled_models:
        raise ValueError("At least one model function must be provided (not None)")
    
    logging.info(f"Enabled models: {list(enabled_models.keys())}")

    # STAGE 1: Run all models with default parameters to calculate average median distance
    logging.info("STAGE 1: Running all models with default parameters...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_train_samples = X_train.shape[0]
    y_train_combined = np.stack([y_train_lat, y_train_lon], axis=1)

    # Track average median distances across folds for each model
    model_avg_median_distances = {}
    
    for model_key, config in enabled_models.items():
        logging.info(f"Running {config['name']} with default parameters...")
        fold_median_distances = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train_lat, y_fold_val_lat = y_train_lat[train_idx], y_train_lat[val_idx]
            y_fold_train_lon, y_fold_val_lon = y_train_lon[train_idx], y_train_lon[val_idx]
            y_fold_train_coords, y_fold_val_coords = y_train_coords[train_idx], y_train_coords[val_idx]
            y_fold_val_combined = np.stack([y_fold_val_lat, y_fold_val_lon], axis=1)
            
            try:
                if config['prediction_type'] == "sequential":
                    # Predict latitude first
                    lat_result = config['function'](
                        X_fold_train, y_fold_train_lat, X_fold_val, y_fold_val_lat,
                        tune_hyperparams=False, params=None, verbose=False
                    )
                    
                    if lat_result.get('skipped', False):
                        fold_median_distances.append(float('inf'))
                        continue
                        
                    lat_pred_train = lat_result['model'].predict(X_fold_train)
                    lat_pred_val = lat_result['predictions']

                    # Augment features with latitude predictions
                    X_fold_train_aug = np.hstack([X_fold_train, lat_pred_train.reshape(-1, 1)])
                    X_fold_val_aug = np.hstack([X_fold_val, lat_pred_val.reshape(-1, 1)])

                    # Predict longitude
                    lon_result = config['function'](
                        X_fold_train_aug, y_fold_train_lon, X_fold_val_aug, y_fold_val_lon,
                        tune_hyperparams=False, params=None, verbose=False
                    )
                    
                    if lon_result.get('skipped', False):
                        fold_median_distances.append(float('inf'))
                        continue
                        
                    lon_pred_val = lon_result['predictions']
                    val_predictions = np.stack([lat_pred_val, lon_pred_val], axis=1)

                elif config['prediction_type'] == 'xyz':
                    fold_result = config['function'](
                        X_fold_train, y_fold_train_coords, X_fold_val, y_fold_val_coords,
                        tune_hyperparams=False, params=None, verbose=False
                    )
                    
                    if fold_result.get('skipped', False):
                        fold_median_distances.append(float('inf'))
                        continue
                    
                    xyz_pred = fold_result['predictions']
                    # Ensure predictions are 2D
                    if xyz_pred.ndim == 1:
                        xyz_pred = xyz_pred.reshape(-1, 3)
                    xyz_rescaled = coord_scaler.inverse_transform(xyz_pred)
                    val_predictions = xyz_to_latlon(xyz_rescaled)
                
                # Calculate median distance for this fold
                distances = haversine_distance(
                    y_fold_val_combined[:, 0], y_fold_val_combined[:, 1],
                    val_predictions[:, 0], val_predictions[:, 1]
                )
                fold_median_distances.append(np.median(distances))
                logging.info(f"  {config['name']} fold {fold+1} median distance: {np.median(distances):.2f} km")
                
            except Exception as e:
                logging.error(f"Error running {config['name']} on fold {fold+1}: {e}")
                fold_median_distances.append(float('inf'))
        
        model_avg_median_distances[model_key] = np.mean(fold_median_distances)
        logging.info(f"{config['name']} average median distance: {model_avg_median_distances[model_key]:.2f} km")
    
    # Select best model by lowest average median distance
    best_model = min(model_avg_median_distances, key=model_avg_median_distances.get)
    logging.info(f"Best model by average median distance: {best_model}")
    logging.info(f"Model average median distances: {model_avg_median_distances}")
    
    # STAGE 2: Hyperparameter tuning only for the best model
    best_params = None
    if tune_hyperparams:
        logging.info("STAGE 2: Tuning hyperparameters for the best model...")
        X_train_hyper, X_test_hyper, y_train_hyper_lat, y_test_hyper_lat = train_test_split(
            X_train, y_train_lat, test_size=0.2, random_state=101
        )
        _, _, y_train_hyper_lon, y_test_hyper_lon = train_test_split(
            X_train, y_train_lon, test_size=0.2, random_state=101
        )
        _, _, y_train_hyper_coords, y_test_hyper_coords = train_test_split(
            X_train, y_train_coords, test_size=0.2, random_state=101
        )
        config = enabled_models[best_model]
        logging.info(f"Tuning {config['name']} hyperparameters...")
        try:
            if best_model == 'tabpfn':
                # For TabPFN, just set max_time to highest value, no tuning
                max_time = max(config.get('tune_params', {}).get('max_time_options', [30, 60, 120, 180]))
                best_params = {'max_time': max_time}
                logging.info(f"TabPFN will use max_time = {max_time} (no tuning needed)")
            elif config['prediction_type'] == 'sequential':
                tune_params = config['tune_params'].copy()
                tune_params.update({
                    'tune_hyperparams': True,
                    'verbose': True
                })
                result = config['function'](
                    X_train_hyper, y_train_hyper_lat, X_test_hyper, y_test_hyper_lat,
                    **tune_params
                )
                best_params = result.get('params')
            elif config['prediction_type'] == 'xyz':
                tune_params = config['tune_params'].copy()
                tune_params.update({
                    'tune_hyperparams': True,
                    'verbose': True
                })
                result = config['function'](
                    X_train_hyper, y_train_hyper_coords, X_test_hyper, y_test_hyper_coords,
                    **tune_params
                )
                best_params = result.get('params')
            logging.info(f"Best {config['name']} params: {best_params}")
        except Exception as e:
            logging.error(f"Error tuning {config['name']}: {e}")
            best_params = None
    else:
        # If not tuning, TabPFN still gets highest max_time
        if best_model == 'tabpfn':
            max_time = max(enabled_models['tabpfn'].get('tune_params', {}).get('max_time_options', [30, 60, 120, 180]))
            best_params = {'max_time': max_time}
            logging.info(f"TabPFN will use max_time = {max_time}")
        else:
            best_params = None

    # STAGE 3: Final training with tuned parameters
    logging.info("STAGE 3: Final training with tuned parameters...")
    
    config = enabled_models[best_model]
    
    try:
        if config['prediction_type'] == 'sequential':
            # Sequential prediction with tuned params
            lat_result = config['function'](
                X_train, y_train_lat, X_test, y_test_lat,
                tune_hyperparams=False, params=best_params, verbose=True
            )
            
            if lat_result.get('skipped', False):
                raise ValueError(f"Latitude prediction failed for {config['name']}")
                
            lat_pred_train = lat_result['model'].predict(X_train)
            lat_pred_test = lat_result['predictions']
            
            # Augment features with latitude predictions
            X_train_aug = np.hstack([X_train, lat_pred_train.reshape(-1, 1)])
            X_test_aug = np.hstack([X_test, lat_pred_test.reshape(-1, 1)])
            
            lon_result = config['function'](
                X_train_aug, y_train_lon, X_test_aug, y_test_lon,
                tune_hyperparams=False, params=best_params, verbose=True
            )
            
            if lon_result.get('skipped', False):
                raise ValueError(f"Longitude prediction failed for {config['name']}")
                
            lon_pred_test = lon_result['predictions']
            test_preds = np.stack([lat_pred_test, lon_pred_test], axis=1)

        elif config['prediction_type'] == 'xyz':
            result = config['function'](
                X_train, y_train_coords, X_test, y_test_coords,
                tune_hyperparams=False, params=best_params, verbose=True
            )
            
            if result.get('skipped', False):
                raise ValueError(f"XYZ prediction failed for {config['name']}")
            
            xyz_pred = result['predictions']
            # Ensure predictions are 2D
            if xyz_pred.ndim == 1:
                xyz_pred = xyz_pred.reshape(-1, 3)
            xyz_rescaled = coord_scaler.inverse_transform(xyz_pred)
            test_preds = xyz_to_latlon(xyz_rescaled)
            
    except Exception as e:
        logging.error(f"Error training final {config['name']} model: {e}")
        raise

    # Calculate distance metrics
    def calculate_distance_metrics(y_true, y_pred):
        distances = haversine_distance(y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1])
        return {
            'median_distance': np.median(distances),
            'mean_distance': np.mean(distances),
            'percentile_95': np.percentile(distances, 95),
            'percentile_99': np.percentile(distances, 99),
            'distances': distances
        }

    y_test_combined = np.stack([y_test_lat, y_test_lon], axis=1)
    test_metrics = calculate_distance_metrics(y_test_combined, test_preds)

    logging.info(f"\nSummary:")
    logging.info(f"- Used model: {best_model}")
    logging.info(f"- Test median distance: {test_metrics['median_distance']:.2f} km")
    logging.info(f"- Test mean distance: {test_metrics['mean_distance']:.2f} km")
    logging.info(f"- Test 95th percentile: {test_metrics['percentile_95']:.2f} km")

    return {
        'test_preds': test_preds,
        'test_metrics': test_metrics,
        'enabled_models': [best_model],
        'best_model': best_model,
        'best_params': best_params
    }

# Process data
df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
processed_data = process_data_hierarchical(df)

X_cont = processed_data['x_cont']
y_cont = processed_data['y_continent']
y_cities = processed_data['y_city']
y_coords = processed_data['y_coords']
y_latitude = processed_data['y_latitude']
y_longitude = processed_data['y_longitude']


split_data = hierarchical_split(
    X_cont,
    y_cont,
    y_cities,
    y_coords,
    processed_data['y_latitude'],
    processed_data['y_longitude']
)

# Original feautres
X_train_cont, X_test_cont = split_data['X_train'], split_data['X_test']
# Train and test for continent
y_train_cont, y_test_cont = split_data['y_cont_train'], split_data['y_cont_test']
# Train and test for cities
y_train_city, y_test_city = split_data['y_city_train'], split_data['y_city_test']
# Train and test for latitude
y_train_lat, y_test_lat = split_data['y_lat_train'], split_data['y_lat_test']
# Train and test for longitude
y_train_lon, y_test_lon = split_data['y_lon_train'], split_data['y_lon_test']
# Train and test for co-ordinates
y_train_coords, y_test_coords = split_data['y_coords_train'],  split_data['y_coords_test']

# Continent layer
continent_model, meta_X_train_cont, meta_X_test_cont, cont_train_preds, cont_test_preds = train_hierarchical_layer(
    X_train=X_train_cont,
    X_test=X_test_cont,
    y_train=y_train_cont,
    y_test=y_test_cont,
    run_xgboost_classifier=run_xgboost_classifier,
    run_grownet_classifier=None,
    run_nn_classifier=None,
    run_tabpfn_classifier=run_tabpfn_classifier,
    run_lightgbm_classifier=run_lightgbm_classifier,
    run_catboost_classifier=None,
    tune_hyperparams=False,
    apply_smote=True,
    n_splits=5,
    accuracy_threshold=0.93  # 91% for continent
)

# City layer 
X_train_city = np.hstack([X_train_cont,meta_X_train_cont])
X_test_city = np.hstack([X_test_cont,meta_X_test_cont])

city_model, meta_X_train_city, meta_X_test_city, city_train_preds, city_test_preds = train_hierarchical_layer(
    X_train=X_train_city,
    X_test=X_test_city,
    y_train=y_train_city,
    y_test=y_test_city,
    run_xgboost_classifier=run_xgboost_classifier,
    run_grownet_classifier=None,
    run_lightgbm_classifier=run_lightgbm_classifier,
    run_catboost_classifier=None,
    run_nn_classifier=None,
    run_tabpfn_classifier=run_tabpfn_classifier,  # Now handles GPU/CPU automatically
    tune_hyperparams=False,
    apply_smote=False,
    n_splits=5,
    accuracy_threshold=0.91  # 89% for city
)

# Coordinate layer

X_train_coord = np.hstack([X_train_city,meta_X_train_city])
X_test_coord = np.hstack([X_test_city,meta_X_test_city])


coords_results = train_hierarchical_coordinate_layer(
    X_train=X_train_coord,
    X_test=X_test_coord,
    y_train_lat=y_train_lat,
    y_train_lon = y_train_lon,
    y_test_lat=y_test_lat,
    y_test_lon=y_test_lon,
    y_train_coords=y_train_coords,
    y_test_coords=y_test_coords,
    coord_scaler=processed_data['encoders']['coord'],
    run_xgboost_regressor=run_xgboost_regressor,
    run_tabpfn_regressor=run_tabpfn_regressor,
    run_nn_regressor=None,
    run_grownet_regressor=None,
    run_lightgbm_regressor=run_lightgbm_regressor,
    run_catboost_regressor=None,
    tune_hyperparams=False,
    n_splits=5
)


# All metrics
save_dir = "saved_results/"
os.makedirs(save_dir,exist_ok=True)
# Continent Layer

logging.info("\nContinent Prediction - Test Set:")
logging.info(classification_report(y_test_cont, cont_test_preds,target_names=processed_data['continents']))
# Save the test predictions
np.save(os.path.join(save_dir, "x_test.npy"), X_test_cont)
np.save(os.path.join(save_dir, "y_test_cont.npy"),y_test_cont)
np.save(os.path.join(save_dir, "y_pred_cont.npy"),cont_test_preds)

# City Layer

logging.info("\nCity Prediction - Test Set:")
logging.info(classification_report(y_test_city,city_test_preds,target_names=processed_data['cities']))
# Save the test predictions
np.save(os.path.join(save_dir,"y_test_city.npy"),y_test_city)
np.save(os.path.join(save_dir,"y_pred_city.npy"),city_test_preds)

# Co-ordinate Layer
logging.info("Coordinate prediction results:")
logging.info(f"Test Median Distance: {coords_results['test_metrics']['median_distance']:.2f} km")
logging.info(f"Test Mean Distance: {coords_results['test_metrics']['mean_distance']:.2f} km")
logging.info(f"Test 95th Percentile: {coords_results['test_metrics']['percentile_95']:.2f} km")

# Save the test predictions
np.save(os.path.join(save_dir,"y_test_coord.npy"),np.stack([y_test_lat,y_test_lon],axis=1).astype(np.float32))
np.save(os.path.join(save_dir,"y_pred_coord.npy"),coords_results['test_preds'])


# Error calculations
def error_calc(test_conts,pred_conts,test_city,pred_city,test_lat,pred_lat,test_lon,pred_lon):
    error_df = pd.DataFrame({
        'true_cont': test_conts,
        'pred_cont': pred_conts,
        'true_city': test_city,
        'pred_city': pred_city,
        'true_lat': test_lat,
        'true_lon': test_lon,
        'pred_lat': pred_lat,
        'pred_lon': pred_lon
    })


    # Assign true contient and city names
    error_df['true_cont_name'] = error_df['true_cont'].map(lambda i: processed_data['continents'][i])
    error_df['pred_cont_name'] = error_df['pred_cont'].map(lambda i: processed_data['continents'][i])

    error_df['true_city_name'] = error_df['true_city'].map(lambda i: processed_data['cities'][i])
    error_df['pred_city_name'] = error_df['pred_city'].map(lambda i: processed_data['cities'][i])

    cont_support_map = dict(zip(np.unique(error_df['true_cont_name'],return_counts=True)[0],np.unique(error_df['true_cont_name'],return_counts=True)[1]))
    city_support_map = dict(zip(np.unique(error_df['true_city_name'],return_counts=True)[0],np.unique(error_df['true_city_name'],return_counts=True)[1]))

    # Step 1: Compute the correctness
    error_df['continent_correct'] = error_df['true_cont'] == error_df['pred_cont']
    error_df['city_correct'] = error_df['true_city'] == error_df['pred_city']

    # Step 2: Calculate the haversine distance
    error_df['coord_error'] = haversine_distance(error_df['true_lat'],error_df['true_lon'],error_df['pred_lat'],error_df['pred_lon'])

    # Print the distance error statistics
    print(f"The median distance error is {np.median(error_df['coord_error'].values)}")
    print(f"The mean distance error is {np.mean(error_df['coord_error'].values)}")
    print(f"The max distance error is {np.max(error_df['coord_error'].values)}")

    # Step 3: Group into 4 categories
    def group_label(row):
        if row['continent_correct'] and row['city_correct']:
            return 'C_correct Z_correct'
        elif row['continent_correct'] and not row['city_correct']:
            return 'C_correct Z_wrong'
        elif not row['continent_correct'] and row['city_correct']:
            return 'C_wrong Z_correct'
        else:
            return 'C_wrong Z_wrong'
        
    # Create the error group column
    error_df['error_group'] = error_df.apply(group_label, axis=1)

    # Now we proceed with grouping
    group_stats = error_df.groupby('error_group')['coord_error'].agg([
        ('count','count'),
        ('mean_error_km','mean'),
        ('median_error_km','median')
    ])

    # Step 5: Calculate proportion and expected error.
    """
    P(C=C*) : Probability of contient predicting correct continent
    P(Z=Z*) : Probability of ciry predicting correct city
    E(D|condition) : Expected distance error under that condition

    E(D) = P(C=C*,Z=Z*)*E(D|C=C*,Z=Z*)+ -> ideal condition continent is correct and city is also correct
            P(C=C*,Z!=Z*)*E(D|C=C*,Z!=Z*)+ -> continent is correct and city is wrong
            P(C!=C*,Z=Z*)*E(D|C!=C*,Z=Z*)+ -> city is correct but continent is wrong
            P(C!=C*,Z!=Z*)*E(D|C!=C*,Z!=Z*) -> both cotinent and city are wrong
    """
    total = len(error_df)
    group_stats['proportion'] = group_stats['count'] / total
    group_stats['weighted_error'] = group_stats['mean_error_km'] * group_stats['proportion']
    expected_total_error = group_stats['weighted_error'].sum()
    logging.info(group_stats)
    logging.info(f"Expected Coordinate Error E[D]: {expected_total_error:.2f} km")

    def compute_in_radius_metrics(y_true, y_pred, thresholds=None):
        """
        Compute % of predictions within given distance thresholds
        y_true, y_pred: numpy arrays of shape (N, 2) for [lat, lon]
        thresholds: List of distance thresholds in kilometers (default: [1, 5, 50, 100, 250, 500, 1000, 5000])
        """
        if thresholds is None:
            thresholds = [1, 5, 50, 100, 250, 500, 1000, 5000]

        distances = haversine_distance(
            y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1]
        )

        results = {}
        for r in thresholds:
            percent = np.mean(distances <= r) * 100
            results[f"<{r} km"] = percent

        return results

    metrics = compute_in_radius_metrics(y_true=np.stack([test_lat,test_lon],axis=1), y_pred=np.stack([pred_lat,pred_lon],axis=1))

    logging.info("In-Radius Accuracy Metrics:")
    for k, v in metrics.items():
        logging.info(f"{k:>8}: {v:.2f}%")
        
    def in_radius_by_group(df, group_col, thresholds=[1, 5, 50, 100, 250, 500, 1000, 5000]):
        """
        Compute in-radius accuracy for a group column (continent, city, or continent+city)
        """
        df = df.copy()
        df['coord_error'] = haversine_distance(
            df['true_lat'].values, df['true_lon'].values,
            df['pred_lat'].values, df['pred_lon'].values
        )

        results = {}
        grouped = df.groupby(group_col)

        for group_name, group_df in grouped:
            res = {}
            errors = group_df['coord_error'].values
            for r in thresholds:
                res[f"<{r} km"] = np.mean(errors <= r) * 100  # in %
            results[group_name] = res

        return pd.DataFrame(results).T  # Transpose for better readability
    
    continent_metrics = in_radius_by_group(error_df, group_col='true_cont_name')
    logging.info("In-Radius Accuracy per Continent")
    continent_metrics['continent_support'] = continent_metrics.index.map(cont_support_map)
    logging.info(continent_metrics.round(2))

    city_metrics = in_radius_by_group(error_df, group_col='true_city_name')
    logging.info("In-Radius Accuracy per City")
    city_metrics['city_support'] = city_metrics.index.map(city_support_map)
    logging.info(city_metrics.round(2))

    error_df['continent_city'] = error_df['true_cont_name'] + " / " + error_df['true_city_name']
    cont_city_metrics = in_radius_by_group(error_df, group_col='continent_city')
    cont_city_metrics['continent_support'] = cont_city_metrics.index.map(lambda x :x.split("/")[-1].strip()).map(city_support_map)
    logging.info("In-Radius Accuracy per Continent-City")
    logging.info(cont_city_metrics.round(2))


# Error calculations for all the predictions
logging.info("Starting error calculations...")
error_calc(test_conts=y_test_cont,pred_conts=cont_test_preds,
           test_city=y_test_city,pred_city = city_test_preds,
           test_lat=y_test_lat,pred_lat=coords_results['test_preds'][:,0],
           test_lon=y_test_lon,pred_lon=coords_results['test_preds'][:,1])

# Plot the points on the world map
logging.info("Plotting points on world map...")
plot_points_on_world_map(true_lat = y_test_lat,
                         true_long=y_test_lon,
                         predicted_lat=coords_results['test_preds'][:,0],
                         predicted_long=coords_results['test_preds'][:,1],
                         filename="test.png")