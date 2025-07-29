# The main ensemble model

# Import libraries 
import os

# Import data processing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import other models
from xgboost_ensemble.xgboost_classification import run_xgboost_classifier
from xgboost_ensemble.xgboost_regression import run_xgboost_regressor
from tab_pfn.tab_pfn_classificaiton import run_tabpfn_classifier
from tab_pfn.tab_pfn_regression import run_tabpfn_regressor
from lightgbm_ensemble_model.lightgbm_classification import run_lightgbm_classifier
from catboost_ensemble.catboost_classification import run_catboost_classifier
from grownet.grownet_classification import run_grownet_classifier
from grownet.grownet_regressor import run_grownet_regressor
from simple_nn.nn_classification import run_nn_classifier
from simple_nn.nn_regression import run_nn_regressor
from ft_transformer.ft_transformer_classification import run_ft_transformer_classifier
from encoder.microbe_autoencoder import train_autoencoder, MicrobiomeAutoencoder


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
        run_catboost_classifier = None,
        tune_hyperparams=False,
        apply_smote = False,
        random_state=42,
        n_splits=3,
        accuracy_threshold=0.8):
    """
    Dynamic hierarchical layer that can skip models if they are passed as None.
    
       Args:
        X_train, X_test, y_train, y_test: Training and test data
        processed_data: Processed data (for reference)
        run_xgboost_classifier: XGBoost classifier function or None to skip
        run_grownet_classifier: GrowNet classifier function or None to skip
        run_nn_classifier: Neural Network classifier function or None to skip
        run_tabpfn_classifier: TabPFN classifier function or None to skip
        tune_hyperparams: Whether to tune hyperparameters
        random_state: Random state for reproducibility
        n_splits: Number of CV splits
    
    Returns:
        meta_model, meta_X_train, meta_X_test, train_preds, test_preds
    """
    
    # Define all possible models with their configurations
    model_configs = {
        'xgb': {
            'name': 'XGBoost',
            'function': run_xgboost_classifier,
            'enabled': run_xgboost_classifier is not None
        },
        'grownet': {
            'name': 'GrowNet',
            'function': run_grownet_classifier,
            'enabled': run_grownet_classifier is not None
        },
        'nn': {
            'name': 'Neural Network',
            'function': run_nn_classifier,
            'enabled': run_nn_classifier is not None
        },
        'tabpfn': {
            'name': 'TabPFN',
            'function': run_tabpfn_classifier,
            'enabled': run_tabpfn_classifier is not None
        },
        'lightgbm': {
            'name': 'LightGBM',
            'function':run_lightgbm_classifier,
            'enabled':run_lightgbm_classifier is not None
        },
        'catboost': {
            'name': 'CatBoost',
            'function':run_catboost_classifier,
            'enabled':run_catboost_classifier is not None
        }
    }
    
    # Filter to only enabled models
    enabled_models = {k: v for k, v in model_configs.items() if v['enabled']}
    
    if not enabled_models:
        raise ValueError("At least one model function must be provided (not None)")
    
    logging.info(f"Enabled models: {list(enabled_models.keys())}")
    
    # Split the same dataset to tune the hyperparameters for the model but with different random state
    X_train_hyper, X_test_hyper, y_train_hyper, y_test_hyper = train_test_split(
        X_train, y_train, test_size=0.2, random_state=101, stratify=y_train
    )

    # Hyperparameter tuning for enabled models
    best_params = {}
    
    if tune_hyperparams:
        logging.info("Tuning hyperparameters for the enabled models")
        
        for model_key, config in enabled_models.items():
            logging.info(f"Tuning {config['name']} hyperparameters...")
            
            try:
                if model_key == 'xgb':
                    result = config['function'](
                        X_train_hyper,y_train_hyper, X_test_hyper, y_test_hyper,
                        tune_hyperparams=True, n_trials=50, timeout=1800
                    )
                elif model_key == 'grownet':
                    result = config['function'](
                        X_train_hyper, y_train_hyper, X_test_hyper, y_test_hyper,
                        tune_hyperparams=True, n_trials=50, timeout=1800
                    )
                elif model_key == 'nn':
                    result = config['function'](
                        X_train_hyper, y_train_hyper, X_test_hyper, y_test_hyper,
                        tune_hyperparams=True, n_trials=50, timeout=1800
                    )
                elif model_key == 'tabpfn':
                    result = config['function'](
                        X_train_hyper, y_train_hyper, X_test_hyper, y_test_hyper,
                        tune_hyperparams=True, max_time=300
                    )

                elif model_key == 'lightgbm':
                    result = config['function'](
                        X_train_hyper, y_train_hyper, X_test_hyper, y_test_hyper,
                        tune_hyperparams=True,n_trials=50, max_time=1800
                    )
                
                elif model_key == 'catboost':
                    result = config['function'](
                        X_train_hyper, y_train_hyper, X_test_hyper, y_test_hyper,
                        tune_hyperparams=True,n_trials=50, max_time=1800
                    )
                
                best_params[model_key] = result['params']
                logging.info(f"Best {config['name']} params: {best_params[model_key]}")
                
            except Exception as e:
                logging.error(f"Error tuning {config['name']}: {e}")
                best_params[model_key] = None
    else:
        # Initialize all as None
        best_params = {model_key: None for model_key in enabled_models.keys()}

    # Initialize arrays for storing CV predictions
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_train_samples = X_train.shape[0]
    n_classes = len(np.unique(y_train))

    # Only create OOF arrays for enabled models
    oof_probs = {}
    for model_key in enabled_models.keys():
        oof_probs[model_key] = np.zeros((n_train_samples, n_classes))

    # Track validation accuracies per model across folds
    model_val_accuracies = {model_key: [] for model_key in enabled_models.keys()}

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        
        logging.info(f"\nFold {fold+1}/{n_splits}")

        # Split data for this fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        if apply_smote:
            logging.info("Applying SMOTE to balance the dataset.")
            X_fold_train_balanced, y_fold_train_balanced = SMOTE(random_state=42).fit_resample(X_fold_train,y_fold_train)

        else:
            X_fold_train_balanced,y_fold_train_balanced = X_fold_train,y_fold_train

        # Train each enabled model
        for model_key, config in enabled_models.items():
            logging.info(f"Running {config['name']} model on Fold {fold+1}/{n_splits}")
            
            try:
                fold_result = config['function'](
                        X_fold_train_balanced, y_fold_train_balanced, X_fold_val, y_fold_val,
                        tune_hyperparams=False, params=best_params[model_key])
                
                # Store the out-of-fold predictions
                oof_probs[model_key][val_idx] = fold_result['predicted_probabilities']
                
            except Exception as e:
                logging.error(f"Error running {config['name']} on fold {fold+1}: {e}")
                # Fill with uniform probabilities as fallback
                oof_probs[model_key][val_idx] = np.full((len(val_idx), n_classes), 1.0/n_classes)

    # Create meta training features by concatenating all enabled model predictions
    meta_feature_list = []
    for model_key in enabled_models.keys():
        meta_feature_list.append(oof_probs[model_key])
    
    meta_X_train = np.hstack(meta_feature_list)
    logging.info(f"Meta training features shape: {meta_X_train.shape}")

    # Train final models on the full training data
    logging.info("\nTraining final models on the full training data")
    test_results = {}
    
    for model_key, config in enabled_models.items():
        logging.info(f"Training final {config['name']} model...")
        
        try:
            result = config['function'](
                    X_train, y_train, X_test, y_test, # Passing the original X_train,y_train, X_test and y_test to make predictions.
                    params=best_params[model_key]
                )
           
            test_results[model_key] = result['predicted_probabilities']
            
        except Exception as e:
            logging.error(f"Error training final {config['name']} model: {e}")
            # Fill with uniform probabilities as fallback
            test_results[model_key] = np.full((X_test.shape[0], n_classes), 1.0/n_classes)

    # Create meta test features
    meta_test_feature_list = []
    for model_key in enabled_models.keys():
        probs = test_results[model_key]
        # Ensure proper shape
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        meta_test_feature_list.append(probs)
    
    meta_X_test = np.hstack(meta_test_feature_list)
    logging.info(f"Meta test features shape: {meta_X_test.shape}")

    # Train meta model
    logging.info("\nTraining meta model...")
    meta_model = xgb.XGBClassifier(objective='multi:softprob', random_state=random_state)
    meta_model.fit(meta_X_train, y_train)

    # Make predictions
    train_preds = meta_model.predict(meta_X_train)
    test_preds = meta_model.predict(meta_X_test)

    # Print summary
    logging.info(f"\nSummary:")
    logging.info(f"- Used models: {list(enabled_models.keys())}")
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
        tune_hyperparams = False,
        random_state = 42,
        n_splits = 3):
    """
    Dynamic hierarchical coordinate prediction laer similar to the classification version.
    """

    # Define all possible models with their cofigurations
    model_configs = {
        'xgb':{
            'name':'XGBoost',
            'function':run_xgboost_regressor,
            'enabled': run_xgboost_regressor is not None,
            'prediction_type':'sequential' # XGBoost does sequential lat -> lon prediction
        },
        'grownet':{
            'name':'GrowNet',
            'function':run_grownet_regressor,
            'enabled':run_grownet_regressor is not None,
            'prediction_type':'xyz' # GrowNet predicts xyz coordinates
        },
        'nn':{
            'name':'Neural Network',
            'function': run_nn_regressor,
            'enabled':run_nn_regressor is not None,
            'prediction_type':'xyz' # Neural Network predicts xyz coordinates
        },
        'tabpfn':{
            'name':'Tab PFN',
            'function':run_tabpfn_regressor,
            'enabled':run_tabpfn_regressor is not None,
            'prediction_type':'xyz' # TabPFN predicts xyz coordinates
        }
    }

    # Filter to only the enabled models
    enabled_models = {k: v for k,v in model_configs.items() if v['enabled']}

    if not enabled_models:
        raise ValueError("At least one model function must be provided (not None)")
    
    logging.info(f"Enabled models: {list(enabled_models.keys())}")

    # Split the same data set to tune the hyperparameters for the mode byt with different random state
    X_train_hyper, X_test_hyper, y_train_hyper_lat, y_test_hyper_lat = train_test_split(
        X_train, y_train_lat, test_size=0.2, random_state=101
    )
    _, _, y_train_hyper_lon, y_test_hyper_lon = train_test_split(
        X_train, y_train_lon, test_size=0.2, random_state=101
    )
    _, _, y_train_hyper_coords, y_test_hyper_coords = train_test_split(
        X_train, y_train_coords, test_size=0.2, random_state=101
    )

    # Hyperparameter tuning for enabled models
    best_params = {}
    
    if tune_hyperparams:
        logging.info("Tuning hyperparameters for the enabled models")
        
        for model_key, config in enabled_models.items():
            logging.info(f"Tuning {config['name']} hyperparameters...")
            
            # Write the tuning function here, Keep this empty now as I have not added tuning to the other scripts.
    else:
        # Initialiaze all as None
        best_params = {model_key: None for model_key in enabled_models.keys()}

    # Initialize arrays for storing CV predictions
    kf = KFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    n_train_samples = X_train.shape[0]

    # Only create oof arrays for enabled models (each model predictions lat and lon)
    oof_predictions = {}
    for model_key in enabled_models.keys():
        oof_predictions[model_key] = np.zeros((n_train_samples,2)) # One for lat and one for long

    # Cross validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):

        logging.info(f"\nFold {fold+1}/{n_splits}")

        # SPlit data for this fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train_lat, y_fold_val_lat = y_train_lat[train_idx], y_train_lat[val_idx]
        y_fold_train_lon, y_fold_val_lon = y_train_lon[train_idx], y_train_lon[val_idx]
        y_fold_train_coords, y_fold_val_coords = y_train_coords[train_idx], y_train_coords[val_idx]

        # Train each enabled model
        for model_key, config in enabled_models.items():
            logging.info(f"Running {config['name']} model on Fold {fold+1}/{n_splits}")

            try:
                if model_key == "xgb":
                    # XGBoost sequential prediction: lat first, then augment and predict lon

                    # Predict latitude
                    lat_result = config['function'](
                        X_fold_train,y_fold_train_lat,X_fold_val,y_fold_val_lat,tune_hyperparams=False,params=best_params[model_key]
                    )
                    lat_pred_train = lat_result['model'].predict(X_fold_train)
                    lat_pred_val = lat_result['predictions']

                    # Augment features with predicted latitude for longitue prediction
                    X_fold_train_aug = np.hstack([X_fold_train,lat_pred_train.reshape(-1,1)])
                    X_fold_val_aug = np.hstack([X_fold_val,lat_pred_val.reshape(-1,1)])

                    # Predict longitude
                    lon_result = config['function'](
                        X_fold_train_aug,y_fold_train_lon,X_fold_val_aug,y_fold_val_lon, tune_hyperparams=False,params = best_params[model_key]
                    )
                    lon_pred_val = lon_result['predictions']

                    # Store predictions
                    oof_predictions[model_key][val_idx,0] = lat_pred_val
                    oof_predictions[model_key][val_idx,1] = lon_pred_val

                elif model_key in ['grownet','nn','tabpfn']:
                    # XYZ predictions models
                    fold_result = config['function'](
                        X_fold_train,y_fold_train_coords,X_fold_val,y_fold_val_coords,
                        tune_hyperparams=False,params = best_params[model_key]
                    )

                    # Convert XYZ to lat/lon
                    xyz_pred = fold_result['predictions']
                    xyz_rescaled = coord_scaler.inverse_transform(xyz_pred)
                    latlon_pred = xyz_to_latlon(xyz_rescaled)

                    # Store predictions
                    oof_predictions[model_key][val_idx] = latlon_pred
                
            except Exception as e:
                logging.error(f"Error running {config['name']} on fold {fold+1}: {e}")
                # Fill with uniform probabilities as fallback
                exit()

    # Crate meta training features by concatenating all enabled model predictions
    meta_features_list = []
    for model_key in enabled_models.keys():
        meta_features_list.append(oof_predictions[model_key])

    meta_X_train = np.hstack(meta_features_list)
    logging.info(f"Meta training feature shape :{meta_X_train.shape}")

    # Train final models on the full training data
    logging.info("\nTraining final models on the full training data")
    test_results = {}
    
    for model_key, config in enabled_models.items():
        logging.info(f"Training final {config['name']} model...")
        
        try:
            if model_key == 'xgb':
                # XGBoost sequential prediction on full data
                lat_result = config['function'](
                    X_train, y_train_lat, X_test, y_test_lat,
                    params=best_params[model_key]
                )
                lat_pred_train = lat_result['model'].predict(X_train)
                lat_pred_test = lat_result['predictions']
                
                # Augment and predict longitude
                X_train_aug = np.hstack([X_train, lat_pred_train.reshape(-1, 1)])
                X_test_aug = np.hstack([X_test, lat_pred_test.reshape(-1, 1)])
                
                lon_result = config['function'](
                    X_train_aug, y_train_lon, X_test_aug, y_test_lon,
                    params=best_params[model_key]
                )
                lon_pred_test = lon_result['predictions']
                
                test_results[model_key] = np.stack([lat_pred_test, lon_pred_test], axis=1)
                
            elif model_key in ['grownet', 'nn', 'tabpfn']:
                # XYZ prediction models
                result = config['function'](
                    X_train, y_train_coords, X_test, y_test_coords,
                    params=best_params[model_key]
                )
                
                xyz_pred = result['predictions']
                xyz_rescaled = coord_scaler.inverse_transform(xyz_pred)
                latlon_pred = xyz_to_latlon(xyz_rescaled)
                
                test_results[model_key] = latlon_pred
                
        except Exception as e:
            logging.error(f"Error training final {config['name']} model : {e}")
            exit()

    # Create meta test features
    meta_test_feature_list = []
    for model_key in enabled_models.keys():
        preds = test_results[model_key]
        meta_test_feature_list.append(preds)
    
    meta_X_test = np.hstack(meta_test_feature_list)
    logging.info(f"Meta test features shape: {meta_X_test.shape}")

    # Train meta models (separate for lat and lon)
    logging.info("\nTraining meta models...")
    
    # Prepare targets
    y_train_combined = np.stack([y_train_lat, y_train_lon], axis=1)
    y_test_combined = np.stack([y_test_lat, y_test_lon], axis=1)
    
    # Train separate XGBoost models for latitude and longitude
    meta_lat_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    meta_lon_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    
    meta_lat_model.fit(meta_X_train, y_train_lat)
    meta_lon_model.fit(meta_X_train, y_train_lon)

    # Make predictions
    train_lat_preds = meta_lat_model.predict(meta_X_train)
    train_lon_preds = meta_lon_model.predict(meta_X_train)
    train_preds = np.stack([train_lat_preds, train_lon_preds], axis=1)
    
    test_lat_preds = meta_lat_model.predict(meta_X_test)
    test_lon_preds = meta_lon_model.predict(meta_X_test)
    test_preds = np.stack([test_lat_preds, test_lon_preds], axis=1)

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

    train_metrics = calculate_distance_metrics(y_train_combined, train_preds)
    test_metrics = calculate_distance_metrics(y_test_combined, test_preds)

    # Print summary
    logging.info(f"\nSummary:")
    logging.info(f"- Used models: {list(enabled_models.keys())}")
    logging.info(f"- Meta features: {meta_X_train.shape[1]}")
    logging.info(f"- Meta model train median distance: {train_metrics['median_distance']:.2f} km")
    logging.info(f"- Meta model test median distance: {test_metrics['median_distance']:.2f} km")
    logging.info(f"- Meta model train mean distance: {train_metrics['mean_distance']:.2f} km")
    logging.info(f"- Meta model test mean distance: {test_metrics['mean_distance']:.2f} km")
    logging.info(f"- Meta model test 95th percentile: {test_metrics['percentile_95']:.2f} km")

    return {
        'meta_lat_model': meta_lat_model,
        'meta_lon_model': meta_lon_model,
        'meta_X_train': meta_X_train,
        'meta_X_test': meta_X_test,
        'train_preds': train_preds,
        'test_preds': test_preds,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'individual_results': test_results,
        'enabled_models': list(enabled_models.keys())
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
    run_lightgbm_classifier=False,
    run_catboost_classifier=None,
    tune_hyperparams=False,
    apply_smote=True,n_splits=5
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
    run_lightgbm_classifier=False,
    run_catboost_classifier=False,
    run_nn_classifier=None,
    run_tabpfn_classifier=None,tune_hyperparams=False,
    apply_smote=False,n_splits=5
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
    run_xgboost_regressor=None,
    run_tabpfn_regressor=run_tabpfn_regressor,
    run_nn_regressor=None,
    run_grownet_regressor=None,
    tune_hyperparams=False,n_splits=5
)


# All metrics
save_dir = "saved_results/"
os.makedirs(save_dir,exist_ok=True)
# Continent Layer
#print("Continent Prediction - Train Set:")
#print(classification_report(y_train_cont, cont_train_preds, target_names=processed_data['continents']))

logging.info("\nContinent Prediction - Test Set:")
logging.info(classification_report(y_test_cont, cont_test_preds,target_names=processed_data['continents']))
# Save the test predictions
np.save(os.path.join(save_dir, "x_test.npy"), X_test_cont)
np.save(os.path.join(save_dir, "y_test_cont.npy"),y_test_cont)
np.save(os.path.join(save_dir, "y_pred_cont.npy"),cont_test_preds)

# City Layer
#print("City Prediction - Train Set:")
#print(classification_report(y_train_city,city_train_preds,target_names=processed_data['cities']))

logging.info("\nCity Prediction - Test Set:")
logging.info(classification_report(y_test_city,city_test_preds))
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
        'true_lon': pred_lat,
        'pred_lat': test_lon,
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
#logging.info("Starting error calculations...")
#error_calc(test_conts=y_test_cont,pred_conts=cont_test_preds,
#           test_city=y_test_city,pred_city = city_test_preds,
#           test_lat=y_test_lat,pred_lat=coords_results['test_preds'][:,0],
#           test_lon=y_test_lon,pred_lon=coords_results['test_preds'][:,1])

# Plot the points on the world map
logging.info("Plotting points on world map...")
plot_points_on_world_map(true_lat = y_test_lat,
                         true_long=y_test_lon,
                         predicted_lat=coords_results['test_preds'][:,0],
                         predicted_long=coords_results['test_preds'][:,1],
                         filename="test.png")