# The main ensemble model

# Import libraries 

# Import data processing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import other models
from xgboost_ensemble.xgboost_classification import run_xgboost_classifier
from xgboost_ensemble.xgboost_regression import run_xgboost_regressor
from tab_pfn.tab_pfn_classificaiton import run_tabpfn_classifier
from tab_pfn.tab_pfn_regression import run_tabpfn_regressor
from grownet.grownet_classification import run_grownet_classifier
from grownet.grownet_regressor import run_grownet_regressor
from simple_nn.nn_classification import run_nn_classifier
from simple_nn.nn_regression import run_nn_regressor
from ft_transformer.ft_transformer_classification import run_ft_transformer_classifier



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

# Training data for continent
# Original feautres
X_train_cont = split_data['X_train']
X_test_cont = split_data['X_test']

# Train and test for continent
y_train_cont = split_data['y_cont_train']
y_test_cont = split_data['y_cont_test']

# Train and test for cities
y_train_city = split_data['y_city_train']
y_test_city = split_data['y_city_test']

# Train and test for latitude
y_train_lat = split_data['y_lat_train']
y_test_lat = split_data['y_lat_test']

# Train and test for longitude
y_train_lon = split_data['y_lon_train']
y_test_lon = split_data['y_lon_test']

# Train and test for co-ordinates
y_train_coords = split_data['y_coords_train']
y_test_coords = split_data['y_coords_test']

def train_hierarchical_layer(
        X_train,
        X_test,
        y_train,
        y_test,
        processed_data,
        run_xgboost_classifier=None,
        run_grownet_classifier=None,
        run_nn_classifier=None,
        run_tabpfn_classifier=None,
        tune_hyperparams=False,
        random_state=42,
        n_splits=3
):
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
        }
    }
    
    # Filter to only enabled models
    enabled_models = {k: v for k, v in model_configs.items() if v['enabled']}
    
    if not enabled_models:
        raise ValueError("At least one model function must be provided (not None)")
    
    print(f"Enabled models: {list(enabled_models.keys())}")
    
    # Split the same dataset to tune the hyperparameters for the model but with different random state
    X_train_hyper, X_test_hyper, y_train_hyper, y_test_hyper = train_test_split(
        X_train, y_train, test_size=0.2, random_state=101, stratify=y_train
    )

    # Hyperparameter tuning for enabled models
    best_params = {}
    
    if tune_hyperparams:
        print("Tuning hyperparameters for the enabled models")
        
        for model_key, config in enabled_models.items():
            print(f"Tuning {config['name']} hyperparameters...")
            
            try:
                if model_key == 'xgb':
                    result = config['function'](
                        X_train_hyper, X_test_hyper, y_train_hyper, y_test_hyper,
                        tune_hyperparams=True, n_trials=50, timeout=1800
                    )
                elif model_key == 'grownet':
                    result = config['function'](
                        X_train_hyper, X_test_hyper, y_train_hyper, y_test_hyper,
                        tune_hyperparams=True, n_trials=50, timeout=1800
                    )
                elif model_key == 'nn':
                    result = config['function'](
                        X_train_hyper, X_test_hyper, y_train_hyper, y_test_hyper,
                        tune_hyperparams=True, n_trials=50, timeout=1800
                    )
                elif model_key == 'tabpfn':
                    result = config['function'](
                        X_train_hyper, y_train_hyper, X_test_hyper, y_test_hyper,
                        tune_hyperparams=True, max_time=300
                    )
                
                best_params[model_key] = result['params']
                print(f"Best {config['name']} params: {best_params[model_key]}")
                
            except Exception as e:
                print(f"Error tuning {config['name']}: {e}")
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

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        
        print(f"\nFold {fold+1}/{n_splits}")

        # Split data for this fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Train each enabled model
        for model_key, config in enabled_models.items():
            print(f"Running {config['name']} model on Fold {fold+1}/{n_splits}")
            
            try:
                if model_key == 'tabpfn':
                    # TabPFN has different signature
                    fold_result = config['function'](
                        X_fold_train, y_fold_train, X_fold_val, y_fold_val,
                        tune_hyperparams=False, params=best_params[model_key]
                    )
                else:
                    # Other models have consistent signature
                    fold_result = config['function'](
                        X_fold_train, y_fold_train, X_fold_val, y_fold_val,
                        tune_hyperparams=False, params=best_params[model_key]
                    )
                
                # Store the out-of-fold predictions
                oof_probs[model_key][val_idx] = fold_result['predicted_probabilities']
                
            except Exception as e:
                print(f"Error running {config['name']} on fold {fold+1}: {e}")
                # Fill with uniform probabilities as fallback
                oof_probs[model_key][val_idx] = np.full((len(val_idx), n_classes), 1.0/n_classes)

    # Create meta training features by concatenating all enabled model predictions
    meta_feature_list = []
    for model_key in enabled_models.keys():
        meta_feature_list.append(oof_probs[model_key])
    
    meta_X_train = np.hstack(meta_feature_list)
    print(f"Meta training features shape: {meta_X_train.shape}")

    # Train final models on the full training data
    print("\\nTraining final models on the full training data")
    test_results = {}
    
    for model_key, config in enabled_models.items():
        print(f"Training final {config['name']} model...")
        
        try:
            if model_key == 'tabpfn':
                # TabPFN has different signature
                result = config['function'](
                    X_train, y_train, X_test, y_test,
                    params=best_params[model_key]
                )
            else:
                # Other models have consistent signature
                result = config['function'](
                    X_train, y_train, X_test, y_test,
                    params=best_params[model_key]
                )
            
            test_results[model_key] = result['predicted_probabilities']
            
        except Exception as e:
            print(f"Error training final {config['name']} model: {e}")
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
    print(f"Meta test features shape: {meta_X_test.shape}")

    # Train meta model
    print("\\nTraining meta model...")
    meta_model = xgb.XGBClassifier(objective='multi:softprob', random_state=random_state)
    meta_model.fit(meta_X_train, y_train)

    # Make predictions
    train_preds = meta_model.predict(meta_X_train)
    test_preds = meta_model.predict(meta_X_test)

    # Print summary
    print(f"\\nSummary:")
    print(f"- Used models: {list(enabled_models.keys())}")
    print(f"- Meta features: {meta_X_train.shape[1]}")
    print(f"- Meta model train accuracy: {accuracy_score(y_train, train_preds):.4f}")
    print(f"- Meta model test accuracy: {accuracy_score(y_test, test_preds):.4f}")

    return meta_model, meta_X_train, meta_X_test, train_preds, test_preds


continent_model, meta_X_train_cont, meta_X_test_cont, train_preds, test_preds = train_hierarchical_layer(
    X_train=X_train_cont,
    X_test=X_test_cont,
    y_train=y_train_cont,
    y_test=y_test_cont,
    processed_data=processed_data,
    run_xgboost_classifier=run_xgboost_classifier,
    run_grownet_classifier=None,
    run_nn_classifier=None,
    run_tabpfn_classifier=run_tabpfn_classifier,
    tune_hyperparams=False
)
print("Continent Prediction - Train Set:")
print(classification_report(y_train_cont, train_preds, target_names=processed_data['continents']))

print("\nContinent Prediction - Test Set:")
print(classification_report(y_test_cont, test_preds,target_names=processed_data['continents']))


X_train_city = np.hstack([X_train_cont,meta_X_train_cont])
X_test_city = np.hstack([X_test_cont,meta_X_test_cont])

city_model, meta_X_train_city, meta_X_test_city, train_preds, test_preds = train_hierarchical_layer(
    X_train=X_train_city,
    X_test=X_test_city,
    y_train=y_train_city,
    y_test=y_test_city,
    processed_data=processed_data,
    run_xgboost_classifier=run_xgboost_classifier,
    run_grownet_classifier=None,
    run_nn_classifier=None,
    run_tabpfn_classifier=None,tune_hyperparams=False
)

print("City Prediction - Train Set:")
print(classification_report(y_train_city,train_preds,target_names=processed_data['cities']))

print("\nCity Prediction - Test Set:")
print(classification_report(y_test_city,test_preds))
#
#X_train_coord = np.hstack([X_train_city,meta_X_train_city])
#X_test_coord = np.hstack([X_test_city,meta_X_test_city])
#
lat_model = run_xgboost_regressor(
    X_train=X_train_city,
    y_train=y_train_lat,
    X_test=X_test_city,
    y_test=y_test_lat,
    tune_hyperparams=False
)

lon_model = run_xgboost_regressor(
    X_train=X_train_cont,
    y_train=y_train_lon,
    X_test=X_test_cont,
    y_test=y_test_lon,
    tune_hyperparams=False
)

print(lat_model['predictions'],lon_model['predictions'])



#results = run_grownet_regressor(
#    X_train=X_train_cont,
#    y_train=y_train_coords,
#    X_test=X_test_cont,
#    y_test=y_test_coords)
#
#print(results['predictions'].shape)
#
#results = run_nn_regressor(
#     X_train=X_train_cont,
#    y_train=y_train_coords,
#    X_test=X_test_cont,
#    y_test=y_test_coords
#)
#
#print(results['predictions'].shape)
#
#
#results = run_tabpfn_regressor(
#    X_train=X_train_cont,
#    y_train=y_train_coords,
#    X_test=X_test_cont,
#    y_test=y_test_coords
#)
#
#xyz = results['xyz_predictions']
#print(xyz.shape)

#xyz_rescaled = processed_data['encoders']['coord'].inverse_transform(xyz)
#
## Rescaled predictions (x, y, z) -> radians
#x, y, z = xyz_rescaled[:, 0], xyz_rescaled[:, 1], xyz_rescaled[:, 2]
#lat_pred_rad = np.arcsin(z)
#lon_pred_rad = np.arctan2(y, x)
#
## Convert to degrees
#lat_pred_deg = np.degrees(lat_pred_rad)
#lon_pred_deg = np.degrees(lon_pred_rad)
#
#
lat_true_deg = y_test_lat
lon_true_deg = y_test_lon
#
from math import radians, sin, cos, sqrt, atan2
#
def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371.0

    # Convert from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  # in kilometers

lat_pred_deg = lat_model['predictions']
lon_pred_deg = lon_model['predictions']


distances = haversine_distance(lat_true_deg, lon_true_deg, lat_pred_deg, lon_pred_deg)

# Median and other stats
print(f"Median Haversine Distance: {np.median(distances):.2f} km")
print(f"Mean Haversine Distance: {np.mean(distances):.2f} km")
print(f"95th Percentile Distance: {np.percentile(distances, 95):.2f} km")
#