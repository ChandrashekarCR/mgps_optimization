# The main ensemble model

# Import libraries 

# Import data processing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt
import seaborn as sns

# Import other models
from xgboost_ensemble.xgboost_classification import XGBoostTuner, run_xgboost_classifier
from random_forest.randomforest_classification import RandomForestTuner
from tab_pfn.tab_pfn_classificaiton import TabPFNModel, run_tabpfn_classifier
from grownet.grownet_classification import GrowNetClassifier, GrowNetTuner, run_grownet_classifier
from simple_nn.nn_classification import run_nn_classifier
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
        'y_coords': y_coords, # This is for neural networks.
        'y_latitude': df['latitude'].values, # This is for XGBoost
        'y_longitude':df['longitude'].values, # This is for XGBoost
        'encoders': {
            'continent': continent_encoder,
            'city': city_encoder,
            'coord': coord_scaler
        },
        'continents': continents,
        'cities': cities
    }

# Process data
df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
processed_data = process_data_hierarchical(df)

print(f"Continent Layer")

X_cont = processed_data['x_cont']
y_cont = processed_data['y_continent']

# Split data into train and test data
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(X_cont, y_cont, test_size=0.2, random_state=42, stratify=y_cont)

# Split the same dataset to tune the hyperparameters for the model but with differernt random state
X_train_cont_hyper,X_test_cont_hyper,y_train_cont_hyper,y_test_cont_hyper = train_test_split(X_cont,y_cont,test_size=0.2,random_state=101,stratify=y_cont)

print("Generating cross-validation predictions for continent layer.")

# Initialize arrays for storing CV predictions
n_train_samples = X_train_cont.shape[0]
n_continent_classes = len(np.unique(y_train_cont))

oof_probs = {
    'xgb':np.zeros((n_train_samples, n_continent_classes)),
    'grownet':np.zeros((n_train_samples, n_continent_classes)),
    'nn': np.zeros((n_train_samples, n_continent_classes))
}


# Tune the hyperparameters once using the subset generated
tune_hyperparams = False
if tune_hyperparams:
    print("Tuning hyperparameters for the contient predictions")
    xgboost_continent_tune_result = run_xgboost_classifier(X_train_cont_hyper,X_test_cont_hyper,y_train_cont_hyper,y_test_cont_hyper,tune_hyperparams=True,
                                         n_trials=50,timeout=1800)
    best_xgboost_cont_params = xgboost_continent_tune_result['params']

    grownet_continent_tune_result = run_grownet_classifier(X_train_cont_hyper,y_train_cont_hyper,X_test_cont_hyper,y_test_cont_hyper,tune_hyperparams=True,
                                                n_trials=50,timeout=1800)
    best_grownet_cont_params = grownet_continent_tune_result['params']

    nn_continent_tune_result = run_nn_classifier(X_train_cont_hyper,y_train_cont_hyper,X_test_cont_hyper,y_test_cont_hyper,tune_hyperparams=True,
                                                 n_trials=50, timeout=1800)
    best_nn_cont_params = nn_continent_tune_result['params']


else:
    best_xgboost_cont_params = None
    best_grownet_cont_params = None
    best_nn_cont_params = None


K = 3
skf = StratifiedKFold(n_splits=K,shuffle=True,random_state=42)
# Store models for test predictions
continent_models = []

# Cross validation for contient predictions
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_cont,y_train_cont)):

    print(f"Fold {fold+1}/{K}")

    # Split data for this fold
    X_fold_train, X_fold_val = X_train_cont[train_idx], X_train_cont[val_idx]
    y_fold_train, y_fold_val = y_train_cont[train_idx], y_train_cont[val_idx]

    # Train the models
    # 1) XGBoost model
    print(f"Running XGBoost model on Fold {fold+1}/{K}")
    xgboost_fold_result = run_xgboost_classifier(X_train=X_fold_train,y_train=y_fold_train,X_test=X_fold_val,y_test=y_fold_val,
                                         tune_hyperparams=False,custom_params=best_xgboost_cont_params)
    
    # 2) Grownet model
    print(f"Running GrowNet model on Fold {fold+1}/{K}")
    grownet_fold_result = run_grownet_classifier(X_train=X_fold_train,y_train=y_fold_train,X_test=X_fold_val,y_test=y_fold_val,
                                         tune_hyperparams=False,params=best_grownet_cont_params)
    
    # 3) Neural Network model
    print(f"Neural network model on Fold {fold+1}/{K}")
    nn_fold_result = run_nn_classifier(X_train=X_fold_train,y_train=y_fold_train,X_test=X_fold_val,y_test=y_fold_val,
                                       tune_hyperparams=False,params=best_nn_cont_params)

    
    # Store the out-of-fold predictions
    oof_probs['xgb'][val_idx] = xgboost_fold_result['predicted_probabilities']
    oof_probs['grownet'][val_idx] = grownet_fold_result['predicted_probabilities']
    oof_probs['nn'][val_idx] = nn_fold_result['predicted_probabilities']


# Train final continent model on full training data for test predictions
print("Training on the entire train dataset")

# 1) XGBoost
xgboost_contient_model = run_xgboost_classifier(X_train_cont,y_train_cont,X_test_cont,y_test_cont,custom_params=best_xgboost_cont_params)
# Get the test predictions for continents
continent_test_xgboost_meta_features = xgboost_contient_model['predicted_probabilities']

# 2) GrowNet
grownet_continent_model = run_grownet_classifier(X_train_cont,y_train_cont,X_test_cont,y_test_cont,params=best_grownet_cont_params,
                                                tune_hyperparams=False)
continent_test_grownet_meta_features = grownet_continent_model['predicted_probabilities']

# 3) Neural Networks
nn_continent_model = run_nn_classifier(X_train_cont,y_train_cont,X_test_cont,y_test_cont,params=best_nn_cont_params,
                                       tune_hyperparams=False)
continent_test_nn_meta_features = nn_continent_model['predicted_probabilities']

# Stacking for the meta model
meta_X_train = np.hstack([oof_probs['xgb'],oof_probs['grownet'],oof_probs['nn']])
meta_X_test = np.hstack([
    continent_test_xgboost_meta_features.reshape(1,-1) if continent_test_xgboost_meta_features.ndim == 1 else continent_test_xgboost_meta_features,
    continent_test_grownet_meta_features.reshape(1,-1) if continent_test_grownet_meta_features.ndim == 1 else continent_test_grownet_meta_features,
    continent_test_nn_meta_features.reshape(1,-1) if continent_test_nn_meta_features.ndim == 1 else continent_test_nn_meta_features
])


# Train the meta model for contient predictions
print("Training the continent meta model")
continent_meta_model = LogisticRegression(multi_class="multinomial",max_iter=1000)
continent_meta_model.fit(meta_X_train,y_train_cont)

# Evaluate continent predictions
continent_train_preds = continent_meta_model.predict(meta_X_train)
continent_test_preds = continent_meta_model.predict(meta_X_test)

print("Continent Prediction - Train Set:")
print(classification_report(y_train_cont, continent_train_preds, target_names=processed_data['continents']))

print("\nContinent Prediction - Test Set:")
print(classification_report(y_test_cont, continent_test_preds, target_names=processed_data['continents']))

print("City Layer")
y_cities = processed_data['y_city']
y_city_train,y_city_test = train_test_split(y_cities,test_size=0.2,random_state=42)

# Augment original features with continent predictions
X_cities_train = np.hstack([X_train_cont,meta_X_train])
X_cities_test = np.hstack([X_test_cont,continent_test_xgboost_meta_features.reshape(1,-1) if continent_test_xgboost_meta_features.ndim == 1 else continent_test_xgboost_meta_features])



## TabPFN model
#tabpfn_tuned = run_tabpfn_classifier(X_train,y_train,X_test,
#                                     y_test,tune_hyperparams=True,
#                                     device='cuda',
#                                     custom_params={'max_time':500})
#tabpfn_model = tabpfn_tuned['model']
#
## Run prediction on the training dataset to get the prbabilities
#prob_tabpfn = tabpfn_model.predict_proba(X_train)
#
## Grownet model
#grownet_model = run_grownet(X_train,y_train, X_test, y_test, tune_hyperparams=True,timeout=1200)
#prob_grownet = grownet_model.predict(X_train)['probabilities']
#
## Neural Network
#nn_model = run_nn_classifier(
#    X_train=X_train,
#    y_train=y_train,
#    X_test=X_test,
#    y_test=y_test,
#    tune_hyperparams=True,n_trials=20,timeout=1000)
#
#prob_nn = nn_model.predict(X_train)['probabilities']

## FT-Transformer
#ft_model = run_ft_transformer_classifier(
#    X_train=X_train,
#    y_train = y_train,
#    X_test=X_test,
#    y_test=y_test,
#    tune_hyperparams=False
#)
#
#print(ft_model.predict(X_train)['probabilities'])