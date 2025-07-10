# The main ensemble model

# Import libraries 

# Import data processing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt
import seaborn as sns

# Import other models
from xgboost_ensemble.xgboost_classification import XGBoostTuner, run_xgboost_classifier
from random_forest.randomforest_classification import RandomForestTuner
from tab_pfn.tab_pfn_classificaiton import TabPFNModel, run_tabpfn_classifier
from grownet.grownet_classification import GrowNetClassifier, GrowNetTuner, run_grownet
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
        'y_latitude': df['latitude'].values,
        'y_longitude':df['longitude'].values,
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


X = processed_data['x_cont']
y = processed_data['y_continent']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost Model
xgboost_tuned = run_xgboost_classifier(X_train, y_train, X_test, y_test, tune_hyperparams=True, n_trials=50, timeout=1800)
xgboost_model = xgboost_tuned['model']

# Run prediciton on the training dataset to get probabilities
prob_xgb = xgboost_model.predict_proba(X_train)

# TabPFN model
tabpfn_tuned = run_tabpfn_classifier(X_train,y_train,X_test,
                                     y_test,tune_hyperparams=True,
                                     device='cuda',
                                     custom_params={'max_time':500})
tabpfn_model = tabpfn_tuned['model']

# Run prediction on the training dataset to get the prbabilities
prob_tabpfn = tabpfn_model.predict_proba(X_train)

# Grownet model
grownet_model = run_grownet(X_train,y_train, X_test, y_test, tune_hyperparams=True,timeout=1200)
prob_grownet = grownet_model.predict(X_train)['probabilities']

# Neural Network
nn_model = run_nn_classifier(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    tune_hyperparams=True,n_trials=20,timeout=1000)

prob_nn = nn_model.predict(X_train)['probabilities']

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

train_meta_features = np.hstack([prob_xgb, prob_tabpfn, prob_grownet, prob_nn])

# Repeat for test set
proba_xgb_test = xgboost_model.predict_proba(X_test)
proba_tabpfn_test = tabpfn_model.predict_proba(X_test)
proba_grownet_test =  grownet_model.predict(X_test)['probabilities']
proba_nn_test = nn_model.predict(X_test)['probabilities']

test_meta_features = np.hstack([proba_xgb_test, proba_tabpfn_test, proba_grownet_test, proba_nn_test])


meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(train_meta_features, y_train)

meta_preds_test = meta_model.predict(test_meta_features)
meta_preds_train = meta_model.predict(train_meta_features)

print(classification_report(y_test, meta_preds_test, target_names=processed_data['continents']))
print(classification_report(y_train,meta_preds_train, target_names=processed_data['continents']))