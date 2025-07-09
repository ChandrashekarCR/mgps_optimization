# The main ensemble model

# Import libraries 

# Import data processing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


import matplotlib.pyplot as plt
import seaborn as sns

# Import other models
from xgboost_ensemble.xgboost_classification import XGBoostTuner
from random_forest.randomforest_classification import RandomForestTuner
from tab_pfn.tab_pfn_classificaiton import TabPFNModel
from grownet.grownet_classification import GrowNetClassifier, GrowNetTuner, run_grownet




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

## Initialize XGBoost tuner
#xgb_tuner = XGBoostTuner(
#        X=X, y=y,
#        X_train=X_train, y_train=y_train,
#        X_test=X_test, y_test=y_test,
#        random_state=42,
#        n_trial=20,  # Increase for better results
#        timeout=1200  # 30 minutes timeout
#    )
#    
## Run complete pipeline
#xgb_results = xgb_tuner.run_complete_pipeline()
#
#
## Initialize Random Forest tuner
#rf_tuner = RandomForestTuner(
#        X=X, y=y,
#        X_train=X_train, y_train=y_train,
#        X_test=X_test, y_test=y_test,
#        random_state=42,
#        n_trial=30,
#        timeout=1000
#    )
#    
### Run complete pipeline
#rf_results = rf_tuner.run_complete_pipeline()
#
#print(xgb_results['accuracy'])
#print(rf_results['accuracy'])


# Run TabPFN pipeline
#tab_model = TabPFNModel(X_train, y_train, X_test, y_test, device='cuda')
#tab_results = tab_model.run_complete_pipeline()
#
#print(tab_results)



# Grownet model
model, grownet_results = run_grownet(X_train,y_train, X_test, y_test, tune_hyperparams=True,timeout=500)

print(grownet_results)

#grownet = GrowNetClassifier(device='cuda')
#grownet.fit(X_train, y_train, X_val=None, y_val=None)  # Will split val internally if not provided
#
## Predict and evaluate
#y_pred = grownet.predict(X_test)
#
#print(y_pred)

#print("\nClassification Report")
#print(classification_report(metrics['targets'], metrics['predictions']))