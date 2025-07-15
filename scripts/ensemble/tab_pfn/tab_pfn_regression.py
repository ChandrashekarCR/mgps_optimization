# Import libraries
import pandas as pd
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_tabpfn_regressor(X_train, y_train, X_test, y_test, tune_hyperparams=False, params=None):
    """
    Runs three separate TabPFNRegressor models to predict x, y, z coordinates.

    Args:
        X_train: Training features
        y_train: Training labels (shape: [n_samples, 3])
        X_test: Test features
        y_test: Test labels (shape: [n_samples, 3])
        params: Optional parameters (like device)
        
    Returns:
        Dictionary with predictions and evaluation metrics
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if params and 'device' in params:
        device = params['device']
    
    print(f"Using device: {device}")
    
    coord_names = ['x', 'y', 'z']
    models = {}
    preds = []
    metrics = {}
    
    for i, coord in enumerate(coord_names):
        print(f"\n----- Predicting {coord.upper()} -----")
        model = TabPFNRegressor(device=device)
        model.fit(X_train, y_train[:, i])
        y_pred = model.predict(X_test)
        preds.append(y_pred)

        mse = mean_squared_error(y_test[:, i], y_pred)
        mae = mean_absolute_error(y_test[:, i], y_pred)
        r2 = r2_score(y_test[:, i], y_pred)

        print(f"{coord.upper()} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        metrics[coord] = {'mse': mse, 'mae': mae, 'r2': r2}
        models[coord] = model
    
    preds = np.stack(preds, axis=1)  # Shape: [n_samples, 3]

    # Optional: convert xyz back to lat/lon
    lat_pred_rad = np.arcsin(preds[:, 2])
    lon_pred_rad = np.arctan2(preds[:, 1], preds[:, 0])
    lat_pred_deg = np.degrees(lat_pred_rad)
    lon_pred_deg = np.degrees(lon_pred_rad)

    return {
        'models': models,
        'predictions': preds,
        'lat_lon_predictions': np.stack([lat_pred_deg, lon_pred_deg], axis=1),
        'metrics': metrics
    }

