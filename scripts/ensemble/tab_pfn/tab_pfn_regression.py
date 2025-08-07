# Import libraries
import pandas as pd
import numpy as np
import torch
import os  # Added for environment variable
from tabpfn import TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_tabpfn_regressor(X_train, y_train, X_test, y_test, tune_hyperparams=False, max_time=60, params=None):
    """
    Runs TabPFNRegressor models to predict x, y, z coordinates.
    Uses AutoTabPFNRegressor for hyperparameter tuning.
    Skips if device is CPU.
    """
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if params and 'device' in params:
        device = params['device']

    if device == 'cpu':
        print("TabPFNRegressor skipped: device is CPU.")
        # Ensure predictions and lat_lon_predictions are arrays of NaN with correct shapes
        n_samples = X_test.shape[0] if hasattr(X_test, "shape") and len(X_test.shape) > 0 else 0
        preds = np.full((n_samples, 3), np.nan)
        lat_lon_preds = np.full((n_samples, 2), np.nan)
        return {
            'models': None,
            'predictions': preds,
            'lat_lon_predictions': lat_lon_preds,
            'metrics': None,
            'skipped': True,
            'reason': 'cpu'
        }

    # Extract max_time from params if provided
    if params and 'max_time' in params:
        max_time = params['max_time']

    coord_names = ['x', 'y', 'z']
    models = {}
    preds = []
    metrics = {}

    try:
        for i, coord in enumerate(coord_names):
            print(f"\n----- Predicting {coord.upper()} -----")
            
            if tune_hyperparams:
                print(f"Using AutoTabPFN for hyperparameter tuning with max_time={max_time}...")
                model = AutoTabPFNRegressor(device=device, max_time=max_time)
            else:
                model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
                
            model.fit(X_train, y_train[:, i])
            y_pred = model.predict(X_test)
            preds.append(y_pred)

            mse = mean_squared_error(y_test[:, i], y_pred)
            mae = mean_absolute_error(y_test[:, i], y_pred)
            r2 = r2_score(y_test[:, i], y_pred)

            print(f"{coord.upper()} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            metrics[coord] = {'mse': mse, 'mae': mae, 'r2': r2}
            models[coord] = model

        preds = np.stack(preds, axis=1)  # Shape: [n_samples, 3]
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
        
    except Exception as e:
        print(f"Error running TabPFNRegressor: {e}")
        n_samples = X_test.shape[0] if hasattr(X_test, "shape") and len(X_test.shape) > 0 else 0
        preds = np.full((n_samples, 3), np.nan)
        lat_lon_preds = np.full((n_samples, 2), np.nan)
        return {
            'models': None,
            'predictions': preds,
            'lat_lon_predictions': lat_lon_preds,
            'metrics': None,
            'skipped': True,
            'reason': f'error: {str(e)}'
        }

