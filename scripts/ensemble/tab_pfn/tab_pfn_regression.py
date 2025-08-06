# Import libraries
import pandas as pd
import numpy as np
import torch
import os  # Added for environment variable
from tabpfn import TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor  # With hyperparameter tuning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try to import the client API for CPU usage
try:
    from tabpfn_client import TabPFNRegressor as TabPFNClientRegressor
    CLIENT_AVAILABLE = True
except ImportError:
    CLIENT_AVAILABLE = False
    print("Warning: tabpfn_client not available. Will skip TabPFN when no GPU is available.")

def run_tabpfn_regressor(X_train, y_train, X_test, y_test, tune_hyperparams=False, 
                        max_time=60, params=None, random_state=42, n_trials=20, verbose=True):
    """
    Runs TabPFNRegressor models to predict x, y, z coordinates.
    Uses TabPFN client API when no GPU is available.
    Uses AutoTabPFNRegressor for hyperparameter tuning.
    """
    try:
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
        
        # Determine device
        use_gpu = torch.cuda.is_available()
        if params and 'device' in params:
            use_gpu = params['device'] in ['cuda', 'gpu']

        # Skip if no GPU and no client available
        if not use_gpu and not CLIENT_AVAILABLE:
            if verbose:
                print("TabPFNRegressor skipped: no GPU available and tabpfn_client not installed.")
            n_samples = X_test.shape[0] if hasattr(X_test, "shape") and len(X_test.shape) > 0 else 0
            output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
            preds = np.full((n_samples, output_dim), np.nan)
            
            return {
                'model': None,
                'predictions': preds,
                'r2_score': -float('inf'),
                'skipped': True,
                'reason': 'no_gpu_no_client',
                'params': params
            }

        # Handle both 1D and multi-dimensional targets
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

        coord_names = ['x', 'y', 'z'][:y_train.shape[1]]  # Adjust based on actual dimensions
        models = {}
        preds = []
        metrics = {}

        for i, coord in enumerate(coord_names):
            if verbose:
                print(f"\n----- Predicting {coord.upper()} -----")
            
            # Hyperparameter tuning with AutoTabPFNRegressor
            if tune_hyperparams and params is None:
                if verbose:
                    print(f"Tuning TabPFN hyperparameters for {coord} using AutoTabPFNRegressor...")
                
                device = 'auto' if use_gpu else 'cpu'
                auto_model = AutoTabPFNRegressor(device=device, max_time=max_time)
                auto_model.fit(X_train, y_train[:, i])
                y_pred = auto_model.predict(X_test)
                model = auto_model
                if verbose:
                    print(f"Using AutoTabPFNRegressor for {coord}")
                
            else:
                # Regular TabPFN usage
                if use_gpu:
                    # Use regular TabPFN with GPU
                    model_params = {'device': 'cuda', 'ignore_pretraining_limits': True}
                    if params and 'max_time' in params:
                        model_params['max_time'] = params['max_time']
                    else:
                        model_params['max_time'] = max_time
                        
                    model = TabPFNRegressor(**model_params)
                    if verbose:
                        print(f"Using TabPFN with GPU for {coord}")
                else:
                    # Use TabPFN client API for CPU
                    model_params = {'device': 'cpu'}
                    if params and 'N_ensemble_configurations' in params:
                        model_params['N_ensemble_configurations'] = params['N_ensemble_configurations']
                    else:
                        model_params['N_ensemble_configurations'] = 10
                        
                    model = TabPFNClientRegressor(**model_params)
                    if verbose:
                        print(f"Using TabPFN client API (CPU) for {coord}")

                model.fit(X_train, y_train[:, i])
                y_pred = model.predict(X_test)

            preds.append(y_pred)

            mse = mean_squared_error(y_test[:, i], y_pred)
            mae = mean_absolute_error(y_test[:, i], y_pred)
            r2 = r2_score(y_test[:, i], y_pred)

            if verbose:
                print(f"{coord.upper()} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            metrics[coord] = {'mse': mse, 'mae': mae, 'r2': r2}
            models[coord] = model

        preds = np.stack(preds, axis=1)  # Shape: [n_samples, n_coords]
        
        # Calculate overall R2 score
        overall_r2 = r2_score(y_test, preds)

        return {
            'model': models,
            'predictions': preds,
            'r2_score': overall_r2,
            'metrics': metrics,
            'params': params,
            'skipped': False
        }
        
    except Exception as e:
        if verbose:
            print(f"Error running TabPFN: {e}")
        n_samples = X_test.shape[0] if hasattr(X_test, "shape") and len(X_test.shape) > 0 else 0
        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
        preds = np.full((n_samples, output_dim), np.nan)
        
        return {
            'model': None,
            'predictions': preds,
            'r2_score': -float('inf'),
            'skipped': True,
            'reason': f'error: {str(e)}',
            'params': params,
            'error': str(e)
        }

