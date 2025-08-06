# Import libraries
import pandas as pd
import numpy as np
import torch
import os  # Added for environment variable
from tabpfn import TabPFNClassifier # Without hyperparameter tuning
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier # With hyperparameter tuning
from sklearn.metrics import classification_report, accuracy_score

# Try to import the client API for CPU usage
try:
    from tabpfn_client import TabPFNClassifier as TabPFNClientClassifier
    CLIENT_AVAILABLE = True
except ImportError:
    CLIENT_AVAILABLE = False
    print("Warning: tabpfn_client not available. Will skip TabPFN when no GPU is available.")

def run_tabpfn_classifier(X_train, y_train, X_test, y_test, tune_hyperparams=False, max_time=60, params=None, random_state=42):
    """
    Run TabPFN classifier with device and class count checks.
    Uses TabPFN client API when no GPU is available.
    Uses AutoTabPFNClassifier for hyperparameter tuning.
    """
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    
    # Determine device availability
    use_gpu = torch.cuda.is_available()
    if params and 'device' in params:
        use_gpu = params['device'] in ['cuda', 'gpu']
    
    n_classes = len(np.unique(y_train))

    # Skip if too many classes
    if n_classes > 30:
        print(f"TabPFNClassifier skipped: number of classes ({n_classes}) exceeds TabPFN's limit.")
        return {
            'model': None,
            'predictions': None,
            'predicted_probabilities': None,
            'accuracy': None,
            'params': params,
            'skipped': True,
            'reason': 'too_many_classes'
        }

    # Skip if no GPU and no client available
    if not use_gpu and not CLIENT_AVAILABLE:
        print("TabPFNClassifier skipped: no GPU available and tabpfn_client not installed.")
        return {
            'model': None,
            'predictions': None,
            'predicted_probabilities': None,
            'accuracy': None,
            'params': params,
            'skipped': True,
            'reason': 'no_gpu_no_client'
        }

    try:
        # Extract max_time from params if provided
        if params and 'max_time' in params:
            max_time = params['max_time']
        
        # Hyperparameter tuning with AutoTabPFNClassifier
        if tune_hyperparams:
            print(f"Using AutoTabPFN for hyperparameter tuning with max_time={max_time}...")
            
            # Use AutoTabPFNClassifier for hyperparameter tuning
            device = 'auto' if use_gpu else 'cpu'
            model = AutoTabPFNClassifier(device=device, max_time=max_time)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            acc = accuracy_score(y_test, preds)
            
            print(f"AutoTabPFN accuracy: {acc:.4f}")
            print("\nAutoTabPFN Classification Report:")
            print(classification_report(y_test, preds))
            
            return {
                'model': model,
                'predictions': preds,
                'predicted_probabilities': probs,
                'accuracy': acc,
                'params': {'max_time': max_time, 'device': device}
            }
        
        # Regular TabPFN usage (default parameters)
        if use_gpu:
            # Use regular TabPFN with GPU (no device parameter)
            model = TabPFNClassifier()
            print("Using TabPFN with GPU")
        else:
            # Use TabPFN client API for CPU
            model = TabPFNClientClassifier()
            print("Using TabPFN client API (CPU)")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        
        print("\nTabPFN Classification Report:")
        print(classification_report(y_test, preds))
        print(f"Accuracy: {acc:.4f}")

        return {
            'model': model,
            'predictions': preds,
            'predicted_probabilities': probs,
            'accuracy': acc,
            'params': params or {}
        }
        
    except Exception as e:
        print(f"Error running TabPFN: {e}")
        return {
            'model': None,
            'predictions': None,
            'predicted_probabilities': None,
            'accuracy': None,
            'params': params,
            'skipped': True,
            'reason': f'error: {str(e)}'
        }

