"""
TabPFN Classification Script for Ensemble Learning

This script provides a TabPFN-based classification pipeline for use in the ensemble learning framework.
It supports both regular and hyperparameter-tuned TabPFN classifiers, device checks, and error handling.
The main functions here are imported and used by the main ensemble script (main.py) to provide TabPFN
as one of the model options for hierarchical classification tasks (e.g., continent/city prediction).

Usage:
- Called by main.py for model selection, training, and prediction.
- Supports both default and tuned hyperparameters.
"""

# Import libraries
import numpy as np
import torch
import os  # Removed unnecessary comment
from tabpfn import TabPFNClassifier # Without hyperparameter tuning
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier # With hyperparameter tuning
from sklearn.metrics import classification_report, accuracy_score

def run_tabpfn_classifier(
    X_train, y_train, X_test, y_test,
    tune_hyperparams=False, max_time=300, params=None, random_state=42,
    verbose=False, max_time_options=None, **kwargs
):
    """
    Run TabPFN classifier with device and class count checks.
    Uses AutoTabPFNClassifier for hyperparameter tuning.
    """
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if params and 'device' in params:
        device = params['device']

    n_classes = len(np.unique(y_train))

    # Skip if too many classes
    if n_classes > 30:
        if verbose:
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

    # Skip if device is CPU
    if device == 'cpu':
        if verbose:
            print("TabPFNClassifier skipped: device is CPU (GPU required).")
        return {
            'model': None,
            'predictions': None,
            'predicted_probabilities': None,
            'accuracy': None,
            'params': params,
            'skipped': True,
            'reason': 'cpu_not_supported'
        }

    try:
        if tune_hyperparams:
            if verbose:
                print(f"Using AutoTabPFN for hyperparameter tuning with max_time=300...")
            model = AutoTabPFNClassifier(device=device, max_time=300)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            acc = accuracy_score(y_test, preds)
            if verbose:
                print(f"AutoTabPFN accuracy: {acc:.4f}")
                print("\nAutoTabPFN Classification Report:")
                print(classification_report(y_test, preds))
            return {
                'model': model,
                'predictions': preds,
                'predicted_probabilities': probs,
                'accuracy': acc,
                'params': {'max_time': 300, 'device': device}
            }

        # Regular TabPFN usage
        if verbose:
            print(f"Using TabPFN on device: {device}")
        model = TabPFNClassifier(device=device)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        if verbose:
            print("\nTabPFN Classification Report:")
            print(classification_report(y_test, preds))
            print(f"Accuracy: {acc:.4f}")
        return {
            'model': model,
            'predictions': preds,
            'predicted_probabilities': probs,
            'accuracy': acc,
            'params': params
        }

    except Exception as e:
        if verbose:
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




