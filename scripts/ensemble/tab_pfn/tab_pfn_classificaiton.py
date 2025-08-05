# Import libraries
import pandas as pd
import numpy as np
import torch
import os  # Added for environment variable
from tabpfn import TabPFNClassifier # Without hyperparameter tuning
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier # With hyperparameter tuning
from tabpfn_extensions.hpo import TunedTabPFNClassifier
from tabpfn_extensions.many_class.many_class_classifier import ManyClassClassifier
from sklearn.metrics import classification_report, accuracy_score

def run_tabpfn_classifier(X_train, y_train, X_test, y_test, tune_hyperparams=False, max_time=60, params=None, random_state=42):
    """
    Run TabPFN classifier with device and class count checks.
    """
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    device = 'cpu'
    if params and 'device' in params:
        device = params['device']
    elif torch.cuda.is_available():
        device = 'cuda'

    n_classes = len(np.unique(y_train))

    # Skip if device is CPU
    if device == 'cpu':
        print("TabPFNClassifier skipped: device is CPU.")
        return {
            'model': None,
            'predictions': None,
            'predicted_probabilities': None,
            'accuracy': None,
            'params': params,
            'skipped': True,
            'reason': 'cpu'
        }

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

    model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
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
        'params': params
    }

