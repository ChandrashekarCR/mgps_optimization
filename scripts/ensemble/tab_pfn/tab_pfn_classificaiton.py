# Import libraries
import pandas as pd
import numpy as np
import torch
from tabpfn import TabPFNClassifier # Without hyperparameter tuning
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier # With hyperparameter tuning
from tabpfn_extensions.hpo import TunedTabPFNClassifier
from tabpfn_extensions.many_class.many_class_classifier import ManyClassClassifier
from sklearn.metrics import classification_report, accuracy_score

def run_tabpfn_classifier(X_train, y_train, X_test, y_test, tune_hyperparams=False, max_time=60, params=None, random_state=42):
    """
    Run TabPFN classifier with proper error handling and device management.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        tune_hyperparams: Whether to tune hyperparameters
        max_time: Maximum time for hyperparameter tuning
        params: Dictionary of parameters
        random_state: Random state for reproducibility
    
    Returns:
        Dictionary containing model, predictions, probabilities, accuracy, and parameters
    """
    
    # Determine device
    device = 'cpu'  # Default to CPU for stability
    if params and 'device' in params:
        device = params['device']
    elif torch.cuda.is_available():
        device = 'cuda'
    
    print(f"Using device: {device}")
    
    # Count unique classes
    n_classes = len(np.unique(y_train))
    print(f"Number of classes: {n_classes}")
    
    try:
        if n_classes > 30:
            print(f"Number of classes = {n_classes} exceeds TabPFN's limit.")
            
            # Try using ManyClassClassifier with proper initialization
            try:
                print("Attempting ManyClassClassifier...")
                
                # Create base TabPFN model
                base_clf = TabPFNClassifier(device=device)
                
                # Create ManyClassClassifier with minimal parameters
                model = ManyClassClassifier(
                    estimator=base_clf,
                    alphabet_size=10,  # Fixed size
                    n_estimators_redundancy=2,  # Reduced redundancy
                    random_state=random_state
                )
                
                # Fit the model
                print("Fitting ManyClassClassifier...")
                model.fit(X_train, y_train)
                
                # Try direct prediction
                print("Making predictions with ManyClassClassifier...")
                preds = model.predict(X_test)
                
                # Try to get probabilities
                try:
                    probs = model.predict_proba(X_test)
                except:
                    print("Warning: predict_proba failed, using dummy probabilities")
                    probs = np.zeros((len(X_test), n_classes))
                    for i, pred in enumerate(preds):
                        probs[i, pred] = 1.0
                    
            except Exception as many_class_error:
                print(f"ManyClassClassifier failed: {many_class_error}")
                print("Falling back to alternative approach for many classes...")
                
                # Alternative 1: Try AutoTabPFNClassifier
                try:
                    print("Trying AutoTabPFNClassifier...")
                    model = AutoTabPFNClassifier(device=device)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    probs = model.predict_proba(X_test)
                    
                except Exception as auto_error:
                    print(f"AutoTabPFNClassifier failed: {auto_error}")
                    
                    # Alternative 2: Use regular TabPFN with reduced classes
                    print("Using regular TabPFN with class reduction...")
                    
                    # Map classes to fewer classes (simple approach)
                    unique_classes = np.unique(y_train)
                    if len(unique_classes) > 10:
                        # Group classes into 10 groups
                        class_groups = {}
                        group_size = len(unique_classes) // 10
                        for i, cls in enumerate(unique_classes):
                            group_id = min(i // group_size, 9)
                            class_groups[cls] = group_id
                        
                        # Map training labels
                        y_train_mapped = np.array([class_groups[cls] for cls in y_train])
                        y_test_mapped = np.array([class_groups[cls] for cls in y_test])
                        
                        # Train on mapped labels
                        model = TabPFNClassifier(device=device)
                        model.fit(X_train, y_train_mapped)
                        
                        # Predict on mapped labels
                        preds_mapped = model.predict(X_test)
                        probs_mapped = model.predict_proba(X_test)
                        
                        # Map back to original classes (approximate)
                        reverse_mapping = {v: k for k, v in class_groups.items()}
                        preds = np.array([reverse_mapping[pred] for pred in preds_mapped])
                        
                        # Create dummy probabilities for original classes
                        probs = np.zeros((len(X_test), n_classes))
                        for i, pred in enumerate(preds):
                            class_idx = np.where(unique_classes == pred)[0][0]
                            probs[i, class_idx] = 1.0
                    else:
                        # Should not reach here, but fallback
                        model = TabPFNClassifier(device=device)
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        probs = model.predict_proba(X_test)
                        
        elif tune_hyperparams:
            print("Running Optuna tuning using TunedTabPFNClassifier...")
            
            # For hyperparameter tuning, use TunedTabPFNClassifier
            model = TunedTabPFNClassifier(
                n_trials=50,
                timeout=max_time,
                metric='accuracy',
                random_state=random_state,
                device=device  # Pass device to tuned classifier
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            
        else:
            # Standard TabPFN classifier
            print("Using standard TabPFNClassifier...")
            model = TabPFNClassifier(device=device)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, preds)
        
        # Print classification report
        print("\nTabPFN Classification Report:")
        print(classification_report(y_test, preds))
        print(f"Accuracy: {acc:.4f}")
        
        # Get best parameters if tuning was used
        best_params = {}
        if tune_hyperparams and hasattr(model, 'best_params_'):
            best_params = model.best_params_
            print("Best tuned parameters:", best_params)
        elif params:
            best_params = params
        
        return {
            'model': model,
            'predictions': preds,
            'predicted_probabilities': probs,
            'accuracy': acc,
            'params': best_params
                            }
        
    except Exception as e:
        print(f"Error in run_tabpfn_classifier: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Final fallback: Use a simple approach
        print("Using final fallback approach...")
        try:
            # Just use regular TabPFN with reduced data or classes
            if n_classes > 10:
                print("Too many classes for TabPFN. Consider using a different classifier.")
                # You could return None or use a different classifier here
                from sklearn.ensemble import RandomForestClassifier
                print("Using RandomForest as fallback...")
                model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)
                
                acc = accuracy_score(y_test, preds)
                return {
                    'model': model,
                    'predictions': preds,
                    'predicted_probabilities': probs,
                    'accuracy': acc,
                    'params': {},
                    'n_classes': n_classes,
                    'fallback_used': True
                }
            else:
                # Regular TabPFN should work
                model = TabPFNClassifier(device='cpu')
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)
                
                acc = accuracy_score(y_test, preds)
                return {
                    'model': model,
                    'predictions': preds,
                    'predicted_probabilities': probs,
                    'accuracy': acc,
                    'params': {},
                    'n_classes': n_classes
                }
        except:
            # If everything fails, re-raise original error
            raise e

