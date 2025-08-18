"""
CatBoost Classification Script for Ensemble Learning

This script provides a CatBoost-based classification pipeline for use in the ensemble learning framework.
It supports hyperparameter tuning via Optuna, training, and evaluation. The main functions and classes here
are imported and used by the main ensemble script (main.py) to provide CatBoost as one of the model options
for hierarchical classification tasks (e.g., continent/city prediction).

Usage:
- Called by main.py for model selection, training, and prediction.
- Supports both default and tuned hyperparameters.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class CatBoostClassifierOptimizer:
    """
    Handles CatBoost hyperparameter tuning, training, and evaluation for classification tasks.
    Used by the ensemble pipeline.
    """
    def __init__(self, X_train, y_train, X_test, y_test, random_state=42, n_trials=20, timeout=1200, cat_features=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.cat_features = cat_features
        self.best_params = None
        self.final_model = None

    def default_params(self):
        # Returns default CatBoost parameters for classification
        return {
            'loss_function': 'MultiClass',
            'iterations': 300,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'random_seed': self.random_state,
            'verbose': False
        }

    def objective(self, trial):
        # Optuna objective for hyperparameter search
        params = {
            'loss_function': 'MultiClass',
            'iterations': trial.suggest_int('iterations', 100, 400),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10.0),
            'random_seed': self.random_state,
            'verbose': False
        }
        model = CatBoostClassifier(**params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        # For categorical features, use .fit(X, y, cat_features=cat_features)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='accuracy', fit_params={'cat_features': self.cat_features})
        return scores.mean()

    def tune(self):
        # Runs Optuna study to find best hyperparameters
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = study.best_params
        self.best_params.update({
            'loss_function': 'MultiClass',
            'random_seed': self.random_state,
            'verbose': False
        })
        return self.best_params

    def train(self, params):
        # Trains CatBoostClassifier with given parameters
        model = CatBoostClassifier(**params)
        model.fit(self.X_train, self.y_train, cat_features=self.cat_features)
        self.final_model = model
        return model

    def evaluate(self, model=None):
        # Evaluates model on test set and prints classification report
        if model is None:
            model = self.final_model
        preds = model.predict(self.X_test)
        probs = model.predict_proba(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print("\nClassification Report:")
        print(classification_report(self.y_test, preds))
        print(f"\nAccuracy: {acc:.4f}")
        return preds, probs, acc


def run_catboost_classifier(X_train, y_train, X_test, y_test, 
                           tune_hyperparams=False, random_state=42, 
                           n_trials=20, timeout=1200, params=None, verbose=False):
    """
    CatBoost classification wrapper for ensemble.
    Called by main.py for hierarchical classification.
    """
    tuner = CatBoostClassifierOptimizer(X_train, y_train, X_test, y_test, 
                         random_state=random_state, n_trials=n_trials,timeout=timeout,cat_features=None)

    if tune_hyperparams:
        best_params = tuner.tune()
        if verbose:
            print("Using tuned parameters:", best_params)
    else:
        best_params = tuner.default_params()
        if params:
            best_params.update(params)
        if verbose:
            print("Using default (or custom) parameters:", best_params)

    model = tuner.train(best_params)
    preds, probs, acc = tuner.evaluate(model) if verbose else (model.predict(X_test), model.predict_proba(X_test), accuracy_score(y_test, model.predict(X_test)))
    
    if verbose:
        return {
            'model': model,
            'predictions': preds,
            'predicted_probabilities': probs,
            'accuracy': acc,
            'params': best_params
        }
    else:
        return {
            'model': model,
            'predictions': preds,
            'predicted_probabilities': probs,
            'accuracy': acc,
            'params': best_params
        }