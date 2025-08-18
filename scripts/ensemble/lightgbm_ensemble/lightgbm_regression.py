"""
LightGBM Regression Script for Ensemble Learning

This script provides a LightGBM-based regression pipeline for use in the ensemble learning framework.
It supports hyperparameter tuning via Optuna, training, and evaluation. The main functions and classes here
are imported and used by the main ensemble script (main.py) to provide LightGBM as one of the model options
for hierarchical coordinate regression tasks (e.g., latitude/longitude prediction).

Usage:
- Called by main.py for model selection, training, and prediction.
- Supports both default and tuned hyperparameters.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class LightGBMRegressorTuner:
    """
    Handles LightGBM hyperparameter tuning, training, and evaluation for regression tasks.
    Used by the ensemble pipeline.
    """
    def __init__(self, X_train, y_train, X_test, y_test,
                 random_state=42, n_trials=20, timeout=1200):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.final_model = None

    def default_params(self):
        # Returns default LightGBM parameters for regression
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
            'n_estimators': 300,
        }

    def objective(self, trial):
        # Optuna objective for hyperparameter search
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int("max_depth", 3, 12),
            'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            'n_estimators': trial.suggest_int("n_estimators", 100, 400),
        }

        model = lgb.LGBMRegressor(**params, random_state=self.random_state, verbose=-1)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring='neg_mean_absolute_error')
        return np.mean(scores)

    def tune(self):
        # Runs Optuna study to find best hyperparameters
        study = optuna.create_study(direction='maximize',
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt'
        })
        return self.best_params

    def train(self, params):
        # Trains LightGBMRegressor with given parameters
        model = lgb.LGBMRegressor(**params, random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        self.final_model = model
        return model

    def evaluate(self, model=None):
        # Evaluates model on test set and prints regression metrics
        if model is None:
            model = self.final_model
        preds = model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        print("\nRegression Report:")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        return preds, mae, r2


def run_lightgbm_regressor(X_train, y_train, X_test, y_test,
                          tune_hyperparams=False, random_state=42,
                          n_trials=20, timeout=1200, params=None, verbose=True):
    """
    LightGBM regression wrapper for ensemble with proper error handling.
    Called by main.py for hierarchical coordinate regression.
    """
    try:
        # Handle multi-dimensional targets
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            if verbose:
                print("Warning: LightGBM doesn't support multi-output regression natively. Using first dimension only.")
            y_train = y_train[:, 0]
            y_test = y_test[:, 0]

        tuner = LightGBMRegressorTuner(X_train, y_train, X_test, y_test,
                                      random_state=random_state, n_trials=n_trials, timeout=timeout)

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
        preds, mae, r2 = tuner.evaluate(model) if verbose else (model.predict(X_test), None, None)

        # Calculate additional metrics if not verbose
        if not verbose:
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

        return {
            'model': model,
            'predictions': preds,
            'mae': mae,
            'r2_score': r2,  # Use r2_score for consistency
            'params': best_params,
            'skipped': False
        }
        
    except Exception as e:
        if verbose:
            print(f"Error in LightGBM regressor: {e}")
        # Return dummy predictions on error
        n_samples = X_test.shape[0]
        dummy_preds = np.zeros(n_samples)
        
        return {
            'model': None,
            'predictions': dummy_preds,
            'mae': float('inf'),
            'r2_score': -float('inf'),
            'params': params,
            'skipped': True,
            'error': str(e)
        }

