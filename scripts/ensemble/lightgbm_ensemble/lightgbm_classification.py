# Import libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class LightGBMTuner:
    def __init__(self, X_train, y_train, X_test, y_test, random_state=42, n_trials=20, timeout=1200):
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
        return {
            'objective': 'multiclass',
            'num_class': len(np.unique(self.y_train)),
            'metric': 'multi_logloss',
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 300,
            'random_state': self.random_state,
            'min_gain_to_split': 1e-3,  # Suppress split gain warning
            'verbose': -1  # Suppress LightGBM output
        }

    def objective(self, trial):
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(self.y_train)),
            'metric': 'multi_logloss',
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int("max_depth", 3, 12),
            'num_leaves': trial.suggest_int("num_leaves", 15, 256),
            'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            'reg_alpha': trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            'n_estimators': trial.suggest_int("n_estimators", 100, 400),
            'random_state': self.random_state,
            'min_gain_to_split': 1e-3,  # Suppress split gain warning
            'verbose': -1  # Suppress LightGBM output
        }
        model = lgb.LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='accuracy')
        return scores.mean()

    def tune(self):
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'multiclass',
            'num_class': len(np.unique(self.y_train)),
            'metric': 'multi_logloss',
            'random_state': self.random_state,
            'min_gain_to_split': 1e-3,  # Suppress split gain warning
            'verbose': -1  # Suppress LightGBM output
        })
        return self.best_params

    def train(self, params):
        model = lgb.LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)
        self.final_model = model
        return model

    def evaluate(self, model=None):
        if model is None:
            model = self.final_model
        preds = model.predict(self.X_test)
        probs = model.predict_proba(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print("\nClassification Report:")
        print(classification_report(self.y_test, preds))
        print(f"\nAccuracy: {acc:.4f}")
        return preds, probs, acc



def run_lightgbm_classifier(X_train, y_train, X_test, y_test, 
                            tune_hyperparams=False, random_state=42,
                            n_trials=20, timeout=1200, params=None, verbose=False):
    tuner = LightGBMTuner(X_train, y_train, X_test, y_test, 
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
