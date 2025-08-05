import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class CatBoostRegressorTuner:
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
        return {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_strength': 1,
            'bagging_temperature': 1,
            'border_count': 254,
            'iterations': 300,
        }

    def objective(self, trial):
        params = {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'depth': trial.suggest_int("depth", 3, 10),
            'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1, 10),
            'random_strength': trial.suggest_float("random_strength", 1e-9, 10, log=True),
            'bagging_temperature': trial.suggest_float("bagging_temperature", 0, 10),
            'border_count': trial.suggest_int("border_count", 1, 255),
            'iterations': trial.suggest_int("iterations", 100, 500),
        }

        model = CatBoostRegressor(**params, random_seed=self.random_state, verbose=False)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring='neg_mean_absolute_error')
        return np.mean(scores)

    def tune(self):
        study = optuna.create_study(direction='maximize',
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = study.best_params
        self.best_params.update({
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
        })
        return self.best_params

    def train(self, params):
        model = CatBoostRegressor(**params, random_seed=self.random_state, verbose=False)
        model.fit(self.X_train, self.y_train)
        self.final_model = model
        return model

    def evaluate(self, model=None):
        if model is None:
            model = self.final_model
        preds = model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        print("\nRegression Report:")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")
        return preds, mae, r2


def run_catboost_regressor(X_train, y_train, X_test, y_test,
                           tune_hyperparams=False, random_state=42,
                           n_trials=20, timeout=1200, params=None):

    tuner = CatBoostRegressorTuner(X_train, y_train, X_test, y_test,
                                   random_state=random_state, n_trials=n_trials, timeout=timeout)

    if tune_hyperparams:
        best_params = tuner.tune()
        print("Using tuned parameters:", best_params)
    else:
        best_params = tuner.default_params()
        if params:
            best_params.update(params)
        print("Using default (or custom) parameters:", best_params)

    model = tuner.train(best_params)
    preds, mae, r2 = tuner.evaluate(model)

    return {
        'model': model,
        'predictions': preds,
        'mae': mae,
        'r2': r2,
        'params': best_params
    }

