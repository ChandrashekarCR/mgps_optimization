# Import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


class XGBoostTuner:
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
            'objective': 'multi:softprob',
            'num_class': len(np.unique(self.y_train)),
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'alpha': 0.0,
            'n_estimators': 300,
        }

    def objective(self, trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(self.y_train)),
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int("max_depth", 3, 12),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
            'gamma': trial.suggest_float("gamma", 0, 5),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'lambda': trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            'n_estimators': trial.suggest_int("n_estimators", 100, 400),
        }

        model = xgb.XGBClassifier(**params, random_state=self.random_state, verbosity=0, use_label_encoder=False)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='accuracy')
        return scores.mean()

    def tune(self):
        study = optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)) # Pruning helps the in stopping bad trials
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'multi:softprob',
            'num_class': len(np.unique(self.y_train)),
            'eval_metric': 'mlogloss',
            'tree_method': 'hist'
        })
        return self.best_params

    def train(self, params):
        model = xgb.XGBClassifier(**params, use_label_encoder=False)
        model.fit(self.X_train, self.y_train)
        self.final_model = model
        return model

    # Evaluate on the test dataset on how the model 
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



def run_xgboost_classifier(X_train, y_train, X_test, y_test, 
                           tune_hyperparams=False, random_state=42, 
                           n_trials=20, timeout=1200, custom_params=None):
    
    tuner = XGBoostTuner(X_train, y_train, X_test, y_test, 
                         random_state=random_state, n_trials=n_trials, timeout=timeout)

    if tune_hyperparams:
        best_params = tuner.tune()
        print("Using tuned parameters:", best_params)
    else:
        best_params = tuner.default_params()
        if custom_params:
            best_params.update(custom_params)
        print("Using default (or custom) parameters:", best_params)

    model = tuner.train(best_params)
    preds, probs, acc = tuner.evaluate(model)
    
    return {
        'model': model,
        'predictions': preds,
        'predicted_probabilities': probs,
        'accuracy': acc,
        'params': best_params
    }







