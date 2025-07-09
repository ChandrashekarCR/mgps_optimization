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
    def __init__(self, X, y, X_train, y_train, X_test, y_test, random_state:int=42,n_trial:int=20,timeout:int=1200):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.n_trial = n_trial
        self.timeout = timeout
        self.best_params = None
        self.final_model = None

    def objective(self, trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(self.y)),
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',  # faster, for CPUs
            'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int("max_depth", 3, 12),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
            'gamma': trial.suggest_float("gamma", 0, 5),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
            'lambda': trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
        }

        model = xgb.XGBClassifier(**params, random_state=self.random_state, verbosity=0, use_label_encoder=False)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='accuracy')
        return scores.mean()

    def get_best_parameters(self):
        """Run the tuning and return best parameters"""
        print(f"Starting hyperparameter tuning with {self.n_trial} trials...")
        
        # Run the tuning
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trial, timeout=self.timeout)


        # Get the best parameters
        best_params = study.best_params
        best_params.update({
            'objective':'multi:softprob',
            'num_class':len(np.unique(self.y)),
            'eval_metric':'mlogloss',
            'tree_method':'hist'
        })

        self.best_params = best_params
        print(f"Best score: {study.best_value:.4f}")
        print(f"Best parameters: {best_params}")

        return best_params
    
    def train_final_model(self,best_params=None):

        """Train the final model with best parameters"""
        if best_params is None:
            if self.best_params is None:
                self.get_best_parameters()
            best_params = self.best_params
        
        print("Training final model with best parameters...")
        self.final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False)
        self.final_model.fit(self.X_train, self.y_train)
        return self.final_model

    def evaluate_final_model(self,final_model=None):
        """Evaluate the final model"""
        if final_model is None:
            if self.final_model is None:
                self.train_final_model()
            final_model = self.final_model
        
        print("Evaluating final model...")
        y_pred = final_model.predict(self.X_test)
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        return y_pred, accuracy
    
    def run_complete_pipeline(self):
        """Run the complete pipeline: tune -> train -> evaluate"""
        print("=" * 60)
        print("STARTING COMPLETE XGBOOST PIPELINE")
        print("=" * 60)
        
        # Step 1: Hyperparameter tuning
        best_params = self.get_best_parameters()
        
        # Step 2: Train final model
        final_model = self.train_final_model(best_params)
        
        # Step 3: Evaluate model
        y_pred, accuracy = self.evaluate_final_model(final_model)
        
        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return {
            'best_params': best_params,
            'final_model': final_model,
            'predictions': y_pred,
            'accuracy': accuracy,
        }







