# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class RandomForestTuner:
    def __init__(self, X, y, X_train, y_train, X_test, y_test, random_state=42, n_trial=20, timeout=1200):
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
            'n_estimators': trial.suggest_int("n_estimators", 50, 500),
            'max_depth': trial.suggest_int("max_depth", 3, 20),
            'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
            'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10),
            'max_features': trial.suggest_categorical("max_features", ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical("bootstrap", [True, False]),
            'criterion': trial.suggest_categorical("criterion", ['gini', 'entropy']),
            'min_impurity_decrease': trial.suggest_float("min_impurity_decrease", 0.0, 0.1),
            'max_leaf_nodes': trial.suggest_int("max_leaf_nodes", 10, 1000),
            'ccp_alpha': trial.suggest_float("ccp_alpha", 0.0, 0.1),
        }
        
        # If bootstrap is False, we need to handle max_samples
        if not params['bootstrap']:
            # Remove max_samples when bootstrap=False as it's not compatible
            pass
        else:
            params['max_samples'] = trial.suggest_float("max_samples", 0.5, 1.0)

        model = RandomForestClassifier(**params, random_state=self.random_state, n_jobs=-1)
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
        
        # Add fixed parameters
        best_params.update({
            'random_state': self.random_state,
            'n_jobs': 4  # Use all available cores
        })

        self.best_params = best_params
        print(f"Best score: {study.best_value:.4f}")
        print(f"Best parameters: {best_params}")

        return best_params
    
    def train_final_model(self, best_params=None):
        """Train the final model with best parameters"""
        if best_params is None:
            if self.best_params is None:
                self.get_best_parameters()
            best_params = self.best_params
        
        print("Training final model with best parameters...")
        self.final_model = RandomForestClassifier(**best_params)
        self.final_model.fit(self.X_train, self.y_train)
        return self.final_model

    def evaluate_final_model(self, final_model=None):
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
        print("STARTING COMPLETE RANDOM FOREST PIPELINE")
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
            'accuracy': accuracy
                }

