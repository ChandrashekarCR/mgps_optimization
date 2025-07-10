# Import libraries
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier # Without hyperparameter tuning
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier # With hyperparameter tuning
from sklearn.metrics import classification_report, accuracy_score

class TabPFNModel:
    def __init__(self, X_train, y_train, X_test, y_test, device='cuda',
                 tune_hyperparams=False, custom_params=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = device
        self.tune_hyperparams = tune_hyperparams
        self.custom_params = custom_params or {}
        self.model = None
        self.predictions = None
        self.predicted_probabilities = None
        self.accuracy = None

    def train(self):
        print("=" * 60)
        print("TRAINING TABPFN MODEL")
        print("=" * 60)

        if self.tune_hyperparams:
            print("Using AutoTabPFNClassifier (with tuning)")
            self.model = AutoTabPFNClassifier(device=self.device, **self.custom_params)
        else:
            print("Using TabPFNClassifier (default, no tuning)")
            self.model = TabPFNClassifier(device=self.device, **self.custom_params)

        self.model.fit(self.X_train, self.y_train)
        print("Training complete.")
        return self.model

    def evaluate(self, model=None):
        if model is None:
            model = self.model

        print("Evaluating TabPFN model...")
        self.predictions = model.predict(self.X_test)
        
        try:
            self.predicted_probabilities = model.predict_proba(self.X_test)
        except:
            self.predicted_probabilities = None

        self.accuracy = accuracy_score(self.y_test, self.predictions)

        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions))
        print(f"\nAccuracy: {self.accuracy:.4f}")

        return self.predictions, self.predicted_probabilities, self.accuracy


def run_tabpfn_classifier(X_train, y_train, X_test, y_test,
                          tune_hyperparams=False, device='cuda',
                          custom_params=None):
    print("=" * 60)
    print("STARTING COMPLETE TABPFN PIPELINE")
    print("=" * 60)

    model_wrapper = TabPFNModel(X_train, y_train, X_test, y_test,
                                device=device,
                                tune_hyperparams=tune_hyperparams,
                                custom_params=custom_params)

    model = model_wrapper.train()
    preds, probs, acc = model_wrapper.evaluate(model)

    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return {
        'model': model,
        'predictions': preds,
        'predicted_probabilities': probs,
        'accuracy': acc
    }
