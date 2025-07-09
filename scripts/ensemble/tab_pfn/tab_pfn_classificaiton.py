# Import libraries
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier # Without hyperparameter tuning
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier # With hyperparameter tuning
from sklearn.metrics import classification_report, accuracy_score

class TabPFNModel:
    def __init__(self, X_train, y_train, X_test, y_test, device='cuda'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = device
        self.model = None
        self.predictions = None
        self.accuracy = None

    def train_model(self):
        print("=" * 60)
        print("TRAINING TABPFN MODEL")
        print("=" * 60)

        self.model = AutoTabPFNClassifier(device=self.device, max_time=120) # Adjust the max_time to get a better prediction.
        self.model.fit(self.X_train, self.y_train)
        print("Training complete.")

        return self.model

    def evaluate_model(self, model=None):
        if model is None:
            model = self.model

        print("Evaluating TabPFN model...")
        self.predictions = model.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.predictions)

        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions))
        print(f"\nAccuracy: {self.accuracy:.4f}")

        return self.predictions, self.accuracy

    def run_complete_pipeline(self):
        print("=" * 60)
        print("STARTING COMPLETE TABPFN PIPELINE")
        print("=" * 60)

        self.train_model()
        preds, acc = self.evaluate_model()

        print("=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return {
            'final_model': self.model,
            'predictions': preds,
            'accuracy': acc
        }
