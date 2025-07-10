# This model is used for classification tasks using state-of-the art FT-Transformer Model
from tab_transformer_pytorch import FTTransformer

# Import libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns
import optuna

from sklearn import preprocessing
from sklearn.metrics import log_loss, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

import time
import warnings
warnings.filterwarnings('ignore')

def default_params():
    return {
        "dim": 32,
        "depth": 6,
        "heads": 8,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "epochs": 200,
        "early_stopping_steps": 30,
        "val_split": 0.2,
        "test_split": 0.2,
        "random_state": 42,
        "gradient_clip": 1.0,
        "scheduler_patience": 10,
        "scheduler_factor": 0.5
    }

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset class for classification
class TrainDataset(Dataset):
    def __init__(self, features, n_targets):
        self.features = features
        self.n_targets = n_targets

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float),
            'n_classes': torch.tensor(self.n_targets[idx], dtype=torch.long)
        }


class FTClassifier:
    def __init__(self, params, device="cpu"):

        self.params = params
        self.device = device
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.best_model_state = None


    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.params["val_split"],
                random_state=self.params["random_state"], stratify=y_train
            )

        # Update pararms with correct dimensions
        self.params['input_dim'] = X_train.shape[1]
        self.params['output_dim'] = len(np.unique(y_train))

        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_dataset = TrainDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=True)

        # Initialize model
        self.model = FTTransformer(
            categories=(),  # No categorical features
            num_continuous=self.params["input_dim"],
            dim=self.params["dim"],
            dim_out=self.params["output_dim"],
            depth=self.params["depth"],
            heads=self.params["heads"],
            attn_dropout=self.params["attn_dropout"],
            ff_dropout=self.params["ff_dropout"]
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.params["lr"], 
            weight_decay=self.params["weight_decay"]
        )
        
        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            patience=self.params["scheduler_patience"], 
            factor=self.params["scheduler_factor"]
        )

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(self.params["epochs"]):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for batch in train_loader:
                x_cont = batch['x']
                targets = batch['n_classes']
                x_cont, targets = x_cont.to(self.device), targets.to(self.device)

                # No categorical features
                x_categ = torch.zeros((x_cont.size(0), 0), dtype=torch.long, device=self.device)


                self.optimizer.zero_grad()
                preds = self.model(x_categ, x_cont)
                loss = self.criterion(preds, targets)
                loss.backward()

                # Gradient clipping
                clip_grad_norm_(self.model.parameters(),self.params['gradient_clip'])

                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(preds.data, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

            train_acc = correct / total

            val_metrics = self.evaluate(X_val, y_val)
            val_acc = val_metrics["accuracy"]

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.params["early_stopping_steps"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Loss = {total_loss/len(train_loader):.4f}")
        
        #Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Best validation accuracy: {best_val_acc:.4f}")
        

    def evaluate(self, X, y):
       
        self.model.eval()
        dataset = TrainDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=False)

        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for batch in loader:
                x_cont = batch['x']
                targets = batch['n_classes']
                x_cont, targets = x_cont.to(self.device), targets.to(self.device)
                
                # No categorical features
                x_categ = torch.empty((x_cont.size(0), 0), dtype=torch.long, device=self.device)

                preds = self.model(x_categ, x_cont)
                probs = F.softmax(preds, dim=1)
                _, predicted = torch.max(preds, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        return {
            "accuracy": acc,
            "predictions": all_preds,
            "targets": all_targets,
            "probabilities": np.array(all_probs)
        }

    def predict(self, X):
       
        self.model.eval()
        dummy_y = np.zeros(X.shape[0])
        dataset = TrainDataset(X, dummy_y)
        loader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=False)

        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                x_cont = batch['x']
                x_cont = x_cont.to(self.device)
                
                # No categorical features
                x_categ = torch.empty((x_cont.size(0), 0), dtype=torch.long, device=self.device)
                
                logits = self.model(x_categ, x_cont)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)
                all_probs.extend(probs)

        return {
            "predictions": all_preds,
            "probabilities": np.array(all_probs)
        }


class FTTransformerTuner:
    def __init__(self, X_train, y_train, X_val, y_val, params=None, device="cpu", n_trials=20, timeout=1200):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.params = params or default_params()
        self.device = device
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.best_score = None

    def objective(self, trial):
        params = self.params.copy()
        params.update({
            "dim": trial.suggest_categorical("dim", [32, 64, 128]),
            "depth": trial.suggest_int("depth", 2, 8),
            "heads": trial.suggest_categorical("heads", [4, 8, 16]),
            "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
            "ff_dropout": trial.suggest_float("ff_dropout", 0.0, 0.3),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "epochs": trial.suggest_int("epochs", 100, 300)
        })

        try:
            model = FTClassifier(params, device=self.device)
            model.fit(self.X_train, self.y_train, X_val=self.X_val, y_val=self.y_val)
            val_metrics = model.evaluate(self.X_val, self.y_val)
            return val_metrics["accuracy"]
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0

    def tune(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)

        self.best_params = study.best_params
        self.best_score = study.best_value
        print(f"Best Accuracy: {self.best_score:.4f}")
        print("Best Hyperparameters:", self.best_params)

        return self.best_params, self.best_score


def run_ft_transformer_classifier(X_train, y_train, X_test, y_test,
                                  device="cuda", tune_hyperparams=False,
                                  params=None, n_trials=20, timeout=1200):

    if params is None:
        params = default_params()
    else:
        default = default_params()
        default.update(params)
        params = default

    if tune_hyperparams:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        tuner = FTTransformerTuner(X_train_split, y_train_split, X_val, y_val, 
                                   params=params, device=device, n_trials=n_trials, timeout=timeout)
        best_params, _ = tuner.tune()
        params.update(best_params)

    print("Training final model with best parameters...")
    model = FTClassifier(params, device=device)
    model.fit(X_train, y_train)
    results = model.evaluate(X_test, y_test)

    print("\nClassification Report:")
    print(classification_report(results["targets"], results["predictions"]))
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    return model