"""
GrowNet Classification Script for Ensemble Learning

This script implements the GrowNet boosting-based neural network for classification tasks (e.g., continent/city prediction)
as part of the ensemble learning pipeline. It supports hyperparameter tuning via Optuna, training, and evaluation.
The main functions and classes here are imported and used by the main ensemble script (main.py) to provide GrowNet
as one of the model options for hierarchical classification.

Usage:
- Called by main.py for model selection, training, and prediction.
- Supports both default and tuned hyperparameters.
"""

# This model is used for classification tasks

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

def grownet_classification_default_params():
    # Returns default parameters for GrowNet classification
    return {
        "hidden_size": 256,
        "num_nets": 10,
        "boost_rate": 0.4,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "epochs_per_stage": 30,
        "early_stopping_steps": 7,
        "gradient_clip": 1.0,
        "val_split": 0.2,
        "test_split": 0.2,
        "random_state": 42,
    }


class GrowNetClassificationTuner:
    """
    Handles GrowNet hyperparameter tuning for classification tasks using Optuna.
    """
    def __init__(self, X_train, y_train, X_val, y_val, params, device="cpu",n_trials = 20, timeout=1200):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.params = params
        self.device = device
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.best_score = None 

    def objective(self,trial):
        # Optuna objective for hyperparameter search (maximize accuracy)
        # Suggest hyperparameters
        params = self.params.copy()
        params.update({
            "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512]),
            "num_nets": trial.suggest_int("num_nets", 10, 30),
            "boost_rate": trial.suggest_float("boost_rate", 0.1, 0.8),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
            "epochs_per_stage": trial.suggest_int("epochs_per_stage", 5, 10),
            "gradient_clip": trial.suggest_float("gradient_clip", 0.5, 2.0),
        })

        # Train model
        model = GrowNetClassifierUnique(params, device=self.device)
        model.fit(self.X_train, self.y_train, X_val=self.X_val, y_val=self.y_val)
        val_metrics = model.evaluate(self.X_val, self.y_val)
        val_acc = val_metrics['class_accuracy']
        return val_acc  # maximize accuracy

    def tune(self):
        # Runs Optuna study to find best hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score


# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset class for continent/city classification
class GrowNetClassificationTrainDataset(Dataset):
    """
    PyTorch Dataset for GrowNet classification training.
    """
    def __init__(self, features, n_targets):
        self.features = features
        self.n_targets = n_targets

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float),
            'n_classes': torch.tensor(self.n_targets[idx], dtype=torch.float)
        }

    
# DynamicNet for classification (ensemble of weak learners)
class GrowNetClassificationDynamicNet(nn.Module):
    """
    Ensemble model for GrowNet classification.
    """
    def __init__(self, c0_classes, lr):
        super(GrowNetClassificationDynamicNet,self).__init__()
        self.models = []
        self.c0_classes = c0_classes
        self.lr = lr

        self.boost_rate = nn.Parameter(torch.tensor(lr,requires_grad=True,device=device))

    def to(self,device):
        self.c0_classes = self.c0_classes.to(device)
        self.boost_rate = self.boost_rate.to(device)
        for m in self.models:
            m.to(device)
    
    def add(self,model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())
        params.append(self.boost_rate)
        return params
    
    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_eval(self):
        for m in self.models:
            m.eval()
    
    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self,x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0_classes = self.c0_classes.repeat(batch,1)
            return None, c0_classes
        
        middle_feat_cum = None
        classes_pred = None

        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, classes_out = m(x,middle_feat_cum)
                    classes_pred = classes_out
                else:
                    middle_feat_cum, classes_out = m(x,middle_feat_cum)
                    classes_pred += classes_out

        final_classes = self.c0_classes + self.boost_rate * classes_pred
        return middle_feat_cum, final_classes
    
    def forward_grad(self,x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0_classes = self.c0_classes.repeat(batch,1)
            return None, c0_classes
        
        middle_feat_cum = None
        classes_pred = None

        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, classes_out = m(x,middle_feat_cum)
                classes_pred = classes_out
            else:
                middle_feat_cum, classes_out = m(x,middle_feat_cum)
                classes_pred += classes_out

        final_classes = self.c0_classes + self.boost_rate * classes_pred
        return middle_feat_cum, final_classes


class GrowNetClassificationMLP(nn.Module):
    """
    MLP weak learner for GrowNet classification.
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(GrowNetClassificationMLP,self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)

        # Simple feedforward layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(0.4)
        self.class_head = nn.Linear(hidden_dim2, output_dim)

    def forward(self,x, lower_f): # In a hierarchical network each new model can receive not just original input but also previously learned features
        if lower_f is not None:
            x = torch.cat([x,lower_f],dim=1)
            x = self.bn(x)

        # Simple feedforward
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.bn1(x)
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.bn2(x)
        shared_features = x

        # Prediction
        n_classes = self.class_head(shared_features)

        return shared_features, n_classes
    
    @classmethod
    def get_model(cls,stage,params):
        # Returns a new MLP for the given stage
        if stage == 0:
            dim_in = params['feat_d']
        else:
            dim_in = params['feat_d'] + params['hidden_size']

        model = cls(
            dim_in,
            params['hidden_size'],
            params['hidden_size'],
            params['n_classes']
        )

        return model


class GrowNetClassifierUnique:
    """
    Main GrowNet classifier interface for training, evaluation, and prediction.
    """
    def __init__(self, params = None, device="cpu"):
        if params is None:
            self.params = grownet_classification_default_params()
        else:
            self.params = params
        self.device = device
        self.net_ensemble = None
        self.class_weights_tensor = None

    def _one_hot(self, y):
        """Ensure y is one-hot encoded."""
        if y.ndim == 1:
            n_classes = np.max(y) + 1
            return np.eye(n_classes)[y]
        return y

    def fit(self, X_train, y_train, X_val=None, y_val = None):
        # Trains the GrowNet ensemble classifier

        # Determine the input feature size
        self.params['feat_d'] = X_train.shape[1]
        
        # One hot encode for this model
        y_train = self._one_hot(y_train)
        if y_val is not None:
            y_val = self._one_hot(y_val)
        self.params["n_classes"] = y_train.shape[1]

        # Determine the number of classes
        self.params['n_classes'] = y_train.shape[1]
        
        # Class weighting to deal with imbalanced datasets
        # Compute and store class weights tensor once
        class_labels_flat = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(class_labels_flat),
            y=class_labels_flat
        )
        self.class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)


        # Split validation if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=self.params['val_split'],random_state=self.params['random_state'],stratify=class_labels_flat)

        train_ds = GrowNetClassificationTrainDataset(X_train, y_train)
        val_ds = GrowNetClassificationTrainDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True)

        print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

        train_loader = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True)
        
        # Init ensemble
        c0_classes = torch.tensor(np.log(np.mean(y_train, axis=0)), dtype=torch.float).unsqueeze(0).to(self.device)
        self.net_ensemble = GrowNetClassificationDynamicNet(c0_classes, self.params['boost_rate'])
        self.net_ensemble.to(self.device)
        
        best_val_loss = float("inf")
        best_stage = 0
        early_stop = 0
        lr = self.params["lr"]
        
        for stage in range(self.params['num_nets']):
            t0 = time.time()
            
            print(f"\nTraining weak learner {stage+1}/{self.params['num_nets']}")
            model = GrowNetClassificationMLP.get_model(stage, self.params).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=self.params['weight_decay'])
            self.net_ensemble.to_train()
            
            stage_train_losses = []
            
            for epoch in range(self.params["epochs_per_stage"]):
                for batch in train_loader:
                    x = batch["x"].to(self.device)
                    targets = batch["n_classes"].to(self.device)
                    
                    with torch.no_grad():
                        _, prev_logits = self.net_ensemble.forward(x)
                        prev_probs = torch.softmax(prev_logits, dim=1)
                        grad = targets - prev_probs
                        hessian = prev_probs * (1 - prev_probs)
                        hessian = hessian.sum(dim=1, keepdim=True)
                    
                    middle_feat, logits = model(x, None if stage == 0 else self.net_ensemble.forward_grad(x)[0])
                    loss_stagewise = nn.MSELoss(reduction="none")
                    boosting_loss = loss_stagewise(self.net_ensemble.boost_rate * logits, grad)
                    boosting_loss = (boosting_loss * hessian).mean()
                    class_loss = F.cross_entropy(logits, torch.argmax(targets, dim=1), weight=self.class_weights_tensor)
                    total_loss = class_loss * boosting_loss # Optionally combine with boosting_loss
                    
                    model.zero_grad()
                    total_loss.backward()
                    clip_grad_norm_(model.parameters(), self.params['gradient_clip'])
                    optimizer.step()
                    stage_train_losses.append(total_loss.item())
            
            self.net_ensemble.add(model)
            avg_stage_loss = np.mean(stage_train_losses)
            print(f"Stage {stage+1} finished | Avg Train Loss: {avg_stage_loss:.5f} | Time: {time.time() - t0:.1f}s")
            
            val_metrics = self.evaluate(X_val, y_val)
            val_loss = val_metrics['class_loss']
            print(f"Validation - Classification Acc: {val_metrics['class_accuracy']:.3f}")
            print(f"Boost rate: {self.net_ensemble.boost_rate.item():.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_stage = stage
                early_stop = 0
            else:
                early_stop += 1
                if early_stop > self.params["early_stopping_steps"]:
                    print("Early stopping!")
                    break
        print(f"\nBest model was at stage {best_stage+1} with Val Loss: {best_val_loss:.5f}")


    def evaluate(self, X, y):
        # Evaluates the classifier and returns metrics
        self.net_ensemble.to_eval()
        all_preds = []
        all_preds_prob = []
        all_targets = []
        class_losses = []
        y = self._one_hot(y)

        loader = DataLoader(GrowNetClassificationTrainDataset(X, y), batch_size=self.params['batch_size'], shuffle=False)
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                targets = batch["n_classes"].to(self.device)
                _, logits = self.net_ensemble.forward(x)
                loss = F.cross_entropy(logits, torch.argmax(targets, dim=1), weight=self.class_weights_tensor)
                class_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labs = torch.argmax(targets, dim=1).cpu().numpy()
                probs = torch.softmax(logits,dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_preds_prob.extend(probs)
                all_targets.extend(labs)
        acc = accuracy_score(all_targets, all_preds)
        return {
            'class_loss': np.mean(class_losses),
            'class_accuracy': acc,
            'predictions': all_preds,
            'probabilities': np.array(all_preds_prob),
            'targets': all_targets
        }
    
    def predict(self, X):
        # Predicts class labels and probabilities for new data
        self.net_ensemble.to_eval()
        all_preds = []
        all_preds_prob = []
        loader = DataLoader(GrowNetClassificationTrainDataset(X, np.zeros((X.shape[0], self.params['n_classes']))), batch_size=self.params['batch_size'], shuffle=False)
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                _, logits = self.net_ensemble.forward(x)
                probs = torch.softmax(logits,dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)
                all_preds_prob.extend(probs)
        return {
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_preds_prob)        
        }
    
def run_grownet_classifier(X_train,y_train,X_test,y_test,params=None,
                tune_hyperparams = False, n_trials=20,timeout=1200, verbose=False):
    """
    GrowNet classification wrapper for ensemble.
    Called by main.py for hierarchical classification.
    """
    # Handle device detection internally
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use default if params not given
    if params is None:
        params = grownet_classification_default_params()
    else:
        default = grownet_classification_default_params()
        default.update(params)
        params = default
    
    if tune_hyperparams:
        # Split validation set from training data
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=42, stratify=y_train)

        tuner = GrowNetClassificationTuner(X_train_split,y_train_split,X_val,y_val,params,device=device,n_trials=n_trials,timeout=timeout)
        best_params, best_score = tuner.tune()
        params.update(best_params)
        if verbose:
            print("Using best params:", params)
    
    # Train final model on full training data
    model = GrowNetClassifierUnique(params,device=device)
    model.fit(X_train,y_train)
    
    results = model.evaluate(X_test,y_test)
    if verbose:
        print("\nClassification Report:")
        print(classification_report(results['targets'], results['predictions']))
        print("\nAccuracy:", results['class_accuracy'])
    
    return {
        'model': model,
        'predictions': results['predictions'],
        'predicted_probabilities': results['probabilities'],
        'accuracy': results['class_accuracy'],
        'params': params
    }
