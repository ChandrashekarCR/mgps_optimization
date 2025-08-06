# This model is used for regression tasks

# Import libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

import time
import warnings
warnings.filterwarnings('ignore')

# Set device globally
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def grownet_regression_default_params():
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
        "n_outputs":3
    }


class GrowNetRegressionTuner:
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
        model = GrowNetRegressorUnique(params, device=self.device)
        model.fit(self.X_train, self.y_train, X_val=self.X_val, y_val=self.y_val)
        val_metrics = model.evaluate(self.X_val, self.y_val)
        return -val_metrics['rmse'] # minimize

    def tune(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score


# Dataset class for regression
class GrowNetRegressionTrainDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float),
            'y': torch.tensor(self.targets[idx], dtype=torch.float)
        }

    
# DynamicNet for regression
class GrowNetRegressionDynamicNet(nn.Module):
    def __init__(self, c0_coords, lr):
        super(GrowNetRegressionDynamicNet, self).__init__()
        self.models = nn.ModuleList()

        self.c0_coords = c0_coords
        self.lr = lr
        self.boost_rate = nn.Parameter(torch.tensor(lr, requires_grad=True, device=device))

    def to(self, device):
        self.c0_coords = self.c0_coords.to(device)
        self.boost_rate = self.boost_rate.to(device)
        for m in self.models:
            m.to(device)

    def add(self, model):
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

    def forward(self, x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0_coords = self.c0_coords.repeat(batch, 1)
            return None, c0_coords

        middle_feat_cum = None
        coords_pred = None
        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, coords_out = m(x, middle_feat_cum)
                    coords_pred = coords_out
                else:
                    middle_feat_cum, coords_out = m(x, middle_feat_cum)
                    coords_pred += coords_out
        final_coords = self.c0_coords + self.boost_rate * coords_pred
        return middle_feat_cum, final_coords

    def forward_grad(self, x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0_coords = self.c0_coords.repeat(batch, 1)
            return None, c0_coords
        middle_feat_cum = None
        coords_pred = None
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, coords_out = m(x, middle_feat_cum)
                coords_pred = coords_out
            else:
                middle_feat_cum, coords_out = m(x, middle_feat_cum)
                coords_pred += coords_out
        final_coords = self.c0_coords + self.boost_rate * coords_pred
        return middle_feat_cum, final_coords


class GrowNetRegressionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(GrowNetRegressionMLP,self).__init__()
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
        self.reg_head = nn.Linear(hidden_dim2, output_dim)

    def forward(self,x, lower_f):
        x = self.bn(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.bn1(x)
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.bn2(x)
        coord_out = self.reg_head(x)
        return None, coord_out

    @classmethod
    def get_model(cls,stage,params):
        dim_in = params['feat_d']
        model = cls(
            dim_in,
            params['hidden_size'],
            params['hidden_size'],
            params['n_outputs']
        )
        return model


class GrowNetRegressorUnique:
    def __init__(self, params = None, device="cpu"):
        if params is None:
            self.params = grownet_regression_default_params()
        else:
            self.params = params
        self.device = device
        self.net_ensemble = None

    def fit(self,X_train,y_train, X_val=None, y_val = None):
        # Split validation if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=self.params['val_split'],random_state=self.params['random_state'])

        train_ds = GrowNetRegressionTrainDataset(X_train, y_train)
        val_ds = GrowNetRegressionTrainDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True)

        print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

        train_loader = DataLoader(train_ds, batch_size=self.params['batch_size'], shuffle=True)
        
        # Init ensemble
        c0 = torch.tensor(np.mean(y_train, axis=0), dtype=torch.float).unsqueeze(0).to(self.device)

        self.net_ensemble = GrowNetRegressionDynamicNet(c0, self.params['boost_rate'])
        self.net_ensemble.to(self.device)
        
        best_val_loss = float("inf")
        best_stage = 0
        early_stop = 0
        lr = self.params["lr"]
        
        for stage in range(self.params['num_nets']):
            t0 = time.time()
            
            print(f"\nTraining weak learner {stage+1}/{self.params['num_nets']}")
            model = GrowNetRegressionMLP.get_model(stage, self.params).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=self.params['weight_decay'])
            self.net_ensemble.to_train()
            
            stage_train_losses = []
            
            for epoch in range(self.params["epochs_per_stage"]):
                for batch in train_loader:
                    x = batch["x"].to(self.device)
                    targets = batch["y"].to(self.device)

                    with torch.no_grad():
                        _, prev_preds = self.net_ensemble.forward(x)
                        grad = targets - prev_preds

                    # Always pass None for lower_f (no feature extraction)
                    middle_feat, preds = model(x, None)
                    loss_stagewise = nn.MSELoss(reduction="none")
                    boosting_loss = loss_stagewise(self.net_ensemble.boost_rate * preds, grad)
                    boosting_loss = boosting_loss.mean()
                    total_loss = boosting_loss
                    
                    model.zero_grad()
                    total_loss.backward()
                    clip_grad_norm_(model.parameters(), self.params['gradient_clip'])
                    optimizer.step()
                    stage_train_losses.append(total_loss.item())
            self.net_ensemble.add(model)
            avg_stage_loss = np.mean(stage_train_losses)
            print(f"Stage {stage+1} finished | Avg Train Loss: {avg_stage_loss:.5f} | Time: {time.time() - t0:.1f}s")
            val_metrics = self.evaluate(X_val, y_val)
            val_loss = val_metrics['rmse']
            print(f"Validation - RMSE: {val_loss:.3f}")
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
        print(f"\nBest model was at stage {best_stage+1} with Val RMSE: {best_val_loss:.5f}")


    def evaluate(self, X, y):
        self.net_ensemble.to_eval()
        all_preds = []
        all_targets = []
        losses = []
        loader = DataLoader(GrowNetRegressionTrainDataset(X, y), batch_size=self.params['batch_size'], shuffle=False)
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                targets = batch["y"].to(self.device)
                _, preds = self.net_ensemble.forward(x)
                loss = F.mse_loss(preds, targets)
                losses.append(loss.item())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)
        return {
            'rmse': rmse,
            'r2': r2,
            'predictions': all_preds,
            'targets': all_targets
        }

    def predict(self, X):
        self.net_ensemble.to_eval()
        all_preds = []
        loader = DataLoader(GrowNetRegressionTrainDataset(X, np.zeros((X.shape[0], self.params['n_outputs']))), batch_size=self.params['batch_size'], shuffle=False)
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                _, preds = self.net_ensemble.forward(x)
                all_preds.append(preds.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        return all_preds
    
def run_grownet_regressor(X_train, y_train, X_test, y_test, params=None,
                          tune_hyperparams=False, n_trials=20, timeout=1200, 
                          device=None, verbose=True):
    """Run GrowNet regressor with proper error handling and interface consistency"""
    
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if verbose:
            print(f"Running GrowNet regressor on device: {device}")
            
        if params is None:
            params = grownet_regression_default_params()
        else:
            default = grownet_regression_default_params()
            default.update(params)
            params = default
            
        # Handle both 1D and multi-dimensional targets
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            
        params['feat_d'] = X_train.shape[1]
        params['n_outputs'] = y_train.shape[1]
        
        if tune_hyperparams:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            tuner = GrowNetRegressionTuner(X_train_split, y_train_split, X_val, y_val, 
                                         params, device=device, n_trials=n_trials, timeout=timeout)
            best_params, best_score = tuner.tune()
            params.update(best_params)
            if verbose:
                print("Using best params:", params)
                
        model = GrowNetRegressorUnique(params, device=device)
        model.fit(X_train, y_train)
        results = model.evaluate(X_test, y_test)
        
        if verbose:
            print("\nRegression Report:")
            print(f"RMSE: {results['rmse']:.4f}")
            print(f"R2 Score: {results['r2']:.4f}")
            
        return {
            'model': model,
            'predictions': results['predictions'],
            'rmse': results['rmse'],
            'r2_score': results['r2'],  # Use r2_score for consistency
            'params': params,
            'skipped': False
        }
        
    except Exception as e:
        if verbose:
            print(f"Error in GrowNet regressor: {e}")
        # Return dummy predictions on error
        n_samples = X_test.shape[0]
        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
        dummy_preds = np.zeros((n_samples, output_dim))
        
        return {
            'model': None,
            'predictions': dummy_preds,
            'rmse': float('inf'),
            'r2_score': -float('inf'),
            'params': params,
            'skipped': True,
            'error': str(e)
        }

