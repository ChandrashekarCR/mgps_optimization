# This model is used for regression tasks with a simple neural network

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

import time
import warnings
warnings.filterwarnings('ignore')


def default_regression_params():
    return {
        "input_dim": 200,
        "hidden_dim": [128, 64],
        "output_dim": 3,
        "use_batch_norm": True,
        "initial_dropout": 0.2,
        "final_dropout": 0.5,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "epochs": 400,
        "early_stopping_steps": 50,
        "gradient_clip": 1.0,
        "val_split": 0.2,
        "random_state": 42,
    }


class NNRegressionTuner:
    def __init__(self, X_train, y_train, X_val=None, y_val=None, params=None, device="cpu", n_trials=20, timeout=1200):
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

    def objective(self, trial):
        params = self.params.copy()
        params.update({
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim",
                [
                    [64],
                    [128],
                    [128, 64],
                    [256, 128, 64],
                    [256, 128],
                    [512, 256, 128, 64]
                ]
            ),
            "initial_dropout": trial.suggest_float("initial_dropout", 0.1, 0.3),
            "final_dropout": trial.suggest_float("final_dropout", 0.5, 0.8),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
            "gradient_clip": trial.suggest_float("gradient_clip", 0.5, 2.0),
        })

        # Train model
        model = NNRegressor(params, device=self.device)
        model.fit(self.X_train, self.y_train, X_val=self.X_val, y_val=self.y_val)
        val_metrics = model.evaluate(self.X_val, self.y_val)
        # Use negative MSE for maximization (Optuna maximizes by default)
        val_mse = val_metrics['mse']
        return -val_mse

    def tune(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"Best score (negative MSE): {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class RegressionTrainDataset(Dataset):
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

   

# Neural Network for Regression
class RegressionNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[128, 64],
                 use_batch_norm=True, initial_dropout=0.2, final_dropout=0.5, random_state=42):
        super().__init__()
        self.input_size = input_dim
        self.output_size = output_dim
        self.hidden_dim = hidden_dim
        self.use_batch_norm = use_batch_norm

        torch.manual_seed(random_state)

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        layer_sizes = [input_dim] + hidden_dim + [output_dim]
        dropout_rates = np.linspace(initial_dropout, final_dropout, len(hidden_dim))

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2 and use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rates[i]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
        return self.layers[-1](x)  # Output layer: no activation



class NNRegressor:
    def __init__(self, params=None, device="cpu"):
        if params is None:
            self.params = default_regression_params()
        else:
            self.params = params
        self.device = device
        self.model = None
        self.best_model_state = None
        self.target_scaler = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        print("Fitting the model...")

        # Update the parameters to this input
        self.params['input_dim'] = X_train.shape[1]
        self.params['output_dim'] = y_train.shape[1]  

        
        # Split if validation is not given
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.params['val_split'],
                random_state=self.params['random_state']
            )
        
        # Create datasets and dataloaders
        train_dataset = RegressionTrainDataset(X_train, y_train)
        val_dataset = RegressionTrainDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'], shuffle=False)

        print(f"Train size {len(train_dataset)}, Val size {len(val_dataset)}")

        # Initialize model
        self.model = RegressionNeuralNetwork(
            input_dim=self.params['input_dim'],
            output_dim=self.params['output_dim'],
            hidden_dim=self.params['hidden_dim'],
            use_batch_norm=self.params['use_batch_norm'],
            initial_dropout=self.params['initial_dropout'],
            final_dropout=self.params['final_dropout'],
            random_state=self.params['random_state']
        ).to(self.device)

        # Loss function and optimizer for regression
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay']
        )
        
        best_val_loss = float('inf')
        early_stopping_counter = 0

        train_losses = []
        val_losses = []

        print("Starting training...")
        for epoch in range(self.params['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                features = batch['x'].to(self.device)
                targets = batch['y'].to(self.device)

                # Zero optimizer
                optimizer.zero_grad()

                # Forward pass
                preds = self.model(features)

                # Calculate loss
                loss = criterion(preds, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.params['gradient_clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.params['gradient_clip'])

                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch['x'].to(self.device)
                    targets = batch['y'].to(self.device)

                    preds = self.model(features)
                    loss = criterion(preds, targets)
                    val_loss += loss.item()

            # Calculate averages
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{self.params["epochs"]}], '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                # Save best model state
                self.best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.params['early_stopping_steps']:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def evaluate(self, X, y):
        """Evaluate the model"""
        self.model.eval()
        all_preds = []
        all_targets = []
        losses = []

        dataset = RegressionTrainDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)
        criterion = nn.MSELoss()

        with torch.no_grad():
            for batch in loader:
                features = batch['x'].to(self.device)
                targets = batch['y'].to(self.device)

                preds = self.model(features)
                loss = criterion(preds, targets)
                losses.append(loss.item())

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Convert back to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Calculate metrics
        mse = mean_squared_error(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse),
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def predict(self, X):
        """Make predictions on new data"""
        self.model.eval()
        all_preds = []
        
        # Create dummy targets for dataset
        dummy_targets = np.zeros(X.shape[0])
        dataset = RegressionTrainDataset(X, dummy_targets)
        loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                features = batch['x'].to(self.device)
                outputs = self.model(features).squeeze()
                all_preds.extend(outputs.cpu().numpy())
        
        # Convert to numpy array and reshape
        all_preds = np.array(all_preds)
        
        return all_preds


def run_nn_regressor(X_train, y_train, X_test, y_test, device=None,
                     tune_hyperparams=False, params=None,
                     n_trials=20, timeout=1200):
    """Run the neural network regressor"""
    
    # Set device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running neural network regressor on device: {device}")
    
    # Use default if params not given
    if params is None:
        params = default_regression_params()
    else:
        default = default_regression_params()
        default.update(params)
        params = default

    # Update input dimension based on actual data
    params['input_dim'] = X_train.shape[1]
    params['output_dim'] = 1  # Single output for regression

    if tune_hyperparams:
        # Split validation set from training data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        tuner = NNRegressionTuner(X_train_split, y_train_split, X_val, y_val, 
                       params, device=device, n_trials=n_trials, timeout=timeout)
        best_params, best_score = tuner.tune()
        params.update(best_params)
        print("Using best params:", params)

    # Train final model on full training data
    model = NNRegressor(params, device=device)
    model.fit(X_train, y_train)

    # Evaluate on test set
    results = model.evaluate(X_test, y_test)
    
    print(f"\nRegression Results:")
    print(f"MSE: {results['mse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R2: {results['r2']:.4f}")
    
    return {
        'model': model,
        'predictions': results['predictions'],
        'mse': results['mse'],
        'mae': results['mae'],
        'rmse': results['rmse'],
        'r2': results['r2'],
        'params': params
    }