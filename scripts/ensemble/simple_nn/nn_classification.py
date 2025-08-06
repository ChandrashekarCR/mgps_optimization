# This model is used for classification tasks with a simple neural network

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


def default_classification_params():
    return {
        "input_dim": 200,
        "hidden_dim": [128, 64],
        "output_dim": 7,
        "use_batch_norm": True,
        "initial_dropout": 0.3,
        "final_dropout": 0.8,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "epochs": 400,
        "early_stopping_steps": 20,
        "gradient_clip": 1.0,
        "val_split": 0.2,
        "test_split": 0.2,
        "random_state": 42,
    }


class NNClassificationTuner:
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
        model = NNClassifier(params, device=self.device)
        model.fit(self.X_train, self.y_train, X_val=self.X_val, y_val=self.y_val)
        val_metrics = model.evaluate(self.X_val, self.y_val)
        val_acc = val_metrics['class_accuracy']
        return val_acc

    def tune(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset class for classification
class ClassificationTrainDataset(Dataset):
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
   

# Neural Network
class ClassificationNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = [128,64], use_batch_norm=True,
                  initial_dropout:float = 0.2, final_dropout:float =0.7, random_state=42):
        super(ClassificationNeuralNetwork,self).__init__()

        """
        Initialize Continent architecture using Pytorch modules.

        Parameters:
        - input_size: Number of input features. In this case it is the GITs. # 200
        - hidden_layers: List of hidden layers # 128, 64 are the default
        - output_size: Number of classes
        - dropout_rate: [0.2, 0.7]
        - random_state: Random state for reporducibility
        
        """
        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_dim
        self.dropout_initial = initial_dropout
        self.dropout_final = final_dropout
        self.use_batch_norm = use_batch_norm

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
       

        # Build the neural network
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Create dynamic doprout rates
        dropout_rates = np.linspace(initial_dropout,final_dropout, len(hidden_dim))

        # Create the layer architecture
        layer_sizes = [input_dim] + hidden_dim + [output_dim]

        for i in range(len(layer_sizes)-1):
            # Add the linear layers first
            self.layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))

            # Add batch normalization for hidden layers only and not for the output layers
            if i < len(layer_sizes) - 2 and self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i+1]))

            # Add dropout for hidden layers onyl and not for the output layers
            if i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rates[i]))

    def forward(self,x):
        """
        Forward propagations through the network
        
        Parameters:
        - x: Input tensor        
        """


        current_input = x

        # Forward pass through the hidden layers
        for i, (layer, dropout) in enumerate(zip(self.layers[:-1],self.dropouts)):
            # Linear transformations
            z = layer(current_input)

            # Batch normalization if enabled
            if self.use_batch_norm:
                z = self.batch_norms[i](z)

            # Acitvation function
            a = F.relu(z)

            # Apply dropout only during training
            if i < len(self.dropouts):
                a = dropout(a) if self.training else a # Apply dropout only during training
            
            current_input = a

        # Output layer (no activation for regression)
        output = self.layers[-1](current_input)

        return output


class NNClassifier:
    def __init__(self, params=None, device="cpu"):
        if params is None:
            self.params = default_classification_params()
        else:
            self.params = params
        self.device = device
        self.model = None
        self.class_weight_tensor = None
        self.best_model_state = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """ Train the model"""
        print("Fit the model...")

        # Update the parameters to this input
        self.params['input_dim'] = X_train.shape[1]
        self.params['output_dim'] = len(np.unique(y_train))


        # Compute the class weights
        class_weights = compute_class_weight(class_weight="balanced",classes=np.unique(y_train),y=y_train)
        self.class_weight_tensor = torch.tensor(class_weights,dtype=torch.float32).to(self.device)

        # Split if validation is not given
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=self.params['val_split'],
                                                              random_state=self.params['random_state'], stratify=y_train)
        
        # Create datasets and dataloaders
        train_dataset = ClassificationTrainDataset(X_train,y_train)
        val_dataset = ClassificationTrainDataset(X_val,y_val)

        train_loader = DataLoader(train_dataset,batch_size=self.params['batch_size'],shuffle=True)
        val_loader = DataLoader(val_dataset,batch_size=self.params['batch_size'],shuffle=False)

        print(f"Train size {len(train_dataset)}, Val size {len(val_dataset)}")
        

        # Initialize model
        self.model = ClassificationNeuralNetwork(
            input_dim=self.params['input_dim'],
            output_dim=self.params['output_dim'],
            hidden_dim=self.params['hidden_dim'],
            use_batch_norm=self.params['use_batch_norm'],
            initial_dropout=self.params['initial_dropout'],
            final_dropout=self.params['final_dropout'],
            random_state=self.params['random_state']
        ).to(self.device)

        # Loss function and evaluation for classification
        criterion_classification = nn.CrossEntropyLoss(weight=self.class_weight_tensor)
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.params['lr'],
                                     weight_decay=self.params['weight_decay'])
        
        best_val_loss = float('inf')
        early_stopping_counter = 0

        train_losses = []
        val_losses = []

        print("Strarting training.....")
        for epoch in range(self.params['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                features = batch['x'].to(self.device)
                targets = batch['n_classes'].to(self.device)

                # Set optimizer
                optimizer.zero_grad()

                # Forward pass
                preds = self.model(features)

                # Calculate loss
                classification_loss = criterion_classification(preds,targets)

                # Combined loss - adjust weight of the reconstruction loss
                total_loss = classification_loss 

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                if self.params['gradient_clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.params['gradient_clip'])

                optimizer.step()


                train_loss += total_loss.item()

                # Calcualte metrics for the epoch
                _, predicted = torch.max(preds.data,1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch['x'].to(self.device)
                    targets = batch['n_classes'].to(self.device)

                    preds = self.model(features)

                    classification_loss = criterion_classification(preds,targets)
                    total_loss = classification_loss 

                    val_loss += total_loss.item()

                    _, predicted = torch.max(preds.data,1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()

            # Calculate averages
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total

            train_losses.append(train_loss)
            val_losses.append(val_loss)


            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{self.params["epochs"]}], '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

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

    def evaluate(self, X, y):
        """
        Evaluate the model
        """

        self.model.eval()
        all_preds = []
        all_targets = []
        class_lossses = []
        all_preds_prob = []

        dataset = ClassificationTrainDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)

        criterion = nn.CrossEntropyLoss(weight=self.class_weight_tensor)

        with torch.no_grad():
            for batch in loader:
                features = batch['x'].to(self.device)
                targets = batch['n_classes'].to(self.device)

                preds = self.model(features)
                loss = criterion(preds,targets)
                class_lossses.append(loss.item())

                probs = F.softmax(preds, dim=1).cpu().numpy()
                _, predicted = torch.max(preds,1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_preds_prob.extend(probs)

        acc = accuracy_score(all_targets, all_preds)

        return {
                'class_loss': np.mean(class_lossses),
                'class_accuracy': acc,
                'probabilities':np.array(all_preds_prob),
                'predictions': all_preds,
                'targets': all_targets
            }
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        
        self.model.eval()
        all_preds = []
        all_preds_prob = []
        
        # Create dummy targets for dataset
        dummy_targets = np.zeros(X.shape[0])
        dataset = ClassificationTrainDataset(X, dummy_targets)
        loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                features = batch['x'].to(self.device)
                outputs = self.model(features)
                
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                all_preds.extend(preds)
                all_preds_prob.extend(probs)
        
        # Convert back to original labels
        
        return {
            'predictions': all_preds,
            'probabilities': np.array(all_preds_prob)
        }
    

def run_nn_classifier(X_train, y_train, X_test, y_test, device=None,
                      tune_hyperparams=False, params=None,
                      n_trials=20, timeout=1200):
    """Run the neural network classifier"""
    
    # Set device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running neural network classifier on device: {device}")
    
    # Use default if params not given
    if params is None:
        params = default_classification_params()
    else:
        default = default_classification_params()
        default.update(params)
        params = default

    # Update input dimension based on actual data
    params['input_dim'] = X_train.shape[1]
    params['output_dim'] = len(np.unique(y_train))

        
    if tune_hyperparams:
        # Split validation set from training data
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train,y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        tuner = NNClassificationTuner(X_train_split, y_train_split, X_val, y_val, 
                               params, device=device, n_trials=n_trials, timeout=timeout)
        best_params, best_score = tuner.tune()
        params.update(best_params)
        print("Using best params:", params)

    # Train final model on full training data
    model = NNClassifier(params, device=device)
    model.fit(X_train, y_train)

    results = model.evaluate(X_test, y_test)
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