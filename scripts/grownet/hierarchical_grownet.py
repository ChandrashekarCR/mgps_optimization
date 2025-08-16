# Import Libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns
import optuna

# --- Scikit-learn imports for preprocessing and metrics ---
from sklearn import preprocessing
from sklearn.metrics import log_loss, classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- PyTorch imports for deep learning ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

import time
import warnings
warnings.filterwarnings('ignore')

# =========================
# Default Parameters
# =========================
def default_params():
    """
    Returns default parameters for hierarchical GrowNet model.
    Includes architecture, boosting, training, and reproducibility settings.
    """
    return {
        # === Architecture ===
        "feat_d": 200,             # Input feature size (species abundance, etc.)
        "hidden_size": 256,        # Hidden layer size for MLP
        "n_continents": 7,         # Set automatically based on data
        "n_cities": None,          # Set based on city_target shape
        "coord_dim": 3,            # x, y, z coordinates

        # === GrowNet boosting ===
        "num_nets": 30,            # Number of weak learners to add
        "boost_rate": 0.4,         # Boosting rate (learnable in net_ensemble)

        # === Training parameters ===
        "lr": 1e-3,                # Base learning rate for weak learners
        "weight_decay": 1e-4,      # L2 regularization
        "batch_size": 128,         # Mini-batch size
        "epochs_per_stage": 20,    # Epochs per weak learner
        "correct_epoch": 5,        # Fine-tuning epochs for uncertainty correction
        "early_stopping_steps": 5, # Stop training if no improvement for N learners
        "gradient_clip": 1.0,      # Gradient clipping norm

        # === Reproducibility ===
        "random_seed": 42,         # Reproducibility

        # === Dropout rates ===
        "dropout1": 0.2,
        "dropout2": 0.4,
        "dropout3": 0.2,  # New for extra layer
    }

params = default_params()

# =========================
# Device Setup & Seeding
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def seed_everything(seed=42):
    """
    Sets seeds for reproducibility across numpy, random, torch, and CUDA.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

# =========================
# Dataset Classes
# =========================

class HierarchicalTrainDataset:
    """
    PyTorch-style dataset for hierarchical training.
    Returns features and all targets for each sample.
    """
    def __init__(self, features, continent_targets, city_targets, coord_targets):
        self.features = features
        self.continent_targets = continent_targets
        self.city_targets = city_targets
        self.coord_targets = coord_targets

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self,idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float),
            'continent': torch.tensor(self.continent_targets[idx], dtype=torch.float),
            'city': torch.tensor(self.city_targets[idx],dtype=torch.float),
            'coords': torch.tensor(self.coord_targets[idx],dtype=torch.float)
        }
    
class HierarchicalTestDataset:
    """
    PyTorch-style dataset for hierarchical testing/inference.
    Returns only features for each sample.
    """
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float)
        }
    
# =========================
# Hierarchical GrowNet Model
# =========================

class HierarchicalDynamicNet(nn.Module):
    """
    Ensemble of weak learners for hierarchical GrowNet.
    Supports boosting, forward pass, and parameter management.
    """
    def __init__(self, c0_continent, c0_city, c0_coords, lr):
        super(HierarchicalDynamicNet,self).__init__()
        self.models = []
        self.c0_continent = c0_continent
        self.c0_city = c0_city
        self.c0_coords = c0_coords
        self.lr = lr
        self.boost_rate = nn.Parameter(torch.tensor(lr, requires_grad=True, device=device))

    def to(self,device):
        self.c0_continent = self.c0_continent.to(device)
        self.c0_city = self.c0_city.to(device)
        self.c0_coords = self.c0_coords.to(device)
        self.boost_rate = self.boost_rate.to(device)
        for m in self.models:
            m.to(device)

    def add(self,model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())
        params += [self.boost_rate]
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

    def forward(self, x, return_intermediates=False):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0_continent = self.c0_continent.repeat(batch,1)
            c0_city = self.c0_city.repeat(batch, 1)
            c0_coords = self.c0_coords.repeat(batch, 1)

            if return_intermediates:
                return None, c0_continent, c0_city, c0_coords, None, None
            return c0_continent, c0_city, c0_coords
        
        middle_feat_cum = None
        continent_pred = None
        city_pred = None
        coord_pred = None

        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, cont_out, city_out, coord_out = m(x, middle_feat_cum)
                    continent_pred = cont_out
                    city_pred = city_out
                    coord_pred = coord_out
                else:
                    middle_feat_cum, cont_out, city_out, coord_out = m(x, middle_feat_cum)
                    continent_pred += cont_out
                    city_pred += city_out
                    coord_pred += coord_out

        final_continent = self.c0_continent + self.boost_rate * continent_pred
        final_city = self.c0_city + self.boost_rate * city_pred
        final_coords = self.c0_coords + self.boost_rate * coord_pred
        
        if return_intermediates:
            return middle_feat_cum, final_continent, final_city, final_coords, continent_pred, city_pred
        return final_continent, final_city, final_coords

    def forward_grad(self, x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0_continent = self.c0_continent.repeat(batch, 1)
            c0_city = self.c0_city.repeat(batch, 1)
            c0_coords = self.c0_coords.repeat(batch, 1)
            return None, c0_continent, c0_city, c0_coords

        middle_feat_cum = None
        continent_pred = None
        city_pred = None
        coord_pred = None
        
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, cont_out, city_out, coord_out = m(x, middle_feat_cum)
                continent_pred = cont_out
                city_pred = city_out
                coord_pred = coord_out
            else:
                middle_feat_cum, cont_out, city_out, coord_out = m(x, middle_feat_cum)
                continent_pred += cont_out
                city_pred += city_out
                coord_pred += coord_out

        final_continent = self.c0_continent + self.boost_rate * continent_pred
        final_city = self.c0_city + self.boost_rate * city_pred
        final_coords = self.c0_coords + self.boost_rate * coord_pred
        
        return middle_feat_cum, final_continent, final_city, final_coords


# =========================
# Hierarchical MLP (Weak Learner)
# =========================

class HierarchicalMLP(nn.Module):
    """
    MLP for hierarchical prediction.
    - Continent head
    - City head (uses continent predictions)
    - Coordinate head (uses continent and city predictions)
    """
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, n_continents, n_cities, coord_dim, dropout1, dropout2):
        super(HierarchicalMLP, self).__init__()
        # Input processing
        self.bn_in = nn.BatchNorm1d(dim_in)
        
        # First hidden layer
        self.fc1 = nn.Linear(dim_in, dim_hidden1)
        self.bn1 = nn.BatchNorm1d(dim_hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout1)
        
        # Second hidden layer
        self.fc2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.bn2 = nn.BatchNorm1d(dim_hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout2)
        
        # Prediction heads
        # Continent head
        self.continent_head = nn.Linear(dim_hidden2, n_continents)
        
        # City head (uses features + continent predictions)
        self.city_head = nn.Linear(dim_hidden2 + n_continents, n_cities)
        
        # Coordinate head (uses features + continent + city predictions)
        self.coord_head = nn.Linear(dim_hidden2 + n_continents + n_cities, coord_dim)

    def forward(self, x, lower_f):
        # Combine input with features from previous weak learners (if any)
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
        
        # Input normalization
        x = self.bn_in(x)
        
        # First layer
        h1 = self.fc1(x)
        h1 = self.bn1(h1)
        h1 = self.relu1(h1)
        h1 = self.dropout1(h1)
        
        # Second layer with residual connection if dimensions match
        h2 = self.fc2(h1)
        h2 = self.bn2(h2)
        h2 = self.relu2(h2)
        h2 = self.dropout2(h2)
        
        # Extract features for the ensemble
        features = h2
        
        # Continent prediction
        continent_out = self.continent_head(features)
        continent_probs = torch.softmax(continent_out, dim=1)
        
        # City prediction (augmented with continent)
        city_input = torch.cat([features, continent_probs], dim=1)
        city_out = self.city_head(city_input)
        city_probs = torch.softmax(city_out, dim=1)
        
        # Coordinate prediction (augmented with continent and city)
        coord_input = torch.cat([features, continent_probs, city_probs], dim=1)
        coord_out = self.coord_head(coord_input)
        
        return features, continent_out, city_out, coord_out

    @classmethod
    def get_model(cls, stage, params):
        if stage == 0:
            dim_in = params["feat_d"]
        else:
            dim_in = params["feat_d"] + params["hidden_size"]
        
        model = cls(
            dim_in, 
            params["hidden_size"], 
            params["hidden_size"],
            params["n_continents"], 
            params["n_cities"], 
            params["coord_dim"],
            params.get("dropout1", 0.2),
            params.get("dropout2", 0.4)
        )
        return model   

def get_optim(params, lr, weight_decay):
    """
    Returns Adam optimizer for given parameters.
    """
    optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
    return optimizer


def compute_hierarchical_loss(continent_pred, city_pred, coord_pred, 
                              continent_target, city_target, coord_target, 
                              continent_class_weights=None, city_class_weights=None,
                              continent_loss_weight=1.0, city_loss_weight=1.0, coord_loss_weight=1.0):
    """
    Computes weighted loss for all three hierarchical tasks.
    """
    continent_loss = F.cross_entropy(continent_pred, torch.argmax(continent_target, dim=1),
                                     weight=continent_class_weights)
    city_loss = F.cross_entropy(city_pred, torch.argmax(city_target, dim=1),
                                weight=city_class_weights)
    coord_loss = F.mse_loss(coord_pred, coord_target)

    total_loss = (
        continent_loss_weight * continent_loss +
        city_loss_weight * city_loss +
        coord_loss_weight * coord_loss
    )
    
    return total_loss, continent_loss, city_loss, coord_loss


def evaluate_hierarchical_model(net_ensemble, test_loader,continent_class_weights_tensor,city_class_weights_tensor,
                               continent_loss_weight=1.0, city_loss_weight=1.0, coord_loss_weight=1.0):
    """
    Evaluates the hierarchical GrowNet model on all tasks.
    Returns losses, accuracies, and predictions/targets.
    """
    net_ensemble.to_eval()
    
    continent_losses = []
    city_losses = []
    coord_losses = []
    
    continent_preds = []
    city_preds = []
    coord_preds = []
    
    continent_targets = []
    city_targets = []
    coord_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            continent_target = batch["continent"].to(device)
            city_target = batch["city"].to(device)
            coord_target = batch["coords"].to(device)
            
            continent_pred, city_pred, coord_pred = net_ensemble.forward(x)
            
            # Compute losses using weights - removed log_sigma parameters
            total_loss, cont_loss, city_loss, coord_loss = compute_hierarchical_loss(
                continent_pred, city_pred, coord_pred,
                continent_target, city_target, coord_target,
                continent_class_weights=continent_class_weights_tensor,
                city_class_weights=city_class_weights_tensor,
                continent_loss_weight=continent_loss_weight,
                city_loss_weight=city_loss_weight,
                coord_loss_weight=coord_loss_weight
            )
            
            continent_losses.append(cont_loss.item())
            city_losses.append(city_loss.item())
            coord_losses.append(coord_loss.item())
            
            # Store predictions for metrics
            continent_preds.extend(torch.argmax(continent_pred, dim=1).cpu().numpy())
            city_preds.extend(torch.argmax(city_pred, dim=1).cpu().numpy())
            coord_preds.extend(coord_pred.cpu().numpy())
            
            continent_targets.extend(torch.argmax(continent_target, dim=1).cpu().numpy())
            city_targets.extend(torch.argmax(city_target, dim=1).cpu().numpy())
            coord_targets.extend(coord_target.cpu().numpy())
    
    return {
        'continent_loss': np.mean(continent_losses),
        'city_loss': np.mean(city_losses),
        'coord_loss': np.mean(coord_losses),
        'continent_acc': accuracy_score(continent_targets, continent_preds),
        'city_acc': accuracy_score(city_targets, city_preds),
        'coord_mse': mean_squared_error(coord_targets, coord_preds),
        'predictions': {
            'continent': continent_preds,
            'city': city_preds,
            'coords': coord_preds
        },
        'targets': {
            'continent': continent_targets,
            'city': city_targets,
            'coords': coord_targets
        }
    }



# =========================
# Training Function
# =========================

def train_hierarchical_grownet(x_data, continent_targets, city_targets, coord_targets, params):
    """
    Trains the hierarchical GrowNet model.
    Handles class weights, data splits, boosting, and corrective steps.
    Returns trained model and test metrics.
    """

    # Continent class weights to deal with data imbalance
    continent_labels_flat = np.argmax(continent_targets, axis=1)
    continent_class_weights = compute_class_weight(class_weight="balanced",
                                                   classes=np.unique(continent_labels_flat),
                                                   y = continent_labels_flat)
    continent_class_weight_tensor = torch.tensor(continent_class_weights,dtype=torch.float32).to(device)

    # City class weights
    city_labels_flat = np.argmax(city_targets, axis=1)
    city_class_weights = compute_class_weight(class_weight='balanced',
                                          classes=np.unique(city_labels_flat),
                                          y=city_labels_flat)
    city_class_weights_tensor = torch.tensor(city_class_weights, dtype=torch.float32).to(device)


    
    # Update parameters
    params["n_cities"] = city_targets.shape[1]
    
    # Split data
    X_train, X_test, cont_train, cont_test, city_train, city_test, coord_train, coord_test = train_test_split(
        x_data, continent_targets, city_targets, coord_targets, 
        test_size=0.2, random_state=42, stratify=continent_targets
    )
    
    X_train, X_val, cont_train, cont_val, city_train, city_val, coord_train, coord_val = train_test_split(
        X_train, cont_train, city_train, coord_train, 
        test_size=0.2, random_state=42, stratify=cont_train
    )
    
    # Create datasets
    train_ds = HierarchicalTrainDataset(X_train, cont_train, city_train, coord_train)
    val_ds = HierarchicalTrainDataset(X_val, cont_val, city_val, coord_val)
    test_ds = HierarchicalTrainDataset(X_test, cont_test, city_test, coord_test)
    
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False)
    
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    
    # Initialize hierarchical GrowNet
    c0_continent = torch.tensor(np.log(np.mean(cont_train, axis=0)), dtype=torch.float).unsqueeze(0).to(device)
    c0_city = torch.tensor(np.log(np.mean(city_train, axis=0)), dtype=torch.float).unsqueeze(0).to(device)
    c0_coords = torch.tensor(np.mean(coord_train, axis=0), dtype=torch.float).unsqueeze(0).to(device)
    
    net_ensemble = HierarchicalDynamicNet(c0_continent, c0_city, c0_coords, params["boost_rate"])
    net_ensemble.to(device)
    
    # Loss functions
    loss_stagewise = nn.MSELoss(reduction="none")
    
    best_val_loss = float("inf")
    best_stage = 0
    early_stop = 0
    lr = params["lr"]
    
    continent_loss_weight = params.get("continent_loss_weight", 2.0)
    city_loss_weight = params.get("city_loss_weight", 1.0)
    coord_loss_weight = params.get("coord_loss_weight", 0.5)
    
    # Initial evaluation
    initial_metrics = evaluate_hierarchical_model(
        net_ensemble, val_loader, continent_class_weight_tensor, city_class_weights_tensor,
        continent_loss_weight=continent_loss_weight,
        city_loss_weight=city_loss_weight,
        coord_loss_weight=coord_loss_weight
    )
    print(f"Initial - Continent Acc: {initial_metrics['continent_acc']:.3f}, "
          f"City Acc: {initial_metrics['city_acc']:.3f}, "
          f"Coord MSE: {initial_metrics['coord_mse']:.5f}")
    
    # Training loop
    for stage in range(params["num_nets"]):
        t0 = time.time()
        print(f"\nTraining weak learner {stage+1}/{params['num_nets']}")
        
       
        model = HierarchicalMLP.get_model(stage, params).to(device)
        optimizer = get_optim(model.parameters(), lr, params["weight_decay"])
        # Add cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs_per_stage"])
        net_ensemble.to_train()
        
        stage_train_losses = []
        
        for epoch in range(params["epochs_per_stage"]):
            for batch in train_loader:
                x = batch["x"].to(device)
                continent_target = batch["continent"].to(device)
                city_target = batch["city"].to(device)
                coord_target = batch["coords"].to(device)
                
                with torch.no_grad():
                    continent_prev, city_prev, coord_prev = net_ensemble.forward(x)
                    
                    # Compute gradients for each task
                    continent_probs = torch.softmax(continent_prev, dim=1)
                    continent_grad = continent_target - continent_probs
                    
                    city_probs = torch.softmax(city_prev, dim=1)
                    city_grad = city_target - city_probs
                    
                    coord_grad = coord_target - coord_prev
                    
                    # Hessian approximations
                    continent_hessian = continent_probs * (1 - continent_probs)
                    continent_hessian = continent_hessian.sum(dim=1, keepdim=True)
                    
                    city_hessian = city_probs * (1 - city_probs)
                    city_hessian = city_hessian.sum(dim=1, keepdim=True)
                
                # Forward pass through current model
                middle_feat, continent_out, city_out, coord_out = model(
                    x, None if stage == 0 else net_ensemble.forward_grad(x)[0]
                )
                
                # Compute boosting losses
                continent_loss = loss_stagewise(net_ensemble.boost_rate * continent_out, continent_grad)
                continent_loss = (continent_loss * continent_hessian).mean()
                
                city_loss = loss_stagewise(net_ensemble.boost_rate * city_out, city_grad)
                city_loss = (city_loss * city_hessian).mean()
                
                coord_loss = loss_stagewise(net_ensemble.boost_rate * coord_out, coord_grad).mean()
                
                # Use weighted sum for total loss - removed log_sigma parameters
                total_loss, _, _, _ = compute_hierarchical_loss(
                    continent_out, city_out, coord_out,
                    continent_target, city_target, coord_target,
                    continent_class_weights=continent_class_weight_tensor,
                    city_class_weights=city_class_weights_tensor,
                    continent_loss_weight=continent_loss_weight,
                    city_loss_weight=city_loss_weight,
                    coord_loss_weight=coord_loss_weight
                )
                
                model.zero_grad()
                total_loss.backward()

                # Gradient clipping
                clip_grad_norm_(model.parameters(),params['gradient_clip'])

                optimizer.step()
                stage_train_losses.append(total_loss.item())
            scheduler.step()
        net_ensemble.add(model)
        avg_stage_loss = np.mean(stage_train_losses)
        print(f"Stage {stage+1} finished | Avg Train Loss: {avg_stage_loss:.5f} | Time: {time.time() - t0:.1f}s")
        
        # Corrective step
        if stage > 0:
            if stage % 3 == 0:
                lr /= 2

            # Only use corrective optimizer for net_ensemble.parameters()
            corrective_optimizer = get_optim(net_ensemble.parameters(), lr/2, params["weight_decay"])
            corrective_losses = []
            
            for _ in range(params["correct_epoch"]):
                for batch in train_loader:
                    x = batch["x"].to(device)
                    continent_target = batch["continent"].to(device)
                    city_target = batch["city"].to(device)
                    coord_target = batch["coords"].to(device)
                    
                    middle_feat, continent_pred, city_pred, coord_pred = net_ensemble.forward_grad(x)
                    
                    total_loss, _, _, _ = compute_hierarchical_loss(
                        continent_pred, city_pred, coord_pred,
                        continent_target, city_target, coord_target,
                        continent_class_weights=continent_class_weight_tensor,
                        city_class_weights=city_class_weights_tensor,
                        continent_loss_weight=continent_loss_weight,
                        city_loss_weight=city_loss_weight,
                        coord_loss_weight=coord_loss_weight
                    )
                    
                    corrective_optimizer.zero_grad()
                    total_loss.backward()
                    clip_grad_norm_(net_ensemble.parameters(), params['gradient_clip'])
                    corrective_optimizer.step()
                    corrective_losses.append(total_loss.item())
            
            print(f"Corrective step avg loss: {np.mean(corrective_losses):.5f}")
        
        # Validation
        val_metrics = evaluate_hierarchical_model(
            net_ensemble, val_loader, continent_class_weight_tensor, city_class_weights_tensor,
            continent_loss_weight=continent_loss_weight,
            city_loss_weight=city_loss_weight,
            coord_loss_weight=coord_loss_weight
        )
        val_loss = val_metrics['continent_loss'] + val_metrics['city_loss'] + val_metrics['coord_loss']
        
        print(f"Validation - Continent Acc: {val_metrics['continent_acc']:.3f}, "
              f"City Acc: {val_metrics['city_acc']:.3f}, "
              f"Coord MSE: {val_metrics['coord_mse']:.5f}")
        print(f"Boost rate: {net_ensemble.boost_rate.item():.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_stage = stage
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > params["early_stopping_steps"]:
                print("Early stopping!")
                break
    
    print(f"\nBest model was at stage {best_stage+1} with Val Loss: {best_val_loss:.5f}")
    
    # Final evaluation on test set
    test_metrics = evaluate_hierarchical_model(
        net_ensemble, test_loader, continent_class_weight_tensor, city_class_weights_tensor,
        continent_loss_weight=continent_loss_weight,
        city_loss_weight=city_loss_weight,
        coord_loss_weight=coord_loss_weight
    )
    print(f"\nFinal Test Results:")
    print(f"Continent Accuracy: {test_metrics['continent_acc']:.3f}")
    print(f"City Accuracy: {test_metrics['city_acc']:.3f}")
    print(f"Coordinate MSE: {test_metrics['coord_mse']:.5f}")
    
    return net_ensemble, test_metrics

# =========================
# Hyperparameter Tuning Class
# =========================

class HierarchicalGrowNetTuner:
    """
    Optuna tuner for hierarchical GrowNet.
    Tunes architecture, boosting, training, and loss weights.
    """
    def __init__(self, X_train, continent_targets, city_targets, coord_targets, params, device="cpu", n_trials=20, timeout=1200):
        self.X_train = X_train
        self.continent_targets = continent_targets
        self.city_targets = city_targets
        self.coord_targets = coord_targets
        self.params = params
        self.device = device
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.best_score = None

    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter search.
        Uses continent accuracy for maximization.
        """
        params = self.params.copy()
        # Enforce hierarchical relationship: continent_weight > city_weight > coord_weight
        continent_weight = trial.suggest_float("continent_weight", 1.0, 2.0)
        city_weight = trial.suggest_float("city_weight", 0.5, continent_weight - 0.05)
        coord_weight = trial.suggest_float("coord_weight", 0.05, city_weight - 0.05)
        
        params.update({
            "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512]),
            "num_nets": trial.suggest_int("num_nets", 10, 30),
            "boost_rate": trial.suggest_float("boost_rate", 0.1, 0.8),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-2),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-3),
            "epochs_per_stage": trial.suggest_int("epochs_per_stage", 5, 10),
            "gradient_clip": trial.suggest_float("gradient_clip", 0.5, 2.0),
            "continent_loss_weight": continent_weight,
            "city_loss_weight": city_weight,
            "coord_loss_weight": coord_weight,
        })
        # Use a fixed split for validation
        X_train, X_val, cont_train, cont_val, city_train, city_val, coord_train, coord_val = train_test_split(
            self.X_train, self.continent_targets, self.city_targets, self.coord_targets,
            test_size=0.2, random_state=42, stratify=self.continent_targets
        )
        try:
            net, metrics = train_hierarchical_grownet(
                X_train, cont_train, city_train, coord_train, params
            )
            val_acc = metrics['continent_acc']
        except Exception as e:
            val_acc = 0.0
        return val_acc

    def tune(self):
        """
        Runs Optuna study to find best hyperparameters.
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = study.best_params
        self.best_score = study.best_value
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        return self.best_params, self.best_score

# =========================
# Data Processing Function
# =========================

def process_hierarchical_data(df):
    """
    Processes input DataFrame for hierarchical learning.
    - Extracts features
    - Encodes continent/city labels (one-hot)
    - Computes and scales coordinates
    Returns arrays and encoders.
    """
    # Separate features and targets
    feature_cols = [col for col in df.columns if col not in ['city', 'continent', 'latitude', 'longitude']]
    x_data = df[feature_cols].values
    
    # Encode continent labels
    continent_encoder = LabelEncoder()
    continent_labels = continent_encoder.fit_transform(df['continent'])
    continent_onehot = np.eye(len(continent_encoder.classes_))[continent_labels]
    
    # Encode city labels
    city_encoder = LabelEncoder()
    city_labels = city_encoder.fit_transform(df['city'])
    city_onehot = np.eye(len(city_encoder.classes_))[city_labels]
    
    # Encode co-ordinates
    coordinate_encoder = StandardScaler()
    df['latitude_rad'] = np.deg2rad(df['latitude'])
    df['longitude_rad'] = np.deg2rad(df['longitude'])

    # Calculate x, y, z coordinates -  Converting polar co-ordinates into cartesian co-ordinates
    df['x'] = np.cos(df['latitude_rad']) * np.cos(df['longitude_rad'])
    df['y'] = np.cos(df['latitude_rad']) * np.sin(df['longitude_rad'])
    df['z'] = np.sin(df['latitude_rad'])

    # Scale the x, y, z coordinates together
    df[['scaled_x','scaled_y','scaled_z']] = coordinate_encoder.fit_transform (df[['x','y','z']])

    # Coordinate targets (assuming you have scaled_x, scaled_y, scaled_z columns)
    if all(col in df.columns for col in ['scaled_x', 'scaled_y', 'scaled_z']):
        coord_targets = df[['scaled_x', 'scaled_y', 'scaled_z']].values
    
    return (x_data, continent_onehot, city_onehot, coord_targets, 
            continent_encoder, city_encoder,coordinate_encoder)


# =========================
# Main Pipeline
# =========================

# Load and process data
df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
x_data, continent_targets, city_targets, coord_targets, continent_encoder, city_encoder, coordinate_encoder = process_hierarchical_data(df)

# Example usage for hyperparameter tuning:
# tuner = HierarchicalGrowNetTuner(x_data, continent_targets, city_targets, coord_targets, params, device=device, n_trials=20, timeout=1200)
# best_params, best_score = tuner.tune()
# params.update(best_params)

# Train model
trained_mode,metrics = train_hierarchical_grownet(x_data,
                                                  continent_targets,
                                                  city_targets,
                                                  coord_targets, params)


# =========================
# Evaluation & Error Analysis
# =========================

print("\nClassification Report - Continent")
print(classification_report(
    metrics['targets']['continent'],
    metrics['predictions']['continent'],
    target_names=list(dict(zip(continent_encoder.transform(continent_encoder.classes_), continent_encoder.classes_)).values())
))

print("\nClassification Report - City")
print(classification_report(
    metrics['targets']['city'],
    metrics['predictions']['city'],
    target_names=list(dict(zip(city_encoder.transform(city_encoder.classes_), city_encoder.classes_)).values())
))

# --- Error calculation and analysis section (added for comparison with main.py) ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (in km).
    """
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def xyz_to_latlon(xyz_coords):
    """
    Convert the XYZ coordinates to latitude and longitude.
    """
    x, y, z = xyz_coords[:, 0], xyz_coords[:, 1], xyz_coords[:, 2]
    lat_rad = np.arcsin(np.clip(z, -1, 1))
    lon_rad = np.arctan2(y, x)
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    return np.stack([lat_deg, lon_deg], axis=1)

def error_calc_hierarchical_grownet(metrics, continent_encoder, city_encoder, coord_scaler):
    """
    Calculates error metrics for hierarchical GrowNet predictions.
    - Classification correctness
    - Haversine distance errors
    - Grouped error statistics
    - In-radius accuracy metrics
    """
    # Prepare error dataframe
    true_cont = np.array(metrics['targets']['continent'])
    pred_cont = np.array(metrics['predictions']['continent'])
    true_city = np.array(metrics['targets']['city'])
    pred_city = np.array(metrics['predictions']['city'])
    # True and predicted coordinates (xyz, scaled)
    true_xyz = np.array(metrics['targets']['coords'])
    pred_xyz = np.array(metrics['predictions']['coords'])
    # Inverse transform to original xyz
    true_xyz_orig = coord_scaler.inverse_transform(true_xyz)
    pred_xyz_orig = coord_scaler.inverse_transform(pred_xyz)
    # Convert to lat/lon
    true_latlon = xyz_to_latlon(true_xyz_orig)
    pred_latlon = xyz_to_latlon(pred_xyz_orig)
    # Build error dataframe
    error_df = pd.DataFrame({
        'true_cont': true_cont,
        'pred_cont': pred_cont,
        'true_city': true_city,
        'pred_city': pred_city,
        'true_lat': true_latlon[:, 0],
        'true_lon': true_latlon[:, 1],
        'pred_lat': pred_latlon[:, 0],
        'pred_lon': pred_latlon[:, 1]
    })
    # Assign names
    continents = dict(zip(continent_encoder.transform(continent_encoder.classes_), continent_encoder.classes_))
    cities = dict(zip(city_encoder.transform(city_encoder.classes_), city_encoder.classes_))
    error_df['true_cont_name'] = error_df['true_cont'].map(continents)
    error_df['pred_cont_name'] = error_df['pred_cont'].map(continents)
    error_df['true_city_name'] = error_df['true_city'].map(cities)
    error_df['pred_city_name'] = error_df['pred_city'].map(cities)
    # Support maps
    cont_support_map = error_df['true_cont_name'].value_counts().to_dict()
    city_support_map = error_df['true_city_name'].value_counts().to_dict()
    # Correctness
    error_df['continent_correct'] = error_df['true_cont'] == error_df['pred_cont']
    error_df['city_correct'] = error_df['true_city'] == error_df['pred_city']
    # Haversine error
    error_df['coord_error'] = haversine_distance(error_df['true_lat'], error_df['true_lon'], error_df['pred_lat'], error_df['pred_lon'])
    print(f'The median distance error is {np.median(error_df["coord_error"].values):.2f} km')
    print(f'The mean distance error is {np.mean(error_df["coord_error"].values):.2f} km')
    print(f'The max distance error is {np.max(error_df["coord_error"].values):.2f} km')
    # Error grouping
    def group_label(row):
        if row['continent_correct'] and row['city_correct']:
            return 'C_correct Z_correct'
        elif row['continent_correct'] and not row['city_correct']:
            return 'C_correct Z_wrong'
        elif not row['continent_correct'] and row['city_correct']:
            return 'C_wrong Z_correct'
        else:
            return 'C_wrong Z_wrong'
    error_df['error_group'] = error_df.apply(group_label, axis=1)
    group_stats = error_df.groupby('error_group')['coord_error'].agg([
        ('count', 'count'),
        ('mean_error_km', 'mean'),
        ('median_error_km', 'median')
    ])
    total = len(error_df)
    group_stats['proportion'] = group_stats['count'] / total
    group_stats['weighted_error'] = group_stats['mean_error_km'] * group_stats['proportion']
    expected_total_error = group_stats['weighted_error'].sum()
    print(group_stats)
    print(f"Expected Coordinate Error E[D]: {expected_total_error:.2f} km")
    # In-radius metrics
    def compute_in_radius_metrics(y_true, y_pred, thresholds=None):
        if thresholds is None:
            thresholds = [1, 5, 50, 100, 250, 500, 1000, 5000]
        distances = haversine_distance(y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1])
        results = {}
        for r in thresholds:
            percent = np.mean(distances <= r) * 100
            results[f"<{r} km"] = percent
        return results
    metrics_inradius = compute_in_radius_metrics(true_latlon, pred_latlon)
    print("In-Radius Accuracy Metrics:")
    for k, v in metrics_inradius.items():
        print(f"{k:>8}: {v:.2f}%")
    def in_radius_by_group(df, group_col, thresholds=[1, 5, 50, 100, 250, 500, 1000, 5000]):
        df = df.copy()
        df['coord_error'] = haversine_distance(
            df['true_lat'].values, df['true_lon'].values,
            df['pred_lat'].values, df['pred_lon'].values
        )
        results = {}
        grouped = df.groupby(group_col)
        for group_name, group_df in grouped:
            res = {}
            errors = group_df['coord_error'].values
            for r in thresholds:
                res[f"<{r} km"] = np.mean(errors <= r) * 100
            results[group_name] = res
        return pd.DataFrame(results).T
    continent_metrics = in_radius_by_group(error_df, group_col='true_cont_name')
    continent_metrics['continent_support'] = continent_metrics.index.map(cont_support_map)
    print("In-Radius Accuracy per Continent")
    print(continent_metrics.round(2))
    city_metrics = in_radius_by_group(error_df, group_col='true_city_name')
    city_metrics['city_support'] = city_metrics.index.map(city_support_map)
    print("In-Radius Accuracy per City")
    print(city_metrics.round(2))
    error_df['continent_city'] = error_df['true_cont_name'] + " / " + error_df['true_city_name']
    cont_city_metrics = in_radius_by_group(error_df, group_col='continent_city')
    cont_city_metrics['city_support'] = cont_city_metrics.index.map(lambda x: x.split("/")[-1].strip()).map(city_support_map)
    print("In-Radius Accuracy per Continent-City")
    print(cont_city_metrics.round(2))

    # Print the R2, MAE and RMSE for the coordinate predictions, using true coordinates
    from sklearn.metrics import mean_absolute_error, r2_score
    true_coords = np.stack([error_df['true_lat'].values, error_df['true_lon'].values]).T
    pred_coords = np.stack([error_df['pred_lat'].values, error_df['pred_lon'].values]).T
    r2 = r2_score(true_coords, pred_coords)
    print(f"Coordinate Prediction Metrics (degrees):") # R2 value after converting to lat/lon
    print(f"R^2:  {r2:.5f}")


# --- Run error calculation for hierarchical GrowNet ---
print("\n--- Hierarchical GrowNet Error Analysis ---")
error_calc_hierarchical_grownet(metrics, continent_encoder, city_encoder, coordinate_encoder)

# Print coordinate regression metrics
from sklearn.metrics import mean_squared_error, r2_score

true_coords = np.array(metrics['targets']['coords'])
pred_coords = np.array(metrics['predictions']['coords'])

coord_mse = mean_squared_error(true_coords, pred_coords)
coord_rmse = np.sqrt(coord_mse)
coord_r2 = r2_score(true_coords, pred_coords)

print(f"\nCoordinate Regression Metrics:")
print(f"R^2:  {coord_r2:.5f}")





