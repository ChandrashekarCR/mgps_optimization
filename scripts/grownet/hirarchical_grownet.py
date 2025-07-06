# Import Libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss, classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import time
import warnings
warnings.filterwarnings('ignore')


# Enhanced Parameters for Hierarchical Model
params = {
    "feat_d": 200,
    "hidden_size": 256,
    "n_continents": 7,
    "n_cities": None,  # Will be set based on data
    "coord_dim": 3,    # x, y, z coordinates
    "num_nets": 30,
    "boost_rate": 0.4,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "epochs_per_stage": 10,
    "correct_epoch": 5,
    "early_stopping_steps": 5,
    # Loss weighting - adaptive scheme
    "continent_weight": 3.0,
    "city_weight": 2.0,
    "coord_weight": 1.0,
    "weight_decay_factor": 0.95,  # Decay earlier stage weights as training progresses
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# Enhanced Dataset class for hierarchical targets
class HierarchicalTrainDataset:
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
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float)
        }
    
# Enhanced DynamicNet for hierarchical leanring
class HierarchicalDynamicNet:
    def __init__(self, c0_continent, c0_city, c0_coords, lr):
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
    

# Enhanced MLP for hierarchical prediction
class HierarchicalMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, n_continents, n_cities, coord_dim):
        super(HierarchicalMLP, self).__init__()
        self.bn = nn.BatchNorm1d(dim_in)
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim_in, dim_hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden1),
            nn.Dropout(0.4),
            nn.Linear(dim_hidden1, dim_hidden2),
            nn.ReLU()
        )
        
        # Continent head
        self.continent_head = nn.Linear(dim_hidden2, n_continents)
        
        # City head (uses continent predictions)
        self.city_head = nn.Linear(dim_hidden2 + n_continents, n_cities)
        
        # Coordinate head (uses both continent and city predictions)
        self.coord_head = nn.Linear(dim_hidden2 + n_continents + n_cities, coord_dim)

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn(x)
        
        # Extract shared features
        shared_features = self.feature_extractor(x)
        
        # Continent prediction
        continent_out = self.continent_head(shared_features)
        continent_probs = torch.softmax(continent_out, dim=1)
        
        # City prediction (augmented with continent)
        city_input = torch.cat([shared_features, continent_probs], dim=1)
        city_out = self.city_head(city_input)
        city_probs = torch.softmax(city_out, dim=1)
        
        # Coordinate prediction (augmented with continent and city)
        coord_input = torch.cat([shared_features, continent_probs, city_probs], dim=1)
        coord_out = self.coord_head(coord_input)
        
        return shared_features, continent_out, city_out, coord_out

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
            params["coord_dim"]
        )
        return model   

def get_optim(params, lr, weight_decay):
    optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
    return optimizer

def compute_hierarchical_loss(continent_pred, city_pred, coord_pred, 
                            continent_target, city_target, coord_target, 
                            continent_weight, city_weight, coord_weight):
    """Compute weighted hierarchical loss"""
    continent_loss = F.cross_entropy(continent_pred, torch.argmax(continent_target, dim=1))
    city_loss = F.cross_entropy(city_pred, torch.argmax(city_target, dim=1))
    coord_loss = F.mse_loss(coord_pred, coord_target)
    
    total_loss = (continent_weight * continent_loss + 
                 city_weight * city_loss + 
                 coord_weight * coord_loss)
    
    return total_loss, continent_loss, city_loss, coord_loss


def evaluate_hierarchical_model(net_ensemble, test_loader, params):
    """Evaluate the hierarchical model on all tasks"""
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
            
            # Compute losses
            total_loss, cont_loss, city_loss, coord_loss = compute_hierarchical_loss(
                continent_pred, city_pred, coord_pred,
                continent_target, city_target, coord_target,
                params["continent_weight"], params["city_weight"], params["coord_weight"]
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


# Training function
def train_hierarchical_grownet(x_data, continent_targets, city_targets, coord_targets, params):
    """Train the hierarchical GrowNet model"""
    
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
    
    # Initial evaluation
    initial_metrics = evaluate_hierarchical_model(net_ensemble, val_loader, params)
    print(f"Initial - Continent Acc: {initial_metrics['continent_acc']:.3f}, "
          f"City Acc: {initial_metrics['city_acc']:.3f}, "
          f"Coord MSE: {initial_metrics['coord_mse']:.5f}")
    
    # Training loop
    for stage in range(params["num_nets"]):
        t0 = time.time()
        print(f"\nTraining weak learner {stage+1}/{params['num_nets']}")
        
        # Adaptive weight decay
        current_continent_weight = params["continent_weight"] * (params["weight_decay_factor"] ** stage)
        current_city_weight = params["city_weight"] * (params["weight_decay_factor"] ** stage)
        current_coord_weight = params["coord_weight"]
        
        model = HierarchicalMLP.get_model(stage, params).to(device)
        optimizer = get_optim(model.parameters(), lr, params["weight_decay"])
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
                
                # Combined loss
                total_loss = (current_continent_weight * continent_loss + 
                             current_city_weight * city_loss + 
                             current_coord_weight * coord_loss)
                
                model.zero_grad()
                total_loss.backward()
                optimizer.step()
                stage_train_losses.append(total_loss.item())
        
        net_ensemble.add(model)
        avg_stage_loss = np.mean(stage_train_losses)
        print(f"Stage {stage+1} finished | Avg Train Loss: {avg_stage_loss:.5f} | Time: {time.time() - t0:.1f}s")
        
        # Corrective step
        if stage > 0:
            if stage % 3 == 0:
                lr /= 2
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
                        current_continent_weight, current_city_weight, current_coord_weight
                    )
                    
                    corrective_optimizer.zero_grad()
                    total_loss.backward()
                    corrective_optimizer.step()
                    corrective_losses.append(total_loss.item())
            
            print(f"Corrective step avg loss: {np.mean(corrective_losses):.5f}")
        
        # Validation
        val_metrics = evaluate_hierarchical_model(net_ensemble, val_loader, params)
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
    test_metrics = evaluate_hierarchical_model(net_ensemble, test_loader, params)
    print(f"\nFinal Test Results:")
    print(f"Continent Accuracy: {test_metrics['continent_acc']:.3f}")
    print(f"City Accuracy: {test_metrics['city_acc']:.3f}")
    print(f"Coordinate MSE: {test_metrics['coord_mse']:.5f}")
    
    return net_ensemble, test_metrics

# Data processing function
def process_hierarchical_data(df):
    """Process data for hierarchical learning"""
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


# Process data 
df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")

x_data, continent_targets, city_targets, coord_targets, continent_encoder, city_encoder, coordinate_encoder = process_hierarchical_data(df)

# Train model
trained_mode,metrics = train_hierarchical_grownet(x_data,
                                                  continent_targets,
                                                  city_targets,
                                                  coord_targets, params)


# Evaluation on the test metrics
print("\nClassification Report - Continent")
print(classification_report(metrics['targets']['continent'], metrics['predictions']['continent'],target_names=dict(zip(continent_encoder.transform(continent_encoder.classes_), continent_encoder.classes_)).values()))

print("\nConfusion Matrix - Continent")
print(confusion_matrix(metrics['targets']['continent'], metrics['predictions']['continent']))

print("\nClassification Report - City")
print(classification_report(metrics['targets']['city'], metrics['predictions']['city'],target_names=dict(zip(city_encoder.transform(city_encoder.classes_), city_encoder.classes_)).values()))

print("\nConfusion Matrix - City")
print(confusion_matrix(metrics['targets']['city'], metrics['predictions']['city']))


