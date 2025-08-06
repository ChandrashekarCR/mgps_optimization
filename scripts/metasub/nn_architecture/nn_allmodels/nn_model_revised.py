# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import copy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from torch.nn.utils import clip_grad_norm_


# Create a dynamic architecutre in pytorch
# First model - In this model, I made a neural network separate for each hierarchy. The same architechture is followed for 
# continent level prediction, city level prediction, cordinate level prediction.

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data processing function for hierarchical model
def process_data_hierarchical(df):
    """Process data for hierarchical prediction"""
    # Process continuous features
    cont_cols = [col for col in df.columns if col not in [
        'latitude', 'longitude',
        'latitude_rad', 'longitude_rad', 'x', 'y', 'z',
        'scaled_x', 'scaled_y', 'scaled_z', 'continent', 'city'
    ]]
    
    # Get the features
    x_cont = df[cont_cols].values
    
    # Encode continent labels
    continent_encoder = LabelEncoder()
    y_continent = continent_encoder.fit_transform(df['continent'].values)
    
    # Encode city labels
    city_encoder = LabelEncoder()
    y_city = city_encoder.fit_transform(df['city'].values)
    
    # Calculate coordinates if not already present
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        df['latitude_rad'] = np.deg2rad(df['latitude'])
        df['longitude_rad'] = np.deg2rad(df['longitude'])
        df['x'] = np.cos(df['latitude_rad']) * np.cos(df['longitude_rad'])
        df['y'] = np.cos(df['latitude_rad']) * np.sin(df['longitude_rad'])
        df['z'] = np.sin(df['latitude_rad'])
    
    # Scale coordinates
    coord_scaler = StandardScaler()
    y_coords = coord_scaler.fit_transform(df[['x', 'y', 'z']].values)
    
    continents = continent_encoder.classes_
    cities = city_encoder.classes_
    
    print(f"Continents: {len(continents)} ({continents})")
    print(f"Cities: {len(cities)}")
    print(f"Continuous features: {len(cont_cols)}")
    
    return {
        'x_cont': x_cont,
        'y_continent': y_continent,
        'y_city': y_city,
        'y_coords': y_coords, # This is for neural networks. Scaling is required
        'y_latitude': df['latitude'].values, # This is for XGBoost, we don't need to scale this
        'y_longitude':df['longitude'].values, # This is for XGBoost, we don't need to scale this
        'encoders': {
            'continent': continent_encoder,
            'city': city_encoder,
            'coord': coord_scaler
        },
        'continents': continents,
        'cities': cities
    }

# Hierarchial split to keep track of the indices
def hierarchical_split(X_cont, y_continent, y_city, y_coords, y_lat, y_lon, test_size=0.2, random_state=42):
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X_cont, y_continent))

    return {
        'X_train': X_cont[train_idx],
        'X_test': X_cont[test_idx],
        'y_cont_train': y_continent[train_idx],
        'y_cont_test': y_continent[test_idx],
        'y_city_train': y_city[train_idx],
        'y_city_test': y_city[test_idx],
        'y_coords_train': y_coords[train_idx],
        'y_coords_test': y_coords[test_idx],
        'y_lat_train': y_lat[train_idx],
        'y_lat_test': y_lat[test_idx],
        'y_lon_train': y_lon[train_idx],
        'y_lon_test': y_lon[test_idx],
        'train_idx': train_idx,
        'test_idx': test_idx
    }

# Process data
df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
processed_data = process_data_hierarchical(df)

X_cont = processed_data['x_cont']
y_cont = processed_data['y_continent']
y_cities = processed_data['y_city']
y_coords = processed_data['y_coords']
y_latitude = processed_data['y_latitude']
y_longitude = processed_data['y_longitude']


split_data = hierarchical_split(
    X_cont,
    y_cont,
    y_cities,
    y_coords,
    processed_data['y_latitude'],
    processed_data['y_longitude']
)

# Original feautres
X_train_cont, X_test_cont = split_data['X_train'], split_data['X_test']
# Train and test for continent
y_train_cont, y_test_cont = split_data['y_cont_train'], split_data['y_cont_test']
# Train and test for cities
y_train_city, y_test_city = split_data['y_city_train'], split_data['y_city_test']
# Train and test for latitude
y_train_lat, y_test_lat = split_data['y_lat_train'], split_data['y_lat_test']
# Train and test for longitude
y_train_lon, y_test_lon = split_data['y_lon_train'], split_data['y_lon_test']
# Train and test for co-ordinates
y_train_coords, y_test_coords = split_data['y_coords_train'],  split_data['y_coords_test']



# Neural network architecture
# ContinentLayer
class ClassificationLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = [128,64], use_batch_norm=True,
                  initial_dropout:float = 0.2, final_dropout:float =0.7, random_state=42):
        super(ClassificationLayer,self).__init__()

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

# Coordinate Layer
class CoordinateLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[256, 128, 64],
                 use_batch_norm=True, initial_dropout=0.2, final_dropout=0.5, random_state=42):
        super(CoordinateLayer,self).__init__()

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

# Dataset class for classification
class TrainDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float),
            'n_classes': torch.tensor(self.targets[idx], dtype=torch.long),
            'y': torch.tensor(self.targets[idx], dtype=torch.float)
        }

# Default parameters for the neural network
def default_params():
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

# Neural network tuner using Optuna
class NNTuner:
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

        # Determine if this is a classification or regression task
        # Classification: y_train is 1D integer, Regression: y_train is 2D or float
        if self.y_train.ndim == 1 and np.issubdtype(self.y_train.dtype, np.integer):
            model = NNClassifier(params, device=self.device)
            model.fit(self.X_train, self.y_train, X_val=self.X_val, y_val=self.y_val)
            val_metrics = model.evaluate(self.X_val, self.y_val)
            val_acc = val_metrics['class_accuracy']
            return val_acc
        else:
            model = NNRegressor(params, device=self.device)
            model.fit(self.X_train, self.y_train, X_val=self.X_val, y_val=self.y_val)
            val_metrics = model.evaluate(self.X_val, self.y_val)
            # Use negative RMSE for maximization
            return -val_metrics['rmse']

    def tune(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score
    
class NNClassifier:
    def __init__(self, params=None, device = "cpu",model=None):
        if params is None:
            self.params = params
        else:
            self.params = params
        self.device = device
        self.model = model
        self.class_weight_tensor = None
        self.best_model_state = None
    
    def fit(self, X_train, y_train, X_val=None,y_val=None):
        """ Train the model"""
        print("Fit the model...")

        # Update the parameters to this input
        self.params['input_dim'] = X_train.shape[1]
        self.params['output_dim'] = len(np.unique(y_train))


        # Only compute class weights if y_train is 1D (classification)
        if y_train.ndim == 1:
            class_weights = compute_class_weight(class_weight="balanced",classes=np.unique(y_train),y=y_train)
            self.class_weight_tensor = torch.tensor(class_weights,dtype=torch.float32).to(device)
        else:
            self.class_weight_tensor = None

        # Splut is validation is not given
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=self.params['val_split'],
                                                              random_state=self.params['random_state'], stratify=y_train)
        
        # Create datasets and dataloaders
        train_dataset = TrainDataset(X_train,y_train)
        val_dataset = TrainDataset(X_val,y_val)

        train_loader = DataLoader(train_dataset,batch_size=self.params['batch_size'],shuffle=True)
        val_loader = DataLoader(val_dataset,batch_size=self.params['batch_size'],shuffle=False)

        print(f"Train size {len(train_dataset)}, Val size {len(val_dataset)}")
        

        # Initailiaze model
        self.model = ClassificationLayer(
            input_dim=self.params['input_dim'],
            output_dim=self.params['output_dim'],
            hidden_dim=self.params['hidden_dim'],
            use_batch_norm=self.params['use_batch_norm'],
            initial_dropout=self.params['initial_dropout'],
            final_dropout=self.params['final_dropout'],
            random_state=self.params['random_state']
        ).to(device)

        # Loss function and evaluation for classification
        if self.class_weight_tensor is not None:
            criterion_classification = nn.CrossEntropyLoss(weight=self.class_weight_tensor)
        else:
            criterion_classification = nn.CrossEntropyLoss()
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
                features = batch['x'].to(device)
                targets = batch['n_classes'].to(device)

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
                    features = batch['x'].to(device)
                    targets = batch['n_classes'].to(device)

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

    def evaluate(self,X,y):
        """
        Evaluate the model
        
        """

        self.model.eval()
        all_preds = []
        all_targets = []
        class_lossses = []
        all_preds_prob = []

        dataset = TrainDataset(X,y)
        loader = DataLoader(dataset,batch_size=self.params['batch_size'],shuffle=False)

        criterion = nn.CrossEntropyLoss(weight=self.class_weight_tensor)

        with torch.no_grad():
            for batch in loader:
                features = batch['x'].to(device)
                targets = batch['n_classes'].to(device)

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
        dataset = TrainDataset(X, dummy_targets)
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
    
def run_nn_classifier(X_train,y_train, X_test,y_test,device="cuda",
                      tune_hyperparams=False,params=None,
                          n_trials=20,timeout=1200):
        # Use default if params not given
        if params is None:
            params = default_params()
        else:
            default = default_params()
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

            tuner = NNTuner(X_train_split, y_train_split, X_val, y_val, 
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

class NNRegressor:
    def __init__(self, params=None, device="cpu"):
        if params is None:
            self.params = default_params()
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
        train_dataset = TrainDataset(X_train, y_train)
        val_dataset = TrainDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.params['batch_size'], shuffle=False)

        print(f"Train size {len(train_dataset)}, Val size {len(val_dataset)}")

        # Initialize model
        self.model = CoordinateLayer(
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

        dataset = TrainDataset(X, y)
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
        dataset = TrainDataset(X, dummy_targets)
        loader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                features = batch['x'].to(self.device)
                outputs = self.model(features).squeeze()
                all_preds.extend(outputs.cpu().numpy())
        
        # Convert to numpy array and reshape
        all_preds = np.array(all_preds)
        
        return all_preds

def run_nn_regressor(X_train, y_train, X_test, y_test, device="cuda",
                     tune_hyperparams=False, params=None,
                     n_trials=20, timeout=1200):
    """Run the neural network regressor"""
    
    # Use default if params not given
    if params is None:
        params = default_params()
    else:
        default = default_params()
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

        tuner = NNTuner(X_train_split, y_train_split, X_val, y_val, 
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

# Distance between two points on the earth
def haversine_distance(lat1,lon1,lat2,lon2):
    """
    Calculate the great circle distance between two points on the earth
    """
    # Radius of the earth
    R = 6371.0

    # Convert from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2) **2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c # in kilometers

# Converting cartesian co-ordinates values to latitude and longitude
def xyz_to_latlon(xyz_coords):
    """
    Convert the XYZ coordinates to latitude and longitude
    """
    x,y,z = xyz_coords[:,0],xyz_coords[:,1],xyz_coords[:,2]

    # Convert to latitude and longitude
    lat_rad = np.arcsin(np.clip(z,-1,1)) # Clip to avoid numerical issues
    lon_rad = np.arctan2(y,x)

    # Convert to degrees
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)

    return np.stack([lat_deg,lon_deg],axis=1)

def run_hierarchical_nn_model(X_train, X_test, y_train_cont, y_test_cont, 

                              y_train_city, y_test_city, y_train_coords, y_test_coords,
                              device="cuda", tune_hyperparams=False, n_trials=20, timeout=1200):
    """
    Run hierarchical neural network model that:
    1. Predicts continents
    2. Uses continent probabilities to help predict cities
    3. Uses both continent and city probabilities to predict coordinates
    
    Returns:
        Dictionary with all models and predictions
    """
    print("\n===== HIERARCHICAL MODEL: LEVEL 1 - CONTINENT PREDICTION =====")
    # Step 1: Train and predict continents
    continent_params = {
        "hidden_dim": [128, 64],
        "output_dim": len(np.unique(y_train_cont)),
        "use_batch_norm": True,
        "initial_dropout": 0.3,
        "final_dropout": 0.7,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "epochs": 400,
        "early_stopping_steps": 20,
        "gradient_clip": 1.0
    }
    
    continent_results = run_nn_classifier(
        X_train, y_train_cont, X_test, y_test_cont,
        device=device, tune_hyperparams=tune_hyperparams, 
        params=continent_params, n_trials=n_trials, timeout=timeout
    )
    
    # Get continent probabilities for training and test sets
    continent_model = continent_results['model']
    
    # Create augmented features with continent probabilities
    train_cont_probs = continent_model.predict(X_train)['probabilities']
    test_cont_probs = continent_model.predict(X_test)['probabilities']
    
    X_train_aug = np.hstack((X_train, train_cont_probs))
    X_test_aug = np.hstack((X_test, test_cont_probs))
    
    print(f"Augmented feature shape: {X_train_aug.shape} (original + {train_cont_probs.shape[1]} continent probabilities)")
    
    print("\n===== HIERARCHICAL MODEL: LEVEL 2 - CITY PREDICTION =====")
    # Step 2: Train and predict cities using augmented features
    city_params = {
        "hidden_dim": [256, 128, 64],
        "output_dim": len(np.unique(y_train_city)),
        "use_batch_norm": True,
        "initial_dropout": 0.3,
        "final_dropout": 0.7,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "epochs": 400,
        "early_stopping_steps": 20,
        "gradient_clip": 1.0
    }
    
    city_results = run_nn_classifier(
        X_train_aug, y_train_city, X_test_aug, y_test_city,
        device=device, tune_hyperparams=tune_hyperparams, 
        params=city_params, n_trials=n_trials, timeout=timeout
    )
    
    # Get city probabilities
    city_model = city_results['model']
    train_city_probs = city_model.predict(X_train_aug)['probabilities']
    test_city_probs = city_model.predict(X_test_aug)['probabilities']
    
    # Create fully augmented features with both continent and city probabilities
    X_train_full_aug = np.hstack((X_train, train_cont_probs, train_city_probs))
    X_test_full_aug = np.hstack((X_test, test_cont_probs, test_city_probs))
    
    print(f"Fully augmented feature shape: {X_train_full_aug.shape} "+
          f"(original + {train_cont_probs.shape[1]} continent + {train_city_probs.shape[1]} city probabilities)")
    
    print("\n===== HIERARCHICAL MODEL: LEVEL 3 - COORDINATE PREDICTION =====")
    # Step 3: Train and predict coordinates using fully augmented features
    coord_params = {
        "hidden_dim": [256, 128, 64],
        "output_dim": y_train_coords.shape[1],  # 3D coordinates
        "use_batch_norm": True,
        "initial_dropout": 0.2,
        "final_dropout": 0.5,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "epochs": 600,
        "early_stopping_steps": 30,
        "gradient_clip": 1.0
    }
    
    # Use run_nn_regressor for coordinates
    coord_results = run_nn_regressor(
        X_train_full_aug, y_train_coords, X_test_full_aug, y_test_coords,
        device=device, tune_hyperparams=tune_hyperparams, 
        params=coord_params, n_trials=n_trials, timeout=timeout
    )
    
    return {
        'continent_model': continent_model,
        'city_model': city_model,
        'coordinate_model': coord_results['model'],
        'continent_accuracy': continent_results['accuracy'],
        'city_accuracy': city_results['accuracy'],
        'coordinate_rmse': coord_results['rmse'],
        'coordinate_r2': coord_results['r2'],
        'continent_predictions': continent_results['predictions'],
        'city_predictions': city_results['predictions'],
        'coordinate_predictions': coord_results['predictions']
    }


# Run the hierarchical model
hierarchical_results = run_hierarchical_nn_model(
    X_train_cont, X_test_cont,
    y_train_cont, y_test_cont,
    y_train_city, y_test_city,
    y_train_coords, y_test_coords,
    device=device, 
    tune_hyperparams=False,
    n_trials=20, 
    timeout=100
)


# Error calculations
def error_calc(test_conts, pred_conts, test_city, pred_city, test_lat, test_lon, pred_lat, pred_lon):
    # Notice the corrected parameter order to match function implementation
    error_df = pd.DataFrame({
        'true_cont': test_conts,
        'pred_cont': pred_conts,
        'true_city': test_city,
        'pred_city': pred_city,
        'true_lat': test_lat,
        'true_lon': test_lon,
        'pred_lat': pred_lat,
        'pred_lon': pred_lon
    })

    # Assign true continent and city names
    error_df['true_cont_name'] = error_df['true_cont'].map(lambda i: processed_data['continents'][i])
    error_df['pred_cont_name'] = error_df['pred_cont'].map(lambda i: processed_data['continents'][i])

    error_df['true_city_name'] = error_df['true_city'].map(lambda i: processed_data['cities'][i])
    error_df['pred_city_name'] = error_df['pred_city'].map(lambda i: processed_data['cities'][i])

    cont_support_map = dict(zip(np.unique(error_df['true_cont_name'], return_counts=True)[0], np.unique(error_df['true_cont_name'], return_counts=True)[1]))
    city_support_map = dict(zip(np.unique(error_df['true_city_name'], return_counts=True)[0], np.unique(error_df['true_city_name'], return_counts=True)[1]))

    # Step 1: Compute the correctness
    error_df['continent_correct'] = error_df['true_cont'] == error_df['pred_cont']
    error_df['city_correct'] = error_df['true_city'] == error_df['pred_city']

    # Step 2: Calculate the haversine distance
    error_df['coord_error'] = haversine_distance(error_df['true_lat'], error_df['true_lon'], error_df['pred_lat'], error_df['pred_lon'])

    print(f'The median distance error is {np.median(error_df['coord_error'].values)}')
    print(f'The mean distance error is {np.mean(error_df['coord_error'].values)}')
    print(f'The max distance error is {np.max(error_df['coord_error'].values)}')

    # Step 3: Group into 4 categories
    def group_label(row):
        if row['continent_correct'] and row['city_correct']:
            return 'C_correct Z_correct'
        elif row['continent_correct'] and not row['city_correct']:
            return 'C_correct Z_wrong'
        elif not row['continent_correct'] and row['city_correct']:
            return 'C_wrong Z_correct'
        else:
            return 'C_wrong Z_wrong'
        
    # Create the error group column
    error_df['error_group'] = error_df.apply(group_label, axis=1)

    # Now we proceed with grouping
    group_stats = error_df.groupby('error_group')['coord_error'].agg([
        ('count', 'count'),
        ('mean_error_km', 'mean'),
        ('median_error_km', 'median')
    ])

    # Step 5: Calculate proportion and expected error.
    """
    P(C=C*) : Probability of continent predicting correct continent
    P(Z=Z*) : Probability of city predicting correct city
    E(D|condition) : Expected distance error under that condition

    E(D) = P(C=C*,Z=Z*)*E(D|C=C*,Z=Z*)+ -> ideal condition continent is correct and city is also correct
            P(C=C*,Z!=Z*)*E(D|C=C*,Z!=Z*)+ -> continent is correct and city is wrong
            P(C!=C*,Z=Z*)*E(D|C!=C*,Z=Z*)+ -> city is correct but continent is wrong
            P(C!=C*,Z!=Z*)*E(D|C!=C*,Z!=Z*) -> both continent and city are wrong
    """
    total = len(error_df)
    group_stats['proportion'] = group_stats['count'] / total
    group_stats['weighted_error'] = group_stats['mean_error_km'] * group_stats['proportion']
    expected_total_error = group_stats['weighted_error'].sum()
    print(group_stats)
    print(f"Expected Coordinate Error E[D]: {expected_total_error:.2f} km")

    def compute_in_radius_metrics(y_true, y_pred, thresholds=None):
        """
        Compute % of predictions within given distance thresholds
        y_true, y_pred: numpy arrays of shape (N, 2) for [lat, lon]
        thresholds: List of distance thresholds in kilometers (default: [1, 5, 50, 100, 250, 500, 1000, 5000])
        """
        if thresholds is None:
            thresholds = [1, 5, 50, 100, 250, 500, 1000, 5000]

        distances = haversine_distance(
            y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1]
        )

        results = {}
        for r in thresholds:
            percent = np.mean(distances <= r) * 100
            results[f"<{r} km"] = percent

        return results

    metrics = compute_in_radius_metrics(y_true=np.stack([test_lat, test_lon], axis=1), y_pred=np.stack([pred_lat, pred_lon], axis=1))

    print("In-Radius Accuracy Metrics:")
    for k, v in metrics.items():
        print(f"{k:>8}: {v:.2f}%")
        
    def in_radius_by_group(df, group_col, thresholds=[1, 5, 50, 100, 250, 500, 1000, 5000]):
        """
        Compute in-radius accuracy for a group column (continent, city, or continent+city)
        """
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
                res[f"<{r} km"] = np.mean(errors <= r) * 100  # in %
            results[group_name] = res

        return pd.DataFrame(results).T  # Transpose for better readability
    
    continent_metrics = in_radius_by_group(error_df, group_col='true_cont_name')
    print("In-Radius Accuracy per Continent")
    continent_metrics['continent_support'] = continent_metrics.index.map(cont_support_map)
    print(continent_metrics.round(2))

    city_metrics = in_radius_by_group(error_df, group_col='true_city_name')
    print("In-Radius Accuracy per City")
    city_metrics['city_support'] = city_metrics.index.map(city_support_map)
    print(city_metrics.round(2))

    error_df['continent_city'] = error_df['true_cont_name'] + " / " + error_df['true_city_name']
    cont_city_metrics = in_radius_by_group(error_df, group_col='continent_city')
    cont_city_metrics['continent_support'] = cont_city_metrics.index.map(lambda x: x.split("/")[-1].strip()).map(city_support_map)
    print("In-Radius Accuracy per Continent-City")
    print(cont_city_metrics.round(2))

# Fix coordinate predictions from cartesian to lat/lon
predicted_coords = hierarchical_results['coordinate_predictions']
predicted_latlon = xyz_to_latlon(predicted_coords)

# Error calculations for the hierarchical model predictions
print("Starting error calculations for hierarchical model...")
error_calc(
    test_conts=y_test_cont,
    pred_conts=hierarchical_results['continent_predictions'],
    test_city=y_test_city,
    pred_city=hierarchical_results['city_predictions'],
    test_lat=y_test_lat,
    test_lon=y_test_lon,
    pred_lat=predicted_latlon[:, 0],
    pred_lon=predicted_latlon[:, 1]
)

print("\n===== HIERARCHICAL MODEL RESULTS SUMMARY =====")
print(f"Continent Classification Accuracy: {hierarchical_results['continent_accuracy']:.4f}")
print(f"City Classification Accuracy: {hierarchical_results['city_accuracy']:.4f}")
print(f"Coordinate Prediction RMSE: {hierarchical_results['coordinate_rmse']:.4f}")
print(f"Coordinate Prediction R2: {hierarchical_results['coordinate_r2']:.4f}")