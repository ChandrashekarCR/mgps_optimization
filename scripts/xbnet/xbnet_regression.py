# Import Libraries
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from geopy.distance import geodesic 

def calculate_mae_km(df, predicted_lat_col, predicted_lon_col, true_lat_col, true_lon_col):
    """
    Calculates the Mean Absolute Error (MAE) in kilometers for predicted
    latitude and longitude compared to true values.

    Args:
        df (pd.DataFrame): DataFrame containing predicted and true latitude/longitude columns.
        predicted_lat_col (str): Name of the predicted latitude column.
        predicted_lon_col (str): Name of the predicted longitude column.
        true_lat_col (str): Name of the true latitude column.
        true_lon_col (str): Name of the true longitude column.

    Returns:
        tuple: A tuple containing the Mean Absolute Error for latitude (km)
               and the Mean Absolute Error for longitude (km). Returns (np.nan, np.nan)
               if the true latitude or longitude columns are not found.
    """
    lat_distances_km = []
    lon_distances_km = []

    if true_lat_col in df.columns and true_lon_col in df.columns:
        for idx, row in df.iterrows():
            pred_lat = row[predicted_lat_col]
            pred_lon = row[predicted_lon_col]
            true_lat = row[true_lat_col]
            true_lon = row[true_lon_col]

            if np.isfinite(pred_lat) and np.isfinite(pred_lon) and np.isfinite(true_lat) and np.isfinite(true_lon):
                # Approximate kilometers per degree
                lat_km_per_degree = geodesic((true_lat, pred_lon), (true_lat + 1, pred_lon)).kilometers
                lon_km_per_degree = geodesic((pred_lat, true_lon), (pred_lat, true_lon + 1)).kilometers

                lat_diff_km = abs(pred_lat - true_lat) * lat_km_per_degree
                lon_diff_km = abs(pred_lon - true_lon) * lon_km_per_degree

                lat_distances_km.append(lat_diff_km)
                lon_distances_km.append(lon_diff_km)
            else:
                lat_distances_km.append(np.nan)
                lon_distances_km.append(np.nan)

        mae_lat_km = np.nanmean(np.abs(lat_distances_km))
        mae_lon_km = np.nanmean(np.abs(lon_distances_km))

        return mae_lat_km, mae_lon_km
    else:
        print(f"Warning: '{true_lat_col}' or '{true_lon_col}' columns not found. Cannot calculate MAE in km.")
        return np.nan, np.nan

# Inverse transform xyz cordinates into latitude and longitude values
def inverse_transform_spherical(scaled_xyz, coordinate_scaler):
    """Inverse transforms scaled x, y, z back to latitude and longitude (degrees)."""
    xyz = coordinate_scaler.inverse_transform(scaled_xyz)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    latitude_rad = np.arcsin(np.clip(z, -1, 1))
    longitude_rad = np.arctan2(y, x)
    latitude_deg = np.degrees(latitude_rad)
    longitude_deg = np.degrees(longitude_rad)
    return latitude_deg, longitude_deg


# XBNet_Regression Architecture
class XBNet_Regressor(nn.Module):
    """
    XBNet (Extremely Bossted Neural Network for Regression tasks).
    
    This implementation combines gradient boosted trees with neural networks using Boosted Gradient Descent (BGD) optimixation technique.
    
    
    """
    def __init__(self, input_size, hidden_layers=[400,200],output_size=3,n_estimators=100, max_depth=3, dropout_rate=0.2, use_batch_norm=True, random_state=42):
        super(XBNet_Regressor, self).__init__()
        """
        Initialize XBNet architecture using Pytorch modules.

        Parameters:
        - input_size: Number of input features. In this case it is the GITs. # 200
        - hidden_layers: List of hidden layers # 400, 200
        - output_size: Latitude and longitude
        - n_estimators: Number of estimators for gradient boosting
        - max_depth: Maximum depth for gradient boosted trees
        - dropout_rate: L1 reqularization
        - random_state: Random state for reporducibility
        
        """

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.use_batch_norm = use_batch_norm

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Build the neural network
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Create the layer architecture
        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes)-1):
            # Add the linear layers first
            self.layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))

            # Add batch normalization for hidden layers only and not for the output layers
            if i < len(layer_sizes) - 2 and self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i+1]))

            # Add dropout for hidden layers onyl and not for the output layers
            if i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rate))

        # Initialiaze gradient boosting components
        self.xgb_tree_regressors = {} # Store multiple regressors for muilti-output
        self.feature_importances = {}
        self.layer_outputs = {}

        # Training history
        self.history = []
        self.accuracy_history = []

        # Initiliaze weight for the the first layers 
        self._initialize_weights()


        # Print the architecture for better visualization
        self._print_architecture()


    def _print_architecture(self):
        """Print the network architecture for debugging"""
        print("\nXBNet Architecture:")
        print(f"Input size: {self.input_size}")
        print(f"Batch Normalization: {'Enabled' if self.use_batch_norm else 'Disabled'}")
        
        for i, layer in enumerate(self.layers):
            layer_type = "Hidden" if i < len(self.layers) - 1 else "Output"
            print(f"Layer {i} ({layer_type}): {layer.in_features} -> {layer.out_features}")
            
            if i < len(self.layers) - 1:  # Hidden layers
                if self.use_batch_norm:
                    print(f"  + BatchNorm1d({layer.out_features})")
                print(f"  + ReLU")
                print(f"  + Dropout({self.dropout_rate})")
        
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
        print("-"*50)

    def _initialize_weights(self):
        """ Initialize weights for the neiral network using Xavier initilization."""
        for layer in self.layers[:-1]: # Hidden layers
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        # Ouput layer
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

        # Initialiaze the batch normalization parameters
        if self.use_batch_norm:
            for bn in self.batch_norms:
                nn.init.ones_(bn.weight)
                nn.init.zeros_(bn.bias)
    
    def initialiaze_first_layer_with_feature_importance(self, X,y):
        """
        
        Initialiaze first alyer weights using feauture importance from gradiet boosted trees. For regression, we
        train seperate regresseors for each output dimension.

        Parameters:
        - X: Input feautres (numpy array or tensor)
        - y: Target values (numpy array or tensor) - shape: (n_samples, output_size)
        
        """

        print(f"Initilizaing the first layer with gradient boosted feature importance ....")


        # Convert to numpy if torch tensor
        if torch.is_tensor(X):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X

        if torch.is_tensor(y):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y
        
        # Train separate XGB regressors for each output dimension
        combined_importance = np.zeros(X_np.shape[1])

        for output_dim in range(self.output_size):
            y_dim = y_np[:,output_dim] if y_np.shape[1] > output_dim else y_np[:,0]

            # Initialiaze XGB regressor
            xgb_reg = XGBRegressor(
                n_estimators = self.n_estimators,
                max_depth = self.max_depth,
                random_state = self.random_state,
                verbosity = 0
            )

            # Train on current output dimension
            xgb_reg.fit(X_np,y_dim)
            self.xgb_tree_regressors[output_dim] = xgb_reg

            # Accumulate feature importance
            combined_importance += xgb_reg.feature_importances_
        
        # Average the importance across all output dimesions
        combined_importance = combined_importance / self.output_size

        # Initialiaze the first layer weights with feature importance
        with torch.no_grad():
            first_layer = self.layers[0]
            input_size, first_hidden = first_layer.weight.shape[1], first_layer.weight.shape[0]
            new_weights = torch.zeros_like(first_layer.weight)

            for i in range(first_hidden):
                # Use feautre importnace to initialize each neuron's weight
                imporatance_weights = torch.tensor(combined_importance, dtype=torch.float32)
                
                # Add some noise to the weights
                noise = torch.normal(0,0.001,size=(input_size,))
                new_weights[i,:] = imporatance_weights * (0.1 + noise)

            first_layer.weight.copy_(new_weights)

        print(f"First layer initialized with feature importance. Shape: {first_layer.weight.shape}")

    def forward(self,x, store_activations=False):
        """
        Forward propagations through the network
        
        ParametersL
        - x: Input tensor
        - store_activations: Whether to store the intermediate actvations for tree training
        
        """

        if store_activations:
            self.layer_outputs = {}

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

            if store_activations:
                self.layer_outputs[i] = a.detach().cpu().numpy()

            a = dropout(a) if self.training else a # Apply dropout only during training
            current_input = a

        # Output layer (no activation for regression)
        output = self.layers[-1](current_input)

        return output
    
    def train_trees_on_hidden_layers(self,y):
        """
        
        Train gradient boosted trees on each hidden layer output for regression.

        Parameters:
        -y: Target values 
        
        """

        if torch.is_tensor(y):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y

        # Ensure y is 2D
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        
        self.feature_importances = {}

        for layer_idx, layer_output in self.layer_outputs.items():
            try:
                # For regression, we average importance across all output dimensions
                combined_importance = np.zeros(layer_output.shape[1])
                
                for output_dim in range(min(self.output_size, y_np.shape[1])):
                    y_dim = y_np[:, output_dim]
                    
                    # Train XGB regressor on this layer's output
                    xgb_reg = XGBRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                        verbosity=0
                    )
                    
                    xgb_reg.fit(layer_output, y_dim)
                    combined_importance += xgb_reg.feature_importances_
                
                # Average importance across output dimensions
                self.feature_importances[layer_idx] = combined_importance / min(self.output_size, y_np.shape[1])

            except Exception as e:
                print(f"Warning: Could not train tree on layer {layer_idx}: {e}")
                # Use uniform importance as fallback
                n_features = layer_output.shape[1]
                self.feature_importances[layer_idx] = np.ones(n_features) / n_features

    def update_weights_with_feature_importances(self):

        """
        
        Update weights using feature importance from graident boosted trees. THis is the second step of BGD optimization
        
        """

        with torch.no_grad():
            for layer_idx in self.feature_importances.keys():
                f_importance = self.feature_importances[layer_idx]

                # Apply importance to the weights going OUT of this layer into the next
                target_layer_idx = layer_idx + 1

                if target_layer_idx < len(self.layers):
                    target_layer = self.layers[target_layer_idx]
                    current_weights = target_layer.weight.data

                    # Check dimension compatibility
                    expected_input_size = current_weights.shape[1]

                    if len(f_importance) != expected_input_size:
                        print(f"Skipping layer {layer_idx}: importance size {len(f_importance)} != expected {expected_input_size}")
                        continue

                    # Calculate a conservative scaling factor
                    weight_std = torch.std(current_weights).item()
                    scaling_factor = weight_std * 0.01

                    f_scaled = torch.tensor(f_importance * scaling_factor, 
                                          dtype=torch.float32, 
                                          device=current_weights.device)

                    # Apply importance by scaling the input connections
                    for i in range(current_weights.shape[0]):
                        current_weights[i, :] += f_scaled


class XBNetRegressionTrainer:
    """
    Training class for XBNet regression neural network architecture.
    """

    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5, device=None):
        """
        Initialize trainer for regression.

        Parameters:
        - model: XBNetRegressor model
        - learning_rate: Learning rate for optimization
        - weight_decay: L2 regularization weight
        - device: Device to run training on
        """
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize optimizer and loss function for regression
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()  # Mean Squared Error for regression
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with activation storage
            output = self.model(data,store_activations=True)

            # Compute loss
            loss = self.criterion(output, target)

            # Train trees on hidden layer outputs (BGD Step 1)
            self.model.train_trees_on_hidden_layers(target)

            # Backward pass
            loss.backward()

            # Step 1: Standard gradient descent update
            self.optimizer.step()

            # Step 2: Update weights with feature importance (BGD Step 2)
            self.model.update_weights_with_feature_importances()

            # Statistics
            total_loss += loss.item()
            total_mse += loss.item()  # MSE is our loss function
            total_samples += target.size(0)

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'MSE': f'{total_mse / (batch_idx + 1):.6f}'
            })

        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)

        return avg_loss, avg_mse

    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_samples = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass without activation storage (faster)
                output = self.model(data, store_activations=False)

                # Compute loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                total_mse += loss.item()
                total_samples += target.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)

        return avg_loss, avg_mse

    def fit(self, train_loader, val_loader, epochs=100, patience=40, verbose=True):
        """
        Train the XBNet regression model.

        Parameters:
        - train_loader: Training data loader
        - val_loader: Validation data loader
        - epochs: Number of training epochs
        - patience: Early stopping patience
        - verbose: Whether to print progress
        """

        print(f"Starting XBNet Regression training for {epochs} epochs...")

        train_losses, train_mses = [], []
        val_losses, val_mses = [], []

        best_val_mse = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_mse = self.train_epoch(train_loader, epoch)

            # Validation
            val_loss, val_mse = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Record history
            train_losses.append(train_loss)
            train_mses.append(train_mse)
            val_losses.append(val_loss)
            val_mses.append(val_mse)

            # Early stopping based on validation MSE
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_xbnet_regression_model.pth')
            else:
                patience_counter += 1

            if verbose:
                print(f'Epoch {epoch + 1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.6f}, Train MSE: {train_mse:.6f}')
                print(f'  Val Loss: {val_loss:.6f}, Val MSE: {val_mse:.6f}')
                print(f'  Best Val MSE: {best_val_mse:.6f}')
                print('-' * 60)

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load the best model
        self.model.load_state_dict(torch.load('best_xbnet_regression_model.pth'))

        # Store training history
        self.model.loss_history = train_losses
        self.model.mse_history = train_mses
        self.val_loss_history = val_losses
        self.val_mse_history = val_mses

        print("XBNet Regression training completed!")
        print(f"Best validation MSE: {best_val_mse:.6f}")

    def predict(self, dataloader):
        """Make predictions on a dataset"""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                output = self.model(data, store_activations=False)
                predictions.extend(output.cpu().numpy())

        return np.array(predictions)

    def plot_training_history(self):
        """Plot training and validation history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.model.loss_history, label='Train Loss', color='blue')
        ax1.plot(self.val_loss_history, label='Val Loss', color='red')
        ax1.set_title('XBNet Regression Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_yscale('log')  # Log scale often better for regression losses

        # MSE plot
        ax2.plot(self.model.mse_history, label='Train MSE', color='blue')
        ax2.plot(self.val_mse_history, label='Val MSE', color='red')
        ax2.set_title('XBNet Regression Training and Validation MSE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE')
        ax2.legend()
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.show()


def create_regression_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    """Create PyTorch data loaders for regression"""

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_regression_model(y_true, y_pred, coordinate_names=['X', 'Y', 'Z']):
    """
    Evaluate regression model performance with multiple metrics.
    
    Parameters:
    - y_true: True target values
    - y_pred: Predicted values
    - coordinate_names: Names of the output dimensions
    """
    print("Regression Evaluation Metrics:")
    print("=" * 50)
    
    # Overall metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Overall Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print()
    
    # Per-dimension metrics
    print("Per-dimension Performance:")
    for i, coord_name in enumerate(coordinate_names[:y_true.shape[1]]):
        dim_mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        dim_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        dim_r2 = r2_score(y_true[:, i], y_pred[:, i])
        dim_rmse = np.sqrt(dim_mse)
        
        print(f"  {coord_name} coordinate:")
        print(f"    MSE: {dim_mse:.6f}")
        print(f"    RMSE: {dim_rmse:.6f}")
        print(f"    MAE: {dim_mae:.6f}")
        print(f"    R²: {dim_r2:.6f}")
    print()


# Function to process the data to feed it into the neural network
def process_data(data_path):

    try:
        in_data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")


    # Initialize label and scalers
    le_continent = LabelEncoder()
    le_city = LabelEncoder()
    stdscaler_lat = StandardScaler() 
    stdscaler_long = StandardScaler() 
    coordinate_scaler = StandardScaler()

    
    # Convert all the categorical variables into numbers
    in_data['city_encoding'] = in_data[['city']].apply(le_city.fit_transform)
    in_data['continent_encoding'] = in_data[['continent']].apply(le_continent.fit_transform)
    in_data['lat_scaled'] = stdscaler_lat.fit_transform(in_data[['latitude']])
    in_data['long_scaled'] = stdscaler_long.fit_transform(in_data[['longitude']])

    
    # Another way of scaling latitiude and longitude data. 
    # https://datascience.stackexchange.com/questions/13567/ways-to-deal-with-longitude-latitude-feature 
    # Convert latitude and longitutde into radians
    in_data['latitude_rad'] = np.deg2rad(in_data['latitude'])
    in_data['longitude_rad'] = np.deg2rad(in_data['longitude'])

    # Calculate x, y, z coordinates -  Converting polar co-ordinates into cartesian co-ordinates
    in_data['x'] = np.cos(in_data['latitude_rad']) * np.cos(in_data['longitude_rad'])
    in_data['y'] = np.cos(in_data['latitude_rad']) * np.sin(in_data['longitude_rad'])
    in_data['z'] = np.sin(in_data['latitude_rad'])

    # Scale the x, y, z coordinates together
    in_data[['scaled_x','scaled_y','scaled_z']] = coordinate_scaler.fit_transform (in_data[['x','y','z']])

    # Encoding dictionary for simpler plotting and understanding the results
    continent_encoding_map = dict(zip(le_continent.transform(le_continent.classes_), le_continent.classes_))
    city_encoding_map = dict(zip(le_city.transform(le_city.classes_),le_city.classes_))

    # Define all non-feature columns
    non_feature_columns = [
        'city', 'continent', 'latitude', 'longitude', # Original identifier/target columns
        'city_encoding', 'continent_encoding', # Encoded categorical targets
        'lat_scaled', 'long_scaled', # Old scaled lat/long (if not used as features)
        'latitude_rad', 'longitude_rad', # Intermediate radian values
        'x', 'y', 'z', # Intermediate cartesian coordinates
        'scaled_x', 'scaled_y', 'scaled_z','Unnamed: 0' # Final XYZ targets
    ]

    # Select X by dropping non-feature columns
    # Use errors='ignore' in case some columns don't exist (e.g., if you only keep one scaling method)
    X = in_data.drop(columns=non_feature_columns, errors='ignore').values.astype(np.float32)

    # Define target columns explicitly
    y_columns = ['continent_encoding', 'city_encoding', 'scaled_x','scaled_y','scaled_z']
    y = in_data[y_columns].values.astype(np.float32)

    return in_data, X, y, le_continent, le_city, coordinate_scaler, continent_encoding_map, city_encoding_map


def implement_xbnet_regression():
    """
    Implementation of XBNet for regression tasks.
    """
    
    in_data, X, y, le_continent, le_city, coordinate_scaler, continent_encoding_map, city_encoding_map = process_data("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")


    # Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X[:],y[:,2:],random_state=123,test_size=0.2)
    # Split train into train and validation as well
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=123, test_size=0.2)
    


    print('\nDataset splits:')
    print(f"Training: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"Testing: X={X_test.shape}, y={y_test.shape}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_regression_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128
    )

    # Initialize XBNet regressor
    model = XBNet_Regressor(
        input_size=200,
        hidden_layers=[512, 256, 128],
        output_size=3,  # X, Y, Z coordinates
        n_estimators=100,
        max_depth=5,
        dropout_rate=0.3,
        random_state=42
    )

    # Initialize first layer with feature importance
    model.initialiaze_first_layer_with_feature_importance(X_train, y_train)

    # Initialize trainer
    trainer = XBNetRegressionTrainer(model, learning_rate=0.001, weight_decay=1e-5)

    # Train the model
    trainer.fit(train_loader, val_loader, epochs=400, verbose=True, patience=40)

    # Make predictions on test set
    test_predictions = trainer.predict(test_loader)

    # Evaluate performance
    evaluate_regression_model(y_test, test_predictions, ['X', 'Y', 'Z'])

    true_lat, true_long = inverse_transform_spherical(y_test,coordinate_scaler)
    predicted_lat, predicted_long = inverse_transform_spherical(test_predictions,coordinate_scaler)

    # Create a DataFrame to hold the coordinates
    results_df = pd.DataFrame({
    'true_lat': true_lat,
    'true_long': true_long,
    'pred_lat': predicted_lat,
    'pred_long': predicted_long
    })

    # Call the MAE in km function
    mae_lat_km, mae_long_km = calculate_mae_km(
        results_df,
        predicted_lat_col='pred_lat',
        predicted_lon_col='pred_long',
        true_lat_col='true_lat',
        true_lon_col='true_long'
    )

    print("\nMean Absolute Error (in kilometers):")
    print(f"Latitude MAE: {mae_lat_km:.2f} km")
    print(f"Longitude MAE: {mae_long_km:.2f} km")

    # Plot training history
    trainer.plot_training_history()

    return model, trainer, test_predictions


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run the regression example
    model, trainer, predictions = implement_xbnet_regression()
    print("\nXBNet Regression implementation completed!")


