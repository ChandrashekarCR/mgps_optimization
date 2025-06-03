# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import air
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from check_accuracy_model import plot_losses, plot_confusion_matrix, plot_points_on_world_map, pull_land, plot_predictions_with_coastline, calculate_mae_km
from process_data import process_data

# Neural Network Architecture
class CombinedNeuralNetXYZModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_continent, num_cities,dropout_rate=0.65):
        super(CombinedNeuralNetXYZModel,self).__init__()

        # ReLU activation function
        self.relu = nn.ReLU()

        # Continent Architechture
        self.continent_layer_1 = nn.Linear(input_size,hidden_dim) 
        self.continent_bn_1 = nn.BatchNorm1d(hidden_dim)
        self.continent_dropout_1 = nn.Dropout(dropout_rate)
        self.continent_layer_2 = nn.Linear(hidden_dim,hidden_dim//2)
        self.continent_bn_2 = nn.BatchNorm1d(hidden_dim//2)
        self.continent_layer_3 = nn.Linear(hidden_dim//2,hidden_dim//4)
        self.continent_bn_3 = nn.BatchNorm1d(hidden_dim//4)

        # Continent Prediction
        self.continent_prediction = nn.Linear(hidden_dim//4,num_continent) # Output for 7 different continents

        # City Architecture
        self.city_layer_1 = nn.Linear(input_size+num_continent,hidden_dim) # Concatenate the output of the continent layers
        self.city_bn_1 = nn.BatchNorm1d(hidden_dim)
        self.city_dropout_1 = nn.Dropout(dropout_rate)
        self.city_layer_2 = nn.Linear(hidden_dim,hidden_dim//2)
        self.city_bn_2 = nn.BatchNorm1d(hidden_dim//2)
        self.city_layer_3 = nn.Linear(hidden_dim//2,hidden_dim//4)
        self.city_bn_3 = nn.BatchNorm1d(hidden_dim//4)

        # City Prediction
        self.city_prediction = nn.Linear(hidden_dim//4,num_cities) # Output for 40 different cities

        # XYZ Architecture
        self.xyz_layer_1 = nn.Linear(input_size+num_continent+num_cities,hidden_dim) # Concatenate the output of the continent and cities layers
        self.xyz_layer_bn_1 = nn.BatchNorm1d(hidden_dim)
        self.xyz_dropout_1 = nn.Dropout(dropout_rate)
        self.xyz_layer_2 = nn.Linear(hidden_dim,hidden_dim//2)
        self.xyz_layer_bn_2 = nn.BatchNorm1d(hidden_dim//2)
        self.xyz_layer_3 = nn.Linear(hidden_dim//2,hidden_dim//4)
        self.xyz_layer_bn_3 = nn.BatchNorm1d(hidden_dim//4)
        
        # XYZ Prediction
        self.xyz_prediction = nn.Linear(hidden_dim//4,3) # Three xyz co-ordinates

        # Add learnable uncertainty parameters
        # The learnable sigma parameters (σ1​, σ2​, σ3​) are introduced to allow the model to estimate the uncertainty 
        # associated with its predictions for continents, cities, and XYZ coordinates, respectively.
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0)) # For continent
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0)) # For city
        self.log_sigma3 = nn.Parameter(torch.tensor(0.0)) # For xyz


    
    def forward(self,x):

        # Continent Architecture
        out_continent = self.relu(self.continent_bn_1(self.continent_layer_1(x)))
        out_continent = self.continent_dropout_1(out_continent)
        out_continent = self.relu(self.continent_bn_2(self.continent_layer_2(out_continent)))
        out_continent = self.relu(self.continent_bn_3(self.continent_layer_3(out_continent)))
        
        # Continent Prediction
        continent_predictions = self.continent_prediction(out_continent)
        continent_probs = F.softmax(continent_predictions, dim=1) # Only using the probabilities as concatenated input to the next layer

        # City Architecture
        input_for_city_layer = torch.cat((x,continent_probs),dim=1)
        out_cities = self.relu(self.city_bn_1(self.city_layer_1(input_for_city_layer)))
        out_cities = self.city_dropout_1(out_cities)
        out_cities = self.relu(self.city_bn_2(self.city_layer_2(out_cities)))
        out_cities = self.relu(self.city_bn_3(self.city_layer_3(out_cities)))

        # City Prediction
        city_predictions  = self.city_prediction(out_cities)
        city_probs = F.softmax(city_predictions, dim =1) # Only using the probabilities as concatenated input to the next layer

        # XYZ Architecture
        input_for_xyz_layer = torch.cat((x, continent_probs, city_probs),dim=1)
        out_xyz = self.relu(self.xyz_layer_bn_1(self.xyz_layer_1(input_for_xyz_layer)))
        out_xyz = self.xyz_dropout_1(out_xyz)
        out_xyz = self.relu(self.xyz_layer_bn_2(self.xyz_layer_2(out_xyz)))
        out_xyz = self.relu(self.xyz_layer_bn_3(self.xyz_layer_3(out_xyz)))

        # XYZ Prediction
        xyz_prediction = self.xyz_prediction(out_xyz)

        return continent_predictions, city_predictions, xyz_prediction
    
# Data loading and splitting functions
def load_data(data_path):
    try:
        in_data = pd.read_csv(data_path)
        return in_data
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None

def split_data(in_data, test_size, random_state):
    X = in_data.iloc[:, :200].values.astype(np.float32)
    y = in_data[['continent_encoding', 'city_encoding', 'scaled_x','scaled_y','scaled_z']].values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Custom Dataset class
class CustDat(Dataset):
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        dp = torch.tensor(self.df[idx], dtype=torch.float32)
        targ = torch.tensor(self.target[idx], dtype=torch.float32)
        continent_city = targ[:2].long()
        lat_lon = targ[2:]
        return dp, continent_city, lat_lon

# Function to compute losses and return them
def compute_loss(model, criterion_continent, criterion_cities, criterion_lat_lon,
                 continent_logits, city_logits, xyz_logits, cont_targ, city_targ, xyz_targ):
    
    """Computes the individual and combined loss."""
    loss_continents = criterion_continent(continent_logits, cont_targ)
    loss_cities = criterion_cities(city_logits, city_targ)
    loss_xyz = criterion_lat_lon(xyz_logits, xyz_targ)

    sigma1 = torch.exp(model.log_sigma1)
    sigma2 = torch.exp(model.log_sigma2)
    sigma3 = torch.exp(model.log_sigma3)

    total_loss = (
        (1 / (2 * sigma1**2)) * loss_continents + torch.log(sigma1) +
        (1 / (2 * sigma2**2)) * loss_cities + torch.log(sigma2) +
        (1 / (2 * sigma3**2)) * loss_xyz + torch.log(sigma3)
    )
    return loss_continents, loss_cities, loss_xyz, total_loss

def training_loop(train_dl, val_dl, combined_model, optimizer_combined, scheduler, criterion_continent,
                 criterion_cities, criterion_lat_lon, device, num_epochs):
    
    start_time = time.time()
    train_losses = {'continent': [], 'cities': [], 'xyz': [], 'total': []}
    val_losses = {'continent': [], 'cities': [], 'xyz': [], 'total': []}

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        combined_model.train()
        train_metrics = {'continent': 0, 'cities': 0, 'xyz': 0, 'total': 0}
        
        for data, continent_city, lat_long in train_dl:
            data = data.to(device)
            cont_targ = continent_city[:, 0].long().to(device)
            city_targ = continent_city[:, 1].long().to(device)
            xyz_targ = lat_long.float().to(device)

            # Forward pass
            optimizer_combined.zero_grad()
            continent_logits, city_logits, xyz_logits = combined_model(data)

            # Calculate losses
            loss_continents, loss_cities, loss_xyz, total_loss = compute_loss(
                combined_model, criterion_continent, criterion_cities, criterion_lat_lon,
                continent_logits, city_logits, xyz_logits, cont_targ, city_targ, xyz_targ
            )

            # Backward pass
            total_loss.backward()
            optimizer_combined.step()

            # Accumulate metrics
            train_metrics['continent'] += loss_continents.item()
            train_metrics['cities'] += loss_cities.item()
            train_metrics['xyz'] += loss_xyz.item()
            train_metrics['total'] += total_loss.item()

        # Calculate average training metrics
        num_train_batches = len(train_dl)
        avg_train_losses = {k: v / num_train_batches for k, v in train_metrics.items()}
        for k, v in avg_train_losses.items():
            train_losses[k].append(v)

        # Validation phase
        combined_model.eval()
        val_metrics = {'continent': 0, 'cities': 0, 'xyz': 0, 'total': 0}
        
        with torch.no_grad():
            for data_val, continent_city_val, lat_long_val in val_dl:
                data_val = data_val.to(device)
                cont_targ_val = continent_city_val[:, 0].long().to(device)
                city_targ_val = continent_city_val[:, 1].long().to(device)
                xyz_targ_val = lat_long_val.float().to(device)

                # Forward pass
                continent_logits_val, city_logits_val, xyz_logits_val = combined_model(data_val)

                # Calculate losses
                loss_continents_val, loss_cities_val, loss_xyz_val, total_val_loss = compute_loss(
                    combined_model, criterion_continent, criterion_cities, criterion_lat_lon,
                    continent_logits_val, city_logits_val, xyz_logits_val, cont_targ_val, city_targ_val, xyz_targ_val
                )

                # Accumulate metrics
                val_metrics['continent'] += loss_continents_val.item()
                val_metrics['cities'] += loss_cities_val.item()
                val_metrics['xyz'] += loss_xyz_val.item()
                val_metrics['total'] += total_val_loss.item()

        # Calculate average validation metrics with fallback for empty validation set
        num_val_batches = len(val_dl)
        if num_val_batches > 0:
            avg_val_losses = {k: v / num_val_batches for k, v in val_metrics.items()}
        else:
            avg_val_losses = {k: 0.0 for k in val_metrics.keys()}

        for k, v in avg_val_losses.items():
            val_losses[k].append(v)

        # Report to Ray Tune with fallback values
        tune.report({
            "epoch": epoch + 1,
            "train_loss": avg_train_losses.get('total', 0.0),
            "val_loss": avg_val_losses.get('xyz', 0.0),
            "train_continent_loss": avg_train_losses.get('continent', 0.0),
            "val_continent_loss": avg_val_losses.get('continent', 0.0),
            "train_cities_loss": avg_train_losses.get('cities', 0.0),
            "val_cities_loss": avg_val_losses.get('cities', 0.0),
            "train_xyz_loss": avg_train_losses.get('xyz', 0.0),
            "val_xyz_loss": avg_val_losses.get('xyz', 0.0)
        })

        # Scheduler step with fallback value
        scheduler.step(avg_val_losses.get('total', 0.0))

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"LR: {optimizer_combined.param_groups[0]['lr']:.2e} | "
                  f"Train Loss - Total: {avg_train_losses['total']:.4f}, "
                  f"Continent: {avg_train_losses['continent']:.4f}, "
                  f"Cities: {avg_train_losses['cities']:.4f}, XYZ: {avg_train_losses['xyz']:.4f} | "
                  f"Val Loss - Total: {avg_val_losses['total']:.4f}, "
                  f"Continent: {avg_val_losses['continent']:.4f}, "
                  f"Cities: {avg_val_losses['cities']:.4f}, XYZ: {avg_val_losses['xyz']:.4f} | "
                  f"Time: {epoch_duration:.2f}s")

    total_training_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    return train_losses, val_losses


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

# Check the accuracy of the neural netowrk on train and test data
def check_combined_accuracy(loader, model, coordinate_scaler=None, device="cpu"):
    
    # Testing phase
    # Initialize test phase
    model.eval()

    correct_continent = 0
    correct_cities = 0
    total = 0
    all_prediction_continents = []
    all_prediction_cities = []
    all_prediction_xyz = []
    all_target_continents = []
    all_target_cities = []
    all_target_xyz = []

    with torch.no_grad():
        for batch_idx, (data, continent_city, lat_long_rad) in enumerate(loader):
            batch_size = data.size(0)
            data = data.to(device)
            cont_targ = continent_city[:, 0].long().to(device)
            city_targ = continent_city[:, 1].long().to(device)
            xyz_targ_scaled = lat_long_rad.float().to(device)

            continent_logits, city_logits, xyz_pred_scaled = model(data)

            _, predictions_continent = continent_logits.max(1)
            correct_continent += (predictions_continent == cont_targ).sum().item()
            all_prediction_continents.extend(predictions_continent.detach().cpu().numpy())
            all_target_continents.extend(cont_targ.detach().cpu().numpy())

            _, predictions_cities = city_logits.max(1)
            correct_cities += (predictions_cities == city_targ).sum().item()
            all_prediction_cities.extend(predictions_cities.detach().cpu().numpy())
            all_target_cities.extend(city_targ.detach().cpu().numpy())

            all_prediction_xyz.append(xyz_pred_scaled.detach().cpu().numpy())
            all_target_xyz.append(xyz_targ_scaled.detach().cpu().numpy())

            total += batch_size

    accuracy_continent = correct_continent / total * 100
    accuracy_cities = correct_cities / total * 100

    all_prediction_xyz = np.concatenate(all_prediction_xyz, axis=0)
    all_target_xyz = np.concatenate(all_target_xyz, axis=0)

    predicted_lat_deg, predicted_long_deg = inverse_transform_spherical(all_prediction_xyz, coordinate_scaler)
    target_lat_deg, target_long_deg = inverse_transform_spherical(all_target_xyz, coordinate_scaler)

    predictions_df = pd.DataFrame({
        'predicted_lat': predicted_lat_deg,
        'predicted_lon': predicted_long_deg,
        'true_latitude': target_lat_deg,
        'true_longitude': target_long_deg
    })

    # Call the MAE (km) calculation function
    mae_lat_km, mae_lon_km = calculate_mae_km(
    predictions_df,
    predicted_lat_col='predicted_lat',
    predicted_lon_col='predicted_lon',
    true_lat_col='true_latitude',
    true_lon_col='true_longitude'
    )

    precision_continent = precision_score(all_target_continents, all_prediction_continents, average='weighted', zero_division=0)
    recall_continent = recall_score(all_target_continents, all_prediction_continents, average='weighted', zero_division=0)
    f1_continent = f1_score(all_target_continents, all_prediction_continents, average='weighted', zero_division=0)

    precision_city = precision_score(all_target_cities, all_prediction_cities, average='weighted', zero_division=0)
    recall_city = recall_score(all_target_cities, all_prediction_cities, average='weighted', zero_division=0)
    f1_city = f1_score(all_target_cities, all_prediction_cities, average='weighted', zero_division=0)

    print(f'Combined Model - Continent Accuracy: {accuracy_continent:.2f}%')
    print(f'Combined Model - Continent Precision: {precision_continent:.4f}')
    print(f'Combined Model - Continent Recall: {recall_continent:.4f}')
    print(f'Combined Model - Continent F1-Score: {f1_continent:.4f}')
    print(f'Combined Model - Cities Accuracy: {accuracy_cities:.2f}%')
    print(f'Combined Model - Cities Precision: {precision_city:.4f}')
    print(f'Combined Model - Cities Recall: {recall_city:.4f}')
    print(f'Combined Model - Cities F1-Score: {f1_city:.4f}')
    print(f'Combined Model - Mean Absolute Error (km) - Latitude: {mae_lat_km:.4f}')
    print(f'Combined Model - Mean Absolute Error (km) - Longitude: {mae_lon_km:.4f}')

    return accuracy_continent, accuracy_cities, precision_continent, recall_continent, f1_continent, \
           precision_city, recall_city, f1_city, mae_lat_km, mae_lon_km, \
           all_prediction_continents, all_prediction_cities, np.array([predicted_lat_deg, predicted_long_deg]).T, \
           all_target_continents, all_target_cities, np.array([target_lat_deg, target_long_deg]).T

# Function to run all the hyperparameters using ray
def hyperparameter_train_model(config):
    
    # Load data
    in_data = load_data(args.data_path)
    if in_data is None:
        exit()
    
    # Process data into correct format - Call the script process data
    in_data, le_continent, le_city, stdscaler_lat, stdscaler_long, coordinate_scaler, continent_encoding_map, city_encoding_map = process_data(in_data)
       
    # Split data
    X_train, X_test, y_train, y_test = split_data(in_data, test_size=args.test_size, random_state=args.random_state)

    # Split the training data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

    # Create DataLoaders
    train_dl = DataLoader(CustDat(X_train, y_train),
                      batch_size=config['batch_size'], 
                      shuffle=True,
                      num_workers=args.num_workers)
    val_dl = DataLoader(CustDat(X_val, y_val),
                    batch_size=config['batch_size'],
                    shuffle=False,
                    num_workers=args.num_workers)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
        
    # Model initialization
    combined_model = CombinedNeuralNetXYZModel(
        input_size=200,
        hidden_dim=config['layer_size'],
        num_continent=len(in_data['continent_encoding'].unique()),
        num_cities=len(in_data['city_encoding'].unique()),
        dropout_rate=config['dropout_rate']
    ).to(device)

    # Loss functions
    class_counts = in_data['continent_encoding'].value_counts().sort_index().tolist()
    continent_weights = (1 / torch.tensor(class_counts, dtype=torch.float32)).to(device)
    continent_weights = continent_weights / continent_weights.sum()
    
    criterion_continent = nn.CrossEntropyLoss(weight=continent_weights)
    criterion_cities = nn.CrossEntropyLoss()
    criterion_lat_lon = nn.MSELoss()
    
    # Optimizer
    if config['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            combined_model.parameters(), 
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            combined_model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:  # Adam
        optimizer = torch.optim.Adam(
            combined_model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training
    train_losses, val_losses = training_loop(
    train_dl, val_dl, combined_model, optimizer, scheduler,
    criterion_continent, criterion_cities, criterion_lat_lon,
    device, config['epochs']
    )
    
    # Return final metrics
    return {
    "val_loss": val_losses['total'][-1],
    "val_continent_loss": val_losses['continent'][-1],
    "val_cities_loss": val_losses['cities'][-1],
    "val_xyz_loss": val_losses['xyz'][-1]
    }


def trainable(config):
    # This wrapper is needed for Ray Tune
    return hyperparameter_train_model(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hierarchical neural network for location prediction.")
    parser.add_argument('-d',"--data_path", type=str, required=True, help="Path to the input CSV data file.")
    parser.add_argument('-t',"--test_size", type=float, default=0.2, help="Fraction of data to use for testing.")
    parser.add_argument('-r',"--random_state", type=int, default=123, help="Random seed for data splitting.")
    parser.add_argument('-b',"--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument('-lr',"--learning_rate", type=float, default=0.001, help="Learning rate for the optimizers.")
    parser.add_argument('-e',"--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument('-n',"--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument('-p',"--pin_memory", type=bool, default=False, help="Pin memory for DataLoader (improves performance on CUDA).")
    parser.add_argument('-c',"--use_cuda", type=bool, default=False, help="Enable CUDA if available.")
    parser.add_argument('-s',"--save_path", type=str, default=None, help="Path to save the trained models.")

    args = parser.parse_args()

    
    # Initialize Ray with proper resources
    ray.init(num_cpus=20, num_gpus=1 if args.use_cuda else 0)

    config = {
        # Learning parameters
        'lr': tune.loguniform(1e-5, 1e-2),
        'optimizer': tune.choice(['adam', 'sgd','adamw']),
        'momentum': tune.uniform(0.85, 0.99),  # Only used for SGD
        'weight_decay': tune.loguniform(1e-6, 1e-3),  # L2 regularization,

        # Architecture parameters
        'layer_size': tune.qrandint(128, 1024, 32),  # Quantized to multiples of 32
        'dropout_rate': tune.uniform(0.1, 0.7),  # Wider dropout range

        # Training parameters
        'batch_size': tune.choice([32, 64, 128, 256]),
        'epochs': 100,

        # Loss weighting parameters
        'loss_alpha': tune.uniform(0.1, 0.9),  # Weight between continent/city losses
        'xyz_loss_scale': tune.loguniform(0.1, 10),  # Scaling factor for XYZ loss
    
        # Learning rate scheduling
        'lr_patience': tune.randint(3, 10),  # For ReduceLROnPlateau
        'lr_factor': tune.uniform(0.1, 0.5)  # LR reduction factor
    }

    # Configure scheduler
    scheduler = ASHAScheduler(
        metric='val_loss',
        mode='min',
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )


    # Calculate maximum concurrent trials based on resources
    max_concurrent = 5  # Adjust based on your GPU memory (e.g., 4-8 trials per GPU)
    resources_per_trial = {"cpu": 4, "gpu": 0.2}  # 1/5 of GPU per trial if using 5 concurrent

    # Run the tuning with parallel execution
    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=100,
            max_concurrent_trials=max_concurrent
        ),
        param_space=config
    )

    results = tuner.fit()

    # Get the best trial
    best_result = results.get_best_result(metric="val_loss", mode="min")
    print("\nBest trial config:")
    print(best_result.config)
    print("\nBest trial final validation loss:", best_result.metrics["val_loss"])
    
    # Shut down Ray
    ray.shutdown()

# python nn_combined_model_lat_long_hyperparameter.py -d /home/chandru/binp37/results/metasub_training_testing_data.csv -t 0.2 -r 123 -b 128 -n 1 -e 40 -c True



"""
# Load data
    in_data = load_data(args.data_path)
    if in_data is None:
        exit()
    
    # Process data into correct format
    in_data, le_continent, le_city, stdscaler_lat, stdscaler_long, coordinate_scaler, continent_encoding_map, city_encoding_map =  process_data(in_data)
       
    # Split data
    X_train, X_test, y_train, y_test = split_data(in_data, test_size=args.test_size, random_state=args.random_state)

    # Split the training data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

    # Create DataLoaders - Train, Validate and Test
    train_dl = DataLoader(CustDat(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    val_dl = DataLoader(CustDat(X_val, y_val),
                        batch_size=args.batch_size, 
                        shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.pin_memory)

    test_dl = DataLoader(CustDat(X_test, y_test),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.pin_memory)
    
        
    # Hyperparameters
    input_size = 200
    hidden_dim = 736
    num_continent = len(in_data['continent_encoding'].unique())
    num_cities = len(in_data['city_encoding'].unique())
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    class_counts = in_data['continent_encoding'].value_counts().sort_index().tolist()

    # Determine device
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    print(f"Using device: {device}")

    # Initialize the combined network
    combined_model = CombinedNeuralNetXYZModel(input_size,hidden_dim,num_continent,num_cities,dropout_rate=0.3161407615609579,).to(device)

    # Loss functions and optimizers
    continent_weights = 1 /torch.tensor(class_counts,dtype=torch.float32)
    continent_weights = continent_weights /continent_weights.sum()
    criterion_continent = nn.CrossEntropyLoss(weight=continent_weights.to(device))
    criterion_cities = nn.CrossEntropyLoss()
    criterion_lat_lon = nn.MSELoss()
    optimizer_combined = torch.optim.Adam(combined_model.parameters(), lr=learning_rate, weight_decay=2.3857749472052553e-05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_combined,patience=8) # If the loss has not decreased for 5 epochs, then we lower the learning rate

    # Train the models
    train_losses, val_losses = training_loop(train_dl, val_dl,combined_model=combined_model,num_epochs=num_epochs,optimizer_combined=optimizer_combined, scheduler= scheduler,
                  criterion_continent=criterion_continent,criterion_cities=criterion_cities,
                  criterion_lat_lon=criterion_lat_lon,device=device)
    

    # Check accuracy of the model in training and testing
    print("\nFinal Model - Training Accuracy:")
    accuracy_continent_train, accuracy_cities_train, precision_continent_train, recall_continent_train, f1_continent_train, \
    precision_city_train, recall_city_train, f1_city_train, mae_lat_train, mae_long_train, \
    all_predictions_continents_train, all_predictions_cities_train, predicted_lat_long_deg_train, \
    all_labels_continents_train, all_labels_cities_train, targ_lat_long_deg_train = check_combined_accuracy(train_dl, combined_model, coordinate_scaler, device)
    
        
    print("\nFinal Model - Test Accuracy:")
    accuracy_continent_test, accuracy_cities_test, precision_continent_test, recall_continent_test, f1_continent_test, \
    precision_city_test, recall_city_test, f1_city_test, mae_lat_test, mae_long_test, \
    all_predictions_continents_test, all_predictions_cities_test, predicted_lat_long_deg_test, \
    all_labels_continents_test, all_labels_cities_test, targ_lat_long_deg_test = check_combined_accuracy(test_dl, combined_model, coordinate_scaler, device)
    
    

    
    plot_losses(train_losses, val_losses, filename='/home/chandru/binp37/results/plots/metasub/hyperparameter_training_losses_nn_combined_model_lat_long.png')

"""