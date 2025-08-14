# This script is to get the latitude and longitude prediction accuracy higher.

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
    def __init__(self, input_size, hidden_dim, initial_dropout_rate, max_dropout_rate):
        super(CombinedNeuralNetXYZModel, self).__init__()

        self.initial_dropout_rate = initial_dropout_rate
        self.max_dropout_rate = max_dropout_rate
        self.relu = nn.ReLU()

        # XYZ Architecture (deeper)
        self.xyz_layer_1 = nn.Linear(input_size, hidden_dim)
        self.xyz_layer_bn_1 = nn.BatchNorm1d(hidden_dim)
        self.xyz_layer_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.xyz_layer_bn_2 = nn.BatchNorm1d(hidden_dim // 2)
        self.xyz_layer_3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.xyz_layer_bn_3 = nn.BatchNorm1d(hidden_dim // 4)
        self.xyz_layer_4 = nn.Linear(hidden_dim // 4, hidden_dim // 8) # Added layer
        self.xyz_layer_bn_4 = nn.BatchNorm1d(hidden_dim // 8)

        # XYZ Prediction
        self.xyz_prediction = nn.Linear(hidden_dim // 8, 3) # Output for three xyz coordinates

    def forward(self, x, current_dropout_rate):
        out_xyz = self.relu(self.xyz_layer_bn_1(self.xyz_layer_1(x)))
        out_xyz = F.dropout(out_xyz, p=current_dropout_rate, training=self.training)
        out_xyz = self.relu(self.xyz_layer_bn_2(self.xyz_layer_2(out_xyz)))
        out_xyz = F.dropout(out_xyz, p=current_dropout_rate, training=self.training)
        out_xyz = self.relu(self.xyz_layer_bn_3(self.xyz_layer_3(out_xyz)))
        out_xyz = self.relu(self.xyz_layer_bn_4(self.xyz_layer_4(out_xyz))) # Added ReLU and BN
        xyz_prediction = self.xyz_prediction(out_xyz)
        return xyz_prediction
    
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
    


def training_loop(train_dl, val_dl, combined_model, optimizer_combined, criterion_lat_lon, device, num_epochs, patience=10):
    start_time = time.time()
    train_losses = {'xyz': []}
    val_losses = {'xyz': []}

    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Calculate dynamic dropout rate (linearly increasing from initial to max)
        current_dropout_rate = combined_model.initial_dropout_rate + (combined_model.max_dropout_rate - combined_model.initial_dropout_rate) * (epoch / num_epochs)

        # Training phase
        combined_model.train()
        train_metrics = {'xyz': 0}

        for data, continent_city, lat_long in train_dl:
            data = data.to(device)
            xyz_targ_train = lat_long.float().to(device)

            optimizer_combined.zero_grad()
            xyz_logits_train = combined_model(data, current_dropout_rate)
            loss_xyz_train = criterion_lat_lon(xyz_logits_train, xyz_targ_train)
            loss_xyz_train.backward()
            optimizer_combined.step()

            train_metrics['xyz'] += loss_xyz_train.item()

        num_train_batches = len(train_dl)
        avg_train_losses = {k: v / num_train_batches for k, v in train_metrics.items()}
        for k, v in avg_train_losses.items():
            train_losses[k].append(v)

        # Validation phase
        combined_model.eval()
        val_metrics = {'xyz': 0}

        with torch.no_grad():
            for data_val, continent_city_val, lat_long_val in val_dl:
                data_val = data_val.to(device)
                xyz_targ_val = lat_long_val.float().to(device)

                xyz_logits_val = combined_model(data_val, current_dropout_rate)
                loss_xyz_val = criterion_lat_lon(xyz_logits_val, xyz_targ_val)

                val_metrics['xyz'] += loss_xyz_val.item()

        num_val_batches = len(val_dl)
        avg_val_losses = {k: v / num_val_batches for k, v in val_metrics.items()}
        current_val_loss = avg_val_losses['xyz']
        for k, v in avg_val_losses.items():
            val_losses[k].append(v)

        # Early stopping check
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            counter = 0
            best_model_state = combined_model.state_dict() # Save the best model weights
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train XYZ Loss: {avg_train_losses['xyz']:.4f}, "
                  f"Validation XYZ Loss: {current_val_loss:.4f}, "
                  f"Dropout Rate: {current_dropout_rate:.2f}, "
                  f"Time: {epoch_duration:.2f}s, Patience: {counter}/{patience}")

    total_training_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    # Load the best model state
    if best_model_state is not None:
        combined_model.load_state_dict(best_model_state)
        print(f"Loaded best model weights with validation loss: {best_val_loss:.4f}")

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

    all_prediction_xyz = []
    all_target_xyz = []

    # Use the initial dropout rate for evaluation
    evaluation_dropout_rate = model.initial_dropout_rate

    with torch.no_grad():
        for batch_idx, (data, continent_city, lat_long_rad) in enumerate(loader):
            batch_size = data.size(0)
            data = data.to(device)

            xyz_targ_scaled = lat_long_rad.float().to(device)

            xyz_pred_scaled = model(data,evaluation_dropout_rate)

            all_prediction_xyz.append(xyz_pred_scaled.detach().cpu().numpy())
            all_target_xyz.append(xyz_targ_scaled.detach().cpu().numpy())


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


    print(f'Combined Model - Mean Absolute Error (km) - Latitude: {mae_lat_km:.4f}')
    print(f'Combined Model - Mean Absolute Error (km) - Longitude: {mae_lon_km:.4f}')

    return mae_lat_km, mae_lon_km, np.array([predicted_lat_deg, predicted_long_deg]).T, np.array([target_lat_deg, target_long_deg]).T

def augment_data_with_lat_lon_noise(X, y, num_new_samples_factor=0.7, noise_std_latlon=0.001):
    """Adds multiplicative noise to X and Gaussian noise to latitude and longitude (y)."""
    num_original_samples = X.shape[0]
    num_synthetic_samples = int(num_original_samples * num_new_samples_factor)

    # Repeat original X and y for augmentation
    X_repeated = np.repeat(X, int(np.ceil(num_new_samples_factor)), axis=0)[:num_synthetic_samples]
    y_repeated = np.repeat(y, int(np.ceil(num_new_samples_factor)), axis=0)[:num_synthetic_samples]

    # Generate random factors for X
    factors_X = np.random.uniform(low=0.9, high=1.1, size=X_repeated.shape)
    synthetic_X = X_repeated * factors_X

    # Add Gaussian noise to latitude and longitude (scaled_x, scaled_y, scaled_z)
    noise_latlon = np.random.normal(loc=0, scale=noise_std_latlon, size=y_repeated[:, 2:].shape)
    synthetic_y_latlon = y_repeated[:, 2:] + noise_latlon

    # Combine the unchanged continent and city encodings with the noisy lat/lon
    synthetic_y = np.hstack((y_repeated[:, :2], synthetic_y_latlon))

    augmented_X = np.vstack((X, synthetic_X))
    augmented_y = np.vstack((y, synthetic_y))

    return augmented_X, augmented_y



# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hierarchical neural network for location prediction.")
    parser.add_argument('-d',"--data_path", type=str, required=True, help="Path to the input CSV data file.")
    parser.add_argument('-t',"--test_size", type=float, default=0.2, help="Fraction of data to use for testing.")
    parser.add_argument('-r',"--random_state", type=int, default=123, help="Random seed for data splitting.")
    parser.add_argument('-b',"--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument('-lr',"--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizers.")
    parser.add_argument('-e',"--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument('-n',"--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument('-p',"--pin_memory", type=bool, default=False, help="Pin memory for DataLoader (improves performance on CUDA).")
    parser.add_argument('-c',"--use_cuda", type=bool, default=False, help="Enable CUDA if available.")
    parser.add_argument('-s',"--save_path", type=str, default=None, help="Path to save the trained models.")
    parser.add_argument('--patience', type=int, default=20, help='Number of epochs to wait for improvement in validation loss before early stopping.')
    parser.add_argument('--initial_dropout', type=float, default=0.2, help='Initial dropout rate.')
    parser.add_argument('--max_dropout', type=float, default=0.7, help='Maximum dropout rate.')

    args = parser.parse_args()

    # Parameters
    input_size = 200
    test_size = args.test_size
    random_state = args.random_state
    num_workers=args.num_workers 
    pin_memory=args.pin_memory

    # Hyperparameters
    hidden_dim = 128
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    intial_dropout_rate = args.initial_dropout
    max_dropout_rate = args.max_dropout
    batch_size = args.batch_size


    # Load data
    in_data = load_data(args.data_path)
    if in_data is None:
        exit()

    # Process data into correct format
    in_data, le_continent, le_city, stdscaler_lat, stdscaler_long, coordinate_scaler, continent_encoding_map, city_encoding_map =  process_data(in_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(in_data, test_size=test_size, random_state=random_state)

    # Split the training data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    
    # Augment Data
    X_train_augmented, y_train_augmented = augment_data_with_lat_lon_noise(X_train, y_train)
    
    # Create DataLoaders - Train, Validate and Test
    train_dl = DataLoader(CustDat(X_train, y_train),
                              batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)

    val_dl = DataLoader(CustDat(X_val, y_val),
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.pin_memory)

    test_dl = DataLoader(CustDat(X_test, y_test),
                             batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    train_dl_augmented = DataLoader(CustDat(X_train_augmented, y_train_augmented),
                                 batch_size=batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    print(f'Number of sample before augmentation: {X_train.shape[0]}')
    print(f'Number of samples after augmentation: {X_train_augmented.shape[0]}')
   

    # Determine device
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    print(f"Using device: {device}")

    # Initialize the combined network with dynamic dropout
    combined_model = CombinedNeuralNetXYZModel(input_size, hidden_dim,initial_dropout_rate=intial_dropout_rate,
                                                              max_dropout_rate=max_dropout_rate).to(device)
    print(combined_model)

    # Loss functions and optimizers
    criterion_lat_lon = nn.MSELoss()
    optimizer_combined = torch.optim.Adam(combined_model.parameters(), lr=learning_rate)

    # Train the models with dynamic dropout and early stopping
    train_losses, val_losses = training_loop(train_dl_augmented, val_dl, combined_model=combined_model, num_epochs=num_epochs,
        optimizer_combined=optimizer_combined, criterion_lat_lon=criterion_lat_lon, device=device,patience=args.patience)


    # Check accuracy of the model
    print("\nFinal Model - Training Accuracy:")
    mae_lat_train, mae_long_train, predicted_lat_long_deg_train,targ_lat_long_deg_train = check_combined_accuracy(train_dl_augmented, combined_model, coordinate_scaler, device)
    
        
    print("\nFinal Model - Test Accuracy:")
    mae_lat_test, mae_long_test, predicted_lat_long_deg_test, targ_lat_long_deg_test = check_combined_accuracy(test_dl, combined_model, coordinate_scaler, device)
    
    
    # Predictions dataframe
    predictions_df = pd.DataFrame({
        'predicted_lat': predicted_lat_long_deg_test[:,0],
        'predicted_lon': predicted_lat_long_deg_test[:,1],
        'true_latitude': targ_lat_long_deg_test[:,0],
        'true_longitude': targ_lat_long_deg_test[:,1]
    })

       
    # Run the coastline pulling function
    updated_predictions_df = pull_land(
    df=predictions_df.copy(),
    coastline_path="/home/chandru/binp37/data/geopandas/ne_110m_coastline.shp",
    countries_path="/home/chandru/binp37/data/geopandas/ne_110m_admin_0_countries.shp",
    lat_col='predicted_lat',
    lon_col='predicted_lon'
    )


    
    predicted_lat_test, predicted_long_test, true_lat_test, true_long_test = updated_predictions_df['updated_lat'].values, \
                                                                            updated_predictions_df['updated_lon'].values, \
                                                                            updated_predictions_df['true_latitude'].values, \
                                                                            updated_predictions_df['true_longitude'].values


    # Visualizing the results of the final model on the test set
    plot_losses(train_losses, val_losses, filename='/home/chandru/binp37/scripts/metasub/train_val_losses_latlong.png')
    plot_points_on_world_map(true_lat_test, true_long_test, predicted_lat_test, predicted_long_test, filename='/home/chandru/binp37/scripts/metasub/worldmap_latlong.png')



# python nn_latlong_model.py -d ../../results/metasub_training_testing_data.csv -t 0.2 -r 123 -b 128 -n 1 -e 400 -lr 0.0001 -c True
