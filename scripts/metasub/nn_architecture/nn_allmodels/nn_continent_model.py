# This script is to get the continent prediction accuracy higher, because the mGPS algorithm correctly classifies 92% of samples to 
# their city of origin.

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

# Neural Network Architecture with Dynamic Dropout
class CombinedNeuralNetXYZModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_continent, initial_dropout_rate, max_dropout_rate):
        super(CombinedNeuralNetXYZModel, self).__init__()

        self.initial_dropout_rate = initial_dropout_rate
        self.max_dropout_rate = max_dropout_rate
        self.relu = nn.ReLU()

        # Continent Architecture
        self.continent_layer_1 = nn.Linear(input_size, hidden_dim)
        self.continent_bn_1 = nn.BatchNorm1d(hidden_dim)
        self.continent_layer_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.continent_bn_2 = nn.BatchNorm1d(hidden_dim // 2)
        self.continent_layer_3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.continent_bn_3 = nn.BatchNorm1d(hidden_dim // 4)

        # Continent Prediction
        self.continent_prediction = nn.Linear(hidden_dim // 4, num_continent) # Output for 7 different continents

    def forward(self, x, current_dropout_rate):
        # Continent Architecture
        out_continent = self.relu(self.continent_bn_1(self.continent_layer_1(x)))
        out_continent = F.dropout(out_continent, p=current_dropout_rate, training=self.training) # Apply dynamic dropout
        out_continent = self.relu(self.continent_bn_2(self.continent_layer_2(out_continent)))
        out_continent = F.dropout(out_continent, p=current_dropout_rate, training=self.training) # Apply dynamic dropout
        out_continent = self.relu(self.continent_bn_3(self.continent_layer_3(out_continent)))

        # Continent Prediction
        continent_predictions = self.continent_prediction(out_continent)

        return continent_predictions
    
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



def training_loop(train_dl, val_dl, combined_model, optimizer_combined, criterion_continent, device, num_epochs):

    start_time = time.time()
    train_losses = {'continent': []}
    val_losses = {'continent': []}



    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Calculate dynamic dropout rate (linearly increasing from initial to max)
        current_dropout_rate = combined_model.initial_dropout_rate + (combined_model.max_dropout_rate - combined_model.initial_dropout_rate) * (epoch / num_epochs)

        # Training phase
        combined_model.train()
        train_metrics = {'continent': 0}

        for data, continent_city, lat_long in train_dl:
            data = data.to(device)
            cont_targ = continent_city[:, 0].long().to(device)


            # Forward pass with dynamic dropout rate
            optimizer_combined.zero_grad()
            continent_logits_train = combined_model(data, current_dropout_rate)

            # Calculate losses
            loss_continents_train = criterion_continent(continent_logits_train, cont_targ)

            # Backward pass
            loss_continents_train.backward()
            optimizer_combined.step()

            # Accumulate metrics
            train_metrics['continent'] += loss_continents_train.item()

        # Calculate average training metrics
        num_train_batches = len(train_dl)
        avg_train_losses = {k: v / num_train_batches for k, v in train_metrics.items()}
        for k, v in avg_train_losses.items():
            train_losses[k].append(v)

        # Validation phase
        combined_model.eval()
        val_metrics = {'continent': 0}

        with torch.no_grad():
            for data_val, continent_city_val, lat_long_val in val_dl:
                data_val = data_val.to(device)
                cont_targ_val = continent_city_val[:, 0].long().to(device)

                # Forward pass with dynamic dropout rate
                continent_logits_val = combined_model(data_val, current_dropout_rate)

                # Calculate losses
                loss_continents_val = criterion_continent(continent_logits_val,cont_targ_val)

                # Accumulate metrics
                val_metrics['continent'] += loss_continents_val.item()

        # Calculate average validation metrics with fallback for empty validation set
        num_val_batches = len(val_dl)
        if num_val_batches > 0:
            avg_val_losses = {k: v / num_val_batches for k, v in val_metrics.items()}
        else:
            avg_val_losses = {k: 0.0 for k in val_metrics.keys()}

        for k, v in avg_val_losses.items():
            val_losses[k].append(v)


        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Continent Loss: {avg_train_losses['continent']:.4f}, "
                  f"Validation Continent Loss: {avg_val_losses['continent']:.4f}, "
                  f"Dropout Rate: {current_dropout_rate:.2f}, "
                  f"Time: {epoch_duration:.2f}s")

    total_training_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    return train_losses, val_losses

# Check the accuracy of the neural netowrk on train and test data
def check_combined_accuracy(loader, model, device="cpu"):

    # Testing phase
    # Initialize test phase
    model.eval()

    correct_continent = 0
    total = 0
    all_prediction_continents = []
    all_target_continents = []

    # Use the initial dropout rate for evaluation
    evaluation_dropout_rate = model.initial_dropout_rate

    with torch.no_grad():
        for data, continent_city, latlong in loader:
            batch_size = data.size(0)
            data = data.to(device)
            cont_targ = continent_city[:, 0].long().to(device)

            continent_logits = model(data, evaluation_dropout_rate) # Pass the dropout rate

            _, predictions_continent = continent_logits.max(1)
            correct_continent += (predictions_continent == cont_targ).sum().item()
            all_prediction_continents.extend(predictions_continent.detach().cpu().numpy())
            all_target_continents.extend(cont_targ.detach().cpu().numpy())

            total += batch_size

    accuracy_continent = correct_continent / total * 100

    precision_continent = precision_score(all_target_continents, all_prediction_continents, average='weighted', zero_division=0)
    recall_continent = recall_score(all_target_continents, all_prediction_continents, average='weighted', zero_division=0)
    f1_continent = f1_score(all_target_continents, all_prediction_continents, average='weighted', zero_division=0)

    print(f'Combined Model - Continent Accuracy: {accuracy_continent:.2f}%')
    print(f'Combined Model - Continent Precision: {precision_continent:.4f}')
    print(f'Combined Model - Continent Recall: {recall_continent:.4f}')
    print(f'Combined Model - Continent F1-Score: {f1_continent:.4f}')

    return accuracy_continent, precision_continent, recall_continent, f1_continent, all_prediction_continents, all_target_continents

def augment_data_multiplicative_noise(X, y, num_new_samples_factor=0.5, factor_range=(0.9, 1.1)):
    """Multiplies input features X by small random factors to create synthetic data."""
    num_original_samples = X.shape[0]
    num_synthetic_samples = int(num_original_samples * num_new_samples_factor)

    # Repeat original X and y for augmentation
    X_repeated = np.repeat(X, int(np.ceil(num_new_samples_factor)), axis=0)[:num_synthetic_samples]
    y_repeated = np.repeat(y, int(np.ceil(num_new_samples_factor)), axis=0)[:num_synthetic_samples]

    # Generate random factors for each synthetic sample
    factors = np.random.uniform(low=factor_range[0], high=factor_range[1], size=X_repeated.shape)
    synthetic_X = X_repeated * factors

    augmented_X = np.vstack((X, synthetic_X))
    augmented_y = np.vstack((y, y_repeated))

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
    parser.add_argument('--initial_dropout', type=float, default=0.2, help='Initial dropout rate.')
    parser.add_argument('--max_dropout', type=float, default=0.65, help='Maximum dropout rate.')

    args = parser.parse_args()

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

    # Augment Data
    X_train_augmented, y_train_augmented = augment_data_multiplicative_noise(X_train, y_train, num_new_samples_factor=0.3, factor_range=(0.95, 1.05))
    
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
    
    train_dl_augmented = DataLoader(CustDat(X_train_augmented, y_train_augmented),
                                 batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory)


    # Hyperparameters
    input_size = 200
    hidden_dim = 64
    num_continent = len(in_data['continent_encoding'].unique())
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    class_counts = in_data['continent_encoding'].value_counts().sort_index().tolist()

    # Determine device
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    print(f"Using device: {device}")

    # Initialize the combined network with dynamic dropout
    combined_model = CombinedNeuralNetXYZModel(input_size, hidden_dim, num_continent,
                                                              initial_dropout_rate=args.initial_dropout,
                                                              max_dropout_rate=args.max_dropout).to(device)
    print(combined_model)

    # Loss functions and optimizers
    continent_weights = 1 /torch.tensor(class_counts,dtype=torch.float32)
    continent_weights = continent_weights /continent_weights.sum()
    criterion_continent = nn.CrossEntropyLoss(weight=continent_weights.to(device))
    optimizer_combined = torch.optim.Adam(combined_model.parameters(), lr=learning_rate)

    # Train the models with dynamic dropout
    train_losses, val_losses = training_loop(train_dl_augmented, val_dl, combined_model=combined_model, num_epochs=num_epochs,
                                             optimizer_combined=optimizer_combined,
                                             criterion_continent=criterion_continent, device=device)


    # Check accuracy of the model in training and testing
    print("\nFinal Model - Training Accuracy:")
    accuracy_continent_train, precision_continent_train, recall_continent_train, f1_continent_train, \
    all_predictions_continents_train, all_labels_continents_train = check_combined_accuracy(train_dl_augmented, combined_model, device)


    print("\nFinal Model - Test Accuracy:")
    accuracy_continent_test, precision_continent_test, recall_continent_test, f1_continent_test, \
    all_predictions_continents_test, all_labels_continents_test, = check_combined_accuracy(test_dl, combined_model, device)

    # Plot the losses
    plot_losses(train_losses, val_losses, filename='train_validation.png')


    
# python nn_continent_model.py -d ../../results/metasub_training_testing_data.csv -t 0.2 -r 123 -b 32 -n 1 -e 400 -lr 0.0001 -c True

