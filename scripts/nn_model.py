# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import argparse
import os
import matplotlib.pyplot as plt

# Define the neural network modules (as in your original script)
class NeuralNetContinent(nn.Module):
    def __init__(self, input_size_continents, num_continents, dropout_rate=0.5):
        super(NeuralNetContinent, self).__init__()
        self.layer1 = nn.Linear(input_size_continents, 400)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(400, 400)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(400, 200)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(200, num_continents)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout1(out)
        out = self.relu(self.layer2(out))
        out = self.dropout2(out)
        out = self.relu(self.layer3(out))
        out = self.dropout3(out)
        out = self.layer4(out)
        return out

class NeuralNetCities(nn.Module):
    def __init__(self, input_size_cities, num_cities, dropout_rate=0.5):
        super(NeuralNetCities, self).__init__()
        self.layer1 = nn.Linear(input_size_cities, 400)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(400, 400)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(400, 200)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(200, num_cities)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout1(out)
        out = self.relu(self.layer2(out))
        out = self.dropout2(out)
        out = self.relu(self.layer3(out))
        out = self.dropout3(out)
        out = self.layer4(out)
        return out

class NeuralNetLat(nn.Module):
    def __init__(self, input_size_lat, lat_size, dropout_rate=0.5):
        super(NeuralNetLat, self).__init__()
        self.layer1 = nn.Linear(input_size_lat, 400)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(400, 400)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(400, 200)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(200, lat_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout1(out)
        out = self.relu(self.layer2(out))
        out = self.dropout2(out)
        out = self.relu(self.layer3(out))
        out = self.dropout3(out)
        out = self.layer4(out)
        return out

class NeuralNetLong(nn.Module):
    def __init__(self, input_size_long, long_size, dropout_rate=0.5):
        super(NeuralNetLong, self).__init__()
        self.layer1 = nn.Linear(input_size_long, 400)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(400, 400)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(400, 200)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(200, long_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout1(out)
        out = self.relu(self.layer2(out))
        out = self.dropout2(out)
        out = self.relu(self.layer3(out))
        out = self.dropout3(out)
        out = self.layer4(out)
        return out

# Data loading and splitting functions
def load_data(data_path):
    try:
        in_data = pd.read_csv(data_path)
        return in_data
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None

def split_data(in_data, test_size=0.2, random_state=123):
    X = in_data.iloc[:, :200].values.astype(np.float32)
    y = in_data[['continent_encoding', 'city_encoding', 'lat_scaled', 'long_scaled']].values.astype(np.float32)
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
    

# Training loop function
def training_loop(train_dl, nn_continent_model, nn_cities_model, nn_lat_model, nn_long_model,
                  optimizer_continent, optimizer_cities, optimizer_lat, optimizer_long,
                  criterion, criterion_lat_lon, device, num_epochs,train_losses):
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss_continent = 0
        total_loss_cities = 0
        total_loss_lat = 0
        total_loss_long = 0

        for batch_idx, (data, continent_city, lat_long) in enumerate(train_dl):
            data = data.to(device=device)
            continent_city = continent_city.to(device=device)
            lat_long = lat_long.to(device=device)

            # Forward pass continent
            scores_continent = nn_continent_model(data)
            loss_continents = criterion(scores_continent, continent_city[:, 0])
            total_loss_continent += loss_continents.detach().cpu().numpy()

            # Forward pass cities
            in_data_cities = torch.cat((data, scores_continent), 1)
            scores_cities = nn_cities_model(in_data_cities)
            loss_cities = criterion(scores_cities, continent_city[:, 1])
            total_loss_cities += loss_cities.detach().cpu().numpy()

            # Forward pass latitude
            in_data_lat = torch.cat((in_data_cities, scores_cities), 1)
            scores_lat = nn_lat_model(in_data_lat)
            loss_lat = criterion_lat_lon(scores_lat, lat_long[:, 0].unsqueeze(1))
            total_loss_lat += loss_lat.detach().cpu().numpy()

            # Forward pass longitude
            in_data_long = torch.cat((in_data_lat, scores_lat), 1)
            scores_long = nn_long_model(in_data_long)
            loss_long = criterion_lat_lon(scores_long, lat_long[:, 1].unsqueeze(1))
            total_loss_long += loss_long.detach().cpu().numpy()

            # Backward propagation and optimization
            optimizer_long.zero_grad()
            loss_long.backward(retain_graph=True)
            optimizer_long.step()

            optimizer_lat.zero_grad()
            loss_lat.backward(retain_graph=True)
            optimizer_lat.step()

            optimizer_cities.zero_grad()
            loss_cities.backward(retain_graph=True)
            optimizer_cities.step()

            optimizer_continent.zero_grad()
            loss_continents.backward()
            optimizer_continent.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss_continent = total_loss_continent / len(train_dl)
        avg_loss_cities = total_loss_cities / len(train_dl)
        avg_loss_lat = total_loss_lat / len(train_dl)
        avg_loss_long = total_loss_long / len(train_dl)

        train_losses['continent'].append(avg_loss_continent)
        train_losses['cities'].append(avg_loss_cities)
        train_losses['latitude'].append(avg_loss_lat)
        train_losses['longitude'].append(avg_loss_long)

        print(f"\nEpoch {epoch+1}/{num_epochs}, Avg. Loss Continents: {avg_loss_continent:.4f}, Epoch Time: {epoch_duration:.2f} seconds")
        print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss Cities: {avg_loss_cities:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss Latitudes: {avg_loss_lat:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss Longitudes: {avg_loss_long:.4f}")

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")


def check_accuracy(loader, continent_model, cities_model=None, lat_model=None, long_model=None, tolerance_lat=0.1, tolerance_long=0.1, device="cpu"):
    num_correct_continent = 0
    num_samples = 0
    num_correct_cities = 0
    total_absolute_error_lat = 0.0
    total_absolute_error_long = 0.0

    continent_model.eval()
    if cities_model:
        cities_model.eval()
    if lat_model:
        lat_model.eval()
    if long_model:
        long_model.eval()

    with torch.no_grad():
        for (data, continent_city, lat_long) in loader:
            data = data.to(device=device)
            target_continent = continent_city[:, 0].to(device=device)
            target_cities = continent_city[:, 1].to(device=device)
            target_lat = lat_long[:, 0].to(device=device).unsqueeze(1) # Ensure correct shape
            target_long = lat_long[:, 1].to(device=device).unsqueeze(1) # Ensure correct shape

            # Continent predictions
            scores_continent = continent_model(data)
            _, predictions_continent = scores_continent.max(1)
            num_correct_continent += (predictions_continent == target_continent).sum()
            num_samples += predictions_continent.size(0)

            # Cities predictions (if cities model is provided)
            if cities_model:
                scores_continent_for_cities = continent_model(data)
                in_data_cities = torch.cat((data, scores_continent_for_cities), 1)
                scores_cities = cities_model(in_data_cities)
                _, predictions_cities = scores_cities.max(1)
                num_correct_cities += (predictions_cities == target_cities).sum()

            # Latitude predictions (if latitude model is provided)
            if lat_model:
                scores_continent_for_lat = continent_model(data)
                in_data_cities_for_lat = torch.cat((data, scores_continent_for_lat), 1)
                scores_cities_for_lat = cities_model(in_data_cities_for_lat)
                in_data_lat = torch.cat((in_data_cities_for_lat, scores_cities_for_lat), 1)
                predicted_lat = lat_model(in_data_lat)
                absolute_error_lat = torch.abs(predicted_lat - target_lat)
                total_absolute_error_lat += torch.sum(absolute_error_lat).item()

            # Longitude predictions (if longitude model is provided)
            if long_model:
                scores_continent_for_long = continent_model(data)
                scores_cities_for_long = cities_model(torch.cat((data, scores_continent_for_long), 1))
                in_data_lat_for_long = torch.cat((torch.cat((data, scores_continent_for_long), 1), scores_cities_for_long), 1)
                predicted_lat_for_long = lat_model(in_data_lat_for_long)
                in_data_long = torch.cat((in_data_lat_for_long, predicted_lat_for_long), 1)
                predicted_long = long_model(in_data_long)
                absolute_error_long = torch.abs(predicted_long - target_long)
                total_absolute_error_long += torch.sum(absolute_error_long).item()

    accuracy_continent = float(num_correct_continent) / float(num_samples) * 100
    print(f'Continent Model Accuracy: {accuracy_continent:.2f}%')

    if cities_model and num_samples > 0:
        accuracy_cities = float(num_correct_cities) / float(num_samples) * 100
        print(f'Cities Model Accuracy: {accuracy_cities:.2f}%')

    if lat_model and num_samples > 0:
        mean_absolute_error_lat = total_absolute_error_lat / num_samples
        print(f'Latitude Model Mean Absolute Error: {mean_absolute_error_lat:.4f}')

    if long_model and num_samples > 0:
        mean_absolute_error_long = total_absolute_error_long / num_samples
        print(f'Longitude Model Mean Absolute Error: {mean_absolute_error_long:.4f}')

    continent_model.train()
    if cities_model:
        cities_model.train()
    if lat_model:
        lat_model.train()
    if long_model:
        long_model.train()


def plot_losses(train_losses,filename):
    epochs = range(1, len(train_losses['continent']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses['continent'], 'r-', label='Continent Loss')
    plt.plot(epochs, train_losses['cities'], 'b-', label='Cities Loss')
    plt.plot(epochs, train_losses['latitude'], 'g-', label='Latitude Loss')
    plt.plot(epochs, train_losses['longitude'], 'm-', label='Longitude Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename) # Save the plot as an image
    plt.show()


def main(args):
    # Load data
    in_data = load_data(args.data_path)
    if in_data is None:
        return

    # Split data
    X_train, X_test, y_train, y_test = split_data(in_data, test_size=args.test_size, random_state=args.random_state)

    # Create DataLoaders
    train_dl = DataLoader(CustDat(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_dl = DataLoader(CustDat(X_test, y_test),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.pin_memory)

    # Hyperparameters from arguments
    input_size_continents = 200
    input_size_cities = 207
    input_size_lat = 247
    input_size_long = 248
    num_continents = 7
    num_cities = 40
    lat_size = 1
    long_size = 1
    learning_rate = args.learning_rate
    num_epochs = args.epochs

    # Determine device
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    print(f"Using device: {device}")

    # Initialize models
    nn_continent_model = NeuralNetContinent(input_size_continents, num_continents).to(device)
    nn_cities_model = NeuralNetCities(input_size_cities, num_cities).to(device)
    nn_lat_model = NeuralNetLat(input_size_lat, lat_size).to(device)
    nn_long_model = NeuralNetLong(input_size_long, long_size).to(device)

    # Define loss functions and optimizers
    criterion = nn.CrossEntropyLoss()
    criterion_lat_lon = nn.MSELoss()
    optimizer_continent = torch.optim.Adam(nn_continent_model.parameters(), lr=learning_rate)
    optimizer_cities = torch.optim.Adam(nn_cities_model.parameters(), lr=learning_rate)
    optimizer_lat = torch.optim.SGD(nn_lat_model.parameters(), lr=learning_rate)
    optimizer_long = torch.optim.SGD(nn_long_model.parameters(), lr=learning_rate)

    # Train the models
    train_losses = {'continent':[],
                    'cities':[],
                    'latitude':[],
                    'longitude':[]}
    training_loop(train_dl, nn_continent_model, nn_cities_model, nn_lat_model, nn_long_model,
                  optimizer_continent, optimizer_cities, optimizer_lat, optimizer_long,
                  criterion, criterion_lat_lon, device, num_epochs,train_losses=train_losses)


    # Check accuracy
    print("\nTraining Accuracy:")
    check_accuracy(train_dl, nn_continent_model, nn_cities_model, nn_lat_model, nn_long_model, device=device)
    print("\nTest Accuracy:")
    check_accuracy(test_dl, nn_continent_model, nn_cities_model, nn_lat_model, nn_long_model, device=device)  

    plot_losses(train_losses,filename='training_losses.png')


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
    main(args)