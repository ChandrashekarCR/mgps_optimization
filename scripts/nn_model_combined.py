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



class CombinedNeuralNet(nn.Module):
    def __init__(self,input_size):
        super(CombinedNeuralNet,self).__init__()
        # Initial layers shared by all the tasks
        self.layer1 = nn.Linear(input_size,400)
        self.layer2 = nn.Linear(400,400)
        self.layer3 = nn.Linear(400,200)
        self.relu = nn.ReLU()

        # Branch for Continent Prediction
        self.continent_layer = nn.Linear(200,7) # Output for 7 continents
        
        # Layers after continent branch
        self.layer_after_continent = nn.Linear(207,400) # Concatenate previous layers
        self.layer_cities1 = nn.Linear(400,400)
        self.layer_cities2 = nn.Linear(400,200) 
        self.city_layer = nn.Linear(200, 40) # Output for 40 cities

        # Layers after city branch
        self.layer_after_cities = nn.Linear(247,400)  # Concatenate previous layers
        self.layer_lat1 = nn.Linear(400,400)
        self.layer_lat2 = nn.Linear(400,200)
        self.latitude_layer = nn.Linear(200,1) # Ouput for latitude

        # Layers after latitude branch
        self.layer_after_lat = nn.Linear(248, 400) # Concatenate previous layers
        self.layer_long1 = nn.Linear(400,400)
        self.layer_long2 = nn.Linear(400,200)
        self.longitude_layer = nn.Linear(200,1) # Outptu for longitude

    def forward(self,x):
        
        # Shared layers
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        shared_out = self.relu(self.layer3(out))

        # Continent branch
        continent_logits = self.continent_layer(shared_out)

        # City branch
        concat_continent = torch.cat((shared_out, continent_logits), dim=1)
        out_cities = self.relu(self.layer_after_continent(concat_continent))
        out_cities = self.relu(self.layer_cities1(out_cities))
        out_cities = self.relu(self.layer_cities2(out_cities))
        city_logits = self.city_layer(out_cities)

        # Latitude branch
        concat_cities = torch.cat((concat_continent, city_logits), dim=1) # Include continent info as well
        out_lat = self.relu(self.layer_after_cities(concat_cities))
        out_lat = self.relu(self.layer_lat1(out_lat))
        out_lat = self.relu(self.layer_lat2(out_lat))
        latitude_prediction = self.latitude_layer(out_lat)

        # Longitude branch
        concat_lat = torch.cat((concat_cities, latitude_prediction), dim=1) # Include continent and city info
        out_long = self.relu(self.layer_after_lat(concat_lat))
        out_long = self.relu(self.layer_long1(out_long))
        out_long = self.relu(self.layer_long2(out_long))
        longitude_prediction = self.longitude_layer(out_long)

        return continent_logits, city_logits, latitude_prediction, longitude_prediction


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
def training_loop(train_dl, combined_model, optimizer_combined, criterion_continent, 
                  criterion_cities, criterion_lat_lon, device, num_epochs,train_losses):
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss_continent = 0
        total_loss_cities = 0
        total_loss_lat = 0
        total_loss_long = 0

        for batch_idx, (data, continent_city, lat_long) in enumerate(train_dl):
            data = data.to(device)
            target_continent = continent_city[:, 0].to(device)
            target_cities = continent_city[:, 1].to(device)
            target_lat = lat_long[:, 0].unsqueeze(1).to(device)
            target_long = lat_long[:, 1].unsqueeze(1).to(device)

            # Forward pass
            continent_logits, city_logits, latitude_prediction, longitude_prediction = combined_model(data)

            # Calculate losses
            loss_continent = criterion_continent(continent_logits, target_continent)
            total_loss_continent += loss_continent.detach().cpu().numpy()
            loss_cities = criterion_cities(city_logits, target_cities)
            total_loss_cities += loss_cities.detach().cpu().numpy()
            loss_lat = criterion_lat_lon(latitude_prediction, target_lat)
            total_loss_lat += loss_lat.detach().cpu().numpy()
            loss_long = criterion_lat_lon(longitude_prediction, target_long)
            total_loss_long += loss_long.detach().cpu().numpy()
            total_loss = loss_continent + loss_cities + loss_lat + loss_long

            # Backward pass and optimization
            optimizer_combined.zero_grad()
            total_loss.backward()
            optimizer_combined.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Loss (Continent): {loss_continent.item():.4f}, Loss (Cities): {loss_cities.item():.4f}, Loss (Lat): {loss_lat.item():.4f}, Loss (Long): {loss_long.item():.4f}, Epoch Time: {epoch_duration:.2f} seconds")

        avg_loss_continent = total_loss_continent / len(train_dl)
        avg_loss_cities = total_loss_cities / len(train_dl)
        avg_loss_lat = total_loss_lat / len(train_dl)
        avg_loss_long = total_loss_long / len(train_dl)

        train_losses['continent'].append(avg_loss_continent)
        train_losses['cities'].append(avg_loss_cities)
        train_losses['latitude'].append(avg_loss_lat)
        train_losses['longitude'].append(avg_loss_long)

        print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss Continents: {avg_loss_continent:.4f}, Epoch Time: {epoch_duration:.2f} seconds")
        print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss Cities: {avg_loss_cities:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss Latitudes: {avg_loss_lat:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss Longitudes: {avg_loss_long:.4f}\n")

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

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


# Modified check_accuracy function for the combined model
def check_combined_accuracy(loader, model,device):
    num_correct_continent = 0
    num_samples = 0
    num_correct_cities = 0
    total_absolute_error_lat = 0.0
    total_absolute_error_long = 0.0

    model.eval()
    with torch.no_grad():
        for (data, continent_city, lat_long) in loader:
            data = data.to(device)
            target_continent = continent_city[:, 0].to(device)
            target_cities = continent_city[:, 1].to(device)
            target_lat = lat_long[:, 0].unsqueeze(1).to(device)
            target_long = lat_long[:, 1].unsqueeze(1).to(device)

            continent_logits, city_logits, latitude_prediction, longitude_prediction = model(data)

            # Continent Accuracy
            _, predictions_continent = continent_logits.max(1)
            num_correct_continent += (predictions_continent == target_continent).sum()
            num_samples += predictions_continent.size(0)

            # Cities Accuracy
            _, predictions_cities = city_logits.max(1)
            num_correct_cities += (predictions_cities == target_cities).sum()

            # Latitude Error
            absolute_error_lat = torch.abs(latitude_prediction - target_lat)
            total_absolute_error_lat += torch.sum(absolute_error_lat).item()

            # Longitude Error
            absolute_error_long = torch.abs(longitude_prediction - target_long)
            total_absolute_error_long += torch.sum(absolute_error_long).item()

    accuracy_continent = float(num_correct_continent) / float(num_samples) * 100
    accuracy_cities = float(num_correct_cities) / float(num_samples) * 100
    mean_absolute_error_lat = total_absolute_error_lat / num_samples
    mean_absolute_error_long = total_absolute_error_long / num_samples

    print(f'Combined Model - Continent Accuracy: {accuracy_continent:.2f}%')
    print(f'Combined Model - Cities Accuracy: {accuracy_cities:.2f}%')
    print(f'Combined Model - Latitude Mean Absolute Error: {mean_absolute_error_lat:.4f}')
    print(f'Combined Model - Longitude Mean Absolute Error: {mean_absolute_error_long:.4f}')

    model.train()


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

    # Hyperparameters
    input_size = 200
    learning_rate = 0.001
    learning_rate = args.learning_rate
    num_epochs = args.epochs

    # Determine device
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    print(f"Using device: {device}")

    # Initialize the combined network
    combined_model = CombinedNeuralNet(input_size).to(device)

    # Loss functions and optimizers
    criterion_continent = nn.CrossEntropyLoss()
    criterion_cities = nn.CrossEntropyLoss()
    criterion_lat_lon = nn.MSELoss()
    optimizer_combined = torch.optim.Adam(combined_model.parameters(), lr=learning_rate)
    # Train the models
    train_losses = {'continent':[],
                    'cities':[],
                    'latitude':[],
                    'longitude':[]}
    training_loop(train_dl,combined_model=combined_model,num_epochs=num_epochs,optimizer_combined=optimizer_combined,
                  criterion_continent=criterion_continent,criterion_cities=criterion_cities,
                  criterion_lat_lon=criterion_lat_lon,device=device,train_losses=train_losses)


    # Check accuracy of the combined model
    print("\nTraining Accuracy (Combined Model):")
    check_combined_accuracy(train_dl, combined_model,device=device)
    print("\nTest Accuracy (Combined Model):")
    check_combined_accuracy(test_dl, combined_model,device=device)
    plot_losses(train_losses,'training_loss.png')

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