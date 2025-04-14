# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
from check_accuracy_model import plot_losses, plot_confusion_matrix, plot_points_on_world_map
from process_data import process_data

# Define the neural network modules (as in your original script)
# First model - In this model, I made a neural network separate for each hierarchy. The same architechture is followed for 
# continent level prediction, city level prediction, latitude level prediction and longitude level prediction.

# Continent Neural Network
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

# City Neural Network
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

# Latitude Neural Network
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

# Longitude Neural Network
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
                  criterion, criterion_lat_lon, device, num_epochs):
    
    start_time = time.time()

    train_losses = {'continent':[],
                    'cities':[],
                    'latitude':[],
                    'longitude':[]}

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
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss (Continent): {loss_continents.item():.4f}, Loss (Cities): {loss_cities.item():.4f}, Loss (Lat): {loss_lat.item():.4f}, Loss (Long): {loss_long.item():.4f}, Epoch Time: {epoch_duration:.2f} seconds")

        avg_loss_continent = total_loss_continent / len(train_dl)
        avg_loss_cities = total_loss_cities / len(train_dl)
        avg_loss_lat = total_loss_lat / len(train_dl)
        avg_loss_long = total_loss_long / len(train_dl)

        train_losses['continent'].append(avg_loss_continent)
        train_losses['cities'].append(avg_loss_cities)
        train_losses['latitude'].append(avg_loss_lat)
        train_losses['longitude'].append(avg_loss_long)


    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

    return train_losses

# Check the accuracy of the model on a test dataset
def check_accuracy(loader, continent_model, cities_model=None, lat_model=None, long_model=None, coordinate_scaler=None, device="cpu"):
    # Model Evaluation
    continent_model.eval()
    if cities_model:
        cities_model.eval()
    if lat_model:
        lat_model.eval()
    if long_model:
        long_model.eval()

    # Initialize variables for checking accuracy
    correct_continent = 0
    num_samples = 0
    correct_cities = 0
    total_absolute_error_lat = 0.0
    total_absolute_error_long = 0.0
    all_predictions_continents = []
    all_predictions_cities = []
    all_labels_continents = []
    all_labels_cities = []
    all_predicted_lat = []
    all_predicted_long = []
    all_target_lat = []
    all_target_long = []

    with torch.no_grad():
        for batch_idx, (data, continent_city, lat_long) in enumerate(loader):
            batch_size = data.size(0)
            data = data.to(device=device)
            target_continent = continent_city[:, 0].long().to(device=device)
            target_cities = continent_city[:, 1].long().to(device=device)
            target_lat = lat_long[:, 0].float().to(device=device).unsqueeze(1)  # Ensure correct shape
            target_long = lat_long[:, 1].float().to(device=device).unsqueeze(1) # Ensure correct shape

            # Continent predictions
            scores_continent = continent_model(data)
            _, predictions_continent = torch.max(scores_continent, dim=1)
            correct_continent += (predictions_continent == target_continent).sum().item()
            all_predictions_continents.extend(predictions_continent.detach().cpu().numpy())
            all_labels_continents.extend(target_continent.detach().cpu().numpy())

            # Cities predictions (if cities model is provided)
            if cities_model:
                scores_continent_for_cities = continent_model(data)
                in_data_cities = torch.cat((data, scores_continent_for_cities), 1)
                scores_cities = cities_model(in_data_cities)
                _, predictions_cities = torch.max(scores_cities, dim=1)
                correct_cities += (predictions_cities == target_cities).sum().item()
                all_predictions_cities.extend(predictions_cities.detach().cpu().numpy())
                all_labels_cities.extend(target_cities.detach().cpu().numpy())

            # Latitude predictions (if latitude model is provided)
            if lat_model:
                scores_continent_for_lat = continent_model(data)
                in_data_cities_for_lat = torch.cat((data, scores_continent_for_lat), 1)
                scores_cities_for_lat = cities_model(in_data_cities_for_lat)
                in_data_lat = torch.cat((in_data_cities_for_lat, scores_cities_for_lat), 1)
                predicted_lat = lat_model(in_data_lat)
                absolute_error_lat = torch.abs(predicted_lat - target_lat)
                total_absolute_error_lat += torch.sum(absolute_error_lat).item()
                all_predicted_lat.extend(predicted_lat.detach().cpu().numpy())
                all_target_lat.extend(target_lat.detach().cpu().numpy())

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
                all_predicted_long.extend(predicted_long.detach().cpu().numpy())
                all_target_long.extend(target_long.detach().cpu().numpy())

            num_samples += batch_size

    accuracy_continent = float(correct_continent) / float(num_samples) * 100
    print(f'Continent Model Accuracy: {accuracy_continent:.2f}%')

    # Calculate Precision, Recall, and F1-Score for Continents
    precision_continent = precision_score(all_labels_continents, all_predictions_continents, average='weighted', zero_division=0)
    recall_continent = recall_score(all_labels_continents, all_predictions_continents, average='weighted', zero_division=0)
    f1_continent = f1_score(all_labels_continents, all_predictions_continents, average='weighted', zero_division=0)
    print(f'Continent Model Precision: {precision_continent:.4f}')
    print(f'Continent Model Recall (Sensitivity): {recall_continent:.4f}')
    print(f'Continent Model F1-Score: {f1_continent:.4f}')

    if cities_model and num_samples > 0:
        accuracy_cities = float(correct_cities) / float(num_samples) * 100
        print(f'Cities Model Accuracy: {accuracy_cities:.2f}%')
        precision_city = precision_score(all_labels_cities, all_predictions_cities, average='weighted', zero_division=0)
        recall_city = recall_score(all_labels_cities, all_predictions_cities, average='weighted', zero_division=0)
        f1_city = f1_score(all_labels_cities, all_predictions_cities, average='weighted', zero_division=0)
        print(f'Cities Model Precision: {precision_city:.4f}')
        print(f'Cities Model Recall (Sensitivity): {recall_city:.4f}')
        print(f'Cities Model F1-Score: {f1_city:.4f}')

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

    if lat_model and long_model and coordinate_scaler is not None:
        predicted_lat_deg, predicted_long_deg = None, None
        target_lat_deg, target_long_deg = None, None
        try:
            predicted_lat_deg = coordinate_scaler.inverse_transform(np.array(all_predicted_lat).reshape(-1, 1))
            predicted_long_deg = coordinate_scaler.inverse_transform(np.array(all_predicted_long).reshape(-1, 1))
            target_lat_deg = coordinate_scaler.inverse_transform(np.array(all_target_lat).reshape(-1, 1))
            target_long_deg = coordinate_scaler.inverse_transform(np.array(all_target_long).reshape(-1, 1))
            return all_predictions_continents, all_predictions_cities, predicted_lat_deg, predicted_long_deg, all_labels_continents, all_labels_cities, target_lat_deg, target_long_deg
        except Exception as e:
            print(f"Error during inverse transform: {e}")
            return all_predictions_continents, all_predictions_cities, np.array(all_predicted_lat), np.array(all_predicted_long), all_labels_continents, all_labels_cities, np.array(all_target_lat), np.array(all_target_long)

    return all_predictions_continents, all_predictions_cities, np.array(all_predicted_lat), np.array(all_predicted_long), all_labels_continents, all_labels_cities, np.array(all_target_lat), np.array(all_target_long)

if __name__ == "__main__":

    # Define argument parser
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

    # Parse all the arguments
    args = parser.parse_args()
    
    # Load data
    in_data = load_data(args.data_path)
    if in_data is None:
        exit()

    # Process data into correct format
    in_data, le_continent, le_city, stdscaler_lat, stdscaler_long, coordinate_scaler, continent_encoding_map, city_encoding_map =  process_data(in_data)
    
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
    # This is a very clumsy approach. This is because we have manually write all the input sizes. Further models have taken care of this problem.
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

    # Define loss functions
    criterion = nn.CrossEntropyLoss()
    criterion_lat_lon = nn.MSELoss()

    # Define optimizers
    optimizer_continent = torch.optim.Adam(nn_continent_model.parameters(), lr=learning_rate)
    optimizer_cities = torch.optim.Adam(nn_cities_model.parameters(), lr=learning_rate)
    optimizer_lat = torch.optim.SGD(nn_lat_model.parameters(), lr=learning_rate)
    optimizer_long = torch.optim.SGD(nn_long_model.parameters(), lr=learning_rate)

    # Train the models
    train_losses = training_loop(train_dl, nn_continent_model, nn_cities_model, nn_lat_model, nn_long_model,
                  optimizer_continent, optimizer_cities, optimizer_lat, optimizer_long,
                  criterion, criterion_lat_lon, device, num_epochs)

    # Check Training accuracy
    print("\nTraining Accuracy:")
    check_accuracy(train_dl, nn_continent_model, nn_cities_model, nn_lat_model, nn_long_model, device=device)

    # Check Testing Accuracy
    print("\nTest Accuracy:")
    all_predictions_continents, all_predictions_cities, predicted_lat, \
    predicted_long, all_labels_continents, all_labels_cities, \
    target_lat, target_long = check_accuracy(test_dl, nn_continent_model, nn_cities_model, nn_lat_model, nn_long_model, device=device) 

    # Inverse transform the latitude and longitude values
    true_lat = stdscaler_lat.inverse_transform(target_lat)
    true_long = stdscaler_long.inverse_transform(target_long)

    predicted_lat = stdscaler_lat.inverse_transform(predicted_lat)
    predicted_long = stdscaler_lat.inverse_transform(predicted_long) 

    # Use this case when the latitude and longitude values are not scaled
    #true_lat, true_long = target_lat.detach().cpu().numpy(), target_long.detach().cpu().numpy()
    #predicted_lat, predicted_long = predicted_lat.detach().cpu().numpy(), predicted_long.detach().cpu().numpy()

    # Visualizing the results
    plot_losses(train_losses,filename='../results/plots/training_losses_nn_model.png')
    plot_confusion_matrix(all_labels_continents,all_predictions_continents,continent_encoding_map,filename='../results/plots/cofusion_matrix_nn_model.png')
    plot_points_on_world_map(true_lat,true_long,predicted_lat,predicted_long,filename='../results/plots/world_map_nn_model.png')



"""
python nn_model.py -d ../results/metasub_training_testing_data.csv -t 0.2 -r 123 -b 128 -n 1 -e 400 -c True

Training Accuracy:
Continent Model Accuracy: 98.25%
Continent Model Precision: 0.9835
Continent Model Recall (Sensitivity): 0.9825
Continent Model F1-Score: 0.9826
Cities Model Accuracy: 86.18%
Cities Model Precision: 0.8584
Cities Model Recall (Sensitivity): 0.8618
Cities Model F1-Score: 0.8504
Latitude Model Mean Absolute Error: 0.3303
Longitude Model Mean Absolute Error: 0.2753

Test Accuracy:
Continent Model Accuracy: 89.56%
Continent Model Precision: 0.8950
Continent Model Recall (Sensitivity): 0.8956
Continent Model F1-Score: 0.8936
Cities Model Accuracy: 78.75%
Cities Model Precision: 0.7759
Cities Model Recall (Sensitivity): 0.7875
Cities Model F1-Score: 0.7740
Latitude Model Mean Absolute Error: 0.3964
Longitude Model Mean Absolute Error: 0.3445
"""