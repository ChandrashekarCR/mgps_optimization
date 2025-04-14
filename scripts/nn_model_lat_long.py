# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
from process_data import process_data
from check_accuracy_model import plot_losses, plot_confusion_matrix, plot_points_on_world_map

# Define the neural network modules 
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

class NeuralNetLatLong(nn.Module):
    def __init__(self, input_xyz, cordinates, dropout_rate=0.5):
        super(NeuralNetLatLong,self).__init__()
        self.layer1 = nn.Linear(input_xyz, 400)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(400, 400)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(400, 200)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(200, cordinates)
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
    
# Function for training loop
def training_loop(train_dl, continent_model, cities_model, xyz_model,
                  optimizer_continent, optimizer_cities, optimizer_xyz,
                  criterion_continent, criterion_cities, criterion_xyz,
                  num_epochs, device):

    train_losses = {
        'continent': [],
        'cities': [],
        'xyz': []
    }

    for epoch in range(num_epochs):
        total_loss_continent = 0.0
        total_loss_cities = 0.0
        total_loss_xyz = 0.0

        continent_model.train()
        cities_model.train()
        xyz_model.train()

        for batch_idx, (data, continent_city,lat_long) in enumerate(train_dl):
            data = data.to(device)
            cont_targ = continent_city[:, 0].long().to(device)  # continent class
            city_targ = continent_city[:, 1].long().to(device)  # city class
            xyz_targ = lat_long.float().to(device) # x, y, z coords

            ### Train Continent Model ###
            optimizer_continent.zero_grad()
            cont_scores = continent_model(data)
            loss_cont = criterion_continent(cont_scores, cont_targ)
            loss_cont.backward()
            optimizer_continent.step()

            ### Train City Model ###
            optimizer_cities.zero_grad()
            cities_input = torch.cat((data, cont_scores.detach()), dim=1)
            city_scores = cities_model(cities_input)
            loss_city = criterion_cities(city_scores, city_targ)
            loss_city.backward()
            optimizer_cities.step()

            ### Train LatLong (XYZ) Model ###
            optimizer_xyz.zero_grad()
            xyz_input = torch.cat((data, cont_scores.detach(), city_scores.detach()), dim=1)
            xyz_pred = xyz_model(xyz_input)
            loss_xyz = criterion_xyz(xyz_pred, xyz_targ)
            loss_xyz.backward()
            optimizer_xyz.step()

            total_loss_continent += loss_cont.item()
            total_loss_cities += loss_city.item()
            total_loss_xyz += loss_xyz.item()

        # Epoch loss logging
        train_losses['continent'].append(total_loss_continent / len(train_dl))
        train_losses['cities'].append(total_loss_cities / len(train_dl))
        train_losses['xyz'].append(total_loss_xyz / len(train_dl))

        if (epoch+1)%10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Continent Loss: {train_losses['continent'][-1]:.4f}, "
              f"City Loss: {train_losses['cities'][-1]:.4f}, "
              f"XYZ Loss: {train_losses['xyz'][-1]:.4f}")

    return train_losses


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


def check_accuracy(data_loader, continent_model, cities_model, xyz_model, coordinate_scaler, device):
    
    # Model Evaluation
    continent_model.eval()
    cities_model.eval()
    xyz_model.eval()

    # Initiliaze variables for checking accuracy
    correct_cont = 0
    correct_city = 0
    total = 0
    total_abs_error_lat = 0.0
    total_abs_error_long = 0.0
    all_predictions_continents = []
    all_predictions_cities = []
    all_labels_continents = []
    all_labels_cities = []
    all_predicted_xyz = []
    all_target_xyz = []

    with torch.no_grad():
        for batch_idx, (data, continent_city, lat_long_rad) in enumerate(data_loader):
            batch_size = data.size(0)
            data = data.to(device)
            cont_targ = continent_city[:, 0].long().to(device)  # continent class
            city_targ = continent_city[:, 1].long().to(device)  # city class
            xyz_targ_scaled = lat_long_rad.float().to(device) # scaled x, y, z coords

            # ---- Continent Prediction ----
            cont_scores = continent_model(data)
            _, pred_cont = torch.max(cont_scores, dim=1)
            correct_cont += (pred_cont == cont_targ).sum().item()

            # ---- City Prediction ----
            city_input = torch.cat((data, cont_scores.detach()), dim=1)
            city_scores = cities_model(city_input)
            _, pred_city = torch.max(city_scores, dim=1)
            correct_city += (pred_city == city_targ).sum().item()

            # ---- XYZ Prediction ----
            xyz_input = torch.cat((data, cont_scores.detach(), city_scores.detach()), dim=1)
            xyz_pred_scaled = xyz_model(xyz_input)

            all_predicted_xyz.append(xyz_pred_scaled.detach().cpu().numpy())
            all_target_xyz.append(xyz_targ_scaled.detach().cpu().numpy())


            total += batch_size

            # ---- Append all the labels and predictions into the list ----
            all_predictions_continents.extend(pred_cont.detach().cpu().numpy())
            all_labels_continents.extend(cont_targ.detach().cpu().numpy())
            all_predictions_cities.extend(pred_city.detach().cpu().numpy())
            all_labels_cities.extend(city_targ.detach().cpu().numpy())

    accuracy_continent = (correct_cont / total) * 100
    accuracy_cities = (correct_city / total) * 100

    all_predicted_xyz = np.concatenate(all_predicted_xyz, axis=0)
    all_target_xyz = np.concatenate(all_target_xyz, axis=0)

    # Inverse transform scaled xyz to get latitude and longitude
    predicted_lat_deg, predicted_long_deg = inverse_transform_spherical(all_predicted_xyz, coordinate_scaler)
    target_lat_deg, target_long_deg = inverse_transform_spherical(all_target_xyz, coordinate_scaler)


    # Calculate Mean Absolute Error for Latitude and Longitude
    total_abs_error_lat = np.sum(np.abs(predicted_lat_deg - target_lat_deg))
    total_abs_error_long = np.sum(np.abs(predicted_long_deg - target_long_deg))
    mean_absolute_error_lat = total_abs_error_lat / total
    mean_absolute_error_long = total_abs_error_long / total

    # Calculate Precision, Recall, and F1-Score for Continents
    precision_continent = precision_score(all_labels_continents, all_predictions_continents, average='weighted', zero_division=0)
    recall_continent = recall_score(all_labels_continents, all_predictions_continents, average='weighted', zero_division=0)
    f1_continent = f1_score(all_labels_continents, all_predictions_continents, average='weighted', zero_division=0)

    # Calculate Precision, Recall, and F1-Score for Cities
    precision_city = precision_score(all_labels_cities, all_predictions_cities, average='weighted', zero_division=0)
    recall_city = recall_score(all_labels_cities, all_predictions_cities, average='weighted', zero_division=0)
    f1_city = f1_score(all_labels_cities, all_predictions_cities, average='weighted', zero_division=0)

    print(f'Combined Model - Continent Accuracy: {accuracy_continent:.2f}%')
    print(f'Combined Model - Continent Precision: {precision_continent:.4f}')
    print(f'Combined Model - Continent Recall (Sensitivity): {recall_continent:.4f}')
    print(f'Combined Model - Continent F1-Score: {f1_continent:.4f}')

    print(f'Combined Model - Cities Accuracy: {accuracy_cities:.2f}%')
    print(f'Combined Model - Cities Precision: {precision_city:.4f}')
    print(f'Combined Model - Cities Recall (Sensitivity): {recall_city:.4f}')
    print(f'Combined Model - Cities F1-Score: {f1_city:.4f}')

    print(f'Combined Model - Latitude Mean Absolute Error: {mean_absolute_error_lat:.4f}')
    print(f'Combined Model - Longitude Mean Absolute Error: {mean_absolute_error_long:.4f}')

    return all_predictions_continents, all_predictions_cities, np.array([predicted_lat_deg, predicted_long_deg]).T, all_labels_continents, all_labels_cities, np.array([target_lat_deg, target_long_deg]).T


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
    input_size_continents = 200
    input_size_cities = 207
    input_xyz = 247
    num_continents = 7
    num_cities = 40
    cordinates = 3
    learning_rate = args.learning_rate
    num_epochs = args.epochs

    # Determine device
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    print(f"Using device: {device}")

    # Initialize models
    nn_continent_model = NeuralNetContinent(input_size_continents, num_continents).to(device)
    nn_cities_model = NeuralNetCities(input_size_cities, num_cities).to(device)
    nn_latlong_model = NeuralNetLatLong(input_xyz=input_xyz,cordinates=cordinates).to(device)

    # Define loss functions and optimizers
    criterion_continent = nn.CrossEntropyLoss()
    criterion_cities = nn.CrossEntropyLoss()
    criterion_latlong = nn.MSELoss()
    optimizer_continent = torch.optim.Adam(nn_continent_model.parameters(), lr=learning_rate)
    optimizer_cities = torch.optim.Adam(nn_cities_model.parameters(), lr=learning_rate)
    optimizer_latlong = torch.optim.SGD(nn_latlong_model.parameters(), lr=learning_rate)

    # Train the models
    train_losses = training_loop(train_dl, nn_continent_model, nn_cities_model, nn_latlong_model,
              optimizer_continent, optimizer_cities, optimizer_latlong,
              criterion_continent, criterion_cities, criterion_latlong,
              num_epochs, device)



    # Check accuracy
    print("\nTraining Accuracy:")
    check_accuracy(train_dl, nn_continent_model, nn_cities_model, nn_latlong_model, coordinate_scaler, device)
    print("\nTest Accuracy:")
    all_predictions_continents, all_predictions_cities, predicted_lat_long_deg, all_labels_continents, \
    all_labels_cities, targ_lat_long_deg =check_accuracy(test_dl, nn_continent_model, nn_cities_model, \
    nn_latlong_model, coordinate_scaler, device)  

    predicted_lat, predicted_long = predicted_lat_long_deg[:,0], predicted_lat_long_deg[:,1]
    true_lat, true_long = targ_lat_long_deg[:,0], targ_lat_long_deg[:,1]

    # Visualizing the results
    plot_losses(train_losses,filename='../results/plots/training_losses_nn_model_lat_long.png')
    plot_confusion_matrix(all_labels_continents,all_predictions_continents,continent_encoding_map,filename='../results/plots/cofusion_matrix_nn_model_lat_long.png')
    plot_points_on_world_map(true_lat,true_long,predicted_lat,predicted_long,filename='../results/plots/world_map_nn_model_lat_long.png')


"""
python nn_model_lat_long.py -d ../results/metasub_training_testing_data.csv -t 0.2 -r 123 -b 128 -n 1 -e 400 -c True

Training Accuracy:
Combined Model - Continent Accuracy: 98.25%
Combined Model - Continent Precision: 0.9836
Combined Model - Continent Recall (Sensitivity): 0.9825
Combined Model - Continent F1-Score: 0.9826
Combined Model - Cities Accuracy: 84.52%
Combined Model - Cities Precision: 0.8379
Combined Model - Cities Recall (Sensitivity): 0.8452
Combined Model - Cities F1-Score: 0.8285
Combined Model - Latitude Mean Absolute Error: 7.7528
Combined Model - Longitude Mean Absolute Error: 15.0948

Test Accuracy:
Combined Model - Continent Accuracy: 89.31%
Combined Model - Continent Precision: 0.8907
Combined Model - Continent Recall (Sensitivity): 0.8931
Combined Model - Continent F1-Score: 0.8900
Combined Model - Cities Accuracy: 75.92%
Combined Model - Cities Precision: 0.7431
Combined Model - Cities Recall (Sensitivity): 0.7592
Combined Model - Cities F1-Score: 0.7423
Combined Model - Latitude Mean Absolute Error: 8.5982
Combined Model - Longitude Mean Absolute Error: 21.2971


"""