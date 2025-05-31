# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from process_data_marine import process_data
from check_accuracy_model import plot_losses, plot_confusion_matrix, plot_points_on_world_map

# Neural Network Architecture
class CombinedNeuralNetXYZModel(nn.Module):
    def __init__(self, input_size, num_sea,dropout_rate=0.5):
        super(CombinedNeuralNetXYZModel,self).__init__()

        # ReLU activation function
        self.relu = nn.ReLU()

        # Sea Architechture
        self.sea_layer_1 = nn.Linear(input_size,400)
        self.sea_dropout_1 = nn.Dropout(dropout_rate)
        self.sea_layer_2 = nn.Linear(400,400)
        self.sea_layer_3 = nn.Linear(400,200)

        # Sea Prediction
        self.sea_prediction = nn.Linear(200,num_sea) # Output for different seas

        
        # XYZ Architecture
        self.xyz_layer_1 = nn.Linear(input_size+num_sea,400) # Concatenate the output of the sea layer
        self.xyz_dropout_1 = nn.Dropout(dropout_rate)
        self.xyz_layer_2 = nn.Linear(400,400)
        self.xyz_layer_3 = nn.Linear(400,200)
        
        # XYZ Prediction
        self.xyz_prediction = nn.Linear(200,3) # Three xyz co-ordinates

    def forward(self,x):

        # Sea Architecture
        out_sea = self.relu(self.sea_layer_1(x))
        out_sea = self.sea_dropout_1(out_sea)
        out_sea = self.relu(self.sea_layer_2(out_sea))
        out_sea = self.relu(self.sea_layer_3(out_sea))

        # Sea Prediction
        sea_predictions = self.sea_prediction(out_sea)

        # XYZ Architecture
        input_for_xyz_layer = torch.cat((x, sea_predictions),dim=1)
        out_xyz = self.relu(self.xyz_layer_1(input_for_xyz_layer))
        out_xyz = self.xyz_dropout_1(out_xyz)
        out_xyz = self.relu(self.xyz_layer_2(out_xyz))
        out_xyz = self.relu(self.xyz_layer_3(out_xyz))

        # XYZ Prediction
        xyz_prediction = self.xyz_prediction(out_xyz)

        return sea_predictions, xyz_prediction
    

# Data loading and splitting functions
def load_data(data_path):
    try:
        in_data = pd.read_csv(data_path)
        return in_data
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None

def split_data(in_data, test_size=0.2, random_state=123):
    X = in_data.iloc[:,:-14].values.astype(np.float32)
    y = in_data[['sea_encoding','scaled_x','scaled_y','scaled_z']].values.astype(np.float32)
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
        sea = targ[0].long()
        lat_lon = targ[1:]
        return dp, sea, lat_lon



# Function for training loop
def training_loop(train_dl, combined_model, optimizer_combined, criterion_sea,
                                   criterion_lat_lon, device, num_epochs):
    
    start_time = time.time()
    
    train_losses = {'sea':[],
                    'xyz':[],}
        
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss_sea = 0
        total_loss_xyz = 0

        for batch_idx, (data, sea,lat_long) in enumerate(train_dl):
            data = data.to(device)
            sea_targ = sea.long().to(device)  # continent class
            xyz_targ = lat_long.float().to(device) # x, y, z coords
            
            # Forward pass
            sea_logits, xyz_logits = combined_model(data)
            
            # Calculate losses
            loss_sea = criterion_sea(sea_logits, sea_targ)
            total_loss_sea += loss_sea.detach().cpu().numpy()
            loss_xyz = criterion_lat_lon(xyz_logits, xyz_targ)
            total_loss_xyz += loss_xyz.detach().cpu().numpy()
            total_loss = loss_sea + loss_xyz
            
            # Backward pass and optimization
            optimizer_combined.zero_grad()
            total_loss.backward()
            optimizer_combined.step()

    
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss (Sea): {loss_sea.item():.4f}, Loss (XYZ): {loss_xyz.item():.4f}, Epoch Time: {epoch_duration:.2f} seconds")

        avg_loss_sea = total_loss_sea / len(train_dl)
        avg_loss_xyz = total_loss_xyz / len(train_dl)

        train_losses['sea'].append(avg_loss_sea)
        train_losses['xyz'].append(avg_loss_xyz)

    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

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


# Check_accuracy function for the combined model
def check_combined_accuracy(loader, model, coordinate_scaler=None, device="cpu"):
    
    # Model Evaluation
    model.eval()

    # Initiliaze variables for checking accuracy
    correct_sea = 0
    total = 0
    total_abs_error_lat = 0.0
    total_abs_error_long = 0.0
    num_samples = 0
    all_predictions_sea = []
    all_labels_sea = []
    all_predicted_xyz = []
    all_target_xyz = []

    with torch.no_grad():
        for batch_idx, (data, sea, lat_long_rad) in enumerate(loader):
            batch_size = data.size(0)
            data = data.to(device)
            sea_targ = sea.long().to(device)  # continent class
            xyz_targ_scaled = lat_long_rad.float().to(device) # scaled x, y, z coords

            sea_logits, xyz_pred_scaled = model(data)

            # Continent Accuracy
            _, predictions_sea = sea_logits.max(1)
            correct_sea += (predictions_sea == sea_targ).sum().item()
            all_predictions_sea.extend(predictions_sea.detach().cpu().numpy())
            all_labels_sea.extend(sea_targ.detach().cpu().numpy())
            num_samples += predictions_sea.size(0)

            # XYZ Error
            all_predicted_xyz.append(xyz_pred_scaled.detach().cpu().numpy())
            all_target_xyz.append(xyz_targ_scaled.detach().cpu().numpy())

            total += batch_size

            # ---- Append all the labels and predictions into the list ----
            all_predictions_sea.extend(predictions_sea.detach().cpu().numpy())
            all_labels_sea.extend(sea_targ.detach().cpu().numpy())
            

    accuracy_sea = float(correct_sea) / float(num_samples) * 100
    

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
    precision_sea = precision_score(all_labels_sea, all_predictions_sea, average='weighted', zero_division=0)
    recall_sea = recall_score(all_labels_sea, all_predictions_sea, average='weighted', zero_division=0)
    f1_sea = f1_score(all_labels_sea, all_predictions_sea, average='weighted', zero_division=0)

    print(f'Combined Model - Continent Accuracy: {accuracy_sea:.2f}%')
    print(f'Combined Model - Continent Precision: {precision_sea:.4f}')
    print(f'Combined Model - Continent Recall (Sensitivity): {recall_sea:.4f}')
    print(f'Combined Model - Continent F1-Score: {f1_sea:.4f}')

    print(f'Combined Model - Latitude Mean Absolute Error: {mean_absolute_error_lat:.4f}')
    print(f'Combined Model - Longitude Mean Absolute Error: {mean_absolute_error_long:.4f}')

    return accuracy_sea, precision_sea, recall_sea, f1_sea, \
           mean_absolute_error_lat, mean_absolute_error_long, \
           all_predictions_sea, np.array([predicted_lat_deg, predicted_long_deg]).T, \
           all_labels_sea, np.array([target_lat_deg, target_long_deg]).T


if __name__ == "__main__":
    
    # Parse all the arguements
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
    
    # Load the data
    in_data = load_data(args.data_path)
    if in_data is None:
        exit
    
    # Process data into correct format
    in_data, le_sea, stdscaler_lat, stdscaler_long, coordinate_scaler, sea_enconding_map =  process_data(in_data)


    # Split the dataset    
    X_train, X_test, y_train, y_test =  split_data(in_data=in_data)


    # Create DataLoaders
    train_dl = DataLoader(CustDat(X_train, y_train),
                              batch_size=64, shuffle=True,
                              num_workers=1, pin_memory=False)
    test_dl = DataLoader(CustDat(X_test, y_test),
                             batch_size=64, shuffle=False,
                             num_workers=1, pin_memory=False)



    # Hyperpararmeters
    input_size = 400
    num_sea = len(in_data['sea_encoding'].unique())
    learning_rate = args.learning_rate
    num_epochs = args.epochs

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu" # Change this, add an arugment that specifies the use of cpu
    print(f"Using device: {device}")

    # Initialize the combined neural network model
    combined_model = CombinedNeuralNetXYZModel(input_size=400,num_sea=9).to(device=device)
        
    # Loss functions and optimizers
    criterion_sea = nn.CrossEntropyLoss()
    criterion_lat_lon = nn.MSELoss()
    optimizer_combined = torch.optim.Adam(combined_model.parameters(), lr=0.001, weight_decay=0.00001)

    # Train the model for the current fold
    train_losses_fold = training_loop(train_dl, combined_model, optimizer_combined, criterion_sea,
                                       criterion_lat_lon, device, num_epochs)



    # Check accuracy
    print("\nTraining Accuracy:")
    check_combined_accuracy(train_dl, combined_model, coordinate_scaler, device)
    print("\nTest Accuracy:")
    accuracy_sea_test, precision_sea_test, recall_sea_test, f1_sea_test, \
    mae_lat_test, mae_long_test, \
    all_predictions_sea_test,  predicted_lat_long_deg_test, \
    all_labels_sea_test, targ_lat_long_deg_test =check_combined_accuracy(test_dl, combined_model, coordinate_scaler, device) 


    predicted_lat, predicted_long = predicted_lat_long_deg_test[:,0], predicted_lat_long_deg_test[:,1]
    true_lat, true_long = targ_lat_long_deg_test[:,0], targ_lat_long_deg_test[:,1]

    # Visualizing the results
    plot_losses(train_losses_fold,filename='/home/chandru/binp37/results/plots/marine/training_losses_nn_combined_model_lat_long_marine_model.png')
    plot_confusion_matrix(all_labels_sea_test,all_predictions_sea_test,sea_enconding_map,filename='/home/chandru/binp37/results/plots/marine/cofusion_matrix_nn_combined_model_lat_long_marine_model.png')
    plot_points_on_world_map(true_lat,true_long,predicted_lat,predicted_long,filename='/home/chandru/binp37/results/plots/marine/world_map_nn_combined_model_lat_long_marine_model.png')
