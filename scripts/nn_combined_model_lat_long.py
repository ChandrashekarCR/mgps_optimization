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
from check_accuracy_model import plot_losses, plot_confusion_matrix, plot_points_on_world_map, pull_land, plot_predictions_with_coastline, calculate_mae_km
from process_data import process_data


class CombinedNeuralNetXYZModel(nn.Module):
    def __init__(self, input_size, num_continent, num_cities,dropout_rate=0.5):
        super(CombinedNeuralNetXYZModel,self).__init__()

        # ReLU activation function
        self.relu = nn.ReLU()

        # Continent Architechture
        self.continent_layer_1 = nn.Linear(input_size,400)
        self.continent_dropout_1 = nn.Dropout(dropout_rate)
        self.continent_layer_2 = nn.Linear(400,400)
        self.continent_layer_3 = nn.Linear(400,200)

        # Continent Prediction
        self.continent_prediction = nn.Linear(200,num_continent) # Output for 7 different continents

        # City Architecture
        self.city_layer_1 = nn.Linear(input_size+num_continent,400) # Concatenate the output of the continent layers
        self.city_dropout_1 = nn.Dropout(dropout_rate)
        self.city_layer_2 = nn.Linear(400,400)
        self.city_layer_3 = nn.Linear(400,200)

        # City Prediction
        self.city_prediction = nn.Linear(200,num_cities) # Output for 40 different cities

        # XYZ Architecture
        self.xyz_layer_1 = nn.Linear(input_size+num_continent+num_cities,400) # Concatenate the output of the continent and cities layers
        self.xyz_dropout_1 = nn.Dropout(dropout_rate)
        self.xyz_layer_2 = nn.Linear(400,400)
        self.xyz_layer_3 = nn.Linear(400,200)
        
        # XYZ Prediction
        self.xyz_prediction = nn.Linear(200,3) # Three xyz co-ordinates

    def forward(self,x):

        # Continent Architecture
        out_continent = self.relu(self.continent_layer_1(x))
        out_continent = self.continent_dropout_1(out_continent)
        out_continent = self.relu(self.continent_layer_2(out_continent))
        out_continent = self.relu(self.continent_layer_3(out_continent))
        
        # Continent Prediction
        continent_predictions = self.continent_prediction(out_continent)

        # City Architecture
        input_for_city_layer = torch.cat((x,continent_predictions),dim=1)
        out_cities = self.relu(self.city_layer_1(input_for_city_layer))
        out_cities = self.city_dropout_1(out_cities)
        out_cities = self.relu(self.city_layer_2(out_cities))
        out_cities = self.relu(self.city_layer_3(out_cities))

        # City Prediction
        city_predictions  = self.city_prediction(out_cities)

        # XYZ Architecture
        input_for_xyz_layer = torch.cat((x, continent_predictions, city_predictions),dim=1)
        out_xyz = self.relu(self.xyz_layer_1(input_for_xyz_layer))
        out_xyz = self.xyz_dropout_1(out_xyz)
        out_xyz = self.relu(self.xyz_layer_2(out_xyz))
        out_xyz = self.relu(self.xyz_layer_3(out_xyz))

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
def training_loop(train_dl, combined_model, optimizer_combined, criterion_continent, 
                  criterion_cities, criterion_lat_lon, device, num_epochs):
    
    start_time = time.time()
    
    train_losses = {'continent':[],
                    'cities':[],
                    'xyz':[],}
        
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss_continent = 0
        total_loss_cities = 0
        total_loss_xyz = 0

        for batch_idx, (data, continent_city,lat_long) in enumerate(train_dl):
                data = data.to(device)
                cont_targ = continent_city[:, 0].long().to(device)  # continent class
                city_targ = continent_city[:, 1].long().to(device)  # city class
                xyz_targ = lat_long.float().to(device) # x, y, z coords

                # Forward pass
                continent_logits, city_logits, xyz_logits = combined_model(data)

                # Calculate losses
                loss_continents = criterion_continent(continent_logits, cont_targ)
                total_loss_continent += loss_continents.detach().cpu().numpy()
                loss_cities = criterion_cities(city_logits, city_targ)
                total_loss_cities += loss_cities.detach().cpu().numpy()
                loss_xyz = criterion_lat_lon(xyz_logits, xyz_targ)
                total_loss_xyz += loss_xyz.detach().cpu().numpy()
                total_loss = loss_continents + loss_cities + loss_xyz

                # Backward pass and optimization
                optimizer_combined.zero_grad()
                total_loss.backward()
                optimizer_combined.step()

    
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        if (epoch+1) % 80 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss (Continent): {loss_continents.item():.4f}, \
                  Loss (Cities): {loss_cities.item():.4f}, Loss (XYZ): {loss_xyz.item():.4f}, \
                    Epoch Time: {epoch_duration:.2f} seconds")

        avg_loss_continent = total_loss_continent / len(train_dl)
        avg_loss_cities = total_loss_cities / len(train_dl)
        avg_loss_xyz = total_loss_xyz / len(train_dl)

        train_losses['continent'].append(avg_loss_continent)
        train_losses['cities'].append(avg_loss_cities)
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
    correct_continent = 0
    correct_cities = 0
    total = 0
    total_abs_error_lat = 0.0
    total_abs_error_long = 0.0
    num_samples = 0
    all_predictions_continents = []
    all_predictions_cities = []
    all_labels_continents = []
    all_labels_cities = []
    all_predicted_xyz = []
    all_target_xyz = []

    with torch.no_grad():
        for batch_idx, (data, continent_city, lat_long_rad) in enumerate(loader):
            batch_size = data.size(0)
            data = data.to(device)
            cont_targ = continent_city[:, 0].long().to(device)  # continent class
            city_targ = continent_city[:, 1].long().to(device)  # city class
            xyz_targ_scaled = lat_long_rad.float().to(device) # scaled x, y, z coords

            continent_logits, city_logits, xyz_pred_scaled = model(data)

            # Continent Accuracy
            _, predictions_continent = continent_logits.max(1)
            correct_continent += (predictions_continent == cont_targ).sum().item()
            all_predictions_continents.extend(predictions_continent.detach().cpu().numpy())
            all_labels_continents.extend(cont_targ.detach().cpu().numpy())
            num_samples += predictions_continent.size(0)

            # Cities Accuracy
            _, predictions_cities = city_logits.max(1)
            correct_cities += (predictions_cities == city_targ).sum().item()
            all_predictions_cities.extend(predictions_cities.detach().cpu().numpy())
            all_labels_cities.extend(city_targ.detach().cpu().numpy())

            # XYZ Error
            all_predicted_xyz.append(xyz_pred_scaled.detach().cpu().numpy())
            all_target_xyz.append(xyz_targ_scaled.detach().cpu().numpy())

            total += batch_size

            # ---- Append all the labels and predictions into the list ----
            all_predictions_continents.extend(predictions_continent.detach().cpu().numpy())
            all_labels_continents.extend(cont_targ.detach().cpu().numpy())
            all_predictions_cities.extend(predictions_cities.detach().cpu().numpy())
            all_labels_cities.extend(city_targ.detach().cpu().numpy())

    accuracy_continent = float(correct_continent) / float(num_samples) * 100
    accuracy_cities = float(correct_cities) / float(num_samples) * 100
    

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

    return accuracy_continent, accuracy_cities, precision_continent, recall_continent, f1_continent, \
           precision_city, recall_city, f1_city, mean_absolute_error_lat, mean_absolute_error_long, \
           all_predictions_continents, all_predictions_cities, np.array([predicted_lat_deg, predicted_long_deg]).T, \
           all_labels_continents, all_labels_cities, np.array([target_lat_deg, target_long_deg]).T

def cross_validated_training_testing(in_data,X,y,random_state,batch_size,num_workers,pin_memory,learning_rate,epochs):

    # Define the K-Fold split
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Store the validation metrics
    all_fold_train_losses = []
    all_fold_val_results = [] # To store validation metrics for each fold

    for fold, (train_index,val_index) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold+1} ---")

        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        train_dataset_fold = CustDat(X_train_fold,y_train_fold)
        val_dataset_fold = CustDat(X_val_fold, y_val_fold)

        train_dl_fold = DataLoader(train_dataset_fold,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=pin_memory)
        val_dl_fold = DataLoader(val_dataset_fold,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=pin_memory)

        # Initialize model, optimizer and the loss function for each fold

        # Hyperparameters
        input_size = 200
        num_continent = len(in_data['continent_encoding'].unique())
        num_cities = len(in_data['city_encoding'].unique())
        learning_rate = 0.001
        learning_rate = learning_rate
        num_epochs = epochs

        # Determine device
        device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
        print(f"Using device: {device}")


        combined_model = CombinedNeuralNetXYZModel(input_size=input_size, num_continent=num_continent, num_cities=num_cities).to(device=device)
        
        # Loss functions and optimizers
        criterion_continent = nn.CrossEntropyLoss()
        criterion_cities = nn.CrossEntropyLoss()
        criterion_lat_lon = nn.MSELoss()
        optimizer_combined = torch.optim.Adam(combined_model.parameters(), lr=learning_rate)

        # Train the model for the current fold
        train_losses_fold = training_loop(train_dl_fold, combined_model, optimizer_combined, criterion_continent,
                                           criterion_cities, criterion_lat_lon, device, num_epochs=args.epochs)
        all_fold_train_losses.append(train_losses_fold)

        # Evaluate on the validation set for the current fold
        print("\nValidation Accuracy (Fold {}):".format(fold + 1))
        val_continent_accuracy, val_cities_accuracy, val_precision_continent, val_recall_continent, val_f1_continent, \
        val_precision_city, val_recall_city, val_f1_city, val_lat_mae, val_long_mae, \
        val_predictions_continents, val_predictions_cities, val_predicted_lat_long_deg, \
        val_labels_continents, val_labels_cities, val_targ_lat_long_deg = \
            check_combined_accuracy(val_dl_fold, combined_model, coordinate_scaler, device)

        all_fold_val_results.append((val_continent_accuracy, val_cities_accuracy, val_precision_continent,
                                     val_recall_continent, val_f1_continent, val_precision_city,
                                     val_recall_city, val_f1_city, val_lat_mae, val_long_mae))

    # After all folds, average the validation results
    print("\n--- Average Validation Results Across All Folds ---")
    avg_val_continent_accuracy = np.mean([res[0] for res in all_fold_val_results])
    avg_val_cities_accuracy = np.mean([res[1] for res in all_fold_val_results])
    avg_val_precision_continent = np.mean([res[2] for res in all_fold_val_results])
    avg_val_recall_continent = np.mean([res[3] for res in all_fold_val_results])
    avg_val_f1_continent = np.mean([res[4] for res in all_fold_val_results])
    avg_val_precision_city = np.mean([res[5] for res in all_fold_val_results])
    avg_val_recall_city = np.mean([res[6] for res in all_fold_val_results])
    avg_val_f1_city = np.mean([res[7] for res in all_fold_val_results])
    avg_val_lat_mae = np.mean([res[8] for res in all_fold_val_results])
    avg_val_long_mae = np.mean([res[9] for res in all_fold_val_results])

    print(f'Average Validation Continent Accuracy: {avg_val_continent_accuracy:.2f}%')
    print(f'Average Validation Continent Precision: {avg_val_precision_continent:.4f}')
    print(f'Average Validation Continent Recall: {avg_val_recall_continent:.4f}')
    print(f'Average Validation Continent F1-Score: {avg_val_f1_continent:.4f}')
    print(f'Average Validation Cities Accuracy: {avg_val_cities_accuracy:.2f}%')
    print(f'Average Validation Cities Precision: {avg_val_precision_city:.4f}')
    print(f'Average Validation Cities Recall: {avg_val_recall_city:.4f}')
    print(f'Average Validation Cities F1-Score: {avg_val_f1_city:.4f}')
    print(f'Average Validation Latitude Mean Absolute Error: {avg_val_lat_mae:.4f}')
    print(f'Average Validation Longitude Mean Absolute Error: {avg_val_long_mae:.4f}')

    # Finally, train your model on the entire training set and evaluate on the original test set
    print("\n--- Training on the Entire Training Set and Evaluating on Test Set ---")
    train_dataset_full = CustDat(X_train, y_train)
    train_dl_full = DataLoader(train_dataset_full, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)

    final_model = CombinedNeuralNetXYZModel(input_size=input_size, num_continent=num_continent, num_cities=num_cities).to(device)
    optimizer_final = torch.optim.Adam(final_model.parameters(), lr=learning_rate)
    criterion_continent_final = nn.CrossEntropyLoss()
    criterion_cities_final = nn.CrossEntropyLoss()
    criterion_lat_lon_final = nn.MSELoss()

    final_train_losses = training_loop(train_dl_full, final_model, optimizer_final, criterion_continent_final,
                                        criterion_cities_final, criterion_lat_lon_final, device, num_epochs=args.epochs)

    print("\nFinal Model - Training Accuracy:")
    check_combined_accuracy(train_dl_full, final_model, coordinate_scaler, device)
    print("\nFinal Model - Test Accuracy:")
    accuracy_continent_test, accuracy_cities_test, precision_continent_test, recall_continent_test, f1_continent_test, \
    precision_city_test, recall_city_test, f1_city_test, mae_lat_test, mae_long_test, \
    all_predictions_continents_test, all_predictions_cities_test, predicted_lat_long_deg_test, \
    all_labels_continents_test, all_labels_cities_test, targ_lat_long_deg_test = \
            check_combined_accuracy(test_dl, final_model, coordinate_scaler, device)
    
    

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

    # Now call the plotting function
    plot_predictions_with_coastline(
    predictions_df=updated_predictions_df,
    coastline_path="/home/chandru/binp37/data/geopandas/ne_110m_coastline.shp",
    lat_col='predicted_lat',
    lon_col='predicted_lon',
    updated_lat_col='updated_lat',  # Now the updated values are in these columns
    updated_lon_col='updated_lon',
    true_lat_col='true_latitude',
    true_lon_col='true_longitude',
    filename='/home/chandru/binp37/results/plots/predictions_vs_adjusted_coastline_correct.png'
    )

    # Calculate Mean Absolute Error in kilometers
    mae_lat_km, mae_lon_km = calculate_mae_km(
    updated_predictions_df,
    predicted_lat_col='updated_lat',
    predicted_lon_col='updated_lon',
    true_lat_col='true_latitude',
    true_lon_col='true_longitude'
    )


    print(f"\nMean Absolute Error (km) - Latitude: {mae_lat_km:.4f}")
    print(f"Mean Absolute Error (km) - Longitude: {mae_lon_km:.4f}")

    predicted_lat_test, predicted_long_test, true_lat_test, true_long_test = updated_predictions_df['updated_lat'].values, \
                                                                            updated_predictions_df['updated_lon'].values, \
                                                                            updated_predictions_df['true_latitude'].values, \
                                                                            updated_predictions_df['true_longitude'].values


    # Visualizing the results of the final model on the test set
    plot_losses(final_train_losses, filename='../results/plots/final_training_losses_nn_combined_model_lat_long.png')
    plot_confusion_matrix(all_labels_continents_test, all_predictions_continents_test, continent_encoding_map, filename='../results/plots/final_cofusion_matrix_nn_combined_model_lat_long.png')
    plot_confusion_matrix(all_labels_cities_test,all_predictions_cities_test,city_encoding_map,filename='../results/plots/final_confusion_matrix_nn_combined_model_lat_long.png')
    plot_points_on_world_map(true_lat_test, true_long_test, predicted_lat_test, predicted_long_test, filename='../results/plots/final_world_map_nn_combined_model_lat_long.png')




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
    
    cross_validated_training_testing(in_data=in_data,X = in_data.iloc[:, :200].values.astype(np.float32),
                                     y = in_data[['continent_encoding', 'city_encoding', 'scaled_x','scaled_y','scaled_z']].values.astype(np.float32),
                                     random_state=args.random_state,batch_size=args.batch_size,num_workers=args.num_workers,
                                     pin_memory=args.pin_memory,learning_rate=args.learning_rate,epochs=args.epochs)

    '''
    # Hyperparameters
    input_size = 200
    num_continent = len(in_data['continent_encoding'].unique())
    num_cities = len(in_data['city_encoding'].unique())
    learning_rate = 0.001
    learning_rate = args.learning_rate
    num_epochs = args.epochs

    # Determine device
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    print(f"Using device: {device}")

    # Initialize the combined network
    combined_model = CombinedNeuralNetXYZModel(input_size=input_size,num_continent=num_continent,num_cities=num_cities).to(device)

    # Loss functions and optimizers
    criterion_continent = nn.CrossEntropyLoss()
    criterion_cities = nn.CrossEntropyLoss()
    criterion_lat_lon = nn.MSELoss()
    optimizer_combined = torch.optim.Adam(combined_model.parameters(), lr=learning_rate, weight_decay=0.00001)
    
    # Train the models
    train_losses = training_loop(train_dl,combined_model=combined_model,num_epochs=num_epochs,optimizer_combined=optimizer_combined,
                  criterion_continent=criterion_continent,criterion_cities=criterion_cities,
                  criterion_lat_lon=criterion_lat_lon,device=device)
    

    # Check accuracy
    print("\nTraining Accuracy:")
    check_combined_accuracy(train_dl, combined_model, coordinate_scaler, device)
    print("\nTest Accuracy:")
    all_predictions_continents, all_predictions_cities, predicted_lat_long_deg, all_labels_continents, \
    all_labels_cities, targ_lat_long_deg =check_combined_accuracy(test_dl, combined_model, coordinate_scaler, device) 


    predicted_lat, predicted_long = predicted_lat_long_deg[:,0], predicted_lat_long_deg[:,1]
    true_lat, true_long = targ_lat_long_deg[:,0], targ_lat_long_deg[:,1]

    # Visualizing the results
    plot_losses(train_losses,filename='../results/plots/training_losses_nn_combined_model_lat_long.png')
    plot_confusion_matrix(all_labels_continents,all_predictions_continents,continent_encoding_map,filename='../results/plots/cofusion_matrix_nn_combined_model_lat_long.png')
    plot_points_on_world_map(true_lat,true_long,predicted_lat,predicted_long,filename='../results/plots/world_map_nn_combined_model_lat_long.png')

'''
"""
python nn_combined_model_lat_long.py -d ../results/metasub_training_testing_data.csv -t 0.2 -r 123 -b 128 -n 1 -e 400 -c True

Training Accuracy:
Combined Model - Continent Accuracy: 97.76%
Combined Model - Continent Precision: 0.9782
Combined Model - Continent Recall (Sensitivity): 0.9776
Combined Model - Continent F1-Score: 0.9776
Combined Model - Cities Accuracy: 97.14%
Combined Model - Cities Precision: 0.9730
Combined Model - Cities Recall (Sensitivity): 0.9714
Combined Model - Cities F1-Score: 0.9711
Combined Model - Latitude Mean Absolute Error: 1.2849
Combined Model - Longitude Mean Absolute Error: 3.5389

Test Accuracy:
Combined Model - Continent Accuracy: 87.35%
Combined Model - Continent Precision: 0.8692
Combined Model - Continent Recall (Sensitivity): 0.8735
Combined Model - Continent F1-Score: 0.8690
Combined Model - Cities Accuracy: 82.92%
Combined Model - Cities Precision: 0.8296
Combined Model - Cities Recall (Sensitivity): 0.8292
Combined Model - Cities F1-Score: 0.8221
Combined Model - Latitude Mean Absolute Error: 4.5070
Combined Model - Longitude Mean Absolute Error: 14.4377

"""