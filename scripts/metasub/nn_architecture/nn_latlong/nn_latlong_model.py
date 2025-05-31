# This script is to get the latitude and longitude prediction accuracy higher.

# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

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
        return (xyz_prediction,)
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedNeuralNetCNNXYZModel(nn.Module):
    def __init__(self, input_size, hidden_dim, initial_dropout_rate, max_dropout_rate):
        super(CombinedNeuralNetCNNXYZModel, self).__init__()

        self.initial_dropout_rate = initial_dropout_rate
        self.max_dropout_rate = max_dropout_rate
        self.relu = nn.ReLU()

        # Assuming input_size (200) is the length of the 1D sequence of features
        # and we have 1 input channel (since it's a single microbial profile)
        input_channels = 1
        num_features = input_size # This will be 200

        # CNN Layers
        # Conv1d expects input of shape (batch_size, channels, sequence_length)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the size of the flattened layer dynamically
        # num_features after pool1: num_features // 2
        # num_features after pool2: (num_features // 2) // 2 = num_features // 4
        self.flattened_size = 64 * (num_features // 4) # 64 filters * remaining length

        # Fully connected layers for XYZ prediction
        self.fc1 = nn.Linear(self.flattened_size, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim // 2)

        # XYZ Prediction
        self.xyz_prediction = nn.Linear(hidden_dim // 2, 3) # Output for three xyz coordinates

    def forward(self, x, current_dropout_rate):
        # Add a channel dimension to the input if it's not already there.
        # Original: (batch_size, num_features) -> Desired for Conv1d: (batch_size, 1, num_features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # Adds a dimension at index 1

        # CNN Forward Pass
        out = self.pool1(self.relu(self.bn1(self.conv1(x))))
        out = self.pool2(self.relu(self.bn2(self.conv2(out))))

        # Flatten the output for the fully connected layers
        out = out.view(out.size(0), -1) # Flatten (batch_size, channels, length) to (batch_size, channels * length)

        # Fully Connected Layers with dynamic dropout
        out = self.relu(self.bn_fc1(self.fc1(out)))
        out = F.dropout(out, p=current_dropout_rate, training=self.training)
        out = self.relu(self.bn_fc2(self.fc2(out)))
        out = F.dropout(out, p=current_dropout_rate, training=self.training)

        xyz_prediction = self.xyz_prediction(out)

        # Return as a tuple for consistency with a generalized testing loop
        return (xyz_prediction,)
    

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