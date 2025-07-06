# Importing libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from check_accuracy_model import plot_losses, plot_confusion_matrix, plot_points_on_world_map


# Create a dynamic architecutre in pytorch
# First model - In this model, I made a neural network separate for each hierarchy. The same architechture is followed for 
# continent level prediction, city level prediction, cordinate level prediction.

# Continent Neural Network
class ContinentNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = [128,64], use_batch_norm=True, dropout_rate = [0.2,0.7], random_state=42):
        super(ContinentNeuralNetwork,self).__init__()

        """
        Initialize Continent architecture using Pytorch modules.

        Parameters:
        - input_size: Number of input features. In this case it is the GITs. # 200
        - hidden_layers: List of hidden layers # 256, 128 are the default
        - output_size: Number of continents
        - dropout_rate: [0.2, 0.7]
        - random_state: Random state for reporducibility
        
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.use_batch_norm = use_batch_norm

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Build the neural network
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Create the layer architecture
        layer_sizes = [input_size] + hidden_size + [output_size]

        for i in range(len(layer_sizes)-1):
            # Add the linear layers first
            self.layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))

            # Add batch normalization for hidden layers only and not for the output layers
            if i < len(layer_sizes) - 2 and self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i+1]))

            # Add dropout for hidden layers onyl and not for the output layers
            if i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rate))

    def forward(self,x):
        """
        Forward propagations through the network
        
        Parameters:
        - x: Input tensor        
        """

        current_input = x

        # Forward pass through the hidden layers
        for i, (layer, dropout) in enumerate(zip(self.layers[:-1],self.dropouts)):
            # Linear transformations
            z = layer(current_input)

            # Batch normalization if enabled
            if self.use_batch_norm:
                z = self.batch_norms[i](z)

            # Acitvation function
            a = F.relu(z)

            a = dropout(a) if self.training else a # Apply dropout only during training
            current_input = a

        # Output layer (no activation for regression)
        output = self.layers[-1](current_input)

        return output


# City Neural Network
class CityNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = [256,128], use_batch_norm=True, dropout_rate = [0.2,0.7], random_state=42):
        super(CityNeuralNetwork,self).__init__()

        """
        Initialize Continent architecture using Pytorch modules.

        Parameters:
        - input_size: Number of input features. In this case it is the GITs and the predicted probabilities of the continent layer prediciton. # 207
        - hidden_layers: List of hidden layers # 256, 128 are the default
        - output_size: Number of cities
        - dropout_rate: [0.2, 0.7]
        - random_state: Random state for reporducibility
        
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.use_batch_norm = use_batch_norm

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Build the neural network
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Create the layer architecture
        layer_sizes = [input_size] + hidden_size + [output_size]

        for i in range(len(layer_sizes)-1):
            # Add the linear layers first
            self.layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))

            # Add batch normalization for hidden layers only and not for the output layers
            if i < len(layer_sizes) - 2 and self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i+1]))

            # Add dropout for hidden layers onyl and not for the output layers
            if i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rate))


# Cordinate neural network
class CordinateNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = [512,256,128], use_batch_norm=True, dropout_rate = [0.2,0.7], random_state=42):
        super(CordinateNeuralNetwork,self).__init__()

        """
        Initialize Continent architecture using Pytorch modules.

        Parameters:
        - input_size: Number of input features. In this case it is the GITs, predicted continent and predicted cities. # 200 + 7 + 42
        - hidden_layers: List of hidden layers # 512, 256, 128 are the default
        - output_size: Cordinates
        - dropout_rate: [0.2, 0.7] # Dynamic dropout rates
        - random_state: Random state for reporducibility
        
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.use_batch_norm = use_batch_norm

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Build the neural network
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Create the layer architecture
        layer_sizes = [input_size] + hidden_size + [output_size]

        for i in range(len(layer_sizes)-1):
            # Add the linear layers first
            self.layers.append(nn.Linear(layer_sizes[i],layer_sizes[i+1]))

            # Add batch normalization for hidden layers only and not for the output layers
            if i < len(layer_sizes) - 2 and self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i+1]))

            # Add dropout for hidden layers onyl and not for the output layers
            if i < len(layer_sizes) - 2:
                self.dropouts.append(nn.Dropout(dropout_rate))