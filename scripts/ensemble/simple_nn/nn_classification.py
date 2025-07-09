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


# Create a dynamic architecutre in pytorch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset class for continent classification
class TrainDataset:
    def __init__(self, features, n_targets):
        self.features = features
        self.n_targets = n_targets

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.features[idx], dtype=torch.float),
            'n_classes': torch.tensor(self.n_targets[idx], dtype=torch.float)
        }


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim:int, latent_dim:64):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Linear(128,latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,128),
            nn.ReLU(),
            nn.Linear(128,input_dim)
        )
    def forward(self,x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon
    
# Feature Attention
class FeatureAttention(nn.Module):
    def __init__(self, intput_dim):
        super(FeatureAttention,self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(intput_dim,128),
            nn.Tanh(),
            nn.Linear(128,intput_dim)
        )

    def forward(self,x):
        scores = self.attn(x)
        weights = F.softmax(scores,dim=1)
        return x * weights
    

# Continent Neural Network
class ClassificationNeuralNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, hidden_size = [128,64], use_batch_norm=True, dropout_rate = [0.2,0.7], random_state=42):
        super(ClassificationNeuralNetwork,self).__init__()

        """
        Initialize Continent architecture using Pytorch modules.

        Parameters:
        - input_size: Number of input features. In this case it is the GITs. # 200
        - hidden_layers: List of hidden layers # 256, 128 are the default
        - output_size: Number of continents
        - dropout_rate: [0.2, 0.7]
        - random_state: Random state for reporducibility
        
        """
        self.input_size = input_dim
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.output_size = output_dim
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.use_batch_norm = use_batch_norm

        self.autoencoder = Autoencoder(input_dim,latent_dim)
        self.attention = FeatureAttention(input_dim)
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Build the neural network
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Create the layer architecture
        layer_sizes = [input_dim] + hidden_size + [output_dim]

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

        latent, recon = self.autoencoder(x)
        atten_latent = self.attention(latent)

        current_input = atten_latent

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

        return output, recon