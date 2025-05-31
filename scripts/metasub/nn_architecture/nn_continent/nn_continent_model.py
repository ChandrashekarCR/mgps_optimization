# This script is to get the continent prediction accuracy higher, because the mGPS algorithm correctly classifies 92% of samples to 
# their city of origin.

# Importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        return (continent_predictions,)
    
