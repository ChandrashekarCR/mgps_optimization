# Hexagon Prediction Model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder,StandardScaler
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
import h3
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



# Neural Network Architecture for Hexagon Prediction
class HexagonPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_hexagons, initial_dropout_rate, max_dropout_rate):
        super(HexagonPredictionModel, self).__init__()
        
        self.initial_dropout_rate = initial_dropout_rate
        self.max_dropout_rate = max_dropout_rate
        
        # Hexagon prediction architecture
        self.hexagon_layer_1 = nn.Linear(input_size, hidden_dim)
        self.hexagon_bn_1 = nn.BatchNorm1d(hidden_dim)
        self.hexagon_layer_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.hexagon_bn_2 = nn.BatchNorm1d(hidden_dim // 2)
        self.hexagon_layer_3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.hexagon_bn_3 = nn.BatchNorm1d(hidden_dim // 4)
        self.hexagon_prediction = nn.Linear(hidden_dim // 4, num_hexagons)

    def forward(self, x, current_dropout_rate):
        # Hexagon prediction pathway
        out_hexagon = F.relu(self.hexagon_bn_1(self.hexagon_layer_1(x)))
        out_hexagon = F.dropout(out_hexagon, p=current_dropout_rate, training=self.training)
        out_hexagon = F.relu(self.hexagon_bn_2(self.hexagon_layer_2(out_hexagon)))
        out_hexagon = F.dropout(out_hexagon, p=current_dropout_rate, training=self.training)
        out_hexagon = F.relu(self.hexagon_bn_3(self.hexagon_layer_3(out_hexagon)))
        hexagon_predictions = self.hexagon_prediction(out_hexagon)
        
        return hexagon_predictions

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import h3

def process_hexagon_data(data_path, h3_resolution=4, return_unique_hexes=False):
    try:
        in_data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None, None, None, None

    required_cols = ['latitude', 'longitude']
    if not all(col in in_data.columns for col in required_cols):
        print(f"Error: Required columns {required_cols} not found in the data.")
        return None, None, None, None

    # Generate H3 hex index for each row
    in_data['h3_index'] = in_data.apply(
        lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], h3_resolution),
        axis=1
    )

    # Label encode the hexagon indexes
    le_hex = LabelEncoder()
    in_data['hexagon_encoding'] = le_hex.fit_transform(in_data['h3_index'])

    # Non-feature columns to exclude
    non_feature_columns = [
        'city', 'continent', 'latitude', 'longitude',
        'city_encoding', 'continent_encoding',
        'lat_scaled', 'long_scaled',
        'latitude_rad', 'longitude_rad',
        'x', 'y', 'z', 'scaled_x', 'scaled_y', 'scaled_z',
        'h3_index', 'hexagon_encoding'
    ]

    # Drop non-feature columns, safely
    feature_columns = [col for col in in_data.columns if col not in non_feature_columns]
    if not feature_columns:
        print("Warning: No feature columns remaining after dropping non-feature columns.")
        return None, None, None, None

    X = in_data[feature_columns].values.astype(np.float32)
    y = in_data['hexagon_encoding'].values.astype(np.int64)

    if return_unique_hexes:
        unique_hexes = in_data[['h3_index', 'hexagon_encoding']].drop_duplicates()
        return in_data, X, y, le_hex, unique_hexes

    return in_data, X, y, le_hex

# Custom Dataset class for hexagon prediction
class HexagonDataset(Dataset):
    def __init__(self, features, hexagon_labels):
        self.features = features
        self.hexagon_labels = hexagon_labels

    def __len__(self):
        return len(self.hexagon_labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.hexagon_labels[idx], dtype=torch.long)
        )

def train_hexagon_model(train_dl, val_dl, model, optimizer, criterion, device, num_epochs, patience=10):
    train_losses = []
    val_losses = []
    early_stopper = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        current_dropout = model.initial_dropout_rate + \
                         (model.max_dropout_rate - model.initial_dropout_rate) * (epoch / num_epochs)
        
        model.train()
        train_loss = 0.0
        for features, labels in train_dl:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features, current_dropout)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_dl:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features, current_dropout)
                val_loss += criterion(outputs, labels).item()

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Dropout: {current_dropout:.2f}')
        
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(early_stopper.best_model_state)
            break

    return train_losses, val_losses

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('hexagon_model_losses.png')
    plt.show()

# After training, append predicted hex indexes for test set
def add_centroids_and_predictions(in_data, X_test, y_test, model, le_hex, device):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor, model.initial_dropout_rate)
        predicted_labels = torch.argmax(outputs, dim=1).cpu().numpy()

    # Decode to H3 indexes
    true_hexes = le_hex.inverse_transform(y_test)
    pred_hexes = le_hex.inverse_transform(predicted_labels)

    # Get lat/lon centroids
    true_coords = np.array([h3.cell_to_latlng(h) for h in true_hexes])
    pred_coords = np.array([h3.cell_to_latlng(h) for h in pred_hexes])

    result_df = pd.DataFrame({
        'true_hex': true_hexes,
        'pred_hex': pred_hexes,
        'true_lat': true_coords[:, 0],
        'true_lon': true_coords[:, 1],
        'pred_lat': pred_coords[:, 0],
        'pred_lon': pred_coords[:, 1],
        'correct': true_hexes == pred_hexes
    })

    return result_df


import folium
from folium import MacroElement
from jinja2 import Template

def plot_on_folium(result_df, sample_size=500):
    result_df = result_df.sample(min(sample_size, len(result_df)))
    m = folium.Map(location=[0, 0], zoom_start=2)

    for _, row in result_df.iterrows():
        true_boundary = h3.cell_to_boundary(row['true_hex'])
        pred_boundary = h3.cell_to_boundary(row['pred_hex'])

        folium.Polygon(locations=true_boundary, color='green', fill=True, fill_opacity=0.3).add_to(m)
        folium.Polygon(locations=pred_boundary, color='red', fill=True, fill_opacity=0.3).add_to(m)

    # Add a custom legend
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 150px; height: 90px; 
        background-color: white; 
        border:2px solid grey; z-index:9999;
        font-size:14px;
        padding: 10px;
        ">
        <b>Legend</b><br>
        <i style="background:green;opacity:0.6;width:10px;height:10px;display:inline-block;margin-right:5px;"></i>True Location<br>
        <i style="background:red;opacity:0.6;width:10px;height:10px;display:inline-block;margin-right:5px;"></i>Predicted Location
    </div>
    {% endmacro %}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    return m




# Evaluation function for hexagon prediction
def evaluate_hexagon_model(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            outputs = model(features, model.initial_dropout_rate)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f'Hexagon Accuracy: {accuracy*100:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    return accuracy, precision, recall, f1

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network for hexagon prediction.")
    parser.add_argument('-d', '--data_path', type=str, required=True, help="Path to input CSV")
    parser.add_argument('-t', '--test_size', type=float, default=0.2, help="Test set size")
    parser.add_argument('-r', '--resolution', type=int, default=4, help="H3 resolution (4-6 recommended)")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('-e', '--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--initial_dropout', type=float, default=0.2, help="Initial dropout rate")
    parser.add_argument('--max_dropout', type=float, default=0.5, help="Max dropout rate")
    parser.add_argument('--cuda', action='store_true', help="Use GPU if available")
    
    args = parser.parse_args()
    
    # Load and process data
    #in_data, X, y, le_hex = process_hexagon_data(args.data_path, args.resolution)
    in_data, X, y, le_hex, unique_hexes = process_hexagon_data(args.data_path, return_unique_hexes=True)
    print(unique_hexes)
    exit()
    if in_data is None:
        exit()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create dataloaders
    train_ds = HexagonDataset(X_train, y_train)
    val_ds = HexagonDataset(X_val, y_val)
    test_ds = HexagonDataset(X_test, y_test)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Initialize model
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    num_hexagons = len(le_hex.classes_)
    
    model = HexagonPredictionModel(
        input_size=X.shape[1],
        hidden_dim=256,
        num_hexagons=num_hexagons,
        initial_dropout_rate=args.initial_dropout,
        max_dropout_rate=args.max_dropout
    ).to(device)
    
    # Class weighting for imbalanced hexagons
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    print(f"Training hexagon model with {num_hexagons} unique hexagons...")
    train_losses, val_losses = train_hexagon_model(
        train_dl, val_dl, model, optimizer, criterion, device, args.epochs, patience=50
    )

    # Plot loss curves
    plot_losses(train_losses, val_losses)
    
    # Evaluate
    print("\nTraining set evaluation:")
    evaluate_hexagon_model(train_dl, model, device)

    result_df = add_centroids_and_predictions(in_data=in_data,X_test=X_test,y_test=y_test,model=model,le_hex=le_hex,device=device)

    # To display map
    m = plot_on_folium(result_df)
    m.save("hex_predictions_map.html")
    
    print("\nTest set evaluation:")
    evaluate_hexagon_model(test_dl, model, device)


# python nn_hexagon_model.py -d ../../../../results/metasub_training_testing_data.csv -t 0.2 -r 3 -b 32 -e 400 -lr 0.0001 --cuda
