import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
import time
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# --- Data processing (reuse from nn_model_revised.py) ---
def process_data_hierarchical(df):
    cont_cols = [col for col in df.columns if col not in [
        'latitude', 'longitude',
        'latitude_rad', 'longitude_rad', 'x', 'y', 'z',
        'scaled_x', 'scaled_y', 'scaled_z', 'continent', 'city'
    ]]
    x_cont = df[cont_cols].values
    continent_encoder = LabelEncoder()
    y_continent = continent_encoder.fit_transform(df['continent'].values)
    city_encoder = LabelEncoder()
    y_city = city_encoder.fit_transform(df['city'].values)
    if not all(col in df.columns for col in ['x', 'y', 'z']):
        df['latitude_rad'] = np.deg2rad(df['latitude'])
        df['longitude_rad'] = np.deg2rad(df['longitude'])
        df['x'] = np.cos(df['latitude_rad']) * np.cos(df['longitude_rad'])
        df['y'] = np.cos(df['latitude_rad']) * np.sin(df['longitude_rad'])
        df['z'] = np.sin(df['latitude_rad'])
    coord_scaler = StandardScaler()
    y_coords = coord_scaler.fit_transform(df[['x', 'y', 'z']].values)
    continents = continent_encoder.classes_
    cities = city_encoder.classes_
    return {
        'x_cont': x_cont,
        'y_continent': y_continent,
        'y_city': y_city,
        'y_coords': y_coords,
        'y_latitude': df['latitude'].values,
        'y_longitude': df['longitude'].values,
        'encoders': {
            'continent': continent_encoder,
            'city': city_encoder,
            'coord': coord_scaler
        },
        'continents': continents,
        'cities': cities
    }

def hierarchical_split(X_cont, y_continent, y_city, y_coords, y_lat, y_lon, test_size=0.2, random_state=42):
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X_cont, y_continent))
    return {
        'X_train': X_cont[train_idx],
        'X_test': X_cont[test_idx],
        'y_cont_train': y_continent[train_idx],
        'y_cont_test': y_continent[test_idx],
        'y_city_train': y_city[train_idx],
        'y_city_test': y_city[test_idx],
        'y_coords_train': y_coords[train_idx],
        'y_coords_test': y_coords[test_idx],
        'y_lat_train': y_lat[train_idx],
        'y_lat_test': y_lat[test_idx],
        'y_lon_train': y_lon[train_idx],
        'y_lon_test': y_lon[test_idx],
        'train_idx': train_idx,
        'test_idx': test_idx
    }

# --- Dataset ---
class HierarchicalDataset(Dataset):
    def __init__(self, X, y_cont, y_city, y_coords):
        self.X = X
        self.y_cont = y_cont
        self.y_city = y_city
        self.y_coords = y_coords

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y_cont[idx], dtype=torch.long),
            torch.tensor(self.y_city[idx], dtype=torch.long),
            torch.tensor(self.y_coords[idx], dtype=torch.float32)
        )

# --- Modular Combined Model ---
class CombinedHierarchicalNet(nn.Module):
    def __init__(
        self,
        input_dim,
        num_continents,
        num_cities,
        coord_dim=3,
        hidden_dims_cont=[128, 64],
        hidden_dims_city=[256, 128, 64],
        hidden_dims_coord=[256, 128, 64],
        dropout_cont=(0.3, 0.7),
        dropout_city=(0.3, 0.7),
        dropout_coord=(0.2, 0.5),
        use_batch_norm=True
    ):
        super().__init__()
        # Continent branch
        self.continent_branch = self._make_branch(
            input_dim, hidden_dims_cont, num_continents, dropout_cont, use_batch_norm
        )
        # City branch (input: features + continent probs)
        self.city_branch = self._make_branch(
            input_dim + num_continents, hidden_dims_city, num_cities, dropout_city, use_batch_norm
        )
        # Coordinate branch (input: features + continent probs + city probs)
        self.coord_branch = self._make_branch(
            input_dim + num_continents + num_cities, hidden_dims_coord, coord_dim, dropout_coord, use_batch_norm, output_activation=None
        )

    def _make_branch(self, in_dim, hidden_dims, out_dim, dropout_range, use_bn, output_activation=None):
        layers = []
        dims = [in_dim] + hidden_dims
        dropouts = np.linspace(dropout_range[0], dropout_range[1], len(hidden_dims))
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropouts[i]))
        layers.append(nn.Linear(dims[-1], out_dim))
        if output_activation is not None:
            layers.append(output_activation)
        return nn.Sequential(*layers)

    def forward(self, x):
        # Continent
        cont_logits = self.continent_branch(x)
        cont_probs = F.softmax(cont_logits, dim=1)
        # City
        city_input = torch.cat([x, cont_probs], dim=1)
        city_logits = self.city_branch(city_input)
        city_probs = F.softmax(city_logits, dim=1)
        # Coordinates
        coord_input = torch.cat([x, cont_probs, city_probs], dim=1)
        coord_pred = self.coord_branch(coord_input)
        return cont_logits, city_logits, coord_pred

# --- Training Loop ---
def train_combined_hierarchical(
    model, train_loader, val_loader, device,
    continent_weight=None, city_weight=None, coord_weight=None,
    lr=1e-3, weight_decay=1e-5, epochs=600, early_stopping_steps=50,
    print_every=10,
    continent_class_weights=None,
    city_class_weights=None
):
    model = model.to(device)
    # Add optimizer definition here
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # FIX: Only pass tensor weights to CrossEntropyLoss, not scalar weights
    criterion_cont = nn.CrossEntropyLoss(weight=continent_class_weights)
    criterion_city = nn.CrossEntropyLoss(weight=city_class_weights)
    criterion_coord = nn.MSELoss()
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, total_cont, total_city, total_coord = 0, 0, 0, 0
        for X, y_cont, y_city, y_coord in train_loader:
            X = X.to(device)
            y_cont = y_cont.to(device)
            y_city = y_city.to(device)
            y_coord = y_coord.to(device)
            cont_logits, city_logits, coord_pred = model(X)
            loss_cont = criterion_cont(cont_logits, y_cont)
            loss_city = criterion_city(city_logits, y_city)
            loss_coord = criterion_coord(coord_pred, y_coord)
            # Ensure losses are scalars
            if loss_cont.dim() > 0:
                loss_cont = loss_cont.mean()
            if loss_city.dim() > 0:
                loss_city = loss_city.mean()
            if loss_coord.dim() > 0:
                loss_coord = loss_coord.mean()
            # Only static weighting (scalars)
            w1 = 1.0 if continent_weight is None else continent_weight
            w2 = 0.5 if city_weight is None else city_weight
            w3 = 0.2 if coord_weight is None else coord_weight
            loss = w1*loss_cont + w2*loss_city + w3*loss_coord
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_cont += loss_cont.item()
            total_city += loss_city.item()
            total_coord += loss_coord.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, val_cont, val_city, val_coord = 0, 0, 0, 0
            for X, y_cont, y_city, y_coord in val_loader:
                X = X.to(device)
                y_cont = y_cont.to(device)
                y_city = y_city.to(device)
                y_coord = y_coord.to(device)
                cont_logits, city_logits, coord_pred = model(X)
                loss_cont = criterion_cont(cont_logits, y_cont)
                loss_city = criterion_city(city_logits, y_city)
                loss_coord = criterion_coord(coord_pred, y_coord)
                # Ensure losses are scalars
                if loss_cont.dim() > 0:
                    loss_cont = loss_cont.mean()
                if loss_city.dim() > 0:
                    loss_city = loss_city.mean()
                if loss_coord.dim() > 0:
                    loss_coord = loss_coord.mean()
                # Only static weighting (scalars)
                w1 = 1.0 if continent_weight is None else continent_weight
                w2 = 0.5 if city_weight is None else city_weight
                w3 = 0.2 if coord_weight is None else coord_weight
                loss = w1*loss_cont + w2*loss_city + w3*loss_coord
                val_loss += loss.item()
                val_cont += loss_cont.item()
                val_city += loss_city.item()
                val_coord += loss_coord.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
        if epoch % print_every == 0:
            print(f"Epoch {epoch}: Train Loss {avg_loss:.4f} | Val Loss {avg_val_loss:.4f}")
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stopping_steps:
                print(f"Early stopping at epoch {epoch}")
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

# --- Evaluation ---
def evaluate_combined(model, loader, device, coord_scaler=None):
    model.eval()
    all_cont_preds, all_city_preds, all_coord_preds = [], [], []
    all_cont_true, all_city_true, all_coord_true = [], [], []
    with torch.no_grad():
        for X, y_cont, y_city, y_coord in loader:
            X = X.to(device)
            cont_logits, city_logits, coord_pred = model(X)
            cont_pred = torch.argmax(cont_logits, dim=1).cpu().numpy()
            city_pred = torch.argmax(city_logits, dim=1).cpu().numpy()
            all_cont_preds.append(cont_pred)
            all_city_preds.append(city_pred)
            all_coord_preds.append(coord_pred.cpu().numpy())
            all_cont_true.append(y_cont.numpy())
            all_city_true.append(y_city.numpy())
            all_coord_true.append(y_coord.numpy())
    all_cont_preds = np.concatenate(all_cont_preds)
    all_city_preds = np.concatenate(all_city_preds)
    all_coord_preds = np.concatenate(all_coord_preds)
    all_cont_true = np.concatenate(all_cont_true)
    all_city_true = np.concatenate(all_city_true)
    all_coord_true = np.concatenate(all_coord_true)
    # Inverse transform coordinates if scaler is given
    if coord_scaler is not None:
        all_coord_preds = coord_scaler.inverse_transform(all_coord_preds)
        all_coord_true = coord_scaler.inverse_transform(all_coord_true)
    acc_cont = accuracy_score(all_cont_true, all_cont_preds)
    acc_city = accuracy_score(all_city_true, all_city_preds)
    mse_coord = mean_squared_error(all_coord_true, all_coord_preds)
    mae_coord = mean_absolute_error(all_coord_true, all_coord_preds)
    r2_coord = r2_score(all_coord_true, all_coord_preds)
    print("Continent accuracy:", acc_cont)
    print("City accuracy:", acc_city)
    print("Coordinate RMSE:", np.sqrt(mse_coord))
    print("Coordinate MAE:", mae_coord)
    print("Coordinate R2:", r2_coord)
    return {
        "continent_accuracy": acc_cont,
        "city_accuracy": acc_city,
        "coordinate_rmse": np.sqrt(mse_coord),
        "coordinate_mae": mae_coord,
        "coordinate_r2": r2_coord,
        "continent_preds": all_cont_preds,
        "city_preds": all_city_preds,
        "coord_preds": all_coord_preds,
        "continent_true": all_cont_true,
        "city_true": all_city_true,
        "coord_true": all_coord_true
    }

# --- Additional Functions ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (in kilometers)
    """
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2) **2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def xyz_to_latlon(xyz_coords):
    """
    Convert the XYZ coordinates to latitude and longitude
    """
    x, y, z = xyz_coords[:, 0], xyz_coords[:, 1], xyz_coords[:, 2]
    lat_rad = np.arcsin(np.clip(z, -1, 1))
    lon_rad = np.arctan2(y, x)
    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    return np.stack([lat_deg, lon_deg], axis=1)

def compute_in_radius_metrics(y_true, y_pred, thresholds=None):
    """
    Compute % of predictions within given distance thresholds
    y_true, y_pred: numpy arrays of shape (N, 2) for [lat, lon]
    thresholds: List of distance thresholds in kilometers (default: [1, 5, 50, 100, 250, 500, 1000, 5000])
    """
    if thresholds is None:
        thresholds = [1, 5, 50, 100, 250, 500, 1000, 5000]
    distances = haversine_distance(
        y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1]
    )
    results = {}
    for r in thresholds:
        percent = np.mean(distances <= r) * 100
        results[f"<{r} km"] = percent
    return results

class CombinedModelTuner:
    def __init__(self, train_loader, val_loader, device, input_dim, num_continents, num_cities, coord_dim=3, n_trials=20, timeout=1200):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.input_dim = input_dim
        self.num_continents = num_continents
        self.num_cities = num_cities
        self.coord_dim = coord_dim
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.best_score = None

    def objective(self, trial):
        hidden_dims_cont = trial.suggest_categorical("hidden_dims_cont", [[128, 64], [256, 128, 64]])
        hidden_dims_city = trial.suggest_categorical("hidden_dims_city", [[256, 128, 64], [128, 64]])
        hidden_dims_coord = trial.suggest_categorical("hidden_dims_coord", [[256, 128, 64], [128, 64]])
        dropout_cont = (trial.suggest_float("dropout_cont_start", 0.2, 0.5), trial.suggest_float("dropout_cont_end", 0.6, 0.8))
        dropout_city = (trial.suggest_float("dropout_city_start", 0.2, 0.5), trial.suggest_float("dropout_city_end", 0.6, 0.8))
        dropout_coord = (trial.suggest_float("dropout_coord_start", 0.1, 0.3), trial.suggest_float("dropout_coord_end", 0.4, 0.6))
        lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        # Enforce continent_weight > city_weight > coord_weight
        continent_weight = trial.suggest_float("continent_weight", 1.0, 2.0)
        city_weight = trial.suggest_float("city_weight", 0.5, continent_weight - 1e-4)
        coord_weight = trial.suggest_float("coord_weight", 0.05, city_weight - 1e-4)

        model = CombinedHierarchicalNet(
            input_dim=self.input_dim,
            num_continents=self.num_continents,
            num_cities=self.num_cities,
            coord_dim=self.coord_dim,
            hidden_dims_cont=hidden_dims_cont,
            hidden_dims_city=hidden_dims_city,
            hidden_dims_coord=hidden_dims_coord,
            dropout_cont=dropout_cont,
            dropout_city=dropout_city,
            dropout_coord=dropout_coord,
            use_batch_norm=use_batch_norm
        )
        # Use a subset of data for speed
        train_loader = self.train_loader
        val_loader = self.val_loader

        # Do not pass scalar weights as class weights
        model, train_losses, val_losses = train_combined_hierarchical(
            model, train_loader, val_loader, self.device,
            continent_weight=continent_weight,
            city_weight=city_weight,
            coord_weight=coord_weight,
            lr=lr, weight_decay=weight_decay, epochs=600, early_stopping_steps=50, print_every=20,
            continent_class_weights=None,  # <-- always None for tuning
            city_class_weights=None
        )
        results = evaluate_combined(model, val_loader, self.device)
        # Use coordinate RMSE as the main metric (minimize)
        return -results['coordinate_rmse']

    def tune(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = study.best_params
        self.best_score = study.best_value
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        return self.best_params, self.best_score

# --- Example usage ---
if __name__ == "__main__":
    # Load and process data
    df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
    processed = process_data_hierarchical(df)
    X = processed['x_cont']
    y_cont = processed['y_continent']
    y_city = processed['y_city']
    y_coords = processed['y_coords']
    coord_scaler = processed['encoders']['coord']
    split = hierarchical_split(X, y_cont, y_city, y_coords, processed['y_latitude'], processed['y_longitude'])
    X_train, X_test = split['X_train'], split['X_test']
    y_train_cont, y_test_cont = split['y_cont_train'], split['y_cont_test']
    y_train_city, y_test_city = split['y_city_train'], split['y_city_test']
    y_train_coords, y_test_coords = split['y_coords_train'], split['y_coords_test']
    # DataLoaders
    batch_size = 128
    train_ds = HierarchicalDataset(X_train, y_train_cont, y_train_city, y_train_coords)
    test_ds = HierarchicalDataset(X_test, y_test_cont, y_test_city, y_test_coords)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Tuning logic ---
    tune_hyperparams = False  # Set to False to skip tuning
    n_trials = 20
    timeout = 600

    if tune_hyperparams:
        tuner = CombinedModelTuner(
            train_loader, test_loader, device,
            input_dim=X.shape[1],
            num_continents=len(processed['continents']),
            num_cities=len(processed['cities']),
            coord_dim=3,
            n_trials=n_trials,
            timeout=timeout
        )
        best_params, best_score = tuner.tune()
        # Unpack best params for model construction
        model = CombinedHierarchicalNet(
            input_dim=X.shape[1],
            num_continents=len(processed['continents']),
            num_cities=len(processed['cities']),
            coord_dim=3,
            hidden_dims_cont=best_params["hidden_dims_cont"],
            hidden_dims_city=best_params["hidden_dims_city"],
            hidden_dims_coord=best_params["hidden_dims_coord"],
            dropout_cont=(best_params["dropout_cont_start"], best_params["dropout_cont_end"]),
            dropout_city=(best_params["dropout_city_start"], best_params["dropout_city_end"]),
            dropout_coord=(best_params["dropout_coord_start"], best_params["dropout_coord_end"]),
            use_batch_norm=best_params["use_batch_norm"]
        )
        lr = best_params["lr"]
        weight_decay = best_params["weight_decay"]
        batch_size = best_params["batch_size"]
        continent_weight = best_params["continent_weight"]
        city_weight = best_params["city_weight"]
        coord_weight = best_params["coord_weight"]
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    else:
        model = CombinedHierarchicalNet(
            input_dim=X.shape[1],
            num_continents=len(processed['continents']),
            num_cities=len(processed['cities']),
            coord_dim=3,
            hidden_dims_cont=[128, 64],
            hidden_dims_city=[256, 128, 64],
            hidden_dims_coord=[256, 128, 64],
            dropout_cont=(0.3, 0.7),
            dropout_city=(0.3, 0.7),
            dropout_coord=(0.2, 0.5),
            use_batch_norm=True
        )
        lr = 1e-3
        weight_decay = 1e-5
        continent_weight = 1.0
        city_weight = 0.5
        coord_weight = 0.2

    # Class weights for continent (inverse frequency)
    class_counts = np.bincount(y_train_cont)
    continent_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32)
    continent_weights = continent_weights / continent_weights.sum()

    # Train
    model, train_losses, val_losses = train_combined_hierarchical(
        model, train_loader, test_loader, device,
        continent_weight=continent_weight,
        city_weight=city_weight,
        coord_weight=coord_weight,
        lr=lr, weight_decay=weight_decay, epochs=600, early_stopping_steps=50, print_every=10,
        continent_class_weights=continent_weights,  # <-- pass tensor hereopyLoss
        city_class_weights=None
    )
    # Evaluate
    results = evaluate_combined(model, test_loader, device, coord_scaler=coord_scaler)

    # --- Classification Reports ---
    print("\n===== Classification Report: Continent =====")
    print(classification_report(
        results['continent_true'],
        results['continent_preds'],
        target_names=processed['continents']
    ))
    print("\n===== Classification Report: City =====")
    print(classification_report(
        results['city_true'],
        results['city_preds'],
        target_names=processed['cities']
    ))

    # --- Coordinate Error Calculations ---
    # Convert predicted and true coordinates from cartesian to lat/lon
    pred_latlon = xyz_to_latlon(results['coord_preds'])
    true_latlon = xyz_to_latlon(results['coord_true'])

    # Haversine distance error
    coord_errors = haversine_distance(
        true_latlon[:, 0], true_latlon[:, 1],
        pred_latlon[:, 0], pred_latlon[:, 1]
    )
    print("\n===== Coordinate Error Statistics =====")
    print(f"Median distance error: {np.median(coord_errors):.2f} km")
    print(f"Mean distance error: {np.mean(coord_errors):.2f} km")
    print(f"Max distance error: {np.max(coord_errors):.2f} km")

    # --- Expected Error Calculation ---
    # Group by correctness of continent/city predictions
    continent_correct = results['continent_true'] == results['continent_preds']
    city_correct = results['city_true'] == results['city_preds']
    error_group = np.array([
        'C_correct Z_correct' if c and z else
        'C_correct Z_wrong' if c and not z else
        'C_wrong Z_correct' if not c and z else
        'C_wrong Z_wrong'
        for c, z in zip(continent_correct, city_correct)
    ])
    # Build DataFrame for error analysis
    error_df = pd.DataFrame({
        'coord_error': coord_errors,
        'error_group': error_group
    })
    group_stats = error_df.groupby('error_group')['coord_error'].agg([
        ('count', 'count'),
        ('mean_error_km', 'mean'),
        ('median_error_km', 'median')
    ])
    total = len(error_df)
    group_stats['proportion'] = group_stats['count'] / total
    group_stats['weighted_error'] = group_stats['mean_error_km'] * group_stats['proportion']
    expected_total_error = group_stats['weighted_error'].sum()
    print("\n===== Expected Coordinate Error by Group =====")
    print(group_stats)
    print(f"Expected Coordinate Error E[D]: {expected_total_error:.2f} km")

    # In-radius metrics
    metrics = compute_in_radius_metrics(true_latlon, pred_latlon)
    print("\nIn-Radius Accuracy Metrics:")
    for k, v in metrics.items():
        print(f"{k:>8}: {v:.2f}%")
