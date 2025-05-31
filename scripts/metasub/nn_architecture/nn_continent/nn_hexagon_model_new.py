#!/usr/bin/env python3
"""
Global Hexagon Prediction System

This script creates a comprehensive hexagon grid covering the entire world
and trains a neural network to predict hexagon regions based on biological features.

Key Features:
- Global hexagon coverage using H3 library
- Balanced training across all hexagons (including empty ones)
- Proper handling of sparse data distribution
- Advanced model architecture for global prediction
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import h3
import matplotlib.pyplot as plt
import folium
from folium import MacroElement
from jinja2 import Template


class GlobalHexagonGenerator:
    """Generate and manage global hexagon grid."""
    
    def __init__(self, resolution: int = 4):
        """
        Initialize global hexagon generator.
        
        Args:
            resolution: H3 resolution level (0-15, where 4-6 are good for global coverage)
        """
        self.resolution = resolution
        self.global_hexagons = self._generate_global_hexagons()
        print(f"Generated {len(self.global_hexagons)} hexagons at resolution {resolution}")
    
    def _generate_global_hexagons(self) -> Set[str]:
        """Generate all hexagons covering the Earth's surface."""
        # Get all resolution 0 hexagons (base hexagons)
        base_hexagons = h3.get_res0_cells()
        
        # Get all hexagons at target resolution
        all_hexagons = set()
        for base_hex in base_hexagons:
            # Get all children hexagons at target resolution
            children = h3.cell_to_children(base_hex, self.resolution)
            all_hexagons.update(children)
        
        return all_hexagons
    
    def get_hexagon_info(self) -> pd.DataFrame:
        """Get information about all global hexagons."""
        hexagon_data = []
        
        for hex_id in self.global_hexagons:
            lat, lon = h3.cell_to_latlng(hex_id)
            boundary = h3.cell_to_boundary(hex_id)
            area = h3.cell_area(hex_id, unit='km^2')
            
            hexagon_data.append({
                'hex_id': hex_id,
                'center_lat': lat,
                'center_lon': lon,
                'area_km2': area,
                'boundary': boundary
            })
        
        return pd.DataFrame(hexagon_data)


class HexagonDataProcessor:
    """Process data for global hexagon prediction."""
    
    def __init__(self, resolution: int = 4):
        self.resolution = resolution
        self.global_hex_gen = GlobalHexagonGenerator(resolution)
        self.le_hex = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
    def process_data(self, data_path: str, min_samples_per_hex: int = 5) -> Tuple:
        """
        Process data for global hexagon prediction.
        
        Args:
            data_path: Path to input CSV
            min_samples_per_hex: Minimum samples required per hexagon to include in training
            
        Returns:
            Processed data tuple
        """
        # Load data
        try:
            in_data = pd.read_csv(data_path)
            print(f"Loaded {len(in_data)} samples from {data_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Validate required columns
        required_cols = ['latitude', 'longitude']
        if not all(col in in_data.columns for col in required_cols):
            raise ValueError(f"Required columns {required_cols} not found")
        
        # Generate H3 hexagon IDs for each sample
        in_data['hex_id'] = in_data.apply(
            lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], self.resolution),
            axis=1
        )
        
        # Get hexagon sample counts
        hex_counts = in_data['hex_id'].value_counts()
        print(f"Data spans {len(hex_counts)} unique hexagons")
        print(f"Hexagons with >= {min_samples_per_hex} samples: {sum(hex_counts >= min_samples_per_hex)}")
        
        # Filter hexagons with sufficient samples
        valid_hexagons = set(hex_counts[hex_counts >= min_samples_per_hex].index)
        filtered_data = in_data[in_data['hex_id'].isin(valid_hexagons)].copy()
        
        print(f"Using {len(filtered_data)} samples from {len(valid_hexagons)} hexagons")
        
        # Create comprehensive hexagon mapping
        all_global_hexagons = list(self.global_hex_gen.global_hexagons)
        
        # Prioritize hexagons with data, then add global coverage
        hexagons_with_data = list(valid_hexagons)
        remaining_hexagons = [h for h in all_global_hexagons if h not in valid_hexagons]
        
        # Create balanced hexagon set (limit for training efficiency)
        max_global_hexagons = min(10000, len(all_global_hexagons))  # Adjust based on memory
        
        if len(hexagons_with_data) < max_global_hexagons:
            # Add random selection of remaining hexagons
            n_additional = max_global_hexagons - len(hexagons_with_data)
            additional_hexagons = np.random.choice(
                remaining_hexagons, 
                size=min(n_additional, len(remaining_hexagons)), 
                replace=False
            )
            final_hexagon_set = hexagons_with_data + list(additional_hexagons)
        else:
            final_hexagon_set = hexagons_with_data
        
        print(f"Training on {len(final_hexagon_set)} total hexagons ({len(hexagons_with_data)} with data)")
        
        # Encode hexagons
        self.le_hex.fit(final_hexagon_set)
        filtered_data['hex_encoding'] = self.le_hex.transform(filtered_data['hex_id'])
        
        # Prepare features
        feature_columns = self._get_feature_columns(filtered_data)
        X = filtered_data[feature_columns].values.astype(np.float32)
        y = filtered_data['hex_encoding'].values.astype(np.int64)
        
        # Scale features
        X = self.feature_scaler.fit_transform(X)
        
        # Create hexagon metadata
        hex_metadata = self._create_hexagon_metadata(final_hexagon_set)
        
        return filtered_data, X, y, hex_metadata, feature_columns
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get feature columns excluding metadata."""
        exclude_columns = {
            'city', 'continent', 'latitude', 'longitude',
            'city_encoding', 'continent_encoding',
            'lat_scaled', 'long_scaled', 'latitude_rad', 'longitude_rad',
            'x', 'y', 'z', 'scaled_x', 'scaled_y', 'scaled_z',
            'hex_id', 'hex_encoding'
        }
        
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        if not feature_columns:
            raise ValueError("No feature columns found after filtering")
        
        print(f"Using {len(feature_columns)} feature columns")
        return feature_columns
    
    def _create_hexagon_metadata(self, hexagon_set: List[str]) -> pd.DataFrame:
        """Create metadata for hexagon set."""
        metadata = []
        
        for hex_id in hexagon_set:
            lat, lon = h3.cell_to_latlng(hex_id)
            area = h3.cell_area(hex_id, unit='km^2')
            encoding = self.le_hex.transform([hex_id])[0]
            
            metadata.append({
                'hex_id': hex_id,
                'hex_encoding': encoding,
                'center_lat': lat,
                'center_lon': lon,
                'area_km2': area
            })
        
        return pd.DataFrame(metadata)


class GlobalHexagonDataset(Dataset):
    """Dataset for global hexagon prediction with proper sampling."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 sample_weights: Optional[np.ndarray] = None):
        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


class AdvancedHexagonModel(nn.Module):
    """Advanced neural network for global hexagon prediction."""
    
    def __init__(self, input_size: int, num_hexagons: int, 
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        self.num_hexagons = num_hexagons
        self.dropout_rate = dropout_rate
        
        # Build dynamic architecture
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_hexagons))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GlobalHexagonTrainer:
    """Trainer for global hexagon prediction."""
    
    def __init__(self, model: nn.Module, device: str, 
                 learning_rate: float = 0.001, weight_decay: float = 1e-4):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              criterion: nn.Module, num_epochs: int = 100, patience: int = 20) -> Dict:
        """Train the model."""
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, criterion)
            val_loss = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                epoch_time = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                      f"LR: {lr:.2e} | Time: {epoch_time:.1f}s" +
                      (" *" if patience_counter == 0 else ""))
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored best model (val loss: {self.best_val_loss:.4f})")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }


class HexagonEvaluator:
    """Evaluate hexagon prediction performance."""
    
    def __init__(self, model: nn.Module, device: str, le_hex: LabelEncoder):
        self.model = model
        self.device = device
        self.le_hex = le_hex
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model performance."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Top-k accuracy
        all_probs = np.array(all_probs)
        top5_accuracy = self._calculate_topk_accuracy(all_probs, all_labels, k=5)
        top10_accuracy = self._calculate_topk_accuracy(all_probs, all_labels, k=10)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'top5_accuracy': top5_accuracy,
            'top10_accuracy': top10_accuracy,
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs
        }
        
        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy*100:.2f}%")
        print(f"Top-10 Accuracy: {top10_accuracy*100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return results
    
    def _calculate_topk_accuracy(self, probs: np.ndarray, labels: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy."""
        # Ensure labels is a numpy array
        labels = np.array(labels)
        
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = np.any(top_k_preds == labels.reshape(-1, 1), axis=1)
        return correct.mean()
    
    def create_prediction_dataframe(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Create dataframe with predictions and hexagon information."""
        # Get predictions
        test_dataset = GlobalHexagonDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        results = self.evaluate(test_loader)
        
        # Decode hexagon IDs
        true_hex_ids = self.le_hex.inverse_transform(results['true_labels'])
        pred_hex_ids = self.le_hex.inverse_transform(results['predictions'])
        
        # Get coordinates
        true_coords = np.array([h3.cell_to_latlng(h) for h in true_hex_ids])
        pred_coords = np.array([h3.cell_to_latlng(h) for h in pred_hex_ids])
        
        # Calculate distances
        distances = self._calculate_distances(true_coords, pred_coords)
        
        return pd.DataFrame({
            'true_hex_id': true_hex_ids,
            'pred_hex_id': pred_hex_ids,
            'true_lat': true_coords[:, 0],
            'true_lon': true_coords[:, 1],
            'pred_lat': pred_coords[:, 0],
            'pred_lon': pred_coords[:, 1],
            'distance_km': distances,
            'correct_prediction': true_hex_ids == pred_hex_ids,
            'confidence': np.max(results['probabilities'], axis=1)
        })
    
    def _calculate_distances(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Calculate haversine distances between coordinate pairs."""
        lat1, lon1 = np.radians(coords1[:, 0]), np.radians(coords1[:, 1])
        lat2, lon2 = np.radians(coords2[:, 0]), np.radians(coords2[:, 1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c  # Earth radius in km


def create_visualization_map(result_df: pd.DataFrame, sample_size: int = 1000) -> folium.Map:
    """Create Folium map visualization of predictions."""
    # Sample data for visualization
    sample_df = result_df.sample(min(sample_size, len(result_df)))
    
    # Calculate map center
    center_lat = sample_df[['true_lat', 'pred_lat']].mean().mean()
    center_lon = sample_df[['true_lon', 'pred_lon']].mean().mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
    
    # Add hexagon boundaries
    for _, row in sample_df.iterrows():
        # True hexagon (green)
        true_boundary = h3.cell_to_boundary(row['true_hex_id'])
        folium.Polygon(
            locations=true_boundary,
            color='green',
            weight=2,
            fill=True,
            fill_opacity=0.3,
            popup=f"True: {row['true_hex_id'][:8]}..."
        ).add_to(m)
        
        # Predicted hexagon (red/orange based on correctness)
        pred_boundary = h3.cell_to_boundary(row['pred_hex_id'])
        color = 'orange' if row['correct_prediction'] else 'red'
        folium.Polygon(
            locations=pred_boundary,
            color=color,
            weight=2,
            fill=True,
            fill_opacity=0.3,
            popup=f"Pred: {row['pred_hex_id'][:8]}...<br>Distance: {row['distance_km']:.1f}km"
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px;">
        <b>Hexagon Predictions</b><br>
        <i style="background:green;opacity:0.6;width:15px;height:15px;display:inline-block;margin-right:5px;"></i>True Location<br>
        <i style="background:orange;opacity:0.6;width:15px;height:15px;display:inline-block;margin-right:5px;"></i>Correct Prediction<br>
        <i style="background:red;opacity:0.6;width:15px;height:15px;display:inline-block;margin-right:5px;"></i>Incorrect Prediction
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def plot_training_curves(history: Dict) -> None:
    """Plot training and validation loss curves."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training History (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Global Hexagon Prediction System")
    
    # Data parameters
    parser.add_argument('-d', '--data_path', type=str, required=True,
                       help="Path to input CSV file")
    parser.add_argument('-r', '--resolution', type=int, default=4,
                       help="H3 hexagon resolution (3-6 recommended for global)")
    parser.add_argument('--min_samples_per_hex', type=int, default=5,
                       help="Minimum samples per hexagon to include in training")
    
    # Model parameters
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 256, 128],
                       help="Hidden layer dimensions")
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help="Dropout rate")
    
    # Training parameters
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                       help="Batch size")
    parser.add_argument('-e', '--epochs', type=int, default=200,
                       help="Maximum number of epochs")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help="Weight decay for regularization")
    parser.add_argument('--patience', type=int, default=50,
                       help="Early stopping patience")
    
    # System parameters
    parser.add_argument('--cuda', action='store_true',
                       help="Use CUDA if available")
    parser.add_argument('--num_workers', type=int, default=4,
                       help="Number of DataLoader workers")
    
    # Output parameters
    parser.add_argument('--save_model', type=str, default=None,
                       help="Path to save trained model")
    parser.add_argument('--visualize', action='store_true',
                       help="Create visualization map")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GLOBAL HEXAGON PREDICTION SYSTEM")
    print("=" * 80)
    
    # Initialize data processor
    processor = HexagonDataProcessor(args.resolution)
    
    # Process data
    print("Processing data...")
    in_data, X, y, hex_metadata, feature_columns = processor.process_data(
        args.data_path, args.min_samples_per_hex
    )
    
    print(f"Feature dimensions: {X.shape}")
    print(f"Number of hexagon classes: {len(processor.le_hex.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets with balanced sampling
    train_dataset = GlobalHexagonDataset(X_train, y_train)
    val_dataset = GlobalHexagonDataset(X_val, y_val)
    test_dataset = GlobalHexagonDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Setup device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = AdvancedHexagonModel(
        input_size=X.shape[1],
        num_hexagons=len(processor.le_hex.classes_),
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create weighted loss function for class imbalance
    # Fix: Create weights for ALL possible classes, not just those in training data
    num_classes = len(processor.le_hex.classes_)
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    
    # Calculate weights only for classes present in training data
    unique_classes, counts = np.unique(y_train, return_counts=True)
    for class_idx, count in zip(unique_classes, counts):
        # Use inverse frequency weighting
        class_weights[class_idx] = 1.0 / count
    
    # Normalize weights to have mean of 1.0
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Train model
    trainer = GlobalHexagonTrainer(
        model, device, args.learning_rate, args.weight_decay
    )
    
    history = trainer.train(
        train_loader, val_loader, criterion, args.epochs, args.patience
    )
    
    # Plot training curves
    plot_training_curves(history)
    
    # Evaluate model
    evaluator = HexagonEvaluator(model, device, processor.le_hex)
    
    print("\nTest Set Evaluation:")
    test_results = evaluator.evaluate(test_loader)
    
    # Create prediction analysis
    if args.visualize:
        print("\nCreating prediction analysis...")
        result_df = evaluator.create_prediction_dataframe(X_test, y_test)
        
        # Print distance statistics
        print(f"\nPrediction Distance Statistics:")
        print(f"Mean distance: {result_df['distance_km'].mean():.1f} km")
        print(f"Median distance: {result_df['distance_km'].median():.1f} km")
        print(f"90th percentile distance: {result_df['distance_km'].quantile(0.9):.1f} km")
        print(f"Max distance: {result_df['distance_km'].max():.1f} km")
        print(f"Accuracy: {result_df['correct_prediction'].mean()*100:.2f}%")
        
        # Create visualization map
        print("Creating visualization map...")
        map_viz = create_visualization_map(result_df, sample_size=1000)
        map_viz.save('hexagon_predictions_map.html')
        print("Map saved as 'hexagon_predictions_map.html'")
        
        # Save detailed results
        result_df.to_csv('prediction_results.csv', index=False)
        print("Detailed results saved as 'prediction_results.csv'")
    
    # Save model if requested
    if args.save_model:
        model_data = {
            'model_state_dict': model.state_dict(),
            'feature_scaler': processor.feature_scaler,
            'label_encoder': processor.le_hex,
            'feature_columns': feature_columns,
            'resolution': args.resolution,
            'model_config': {
                'input_size': X.shape[1],
                'num_hexagons': len(processor.le_hex.classes_),
                'hidden_dims': args.hidden_dims,
                'dropout_rate': args.dropout_rate
            }
        }
        torch.save(model_data, args.save_model)
        print(f"Model saved to {args.save_model}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Dataset: {args.data_path}")
    print(f"H3 Resolution: {args.resolution}")
    print(f"Total samples: {len(in_data):,}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Number of hexagon classes: {len(processor.le_hex.classes_):,}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training epochs completed: {len(history['train_losses'])}")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Test accuracy: {test_results['accuracy']*100:.2f}%")
    print(f"Test top-5 accuracy: {test_results['top5_accuracy']*100:.2f}%")
    print(f"Test F1-score: {test_results['f1_score']:.4f}")
    
    if args.visualize:
        print(f"Mean prediction distance: {result_df['distance_km'].mean():.1f} km")
    
    print("\nFiles created:")
    print("- training_curves.png")
    if args.visualize:
        print("- hexagon_predictions_map.html")
        print("- prediction_results.csv")
    if args.save_model:
        print(f"- {args.save_model}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()