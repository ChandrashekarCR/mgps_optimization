#!/usr/bin/env python3
"""
Global Hexagon Prediction System

This system:
1. Covers the entire world with hexagonal sections using H3
2. Associates each lat/long coordinate to a hexagon section
3. Predicts hexagon sections based on input biological/environmental features
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import h3
import matplotlib.pyplot as plt
import folium
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')


class GlobalHexagonGrid:
    """Create and manage global hexagon grid covering the entire world."""
    
    def __init__(self, resolution: int = 4):
        """
        Initialize global hexagon grid.
        
        Args:
            resolution: H3 hexagon resolution (3-6 recommended for global coverage)
                       - Resolution 3: ~1,000 km² per hexagon (~500,000 hexagons globally)
                       - Resolution 4: ~100 km² per hexagon (~3M hexagons globally)
                       - Resolution 5: ~10 km² per hexagon (~20M hexagons globally)
        """
        self.resolution = resolution
        print(f"Creating global hexagon grid at resolution {resolution}...")
        
        # Generate all hexagons covering Earth's surface
        self.global_hexagons = self._generate_global_hexagons()
        print(f"Generated {len(self.global_hexagons):,} hexagons covering the entire world")
        
        # Create hexagon metadata
        self.hexagon_info = self._create_hexagon_info()
        print("Global hexagon grid created successfully!")
    
    def _generate_global_hexagons(self) -> Set[str]:
        """Generate all hexagons covering Earth's surface."""
        # Get all base resolution hexagons (resolution 0)
        base_hexagons = h3.get_res0_cells()
        
        # Generate all hexagons at target resolution
        all_hexagons = set()
        for base_hex in base_hexagons:
            # Get all child hexagons at target resolution
            children = h3.cell_to_children(base_hex, self.resolution)
            all_hexagons.update(children)
        
        return all_hexagons
    
    def _create_hexagon_info(self) -> pd.DataFrame:
        """Create comprehensive information about each hexagon."""
        hexagon_data = []
        
        print("Creating hexagon metadata...")
        for i, hex_id in enumerate(self.global_hexagons):
            if i % 10000 == 0:
                print(f"Processing hexagon {i:,}/{len(self.global_hexagons):,}")
            
            # Get hexagon center coordinates
            lat, lon = h3.cell_to_latlng(hex_id)
            
            # Get hexagon area
            area_km2 = h3.cell_area(hex_id, unit='km^2')
            
            # Get hexagon boundary for visualization
            boundary = h3.cell_to_boundary(hex_id)
            
            hexagon_data.append({
                'hex_id': hex_id,
                'center_lat': lat,
                'center_lon': lon,
                'area_km2': area_km2,
                'boundary_coords': boundary
            })
        
        return pd.DataFrame(hexagon_data)
    
    def coordinate_to_hexagon(self, lat: float, lon: float) -> str:
        """Convert latitude/longitude to hexagon ID."""
        return h3.latlng_to_cell(lat, lon, self.resolution)
    
    def hexagon_to_coordinate(self, hex_id: str) -> Tuple[float, float]:
        """Convert hexagon ID to center latitude/longitude."""
        return h3.cell_to_latlng(hex_id)
    
    def get_hexagon_neighbors(self, hex_id: str) -> List[str]:
        """Get neighboring hexagons."""
        return list(h3.grid_ring(hex_id, 1))
    
    def visualize_hexagons(self, hexagon_ids: List[str], center_lat: float = 0, center_lon: float = 0) -> folium.Map:
        """Create a map visualization of specific hexagons."""
        m = folium.Map(location=[center_lat, center_lon], zoom_start=3)
        
        for hex_id in hexagon_ids[:100]:  # Limit for performance
            boundary = h3.cell_to_boundary(hex_id)
            folium.Polygon(
                locations=boundary,
                color='blue',
                weight=2,
                fill=True,
                fill_opacity=0.3,
                popup=f"Hexagon: {hex_id}"
            ).add_to(m)
        
        return m


class HexagonDataProcessor:
    """Process data for hexagon prediction."""
    
    def __init__(self, hexagon_grid: GlobalHexagonGrid):
        self.hexagon_grid = hexagon_grid
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def process_dataset(self, data_path: str, min_samples_per_hexagon: int = 5) -> Tuple:
        """
        Process input dataset for hexagon prediction.
        
        Args:
            data_path: Path to CSV file with latitude, longitude, and feature columns
            min_samples_per_hexagon: Minimum samples required per hexagon
            
        Returns:
            Tuple of processed data (features, labels, metadata)
        """
        print(f"Loading dataset from {data_path}...")
        
        # Load data
        try:
            data = pd.read_csv(data_path)
            print(f"Loaded {len(data):,} samples")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
        
        # Validate required columns
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            raise ValueError("Dataset must contain 'latitude' and 'longitude' columns")
        
        print("Associating coordinates with hexagons...")
        
        # Convert coordinates to hexagon IDs
        data['hex_id'] = data.apply(
            lambda row: self.hexagon_grid.coordinate_to_hexagon(row['latitude'], row['longitude']), 
            axis=1
        )
        
        # Count samples per hexagon
        hexagon_counts = data['hex_id'].value_counts()
        print(f"Data spans {len(hexagon_counts):,} unique hexagons")
        
        # Filter hexagons with sufficient samples
        valid_hexagons = hexagon_counts[hexagon_counts >= min_samples_per_hexagon].index
        filtered_data = data[data['hex_id'].isin(valid_hexagons)].copy()
        
        print(f"Using {len(filtered_data):,} samples from {len(valid_hexagons):,} hexagons")
        print(f"(Hexagons with >= {min_samples_per_hexagon} samples)")
        
        # Encode hexagon labels
        self.label_encoder.fit(valid_hexagons)
        filtered_data['hex_label'] = self.label_encoder.transform(filtered_data['hex_id'])
        
        # Extract features (exclude coordinate and ID columns)
        feature_columns = [col for col in filtered_data.columns 
                          if col not in ['latitude', 'longitude', 'hex_id', 'hex_label', 'continent','city']]
        
        if not feature_columns:
            raise ValueError("No feature columns found in dataset")
        
        print(f"Using {len(feature_columns)} feature columns: {feature_columns}")
        
        # Prepare features and labels
        X = filtered_data[feature_columns].values.astype(np.float32)
        y = filtered_data['hex_label'].values.astype(np.int64)
        
        # Scale features
        X = self.feature_scaler.fit_transform(X)
        
        # Create metadata
        metadata = {
            'feature_columns': feature_columns,
            'num_hexagons': len(valid_hexagons),
            'hexagon_ids': list(valid_hexagons),
            'sample_distribution': hexagon_counts[valid_hexagons].to_dict()
        }
        
        return X, y, filtered_data, metadata


class HexagonDataset(Dataset):
    """PyTorch dataset for hexagon prediction."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class HexagonPredictor(nn.Module):
    """Neural network for predicting hexagon sections."""
    
    def __init__(self, input_size: int, num_hexagons: int, hidden_sizes: List[int] = [512, 256, 128]):
        super(HexagonPredictor, self).__init__()
        
        self.input_size = input_size
        self.num_hexagons = num_hexagons
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_hexagons))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HexagonTrainer:
    """Train the hexagon prediction model."""
    
    def __init__(self, model: nn.Module, device: str, learning_rate: float = 0.001):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 100, patience: int = 15) -> Dict:
        """Train the model with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_hexagon_model.pth')
            else:
                patience_counter += 1
            
            # Print progress
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.4f} | "
                  f"{val_loss:8.4f} | {val_acc:7.4f} | {current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_hexagon_model.pth'))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss
        }


class HexagonEvaluator:
    """Evaluate hexagon prediction performance."""
    
    def __init__(self, model: nn.Module, device: str, hexagon_grid: GlobalHexagonGrid, 
                 label_encoder: LabelEncoder):
        self.model = model
        self.device = device
        self.hexagon_grid = hexagon_grid
        self.label_encoder = label_encoder
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model performance."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                probabilities = F.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate top-k accuracy
        all_probabilities = np.array(all_probabilities)
        top5_accuracy = self._calculate_topk_accuracy(all_probabilities, all_labels, k=5)
        
        # Calculate geographic accuracy (distance-based)
        geographic_accuracy = self._calculate_geographic_accuracy(all_predictions, all_labels)
        
        results = {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'geographic_accuracy': geographic_accuracy,
            'predictions': all_predictions,
            'true_labels': all_labels,
            'probabilities': all_probabilities
        }
        
        print(f"\nEvaluation Results:")
        print(f"Exact Accuracy: {accuracy*100:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy*100:.2f}%")
        print(f"Mean Distance Error: {geographic_accuracy['mean_distance_km']:.1f} km")
        print(f"Median Distance Error: {geographic_accuracy['median_distance_km']:.1f} km")
        
        return results
    
    def _calculate_topk_accuracy(self, probabilities: np.ndarray, labels: List[int], k: int) -> float:
        """Calculate top-k accuracy."""
        labels = np.array(labels)
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        correct = np.any(top_k_preds == labels.reshape(-1, 1), axis=1)
        return correct.mean()
    
    def _calculate_geographic_accuracy(self, predictions: List[int], true_labels: List[int]) -> Dict:
        """Calculate geographic distance-based accuracy."""
        distances = []
        
        for pred, true in zip(predictions, true_labels):
            # Convert labels back to hexagon IDs
            pred_hex = self.label_encoder.inverse_transform([pred])[0]
            true_hex = self.label_encoder.inverse_transform([true])[0]
            
            # Get center coordinates
            pred_lat, pred_lon = self.hexagon_grid.hexagon_to_coordinate(pred_hex)
            true_lat, true_lon = self.hexagon_grid.hexagon_to_coordinate(true_hex)
            
            # Calculate haversine distance
            distance = self._haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
            distances.append(distance)
        
        distances = np.array(distances)
        
        return {
            'mean_distance_km': distances.mean(),
            'median_distance_km': np.median(distances),
            'distances': distances
        }
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points."""
        R = 6371  # Earth radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def predict_hexagon(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict hexagon for given features."""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, prediction = torch.max(outputs, 1)
            
            # Convert back to hexagon ID
            pred_label = prediction.item()
            confidence = probabilities[0][pred_label].item()
            hexagon_id = self.label_encoder.inverse_transform([pred_label])[0]
            
            return hexagon_id, confidence
    
    def create_prediction_map(self, predictions: List[int], true_labels: List[int], 
                            sample_size: int = 50) -> folium.Map:
        """Create a map showing predictions vs true locations."""
        # Sample data for visualization
        indices = np.random.choice(len(predictions), min(sample_size, len(predictions)), replace=False)
        
        # Calculate map center
        sample_hexagons = [self.label_encoder.inverse_transform([true_labels[i]])[0] for i in indices]
        sample_coords = [self.hexagon_grid.hexagon_to_coordinate(hex_id) for hex_id in sample_hexagons]
        center_lat = np.mean([coord[0] for coord in sample_coords])
        center_lon = np.mean([coord[1] for coord in sample_coords])
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=3)
        
        for i in indices:
            # True hexagon
            true_hex = self.label_encoder.inverse_transform([true_labels[i]])[0]
            true_lat, true_lon = self.hexagon_grid.hexagon_to_coordinate(true_hex)
            
            # Predicted hexagon
            pred_hex = self.label_encoder.inverse_transform([predictions[i]])[0]
            pred_lat, pred_lon = self.hexagon_grid.hexagon_to_coordinate(pred_hex)
            
            # Add markers
            folium.Marker([true_lat, true_lon], popup=f"True: {true_hex}", 
                         icon=folium.Icon(color='green')).add_to(m)
            folium.Marker([pred_lat, pred_lon], popup=f"Predicted: {pred_hex}", 
                         icon=folium.Icon(color='red')).add_to(m)
            
            # Add line connecting true and predicted
            folium.PolyLine([[true_lat, true_lon], [pred_lat, pred_lon]], 
                           color='orange', weight=2, opacity=0.7).add_to(m)
        
        return m


def plot_training_history(history: Dict) -> None:
    """Plot training history."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plots
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plots
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Loss (log scale)
    ax3.plot(history['train_losses'], label='Train Loss')
    ax3.plot(history['val_losses'], label='Validation Loss')
    ax3.set_title('Training and Validation Loss (Log Scale)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (Log Scale)')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True)
    
    # Learning rate (if available)
    ax4.text(0.5, 0.5, f'Best Validation Loss: {history["best_val_loss"]:.4f}', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=14)
    ax4.set_title('Training Summary')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Global Hexagon Prediction System")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                       help="Path to CSV file with latitude, longitude, and feature columns")
    parser.add_argument('--resolution', type=int, default=4,
                       help="H3 hexagon resolution (3-6 recommended)")
    parser.add_argument('--min_samples', type=int, default=5,
                       help="Minimum samples per hexagon")
    
    # Model parameters
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[512, 256, 128],
                       help="Hidden layer sizes")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=128,
                       help="Batch size")
    parser.add_argument('--epochs', type=int, default=100,
                       help="Maximum number of epochs")
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                       help="Device to use (cpu, cuda, or auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 80)
    print("GLOBAL HEXAGON PREDICTION SYSTEM")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"H3 Resolution: {args.resolution}")
    
    # Step 1: Create global hexagon grid
    print("\n1. Creating global hexagon grid...")
    hexagon_grid = GlobalHexagonGrid(resolution=args.resolution)
    
    # Step 2: Process dataset
    print("\n2. Processing dataset...")
    processor = HexagonDataProcessor(hexagon_grid)
    X, y, processed_data, metadata = processor.process_dataset(
        args.data_path, args.min_samples
    )
    
    print(f"Features shape: {X.shape}")
    print(f"Number of hexagon classes: {metadata['num_hexagons']}")
    
    # Step 3: Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 4: Create datasets and loaders
    train_dataset = HexagonDataset(X_train, y_train)
    val_dataset = HexagonDataset(X_val, y_val)
    test_dataset = HexagonDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Step 5: Create and train model
    print("\n4. Creating and training model...")
    model = HexagonPredictor(
        input_size=X.shape[1],
        num_hexagons=metadata['num_hexagons'],
        hidden_sizes=args.hidden_sizes
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = HexagonTrainer(model, device, args.learning_rate)
    history = trainer.train(train_loader, val_loader, args.epochs)
    
    # Step 6: Evaluate model
    print("\n5. Evaluating model...")
    evaluator = HexagonEvaluator(model, device, hexagon_grid, processor.label_encoder)
    results = evaluator.evaluate(test_loader)
    
    # Step 7: Create visualizations
    print("\n6. Creating visualizations...")
    plot_training_history(history)
    
    # Create prediction map
    prediction_map = evaluator.create_prediction_map(
        results['predictions'], results['true_labels'], sample_size=20
    )
    prediction_map.save('hexagon_predictions.html')
    print("Prediction map saved as 'hexagon_predictions.html'")
    
    # Step 8: Save model and results
    print("\n7. Saving results...")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_scaler': processor.feature_scaler,
        'label_encoder': processor.label_encoder,
        'metadata': metadata,
        'args': args
    }, 'hexagon_model.pth')
    
    # Save results
    results_df = pd.DataFrame({
        'true_hexagon': processor.label_encoder.inverse_transform(results['true_labels']),
        'predicted_hexagon': processor.label_encoder.inverse_transform(results['predictions']),
        'confidence': np.max(results['probabilities'], axis=1)
    })
    results_df.to_csv('prediction_results.csv', index=False)
    
    print("\n" + "=" * 80)
    print("SYSTEM SUMMARY")
    print("=" * 80)
    print(f"Global hexagons created: {len(hexagon_grid.global_hexagons):,}")
    print(f"Hexagons with data: {metadata['num_hexagons']:,}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Model accuracy: {results['accuracy']*100:.2f}%")
    print(f"Top-5 accuracy: {results['top5_accuracy']*100:.2f}%")
    print(f"Mean distance error: {results['geographic_accuracy']['mean_distance_km']:.1f} km")
    print("\nFiles created:")
    print("- hexagon_model.pth (trained model)")
    print("- prediction_results.csv (detailed predictions)")
    print("- training_history.png (training curves)")
    print("- hexagon_predictions.html (visualization map)")
    print("\nSystem execution completed successfully!")


if __name__ == "__main__":
    main()