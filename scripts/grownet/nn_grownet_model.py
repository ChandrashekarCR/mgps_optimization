# Importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


# Weak learner - simplified architecture
class WeakLearner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.2):
        super(WeakLearner, self).__init__()
        
        # Simpler architecture - single hidden layer
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


class GrowNet(nn.Module):
    """
    Fixed GrowNet implementation with proper gradient boosting
    """
    def __init__(self, input_dim: int, num_classes: int, num_models: int = 20, 
                 hidden_dim: int = 128, lr: float = 0.01, reg_lambda: float = 0.01, 
                 dropout_rate: float = 0.2, device: str = "cpu"):
        super(GrowNet, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_models = num_models
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.dropout = dropout_rate
        self.device = device

        # Store weak learners
        self.weak_learners = nn.ModuleList()
        
        # Store learning rates for each weak learner
        self.alphas = []

        # Move to device
        self.to(device=self.device)

    def add_weak_learner(self, stage: int):
        """Add a new weak learner with appropriate input dimension"""
        # For stage 0: only original features
        # For stage > 0: original features + predictions from previous stages
        if stage == 0:
            input_size = self.input_dim
        else:
            input_size = self.input_dim + stage * self.num_classes
            
        weak_learner = WeakLearner(
            input_dim=input_size,
            hidden_dim=self.hidden_dim, 
            output_dim=self.num_classes,
            dropout_rate=self.dropout
        )
        weak_learner = weak_learner.to(self.device)
        self.weak_learners.append(weak_learner)

    def get_augmented_features(self, x: torch.Tensor, predictions_history: list):
        """
        Create augmented features by concatenating original features with previous predictions
        
        Args:
            x: Original features [batch_size, input_dim]
            predictions_history: List of previous stage predictions
            
        Returns:
            Augmented features
        """
        if len(predictions_history) == 0:
            return x
        
        # Concatenate original features with all previous predictions
        augmented_x = x
        for prev_pred in predictions_history:
            if prev_pred.size(0) == x.size(0):  # Ensure batch sizes match
                augmented_x = torch.cat([augmented_x, prev_pred], dim=1)
        
        return augmented_x
    
    def forward(self, x: torch.Tensor, num_models_used=None):
        """Forward pass through the GrowNet architecture"""
        if num_models_used is None:
            num_models_used = len(self.weak_learners)

        # Initialize final predictions
        predictions = torch.zeros(x.size(0), self.num_classes).to(self.device)
        
        # Store predictions from each stage for feature augmentation
        stage_predictions = []

        # Ensemble predictions from weak learners
        for stage in range(min(num_models_used, len(self.weak_learners))):
            # Get augmented features for current stage
            if stage == 0:
                augmented_x = x
            else:
                augmented_x = self.get_augmented_features(x, stage_predictions)
            
            # Get prediction from current weak learner
            weak_pred = self.weak_learners[stage](augmented_x)
            stage_predictions.append(weak_pred.detach())

            # Apply learner step size (alpha)
            alpha = self.alphas[stage] if stage < len(self.alphas) else 1.0
            predictions += alpha * weak_pred

        return predictions
    
    def compute_gradients(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Compute gradients for multiclass classification using softmax cross-entropy loss
        
        Args:
            predictions: Current ensemble predictions [batch_size, num_classes]
            targets: True labels [batch_size]
        
        Returns:
            gradients: Negative gradients [batch_size, num_classes]
        """
        targets = targets.long()
        
        # Compute softmax probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Compute negative gradient of cross-entropy loss
        # This is what the next weak learner should predict
        gradients = targets_onehot - probs
        
        return gradients

    def line_search_alpha(self, X: torch.Tensor, y: torch.Tensor, 
                         current_pred: torch.Tensor, weak_pred: torch.Tensor):
        """Perform line search to find optimal step size"""
        alpha_candidates = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        best_alpha = 1.0
        best_loss = float('inf')
        
        with torch.no_grad():
            for alpha in alpha_candidates:
                test_pred = current_pred + alpha * weak_pred
                loss = F.cross_entropy(test_pred, y)
                
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha
        
        return best_alpha
    
    def fit_single_weak_learner(self, X: torch.Tensor, gradients: torch.Tensor, 
                               stage_predictions: list, stage: int, epochs: int = 3):
        """
        Fit a single weak learner using gradients
        
        Args:
            X: Input features
            gradients: Target gradients (negative gradients)
            stage_predictions: Predictions from previous stages
            stage: Current stage number
            epochs: Number of epochs to train
        """
        # Get current weak learner
        current_weak_learner = self.weak_learners[-1]
        current_weak_learner.train()

        # Optimizer for current weak learner
        optimizer = optim.Adam(current_weak_learner.parameters(), 
                             lr=self.lr, weight_decay=self.reg_lambda)

        # Get augmented features for current stage
        if stage == 0:
            augmented_X = X
        else:
            augmented_X = self.get_augmented_features(X, stage_predictions)

        # Create dataset (target is the gradients)
        dataset = TensorDataset(augmented_X, gradients)
        dataloader = DataLoader(dataset, batch_size=min(256, len(X)), shuffle=True)

        # Train weak learner
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_grad_target in dataloader:
                batch_x = batch_x.to(self.device)
                batch_grad_target = batch_grad_target.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                pred = current_weak_learner(batch_x)

                # MSE loss between prediction and gradient target
                loss = F.mse_loss(pred, batch_grad_target)

                # Add L2 regularization
                l2_reg = sum(p.pow(2.0).sum() for p in current_weak_learner.parameters())
                loss += self.reg_lambda * l2_reg

                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(current_weak_learner.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, 
            X_val=None, y_val=None, epochs_per_stage=3, verbose=True):
        """
        Fit GrowNet using gradient boosting
        """
        # Convert to tensors if needed
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train).to(self.device)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.LongTensor(y_train).to(self.device)

        if X_val is not None:
            if not isinstance(X_val, torch.Tensor):
                X_val = torch.FloatTensor(X_val).to(self.device)
                y_val = torch.LongTensor(y_val).to(self.device)

        train_losses = []
        val_accuracies = []

        # Initialize ensemble predictions
        ensemble_pred = torch.zeros(X_train.size(0), self.num_classes).to(self.device)
        stage_predictions = []  # Store predictions from each stage

        # Sequential training of weak learners
        for stage in range(self.num_models):
            if verbose:
                print(f"Training weak learner {stage+1}/{self.num_models}")

            # Compute gradients based on current ensemble predictions
            gradients = self.compute_gradients(ensemble_pred, y_train)

            # Add new weak learner
            self.add_weak_learner(stage)

            # Fit the current weak learner using gradients
            self.fit_single_weak_learner(X_train, gradients, stage_predictions, 
                                       stage, epochs_per_stage)

            # Get prediction from the newly trained weak learner
            self.eval()
            with torch.no_grad():
                if stage == 0:
                    augmented_X = X_train
                else:
                    augmented_X = self.get_augmented_features(X_train, stage_predictions)
                
                weak_pred = self.weak_learners[stage](augmented_X)
                stage_predictions.append(weak_pred.detach())

                # Line search for optimal alpha
                optimal_alpha = self.line_search_alpha(X_train, y_train, 
                                                     ensemble_pred, weak_pred)
                self.alphas.append(optimal_alpha)

                # Update ensemble predictions
                ensemble_pred += optimal_alpha * weak_pred

            # Evaluate current performance
            with torch.no_grad():
                train_loss = F.cross_entropy(ensemble_pred, y_train)
                train_losses.append(train_loss.item())

                if X_val is not None:
                    val_pred = self.forward(X_val, num_models_used=stage+1)
                    val_acc = accuracy_score(y_val.cpu().numpy(), 
                                           val_pred.argmax(dim=1).cpu().numpy())
                    val_loss = F.cross_entropy(val_pred, y_val)
                    val_accuracies.append(val_acc)

                    if verbose:
                        print(f"  Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Alpha: {optimal_alpha:.4f}")
                else:
                    train_acc = accuracy_score(y_train.cpu().numpy(), 
                                             ensemble_pred.argmax(dim=1).cpu().numpy())
                    if verbose:
                        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
                              f"Alpha: {optimal_alpha:.4f}")

            # Early stopping based on validation performance
            if X_val is not None and len(val_accuracies) > 5:
                recent_accs = val_accuracies[-5:]
                if max(recent_accs) - min(recent_accs) < 0.001:  # No improvement
                    if verbose:
                        print(f"Early stopping at stage {stage+1}")
                    break
        
        return train_losses, val_accuracies 
    
    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X).to(self.device)
        
        self.eval()
        with torch.no_grad():
            predictions = self.forward(X)
            probabilities = F.softmax(predictions, dim=1)
        
        return probabilities.cpu().numpy()
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


# Example training script
# Read the data 
df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
df = pd.concat([df.iloc[:,:-4],df['continent']],axis=1)
X = df[df.columns[:-1]][:].to_numpy()
y = df[df.columns[-1]][:].to_numpy()

le = LabelEncoder()
y = le.fit_transform(y)

continent_encoding_map = dict(zip(le.transform(le.classes_), le.classes_))

print("Dataset Info:")
print(f"Shape: {X.shape}")
print(f"RSA sum range: [{X.sum(axis=1).min():.3f}, {X.sum(axis=1).max():.3f}]")
print(f"Class distribution: {dict(zip(le.classes_,np.bincount(y)))}")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)


# Initialize GrowNet
input_dim = X_train.shape[1]
num_classes = len(continent_encoding_map)
model = GrowNet(
    input_dim=input_dim,
    num_classes=num_classes,
    num_models=100,           # Number of weak learners
    hidden_dim=64,          # Hidden layer size
    lr=0.01,               # Learning rate
    reg_lambda=0.001,        # L2 regularization
    dropout_rate=0.3,       # Dropout probability
    device=device
)

# Train the model
train_losses, val_accuracies = model.fit(
    X_train_tensor, 
    y_train_tensor,
    X_val=X_val_tensor,
    y_val=y_val_tensor,
    epochs_per_stage=5,     # Train each weak learner for 2 epochs
    verbose=True
)

# Evaluate
def evaluate(model, X, y):
    with torch.no_grad():
        probs = model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        accuracy = (preds == y).mean()
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y, preds, target_names=continent_encoding_map.values()))


print("\nTest set evaluation:")
evaluate(model, X_test, y_test)

