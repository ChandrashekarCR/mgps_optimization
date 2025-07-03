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


def ensure_tensor_types(X, y, device):
    """
    Ensure proper tensor types for PyTorch operations
    """
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    else:
        X = X.float()
    
    if not isinstance(y, torch.Tensor):
        y = torch.LongTensor(y)
    else:
        y = y.long()
    
    return X.to(device), y.to(device)

# Weak learner
# This is a shallow neural network with just two hidden layers
class WeakLearner(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, dropout_rate: float = 0.3):
        super(WeakLearner,self).__init__()

        # Two layer shallow neural network with batch normalization and dropout
        # Input -> hidden -> hidden//2 -> Output
        self.layers = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim,hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim//2,output_dim) # Output raw logits
        )

        # Initialiaze weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        """Forward propagation for weak learners"""
        return self.layers(x)
    

# Grownet model
class GrowNet(nn.Module):
    """
    GrowNet: Gradient Boosting Neural Networks for Multi-Class Classification
    The main things about this network is that -:
        1) It usese shallow two layers deep neural networks instead of decision trees when compared to XGBoost
        2) Implements gradient boositng frameworks with neural networks
        3) Includes second-order statistics (Newton's method approximation)
        4) Has a global corrective step for fine tuning
    """
    def __init__(self, input_dim:int, num_classes:int, num_models:int = 20, hidden_dim:int = 256, lr:float = 0.01,
                 reg_lambda:float = 0.01, dropout_rate:float=0.3,device:str="cpu"):
        super(GrowNet,self).__init__()

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

        # Store learning rates for each weak learner (adaptice)
        self.alphas = []

        # Global corrective network
        self.corrective_net = WeakLearner(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.num_classes,dropout_rate=self.dropout)

        # Initialize first weak learner
        self.add_weak_learner(dropout_rate)

        # Move to device
        self.to(device=self.device)

    def add_weak_learner(self,dropout_rate:float=0.2):
        """Add a aner weak learner to the ensemble"""
        weak_learner = WeakLearner(self.input_dim,self.hidden_dim, self.num_classes,dropout_rate)
        weak_learner = weak_learner.to(self.device)
        self.weak_learners.append(weak_learner)

    def forward(self,x,num_models_used:Optional[int]=None):
        """
        Forward pass through the GrowNet
        
        """

        if num_models_used is None:
            num_models_used = len(self.weak_learners)

        predictions = torch.zeros(x.size(0),self.num_classes).to(self.device)

        # Ensemble predictions from weak learners
        for i in range(min(num_models_used,len(self.weak_learners))):
            weak_pred = self.weak_learners[i](x)
            alpha = self.alphas[i] if i < len(self.alphas) else 1.0
            predictions += alpha * weak_pred

        return predictions
    
    def compute_gradients_and_hessians(self, predictions, targets):
        """
        Computte first and second order gradients for multiclass classification using softmax cross entropy loss function
        """

        # Concer targets to one-hot ecoding first
        targets = targets.long()
        if targets.dim() == 1:
            targets_onehot = F.one_hot(targets,num_classes=self.num_classes).float()
        else:
            targets_onehot = targets

        # Softmax prbabilities
        probs = F.softmax(predictions, dim=1)

        # First order gradients (negative gradient for cross entropy)
        gradients = targets_onehot - probs


        # Second order gradients (Hessian diagonal approximation)
        # For multiclass: H_ii = p_i(1-p_i), H_ij = -p_i*p_j for i≠j
        # We use diagonal approximation: H_ii = p_i(1-p_i)
        hessians = probs * (1 - probs)
        
        return gradients, hessians
    
    def fit_single_weak_learner(self,X,gradients,hessians, epochs=100):
        """
        Fit a single weak learner using gradients and hessians
        """

        # Create current weak learner
        current_weak_learner = self.weak_learners[-1]
        current_weak_learner = current_weak_learner.to(self.device)
        optimizer = optim.Adam(current_weak_learner.parameters(),lr=self.lr,weight_decay=self.reg_lambda)

        X = X.to(self.device)
        gradients = gradients.to(self.device)
        hessians = hessians.to(self.device)

        # Create dataset fro gradient boosting
        dataset = TensorDataset(X, gradients, hessians)
        dataloader = DataLoader(dataset=dataset, batch_size=min(128,len(X)),shuffle=True)

        # Set to model to train mode
        current_weak_learner.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_grad, batch_hess  in dataloader:
                
                # Move batch correct device
                batch_x = batch_x.to(self.device)
                batch_grad = batch_grad.to(self.device)
                batch_hess = batch_hess.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                pred = current_weak_learner(batch_x)

                # Custom loss using gradients and hessians (Newton-Raphson approximation)
                # Loss = 0.5 * sum(grad^2/(hess + lambda))

                loss = -torch.sum(batch_grad*pred) + 0.5 * torch.sum(batch_hess * pred**2)
                loss += self.reg_lambda * sum(p.pow(2.0).sum() for p in current_weak_learner.parameters())

                loss.backward()
                optimizer.step()

                total_loss += loss.item()


        # Compute optimal step size (alpha) using line search
        current_weak_learner.eval()
        with torch.no_grad():
            weak_pred = current_weak_learner(X)
            # SImple alpa calcualtion - can be improve with line search
            alpha = 1.0 / len(self.weak_learners)
            self.alphas.append(alpha)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epoch_per_stage=100, verbose=True):
        """
        Fit GrowNet using gradient boosting
        
        """

        # COnvet to tensors - Just to avoid any potential erros
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train).to(self.device)

        if not isinstance(y_train,torch.Tensor):
            y_train = torch.LongTensor(y_train).to(self.device)

        if X_val is not None and not isinstance(X_val,torch.Tensor):
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.LongTensor(y_val).to(self.device)

        
        train_losses = []
        val_accuracies = []

        for stage in range(self.num_models):
            if verbose:
                print(f"Training weak learner {stage+1}/{self.num_models}")

            # Get current ensemble predictions
            self.eval()
            with torch.no_grad():
                if stage == 0:
                    current_pred = torch.zeros(X_train.size(0),self.num_classes).to(self.device)
                else:
                    current_pred = self.forward(X_train,num_models_used=stage)


            # Compute gradients and hessians
            gradients, hessians = self.compute_gradients_and_hessians(current_pred, y_train)

            # Add new weak learner is not first stage
            if stage > 0:
                self.add_weak_learner()

            # Fit the weak learner
            self.fit_single_weak_learner(X_train, gradients, hessians, epoch_per_stage)


            # Evaluate
            self.eval() 
            with torch.no_grad():
                train_pred = self.forward(X_train, num_models_used=stage+1)
                train_loss = F.cross_entropy(train_pred,y_train)
                train_losses.append(train_loss.item())

                if X_val is not None:
                    val_pred = self.forward(X_val, num_models_used=stage+1)
                    val_acc = accuracy_score(y_val.cpu().numpy(), val_pred.argmax(dim=1).cpu().numpy())
                    val_loss = F.cross_entropy(val_pred,y_val)
                    val_accuracies.append(val_acc)


                    if verbose:
                       print(f"  Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    train_acc = accuracy_score(y_train.cpu().numpy(), train_pred.argmax(dim=1).cpu().numpy())
                    if verbose:
                        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}")
        
        # Global corrective step
        if verbose:
            print("Training global corrective network...")
        
        self.train_corrective_network(X_train, y_train, epochs=400)
        
        return train_losses, val_accuracies 
    

    def train_corrective_network(self, X, y, epochs=100):
        """
        Train the global corrective network
        """

        self.corrective_net = self.corrective_net.to(self.device)
        optimizer = optim.Adam(self.corrective_net.parameters(), lr=self.lr/2, weight_decay=self.reg_lambda)
        
        # Get ensemble predictions
        self.eval()
        with torch.no_grad():
            ensemble_pred = self.forward(X)
        
        # Train corrective network
        self.corrective_net.train()

        X = X.to(self.device)
        y = y.to(self.device)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=min(256, len(X)), shuffle=True)
        
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Get current ensemble prediction
                with torch.no_grad():
                    base_pred = self.forward(batch_x)
                
                # Corrective prediction
                corrective_pred = self.corrective_net(batch_x)
                
                # Final prediction
                final_pred = base_pred + 0.1 * corrective_pred  # Small correction factor
                
                loss = F.cross_entropy(final_pred, batch_y)
                loss.backward()
                optimizer.step()
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X).to(self.device)
        
        self.eval()
        with torch.no_grad():
            base_pred = self.forward(X)
            corrective_pred = self.corrective_net(X)
            final_pred = base_pred + 0.1 * corrective_pred
            probabilities = F.softmax(final_pred, dim=1)
        
        return probabilities.cpu().numpy()
    
    def predict(self, X):
        """Get class predictions"""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


class MicrobiomeGrowNetClassifier:
    """
    Wrapper class for GrowNet specifically designed for microbiome classification
    """
    def __init__(self, 
                 num_classes=7,
                 num_models=20,
                 hidden_dim=128,
                 lr=0.001,
                 reg_lambda=0.01,
                 dropout_rate=0.2,
                 device=None):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.scaler = StandardScaler()
        self.num_classes = num_classes
        self.num_models = num_models
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.dropout_rate = dropout_rate
        self.model = None
        
    def fit(self, X, y, test_size=0.2, epochs_per_stage=100, verbose=True):
        """
        Fit the GrowNet model
        
        Args:
            X: Feature matrix (4070, 200) - RSA values
            y: Target labels (continent labels 0-6)
            test_size: Proportion for validation split
            epochs_per_stage: Epochs to train each weak learner
            verbose: Print training progress
        """

        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Initialize model
        input_dim = X_train.shape[1]
        self.model = GrowNet(
            input_dim=input_dim,
            num_classes=self.num_classes,
            num_models=self.num_models,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            reg_lambda=self.reg_lambda,
            dropout_rate=self.dropout_rate,
            device=self.device
        )
        
        # Train model
        train_losses, val_accuracies = self.model.fit(
            X_train, y_train, X_val, y_val, 
            epoch_per_stage=epochs_per_stage, 
            verbose=verbose
        )
        
        return train_losses, val_accuracies
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        cm = confusion_matrix(y, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def plot_training_curves(self, train_losses, val_accuracies):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training loss
        ax1.plot(train_losses)
        ax1.set_title('Training Loss vs Weak Learners')
        ax1.set_xlabel('Weak Learner')
        ax1.set_ylabel('Cross-Entropy Loss')
        ax1.grid(True)
        
        # Validation accuracy
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy vs Weak Learners')
        ax2.set_xlabel('Weak Learner')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, continent_names=None):
        """Plot confusion matrix"""
        if continent_names is None:
            continent_names = [f'Continent_{i}' for i in range(self.num_classes)]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=continent_names,
                    yticklabels=continent_names)
        plt.title('Confusion Matrix - GrowNet Microbiome Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
# Example usage and comparison with XGBoost
def compare_with_xgboost(X, y, continent_names=None):
    """
    Compare GrowNet with XGBoost for microbiome classification
    """
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier
    
    print("=== GrowNet vs XGBoost Comparison ===\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # GrowNet
    print("Training GrowNet...")
    grownet_model = MicrobiomeGrowNetClassifier(
        num_classes=7,
        num_models=50,  # Fewer models for faster training
        hidden_dim=128,
        lr=0.0001,
        reg_lambda=0.00001,
        dropout_rate=0.4
    )
    
    train_losses, val_accuracies = grownet_model.fit(X_train, y_train, verbose=True,epochs_per_stage=150)
    grownet_results = grownet_model.evaluate(X_test, y_test)
    
    # XGBoost
    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.01,
        random_state=42,
        objective="multi:softmax",
        use_label_encoder=False,
        eval_metric='mlogloss',
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    
    # Results
    print(f"\n=== Results ===")
    print(f"GrowNet Accuracy: {grownet_results['accuracy']:.4f}")
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    print(f"GrowNet Improvement: {grownet_results['accuracy'] - xgb_accuracy:.4f}")
    
    # Plot results
    grownet_model.plot_training_curves(train_losses, val_accuracies)
    grownet_model.plot_confusion_matrix(y_test, grownet_model.predict(X_test), continent_names)
    
    return grownet_model, xgb_model



if __name__ == "__main__":

    # Example training script
    # Read the data 
    df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
    df = pd.concat([df.iloc[:,:-4],df['continent']],axis=1)
    X = df[df.columns[:-1]][:].to_numpy()
    y = df[df.columns[-1]][:].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y)

    continent_encoding_map = dict(zip(le.transform(le.classes_), le.classes_))
    print(continent_encoding_map)

    
    print("Dataset Info:")
    print(f"Shape: {X.shape}")
    print(f"RSA sum range: [{X.sum(axis=1).min():.3f}, {X.sum(axis=1).max():.3f}]")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Run comparison
    grownet_model, xgb_model = compare_with_xgboost(X, y, continent_encoding_map.values())



