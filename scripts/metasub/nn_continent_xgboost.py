import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from xgboost import XGBClassifier
from collections import OrderedDict
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
import os

# --- Import shared scripts ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from scripts.metasub.process_transfrom_data import process_transform_data
from scripts.metasub.accuracy_metrics.calculate_model_accuracy import check_combined_accuracy


class Seq(torch.nn.Sequential):
    '''
    Seq uses sequential module to implement tree boosting in the forward pass.
    '''
    def give(self, xg, num_layers_boosted, ep=0.001):
        '''
        Saves various information into the object for further usage in the training process
        :param xg(object of XGBoostClassifier): Object of XGBoostClassifier
        :param num_layers_boosted(int,optional): Number of layers to be boosted in the neural network.
        :param ep(int,optional): Epsilon for smoothing. Default: 0.001
        '''
        self.xg = xg
        self.epsilon = ep
        self.boosted_layers = OrderedDict()
        self.num_layers_boosted = num_layers_boosted
        
    def forward(self, input, labels, train=True):
        for i, module in enumerate(self):
            input = module(input)
            x0 = input
            if train and hasattr(self, 'xg') and labels is not None:
                if i < self.num_layers_boosted:
                    try:
                        # Apply XGBoost boosting to current layer output
                        feature_importances = self.xg.fit(
                            x0.detach().cpu().numpy(), 
                            labels.detach().cpu().numpy(),
                            eval_metric="mlogloss"
                        ).feature_importances_
                        self.boosted_layers[i] = torch.from_numpy(
                            np.array(feature_importances) + self.epsilon
                        ).to(x0.device)
                    except Exception as e:
                        # Continue without boosting if there's an error
                        pass
        return input

class XBNETClassifier(nn.Module):
    '''
    XBNetClassifier is a model for classification tasks that combines tree-based models with
    neural networks to create a robust architecture.
    
    :param input_size(int): Input feature dimension
    :param hidden_dim(int): Hidden layer dimension
    :param num_classes(int): Number of output classes
    :param num_layers_boosted(int, optional): Number of layers to be boosted. Default: 1
    :param initial_dropout_rate(float): Initial dropout rate
    :param max_dropout_rate(float): Maximum dropout rate
    :param n_estimators(int, optional): Number of XGBoost estimators. Default: 100
    '''
    
    def __init__(self, input_size, hidden_dim, num_classes, num_layers_boosted=1, 
                 initial_dropout_rate=0.1, max_dropout_rate=0.5, n_estimators=100):
        super(XBNETClassifier, self).__init__()
        
        self.name = "Classification"
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers_boosted = num_layers_boosted
        self.initial_dropout_rate = initial_dropout_rate
        self.max_dropout_rate = max_dropout_rate
        self.n_estimators = n_estimators
        self.labels = num_classes  # For compatibility with original interface
        
        # Initialize XGBoost classifier with proper parameters to avoid deprecation warning
        self.xg = XGBClassifier(
            n_estimators=self.n_estimators,
            use_label_encoder=False,  # This fixes the deprecation warning
            eval_metric='mlogloss'    # Specify eval_metric explicitly
        )
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Neural network layers with progressive dimension reduction
        self.layer_1 = nn.Linear(input_size, hidden_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.layer_3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn_3 = nn.BatchNorm1d(hidden_dim // 4)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim // 4, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        # Create layers dictionary for compatibility
        self.layers = OrderedDict()
        self.layers['0'] = self.layer_1
        self.layers['1'] = self.layer_2
        self.layers['2'] = self.layer_3
        
        # Create sequential container for boosting
        seq_layers = OrderedDict()
        seq_layers['0'] = nn.Sequential(self.layer_1, self.bn_1, self.relu)
        seq_layers['1'] = nn.Sequential(self.layer_2, self.bn_2, self.relu)
        seq_layers['2'] = nn.Sequential(self.layer_3, self.bn_3, self.relu)
        
        self.sequential = Seq(seq_layers)
        self.sequential.give(self.xg, self.num_layers_boosted)
        
        # Track initialization status and current labels
        self.weights_initialized = False
        self.feature_importances_ = None
        self.current_labels = None
        
    def get(self, labels):
        '''
        Gets the set of current actual outputs of the inputs (for compatibility)
        :param labels(tensor): Labels of the current set of inputs that are getting processed.
        '''
        self.current_labels = labels
        
    # 4. Initialize XGBoost in the sequential container properly
    def initialize_first_layer_weights(self, X, y):
        if not self.weights_initialized:
            try:
                # Ensure labels are integers
                y = y.astype(int)
                
                # Fit XGBoost to get feature importances
                temp_xg = XGBClassifier(
                    n_estimators=self.n_estimators,
                    use_label_encoder=False,
                    eval_metric="mlogloss"
                )
                temp_xg.fit(X, y)
                feature_importances = temp_xg.feature_importances_
                self.feature_importances_ = feature_importances
                
                # Create weight matrix by repeating feature importances
                weight_matrix = np.tile(feature_importances, (self.hidden_dim, 1))
                
                # Set the first layer weights
                with torch.no_grad():
                    self.layer_1.weight.data = torch.from_numpy(weight_matrix).float()
                
                self.weights_initialized = True
                
            except Exception as e:
                print(f"Warning: Could not initialize weights with XGBoost: {e}")
                self.weights_initialized = True
    
    # 3. Improved forward method with better error handling
    def forward(self, x, train=True):
        """Updated forward method with safe XGBoost integration"""
        # Initialize weights on first forward pass if training data is available
        if not self.weights_initialized and self.current_labels is not None and train:
            try:
                X_np = x.detach().cpu().numpy()
                y_np = self.current_labels.detach().cpu().numpy().flatten().astype(np.int32)
                self.initialize_first_layer_weights(X_np, y_np)
            except Exception as e:
                print(f"Warning: Could not initialize weights: {e}")
                self.weights_initialized = True

        # Apply first layer
        out = self.relu(self.bn_1(self.layer_1(x)))

        # Apply second layer
        out = self.relu(self.bn_2(self.layer_2(out)))

        # Apply third layer
        out = self.relu(self.bn_3(self.layer_3(out)))

        # If we have labels and we're training, apply boosting
        if train and self.current_labels is not None and hasattr(self.sequential, 'xg'):
            try:
                # Apply XGBoost boosting to the current layer output
                x0 = out.detach().cpu().numpy()
                y0 = self.current_labels.detach().cpu().numpy().flatten().astype(np.int32)

                # Validate labels before XGBoost
                unique_labels = np.unique(y0)
                if len(unique_labels) > 1 and x0.shape[0] > len(unique_labels):  # Ensure we have enough samples
                    # Ensure labels are consecutive from 0
                    if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
                        label_map = {old: new for new, old in enumerate(unique_labels)}
                        y0 = np.array([label_map[label] for label in y0], dtype=np.int32)

                    # Additional validation
                    assert y0.min() >= 0, f"Labels must be >= 0, got min: {y0.min()}"
                    assert y0.max() < len(unique_labels), f"Labels must be < num_classes, got max: {y0.max()}, num_classes: {len(unique_labels)}"

                    # Fit XGBoost and get feature importances
                    fitted_xg = safe_xgboost_fit(self.sequential.xg, x0, y0)
                    if fitted_xg is not None:
                        feature_importances = fitted_xg.feature_importances_

                        # Store boosted layers for gradient modification
                        if not hasattr(self.sequential, 'boosted_layers'):
                            self.sequential.boosted_layers = {}

                        self.sequential.boosted_layers[0] = torch.from_numpy(
                            feature_importances + self.sequential.epsilon
                        ).to(out.device)

            except Exception as e:
                # Continue without boosting if there's an error
                print(f"Warning: XGBoost boosting failed: {e}")
                pass
            
        # Final classification
        class_logits = self.classifier(out)

        # Return softmax probabilities for multi-class, raw logits for binary
        if self.labels == 1:
            return torch.sigmoid(class_logits)
        else:
            return self.softmax(class_logits)
    
    def save(self, path):
        '''
        Saves the entire model in the provided path
        :param path(string): Path where model should be saved
        '''
        torch.save(self, path)


class CustDat(Dataset):
    def __init__(self, X, y, task='continent'):
        self.X = X
        self.y = y
        self.task = task
        
        # Extract appropriate labels based on task
        if task == 'continent':
            self.labels = y[:, 0] if len(y.shape) > 1 else y
        elif task == 'city':
            self.labels = y[:, 1] if len(y.shape) > 1 and y.shape[1] > 1 else y
        elif task == 'xyz':
            self.labels = y[:, 2] if len(y.shape) > 1 and y.shape[1] > 2 else y
        else:
            self.labels = y[:, 0] if len(y.shape) > 1 else y
        
        # Ensure labels are proper integers from 0
        self.labels = self.labels.astype(np.int32)
        unique_labels = np.unique(self.labels)
        if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
            label_map = {old: new for new, old in enumerate(unique_labels)}
            self.labels = np.array([label_map[label] for label in self.labels], dtype=np.int32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Ensure labels are proper integers
        label = int(self.labels[idx])
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# --- XBNet Training Loop ---
def xbnet_training_loop(train_dl, val_dl, model, optimizer, criterion, device="cpu",
                       num_epochs=100, patience=10, task_name="classification"):
    '''
    Training function for XBNETClassifier with XGBoost boosting and early stopping
    :param train_dl: Training DataLoader
    :param val_dl: Validation DataLoader  
    :param model: XBNETClassifier model
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param device: Device to run on
    :param num_epochs: Number of epochs
    :param patience: Early stopping patience
    :param task_name: Name of the task for logging
    :return: Training and validation histories
    '''
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    # History tracking
    train_accuracy = []
    train_loss = []
    val_accuracy = []
    val_loss = []
    
    print(f"\nStarting training for {task_name} task...")
    
    for epoch in tqdm(range(1, num_epochs + 1), desc=f"Training {task_name}"):
        epoch_start = time.time()
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        predictions = []
        actuals = []
        
        for inp, out in train_dl:
            inp, out = inp.to(device), out.to(device)
            
            # Set labels for XGBoost boosting
            model.get(out.long())
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(inp.float(), train=True)
            
            # Calculate loss
            if model.labels == 1:
                loss = criterion(y_pred, out.view(-1, 1).float())
            else:
                loss = criterion(y_pred, out.long())
                
            running_loss += loss.item()
            loss.backward()
            
            # Fixed XGBoost gradient boosting modification section
            # Replace the problematic section in your xbnet_training_loop function

            # XGBoost gradient boosting modification
            for i, p in enumerate(model.parameters()):
                if i < model.num_layers_boosted and hasattr(model.sequential, 'boosted_layers'):
                    if i in model.sequential.boosted_layers and p.grad is not None:
                        feature_importance_tensor = model.sequential.boosted_layers[i]
                        
                        # Reshape feature importance tensor to match parameter gradient shape
                        if len(p.grad.shape) == 2:  # Weight matrix
                            # For weight matrices, broadcast across appropriate dimension
                            if feature_importance_tensor.shape[0] == p.grad.shape[1]:
                                # Feature importances match input dimension
                                l0 = feature_importance_tensor.unsqueeze(0).expand_as(p.grad)
                            elif feature_importance_tensor.shape[0] == p.grad.shape[0]:
                                # Feature importances match output dimension
                                l0 = feature_importance_tensor.unsqueeze(1).expand_as(p.grad)
                            else:
                                # Skip if dimensions don't match
                                continue
                        elif len(p.grad.shape) == 1:  # Bias vector
                            # For bias vectors, use mean or first few elements
                            if feature_importance_tensor.shape[0] >= p.grad.shape[0]:
                                l0 = feature_importance_tensor[:p.grad.shape[0]]
                            else:
                                # Repeat the feature importance tensor
                                repeats = (p.grad.shape[0] + feature_importance_tensor.shape[0] - 1) // feature_importance_tensor.shape[0]
                                l0 = feature_importance_tensor.repeat(repeats)[:p.grad.shape[0]]
                        else:
                            # Skip for other tensor shapes
                            continue
                        
                        # Apply gradient modification with proper scaling
                        lMin = torch.min(p.grad)
                        lPower = torch.log(torch.abs(lMin + 1e-8))  # Added small epsilon
                        if lMin != 0:
                            l0 = l0 * 10 ** lPower
                            try:
                                p.grad += l0.to(p.grad.device)
                            except RuntimeError as e:
                                # If there's still a size mismatch, skip this parameter
                                print(f"Warning: Skipping gradient boosting for parameter {i} due to size mismatch: {e}")
                                continue
            
            optimizer.step()
            
            # Calculate training accuracy
            with torch.no_grad():
                outputs = model(inp.float(), train=False)
                total += out.size(0)
                
                if model.labels == 1:
                    predicted = (outputs > 0.5).float().squeeze()
                    correct += (predicted == out.float()).sum().item()
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == out.long()).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                actuals.extend(out.cpu().numpy())
        
        # Record training metrics
        epoch_train_loss = running_loss / len(train_dl)
        epoch_train_acc = 100 * correct / total
        train_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_train_acc)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_actuals = []
        
        with torch.no_grad():
            for inp, out in val_dl:
                inp, out = inp.to(device), out.to(device)
                model.get(out.float())
                
                y_pred = model(inp.float(), train=False)
                
                if model.labels == 1:
                    loss = criterion(y_pred, out.view(-1, 1).float())
                    predicted = (y_pred > 0.5).float().squeeze()
                    val_correct += (predicted == out.float()).sum().item()
                else:
                    loss = criterion(y_pred, out.long())
                    _, predicted = torch.max(y_pred.data, 1)
                    val_correct += (predicted == out.long()).sum().item()
                
                val_running_loss += loss.item()
                val_total += out.size(0)
                val_predictions.extend(predicted.cpu().numpy())
                val_actuals.extend(out.cpu().numpy())
        
        # Record validation metrics
        epoch_val_loss = val_running_loss / len(val_dl)
        epoch_val_acc = 100 * val_correct / val_total
        val_loss.append(epoch_val_loss)
        val_accuracy.append(epoch_val_acc)
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} | "
                  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% | "
                  f"Time: {time.time()-epoch_start:.1f}s")
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
        print(f"Restored best model (val loss: {best_val_loss:.4f})")
    
    # Print final classification report
    print(f"\nFinal {task_name} Training Classification Report:")
    print(classification_report(np.array(actuals), np.array(predictions)))
    
    print(f"\nFinal {task_name} Validation Classification Report:")
    print(classification_report(np.array(val_actuals), np.array(val_predictions)))
    
    return train_loss, val_loss


def check_combined_accuracy(loader, models, device, coordinate_scaler=None, has_continent=True, has_city=True, has_xyz=True):
    '''
    Check accuracy for all enabled tasks
    :param loader: Test DataLoader
    :param models: Dictionary of models for each task
    :param device: Device to run on
    :param coordinate_scaler: Scaler for XYZ coordinates (if applicable)
    :param has_continent: Whether to evaluate continent task
    :param has_city: Whether to evaluate city task
    :param has_xyz: Whether to evaluate xyz task
    :return: Dictionary of test results
    '''
    results = {}
    
    if has_continent and 'continent' in models:
        model = models['continent']
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inp, out in loader:
                inp, out = inp.to(device), out.to(device)
                outputs = model(inp.float(), train=False)
                _, predicted = torch.max(outputs.data, 1)
                total += out.size(0)
                correct += (predicted == out.long()).sum().item()
        
        results['continent_accuracy'] = 100 * correct / total
        print(f"Continent Test Accuracy: {results['continent_accuracy']:.2f}%")
    
    if has_city and 'city' in models:
        model = models['city']
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inp, out in loader:
                inp, out = inp.to(device), out.to(device)
                outputs = model(inp.float(), train=False)
                _, predicted = torch.max(outputs.data, 1)
                total += out.size(0)
                correct += (predicted == out.long()).sum().item()
        
        results['city_accuracy'] = 100 * correct / total
        print(f"City Test Accuracy: {results['city_accuracy']:.2f}%")
    
    if has_xyz and 'xyz' in models:
        model = models['xyz']
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inp, out in loader:
                inp, out = inp.to(device), out.to(device)
                outputs = model(inp.float(), train=False)
                _, predicted = torch.max(outputs.data, 1)
                total += out.size(0)
                correct += (predicted == out.long()).sum().item()
        
        results['xyz_accuracy'] = 100 * correct / total
        print(f"XYZ Test Accuracy: {results['xyz_accuracy']:.2f}%")
    
    return results


def fix_labels_complete(y):
    """Complete fix to ensure labels are consecutive integers from 0"""
    if len(y.shape) == 1:
        # Single column
        unique_vals = np.unique(y)
        unique_vals = unique_vals[~np.isnan(unique_vals)]  # Remove NaN values
        mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
        fixed = np.array([mapping[val] for val in y], dtype=np.int32)
        #print(f"Label mapping: {mapping}")
        #print(f"Fixed labels range: {fixed.min()} to {fixed.max()}, unique: {np.unique(fixed)}")
        return fixed
    else:
        # Multiple columns
        fixed_cols = []
        for col in range(y.shape[1]):
            unique_vals = np.unique(y[:, col])
            unique_vals = unique_vals[~np.isnan(unique_vals)]  # Remove NaN values
            mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
            fixed_col = np.array([mapping[val] for val in y[:, col]], dtype=np.int32)
            fixed_cols.append(fixed_col)
            #print(f"Column {col} mapping: {mapping}")
            #print(f"Column {col} range: {fixed_col.min()} to {fixed_col.max()}, unique: {np.unique(fixed_col)}")
        return np.column_stack(fixed_cols)

# 2. Safe XGBoost wrapper to handle label validation
def safe_xgboost_fit(xg_model, X, y):
    """Safely fit XGBoost with proper label validation"""
    try:
        # Ensure X and y are numpy arrays
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        
        # Flatten y if needed and convert to int32
        y = y.flatten().astype(np.int32)
        
        # Validate labels are in correct range
        unique_labels = np.unique(y)
        expected_labels = np.arange(len(unique_labels))
        
        if not np.array_equal(unique_labels, expected_labels):
            # Remap labels to be consecutive from 0
            label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y], dtype=np.int32)
            print(f"Remapped labels: {label_map}")
        
        # Validate final labels
        assert y.min() == 0, f"Labels must start from 0, got min: {y.min()}"
        assert y.max() == len(np.unique(y)) - 1, f"Labels must be consecutive, got max: {y.max()}, unique count: {len(np.unique(y))}"
        
        # Fit XGBoost
        return xg_model.fit(X, y, eval_metric="mlogloss")
        
    except Exception as e:
        print(f"XGBoost fit failed: {e}")
        return None

# --- Main Code (with Argument Parsing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XBNet classifiers for continent/city/xyz.")
    parser.add_argument("--continent", action="store_true", help="Enable continent prediction")
    parser.add_argument("--city", action="store_true", help="Enable city prediction")
    parser.add_argument("--xyz", action="store_true", help="Enable XYZ prediction")
    parser.add_argument('-d',"--data_path", type=str, required=True, help="Path to the input CSV data file.")
    parser.add_argument('-t',"--test_size", type=float, default=0.2, help="Fraction of data to use for testing.")
    parser.add_argument('-r',"--random_state", type=int, default=123, help="Random seed for data splitting.")
    parser.add_argument('-b',"--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument('-lr',"--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizers.")
    parser.add_argument('-e',"--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument('-n',"--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument('-p',"--pin_memory", type=bool, default=False, help="Pin memory for DataLoader (improves performance on CUDA).")
    parser.add_argument('-c',"--use_cuda", type=bool, default=False, help="Enable CUDA if available.")
    parser.add_argument('-s',"--save_path", type=str, default=None, help="Path to save the trained models.")
    parser.add_argument('--patience', type=int, default=40, help='Number of epochs to wait for improvement in validation loss before early stopping.')
    parser.add_argument('--initial_dropout', type=float, default=0.2, help='Initial dropout rate.')
    parser.add_argument('--max_dropout', type=float, default=0.7, help='Maximum dropout rate.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of XGBoost estimators.')
    
    args = parser.parse_args()

    print(f"Task settings — Continent: {args.continent}, City: {args.city}, XYZ: {args.xyz}")

    # Load and preprocess data
    in_data, X, y, le_continent, le_city, coordinate_scaler, \
        continent_encoding_map, city_encoding_map = process_transform_data.process_data(args.data_path)

    X_train, X_test, y_train, y_test = process_transform_data.split_data(X, y, test_size=0.2, random_state=123)
    X_train, X_val, y_train, y_val = process_transform_data.split_data(X_train, y_train, test_size=0.2, random_state=123)

    # Apply the fix
    y_train = fix_labels_complete(y_train)
    y_val = fix_labels_complete(y_val)
    y_test = fix_labels_complete(y_test)



    hyperparams = {
        "data": args.data_path,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "hidden_dim": 128,
        "initial_dropout": args.initial_dropout,
        "max_dropout": args.max_dropout,
        "patience": args.patience,
        "input_size": X_train.shape[1],
        "n_estimators": args.n_estimators
    }

    # Device
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    print(f"Using device: {device}")

    # Store models and results
    models = {}
    all_results = {}

    # Train models for each enabled task
    if args.continent:
        print("\n" + "="*50)
        print("TRAINING CONTINENT CLASSIFIER")
        print("="*50)
        
        # Create continent-specific datasets
        train_ds_cont = CustDat(X_train, y_train, task='continent')
        val_ds_cont = CustDat(X_val, y_val, task='continent')
        test_ds_cont = CustDat(X_test, y_test, task='continent')
        
        train_dl_cont = DataLoader(train_ds_cont, batch_size=hyperparams['batch_size'], 
                                  shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        val_dl_cont = DataLoader(val_ds_cont, batch_size=hyperparams['batch_size'], 
                                shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        test_dl_cont = DataLoader(test_ds_cont, batch_size=hyperparams['batch_size'], 
                                 shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        
        # Create model
        num_continent = len(in_data['continent_encoding'].unique())
        model_cont = XBNETClassifier(
            input_size=hyperparams['input_size'],
            hidden_dim=hyperparams['hidden_dim'],
            num_classes=num_continent,
            initial_dropout_rate=hyperparams['initial_dropout'],
            max_dropout_rate=hyperparams['max_dropout'],
            n_estimators=hyperparams['n_estimators']
        ).to(device)
        
        # Optimizer and loss
        optimizer_cont = torch.optim.Adam(model_cont.parameters(), lr=hyperparams['learning_rate'], weight_decay=0.0001)
        criterion_cont = nn.CrossEntropyLoss(weight=(1 / torch.tensor(
            in_data['continent_encoding'].value_counts().sort_index().tolist(), dtype=torch.float32)).to(device))
        
        # Train
        train_loss_cont, val_loss_cont = xbnet_training_loop(
            train_dl=train_dl_cont,
            val_dl=val_dl_cont,
            model=model_cont,
            optimizer=optimizer_cont,
            criterion=criterion_cont,
            device=device,
            num_epochs=hyperparams["epochs"],
            patience=hyperparams['patience'],
            task_name="Continent"
        )
        
        models['continent'] = model_cont
        all_results['continent'] = {'train_loss': train_loss_cont, 'val_loss': val_loss_cont}

    if args.city:
        print("\n" + "="*50)
        print("TRAINING CITY CLASSIFIER")
        print("="*50)
        
        # Create city-specific datasets
        train_ds_city = CustDat(X_train, y_train, task='city')
        val_ds_city = CustDat(X_val, y_val, task='city')
        test_ds_city = CustDat(X_test, y_test, task='city')
        
        train_dl_city = DataLoader(train_ds_city, batch_size=hyperparams['batch_size'], 
                                  shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        val_dl_city = DataLoader(val_ds_city, batch_size=hyperparams['batch_size'], 
                                shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        test_dl_city = DataLoader(test_ds_city, batch_size=hyperparams['batch_size'], 
                                 shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        
        # Create model
        num_city = len(in_data['city_encoding'].unique()) if 'city_encoding' in in_data.columns else 10  # fallback
        model_city = XBNETClassifier(
            input_size=hyperparams['input_size'],
            hidden_dim=hyperparams['hidden_dim'],
            num_classes=num_city,
            initial_dropout_rate=hyperparams['initial_dropout'],
            max_dropout_rate=hyperparams['max_dropout'],
            n_estimators=hyperparams['n_estimators']
        ).to(device)
        
        # Optimizer and loss
        optimizer_city = torch.optim.Adam(model_city.parameters(), lr=hyperparams['learning_rate'], weight_decay=0.0001)
        criterion_city = nn.CrossEntropyLoss()
        
        # Train
        train_loss_city, val_loss_city = xbnet_training_loop(
            train_dl=train_dl_city,
            val_dl=val_dl_city,
            model=model_city,
            optimizer=optimizer_city,
            criterion=criterion_city,
            device=device,
            num_epochs=hyperparams["epochs"],
            patience=hyperparams['patience'],
            task_name="City"
        )
        
        models['city'] = model_city
        all_results['city'] = {'train_loss': train_loss_city, 'val_loss': val_loss_city}

    if args.xyz:
        print("\n" + "="*50)
        print("TRAINING XYZ CLASSIFIER")
        print("="*50)
        
        # Create xyz-specific datasets
        train_ds_xyz = CustDat(X_train, y_train, task='xyz')
        val_ds_xyz = CustDat(X_val, y_val, task='xyz')
        test_ds_xyz = CustDat(X_test, y_test, task='xyz')
        
        train_dl_xyz = DataLoader(train_ds_xyz, batch_size=hyperparams['batch_size'], 
                                 shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
        val_dl_xyz = DataLoader(val_ds_xyz, batch_size=hyperparams['batch_size'], 
                               shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        test_dl_xyz = DataLoader(test_ds_xyz, batch_size=hyperparams['batch_size'], 
                                shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        
        # Create model (assuming xyz is also classification)
        num_xyz = len(np.unique(y_train[:, 2])) if len(y_train.shape) > 1 and y_train.shape[1] > 2 else 10  # fallback
        model_xyz = XBNETClassifier(
            input_size=hyperparams['input_size'],
            hidden_dim=hyperparams['hidden_dim'],
            num_classes=num_xyz,
            initial_dropout_rate=hyperparams['initial_dropout'],
            max_dropout_rate=hyperparams['max_dropout'],
            n_estimators=hyperparams['n_estimators']
        ).to(device)
        
        # Optimizer and loss
        optimizer_xyz = torch.optim.Adam(model_xyz.parameters(), lr=hyperparams['learning_rate'], weight_decay=0.0001)
        criterion_xyz = nn.CrossEntropyLoss()
        
        # Train
        train_loss_xyz, val_loss_xyz = xbnet_training_loop(
            train_dl=train_dl_xyz,
            val_dl=val_dl_xyz,
            model=model_xyz,
            optimizer=optimizer_xyz,
            criterion=criterion_xyz,
            device=device,
            num_epochs=hyperparams["epochs"],
            patience=hyperparams['patience'],
            task_name="XYZ"
        )
        
        models['xyz'] = model_xyz
        all_results['xyz'] = {'train_loss': train_loss_xyz, 'val_loss': val_loss_xyz}

    # Run testing loop for all enabled tasks
    print("\n" + "="*50)
    print("TESTING PHASE")
    print("="*50)
    
    test_results = {}
    
    if args.continent and 'continent' in models:
        test_results.update(check_combined_accuracy(
            loader=test_dl_cont,
            models={'continent': models['continent']},
            device=device,
            coordinate_scaler=coordinate_scaler,
            has_continent=True,
            has_city=False,
            has_xyz=False
        ))
    
    if args.city and 'city' in models:
        test_results.update(check_combined_accuracy(
            loader=test_dl_city,
            models={'city': models['city']},
            device=device,
            coordinate_scaler=coordinate_scaler,
            has_continent=False,
            has_city=True,
            has_xyz=False
        ))
    
    if args.xyz and 'xyz' in models:
        test_results.update(check_combined_accuracy(
            loader=test_dl_xyz,
            models={'xyz': models['xyz']},
            device=device,
            coordinate_scaler=coordinate_scaler,
            has_continent=False,
            has_city=False,
            has_xyz=True
        ))

    # Print final results summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    for task, result in test_results.items():
        print(f"{task}: {result:.2f}%")
    
    # Save models if save path is provided
    if args.save_path:
        import os
        os.makedirs(args.save_path, exist_ok=True)
        
        for task_name, model in models.items():
            model_path = os.path.join(args.save_path, f"{task_name}_xbnet_model.pth")
            model.save(model_path)
            print(f"Saved {task_name} model to {model_path}")
    
    # Filter results to exclude predicted/target arrays (as in original code)
    filtered_results = {k: v for k, v in test_results.items() if 'predicted' not in k and 'target' not in k}
    
    print(f"\nFiltered results: {filtered_results}")


# --- Utility Functions for Predictions ---
def predict(model, X):
    '''
    Predicts the output given the input data
    :param model: XBNETClassifier model
    :param X: Feature array for prediction
    :return: Predicted classes
    '''
    X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
    model.eval()
    with torch.no_grad():
        y_pred = model(X.float(), train=False)
        if model.labels == 1:
            return (y_pred > 0.5).float().cpu().numpy()
        else:
            return torch.argmax(y_pred, dim=1).cpu().numpy()


def predict_proba(model, X):
    '''
    Predicts class probabilities given the input data
    :param model: XBNETClassifier model
    :param X: Feature array for prediction
    :return: Predicted probabilities
    '''
    X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
    model.eval()
    with torch.no_grad():
        y_pred = model(X.float(), train=False)
        return y_pred.cpu().numpy()