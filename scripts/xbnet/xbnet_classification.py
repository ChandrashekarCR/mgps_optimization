# Importinh libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
from sklearn.utils.class_weight import compute_sample_weight


class XBNet(nn.Module):
    """
    XBNet (Extremely Boosted Neural Network) implementation from scratch.

    This implementation combines gradient boosted trees with neural networks using Boosted Gradient Descent (BGD) optimzation technique.

    """

    def __init__(self,input_size, hidden_layers=[400,200],num_classes=7, n_estimators=100, max_depth=3, dropout_rate=0.2, sample_weight=None , use_batch_norm = True, random_state=42):
        super(XBNet,self).__init__()
        """
        Initialize XBNet architecture using Pytorch modules.

        Parameters:
        - input_size: Number of input features. In this case it is the GITs. # 200
        - hidden_layers: List of hidden layers # 400, 200
        - n_classes: Number of output classes
        - n_estimators: Number of estimators for gradient boosting
        - max_depth: Maximum depth for gradient boosted trees
        - dropout_rate: L1 reqularization
        - random_state: Random state for reporducibility

        """

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.use_batch_norm = use_batch_norm
        self.sample_weight = sample_weight
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Build the neural network
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Create layer architecture
        layer_sizes = [input_size] + hidden_layers + [num_classes]

        for i in range(len(layer_sizes)-1):
            # Add liner layers
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            # Add batch normalization for hidden layers only and not the output layer
            if i < len(layer_sizes)-2 and self.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i+1]))
            
            # Add dropout for hidden layers only and not th output layer
            if i < len(layer_sizes)-2: # Don't add dropout to ouput layer
                self.dropouts.append(nn.Dropout(dropout_rate))


        # Initialiaze graident boosting components
        self.xgb_tree_initial = None    
        self.feature_importances = {}
        self.layer_outputs = {}

        # Training history
        self.history = []
        self.accuracry_history = []

        # Initialize weights
        self._initialize_weights()

        # Debug: Print the architecture
        self._print_architecture()

    def _print_architecture(self):
        """Print the network architecture for debugging"""
        print("\nXBNet Architecture:")
        print(f"Input size: {self.input_size}")
        print(f"Batch Normalization: {'Enabled' if self.use_batch_norm else 'Disabled'}")
        
        for i, layer in enumerate(self.layers):
            layer_type = "Hidden" if i < len(self.layers) - 1 else "Output"
            print(f"Layer {i} ({layer_type}): {layer.in_features} -> {layer.out_features}")
            
            if i < len(self.layers) - 1:  # Hidden layers
                if self.use_batch_norm:
                    print(f"  + BatchNorm1d({layer.out_features})")
                print(f"  + ReLU")
                print(f"  + Dropout({self.dropout_rate})")
        
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
        print("-"*50)

    def _initialize_weights(self):
        """ Initialize weights for the neiral network using Xavier initilization."""
        for layer in self.layers[:-1]: # Hidden layers
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        # Ouput layer
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

        # Initialiaze the batch normalization parameters
        if self.use_batch_norm:
            for bn in self.batch_norms:
                nn.init.ones_(bn.weight)
                nn.init.zeros_(bn.bias)

    def initialize_first_layer_with_feature_importance(self,X,y):
        """
        Initialize first layer weights using feature importance from gradient boosted tree. This is the most innovative part of XBNet.
        Parameters:
        - X: Input features (numpy array or tensor)
        - y: Number of classes (numpy array or tensor)
        """
        print(f"Initializing the first layer with gradient boosting feautre importance...")

        # Convert to numpu if torch tensor
        if torch.is_tensor(X):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = X
        
        if torch.is_tensor(y):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y

        # Train initial gradient boosted tree
        # Initialize the XGB classifier model
        self.gb_tree_initial = XGBClassifier(objective="multi:softmax",num_class = self.num_classes, n_estimators=self.n_estimators, 
                                             max_depth=self.max_depth,random_state=self.random_state, 
                                             use_label_encoder=False, eval_metric='mlogloss')

        # Train it on the entire dataset
        self.gb_tree_initial.fit(X_np,y_np,sample_weight=self.sample_weight)
        feature_importance = self.gb_tree_initial.feature_importances_

        # Initialize the first layer weights with feature importance
        with torch.no_grad():
            first_layer = self.layers[0]
            input_size, first_hidden = first_layer.weight.shape[1], first_layer.weight.shape[0]
            new_weights = torch.zeros_like(first_layer.weight)
            for i in range(first_hidden):
                # Use feature importance to initialize each neuron's weight
                importance_weights = torch.tensor(feature_importance, dtype=torch.float32)
                # Add some random variation 
                noise = torch.normal(0,0.01, size=(input_size,))
                new_weights[i, :] = importance_weights * (0.1 + noise)
            first_layer.weight.copy_(new_weights)

        print(f"First layer initialized with feature importance. Shape: {first_layer.weight.shape}")

    def forward(self, x, store_activations=False):
        """
        Forward propoagation through the network

        Parameters:
        - x: Input tensor
        - store_activations: Whether to store intermediate activations for tree training

        """
        if store_activations:
            self.layer_outputs = {}


        current_input = x

        # Forward pass through the hidden layers
        for i, (layer, dropout) in enumerate(zip(self.layers[:-1],self.dropouts)):
            
            # Linear transformations
            z = layer(current_input)
            
            # Batch normalization if it is set to True
            if self.use_batch_norm:
                z = self.batch_norms[i](z)

            # Activation function
            a = F.relu(z) # RelU activation for hidden layers

            if store_activations:
                self.layer_outputs[i] = a.detach().cpu().numpy()

            a = dropout(a) if self.training else a # Apply dropout only during training phase
            current_input = a


        output = self.layers[-1](current_input)

        return output


    def train_trees_on_hidden_layers(self,y):
        """
        Train gradient boosted trees on each hidden layer output. This is the part of BGD optimization technique.

        Parameters: 
        - y: Target labels (numpy array or tensor)
        
        """

        if torch.is_tensor(y):
            y_np = y.detach().cpu().numpy()

        else:
            y_np = y

        # ENsure that labels are integers only
        y_np = y_np.astype(int)

        
        self.feature_importances = {}

        for layer_idx, layer_output in self.layer_outputs.items():
            try:
                # Train a new gradient boosted tree on this layer's output
                gb_tree_layer = XGBClassifier(
                    objective='multi:softmax',
                    num_class = self.num_classes,
                    n_estimators= self.n_estimators,
                    max_depth = self.max_depth,
                    random_state = self.random_state,
                    use_label_encoder=False,
                    eval_metric = 'mlogloss',verbosity = 0
                )

                gb_tree_layer.fit(layer_output,y_np,sample_weight=self.sample_weight)
                self.feature_importances[layer_idx] = gb_tree_layer.feature_importances_

            except Exception as e:
                print(f"Warning: Could not train tree on layer {layer_idx}: {e}")
                # Use unifrom importance as fallback option. In case the above tree does not work.
                n_features = layer_output.shape[1]
                self.feature_importances[layer_idx] = np.ones(n_features)/n_features

    def update_weights_with_feature_importance(self):
        """
        Update weights using feature importance from gradient boosted trees. This is the second step of BGD optimization
        Apply feature importance in a way that make sense:
        - Feaute importance from layer i tell us which neuraons in layer i are important
        - This shou;d influence the weights from FROM layer i TO  layer i+1
        
        """
        with torch.no_grad():
            for layer_idx in self.feature_importances.keys(): # Exclude output layer 
                f_importance = self.feature_importances[layer_idx]

                # Apply importance to the weights going OUT of this layer
                # This means updating the wuights of the layer that takes in this layers's output as input
                target_layer_idx = layer_idx + 1

                if target_layer_idx < len(self.layers):
                    target_layer = self.layers[target_layer_idx]
                    current_weights = target_layer.weight.data # Shape: [out_featre, in_features]

                    # The feature importance should match the input features of the target layer
                    expected_input_size = current_weights.shape[1]

                    if len(f_importance) != expected_input_size:
                        print(f"Skipping layer {layer_idx}: importance size {len(f_importance)} != expected {expected_input_size}")

                    # Calculate a conservative scaling factor
                    weight_std = torch.std(current_weights).item()
                    scaling_factor = weight_std * 0.01

                    f_scaled = torch.tensor(f_importance*scaling_factor,dtype=torch.float32,device=current_weights.device)

                    # Apply importnace by scaling the input connections
                    # Each column corrresponds to connections from one input neuron
                    #current_weights *= (1.0 + f_scaled.unsqueeze(0)) # Broadcast across output neurons
                    for i in range(current_weights.shape[0]):
                        current_weights[i,:] += f_scaled


class XBNetTrainer:
    """
    Training class for XBNet neural network architecture.
    
    """

    def __init__(self, model, learning_rate=0.001, weight_deacay=1e-5, device=None):
        """
        Initialize trainer.

        Parameters:
        - model: XBNet model
        - learning_rate: Learning rate for the model to learn
        - weight_decay: L2 regularization weight
        - device: Device to run training on
        """

    
        self.model = model
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialiaze optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_deacay )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

    def train_epoch(self, dataloader, epoch):
        """ Train for one epoch """
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(dataloader,desc=f"Epoch {epoch+1}")

        for batch_idx, (data,target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with activation storage
            output = self.model(data,store_activations=True)

            # Compute loss
            loss = self.criterion(output,target)

            # Train trees on hidden layer outputs (Boosted Gradient Descent 1)
            self.model.train_trees_on_hidden_layers(target)

            # Backward pass
            loss.backward()

            # Step 1: Standard gradient descent update
            self.optimizer.step()

            # Step 2: Update weights with feature importance (BGD Step 2)
            self.model.update_weights_with_feature_importance()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(output.data,1)
            total_samples += target.size(0)
            correct_predictions += (predicted==target).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Accuracy': f'{100. * correct_predictions / total_samples:.2f}%'
            })

        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct_predictions /total_samples

        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """ Validate the model """  
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0 
        total_samples = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass without activation storage (faster)
                output = self.model(data,store_activations = False)

                # Compute loss
                loss = self.criterion(output, target)
                total_loss += loss.item()

                # Predictiosn
                _, predicted = torch.max(output.data, 1)
                total_samples += target.size(0)
                correct_predictions += (predicted==target).sum().item()

        avg_loss = total_loss/ len(dataloader)
        accuracy = 100. * correct_predictions / total_samples

        return avg_loss, accuracy
    
    def fit(self,train_loader, val_loader, epochs=100, patience=40, verbose=True):
        """
        
        Train the XBNet model.

        Parameters:
        - train_loader: Training data loader
        - val_loader: Validation data loader
        - epochs: Number of training epochs
        - verbose: Whether to print the progress or not

        """

        print(f"Starting XBNet training for {epochs} epochs ...")

        train_losses, train_accuracies = [],[]
        val_losses, val_accuracies = [], []

        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader,epoch)

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Record history
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_xbnet_model.pth')
            else:
                patience_counter+=1

            if verbose:
                print(f'Epoch {epoch + 1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'  Best Val Acc: {best_val_accuracy:.2f}%')
                print('-' * 60)

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load the best model
        self.model.load_state_dict(torch.load('best_xbnet_model.pth'))

        # Store training history
        self.model.loss_history = train_losses
        self.model.accuracy_history = train_accuracies
        self.val_loss_history = val_losses
        self.val_accuracy_history = val_accuracies

        print("XBNet training completed!")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}%")


    def predict(self,dataloader):
        """ Make predictions on a dataset """
        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                output = self.model(data, store_activations=False) # We are not training, so we don't need the activation funcitons

                # Get probabilities and predictions
                probs = F.softmax(output,dim=1)
                _, preds = torch.max(output,1)

                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return np.array(predictions), np.array(probabilities)
    
    def plot_training_history(self):
        """ Plot training and validaiton history  """
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))


        # Loss plot
        ax1.plot(self.model.loss_history, label='Train Loss', color='blue')
        ax1.plot(self.val_loss_history, label='Val Loss',color = 'red')
        ax1.set_title('XBNet Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel("Loss")
        ax1.legend()


        # Accuracy Plot
        ax2.plot(self.model.accuracy_history, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracy_history, label='Val Accuracy',color = 'red')
        ax2.set_title('XBNet Training and Validation Accuracies')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel("Accuracy %")
        ax2.legend()

        plt.tight_layout()
        plt.show()


def create_dataloaders(X_train,y_train,X_val, y_val, X_test, y_test, batch_size=64):
    """Create Pytorch data loaders"""

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor,y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor,y_test_tensor)

    # Create Data loaders
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_loader, val_loader, test_loader



def implement_xbnet_pytorch():
    """Run the xbnet model on the datast"""

    # Read the data 
    df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
    df = pd.concat([df.iloc[:,:-4],df['continent']],axis=1)
    x_data = df[df.columns[:-1]][:].to_numpy()
    print(x_data.shape)
    y_data = df[df.columns[-1]][:].to_numpy()
    le = LabelEncoder()
    y_data = le.fit_transform(y_data)
    print(le.classes_)

    # Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(x_data,y_data,random_state=123,test_size=0.2,stratify=y_data)
    # Split train into train and validation as well
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=123, test_size=0.2,stratify=y_train)

    print('Training, Validation and Testing matrices shapes')
    print("\nTraining\n")
    print(X_train.shape, y_train.shape)
    print("\nValidation\n")
    print(X_val.shape, y_val.shape)
    print("\nTesting\n")
    print(X_test.shape, y_test.shape)

    sample_weights = compute_sample_weight(class_weight='balanced',y=y_train)

    # Create dataloader
    train_loader, val_loader, test_loader = create_dataloaders(X_train,y_train, X_val, y_val, X_test, y_test, batch_size=256)

    # Initialiaze XBNet model
    model = XBNet(
        input_size=200,
        hidden_layers=[512,256,256,128],
        num_classes=7,
        n_estimators=100,
        max_depth=5,
        dropout_rate=0.4,
        random_state=42,sample_weight=sample_weights
    )

    # Initialiaze first layer with feature importance
    model.initialize_first_layer_with_feature_importance(X_train,y_train)


    # Initialiaze trainer
    trainer = XBNetTrainer(model,learning_rate=0.001,weight_deacay=1e-5)

    # Train the model
    trainer.fit(train_loader,val_loader,epochs=400,verbose=True,patience=50)

    # Make predictions on test set
    test_predictions, test_probabilities = trainer.predict(test_loader)

    # Evaluate performance
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"\nXBNet Test Accuracy: {test_accuracy:.4f}")

    # Classification Report
    print("\nClassification Report")
    print(classification_report(y_test,test_predictions))

    trainer.plot_training_history()

    return model, trainer, test_predictions, test_probabilities


if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run the test
    model, trainer, predictions, probabilities = implement_xbnet_pytorch()
    print("\nXBNet PyTorch implementation test completed!")
