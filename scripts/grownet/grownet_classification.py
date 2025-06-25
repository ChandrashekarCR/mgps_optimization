# Importing libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import copy
import pandas as pd

# Weak learners
class WeakLearner(nn.Module):
    """
    Shallow neural network used as weak learners in Grownet. This is similar to using small trees in XGBoost.

    """
    def __init__(self, input_size, hidden_size=64, num_class=7, dropout=0.3):
        super(WeakLearner,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size,hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2,num_class) # Output raw logits
        )

    def forward(self,x):
        return self.layers(x)
    
# Grow Net architecture
class GrowNet(nn.Module):
    """
    GrowNet implementation for multi-class classification    
    
    """
    def __init__(self, input_size, hidden_size=128, num_weak_learners = 5, boost_rate = 0.1, num_class=7, dropout=0.3):
        super(GrowNet,self).__init__()
        self.input_size = input_size
        self.num_weak_learners = num_weak_learners
        self.boost_rate = boost_rate
        self.num_classes = num_class

        # Initialize weak learners
        self.weak_learners = nn.ModuleList()

        # First weak learner will use the original feautres
        self.weak_learners.append(
            WeakLearner(input_size=input_size, hidden_size=hidden_size, num_class=num_class, dropout=dropout)
        )

        # Subsequent weak learners will use the original feautres + previous predictions
        for _ in range(1, num_weak_learners):
            self.weak_learners.append(
                WeakLearner(input_size=input_size+num_class,
                            hidden_size=hidden_size,
                            num_class=num_class,
                            dropout=dropout)
            )

        # Learnable boost rates for each weak learner
        self.boost_rate = nn.Parameter(torch.ones(num_weak_learners)*boost_rate)

    def forward(self,x, stage=None):
        """
        Forward pass through GrowNet
        Args:
            x: Input features
            stage: If specified, only compute up to this stage (for training)
                
        """
        batch_size = x.size(0)
        cumulative_ouput = torch.zeros(batch_size,self.num_classes,device=x.device)

        max_stage = stage if stage is not None else self.num_weak_learners

        for i in range(max_stage):
            if i == 0:
                # First weak learner uses the original features
                weak_output = self.weak_learners[i](x)

            else:
                # Subsequent weak learners use orignial feautres + previous predictions
                combined_input = torch.cat([x,cumulative_ouput],dim=1)
                weak_output = self.weak_learners[i](combined_input)

            # Add weighted weak learner output
            cumulative_ouput = cumulative_ouput + self.boost_rate[i] * weak_output

        return cumulative_ouput
    
    def get_stage_output(self,x, stage):
        """Get outptut up to a specific stae"""
        return self.forward(x,stage+1)

# Create Trainer for Grownet
class GrowNetTrainer:
    """
    
    Trainer class for GrowNet with stage-wise training and corrective steps
    
    """ 
    def __init__(self, model, device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)


    def compute_residuals(self, y_true, y_pred):
        """
        
        Compute residuals for classificaiton using cross entorpy gradient
        
        """

        # Convert to probabilities
        y_pred_prob = F.softmax(y_pred,dim=1)

        # Create one-hot encoding
        y_true_onehot = F.one_hot(y_true,num_classes=self.model.num_classes).float()

        # Compute the residulas
        residuals = y_true_onehot - y_pred_prob

        return residuals
        

    def train_stage(self,train_loader, stage, epochs=50,lr=0.001):
        """
        Train a specific stage of GrowNet
        
        """
        # Freeze previous stages furing individual stage training
        for i, weak_learner in enumerate(self.model.weak_learners):
            if i < stage:
                for param in weak_learner.parameters():
                    param.requires_grad = False

            else:
                for param in weak_learner.parameters():
                    param.requires_grad = True

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr = lr, weight_decay=1e-4
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                if stage == 0:
                    # First stage: train on original targets
                    output = self.model.get_stage_output(batch_x, stage)
                    loss = self.criterion(output, batch_y)
                else:
                    # Subsequent stages: train on residuals
                    with torch.no_grad():
                        prev_output = self.model.get_stage_output(batch_x, stage - 1)
                    
                    # Compute current stage output
                    current_output = self.model.get_stage_output(batch_x, stage)
                    
                    # Loss based on final prediction
                    loss = self.criterion(current_output, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            
            if epoch % 10 == 0:
                print(f"  Stage {stage}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
    
    def corrective_step(self, train_loader, epochs=30, lr=0.0005):
        """
        Corrective step: fine-tune all weak learners together
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
        
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"  Corrective step early stopping at epoch {epoch+1}")
                    break
            
            if epoch % 5 == 0:
                print(f"  Corrective Step, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def fit(self, train_loader, val_loader=None, stage_epochs=50, corrective_epochs=30):
        """
        Complete training procedure for GrowNet
        """
        print("Starting GrowNet Training...")
        print(f"Device: {self.device}")
        print(f"Number of weak learners: {self.model.num_weak_learners}")

        # ENsure model is on correct device
        self.model = self.model.to(self.device)
        
        # Stage-wise training
        for stage in range(self.model.num_weak_learners):
            print(f"\nTraining Stage {stage + 1}/{self.model.num_weak_learners}")
            self.train_stage(train_loader, stage, epochs=stage_epochs)
            
            # Validate after each stage if validation data is provided
            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                print(f"  Stage {stage + 1} Validation Accuracy: {val_acc:.4f}")
        
        # Corrective step
        print(f"\nPerforming Corrective Step...")
        self.corrective_step(train_loader, epochs=corrective_epochs)
        
        if val_loader is not None:
            final_val_acc = self.evaluate(val_loader)
            print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        
        print("Training completed!")
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on given data
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_predictions)

        return accuracy
    
    def predict(self, data_loader):
        """
        Make predictions on given data
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        classfication_metrics = classification_report(all_targets,all_predictions)
        confusion_matrix_metrics = confusion_matrix(all_targets,all_predictions)
        
        return accuracy, classfication_metrics, confusion_matrix_metrics  
    


def create_grownet_model(input_dim=200, num_classes=7):
    """
    Create GrowNet model with optimized hyperparameters for your dataset
    """
    model = GrowNet(
        input_size=input_dim,
        num_class=num_classes,
        num_weak_learners=25,  # Increased for better ensemble effect
        hidden_size=128,       # Larger hidden dimension for complex patterns
        boost_rate=0.08,      # Slightly lower boost rate for stability
        dropout=0.4      # Higher dropout for regularization
    )
    return model


def prepare_data(X, y, batch_size=64, test_size=0.2, random_state=42):
    """
    Prepare data for training
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler


# Example training script
# Read the data 
df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
df = pd.concat([df.iloc[:,:-4],df['continent']],axis=1)
x_data = df[df.columns[:-1]][:]
print(x_data.shape)
y_data = df[df.columns[-1]][:]
le = LabelEncoder()
y_data = le.fit_transform(y_data)
print(le.classes_)


# Prepare data
train_loader, val_loader, scaler = prepare_data(x_data, y_data, batch_size=64)

# Create model
model = create_grownet_model(input_dim=200, num_classes=7)

# Create trainer
trainer = GrowNetTrainer(model)

# Train the model
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    stage_epochs=120,      # More epochs per stage
    corrective_epochs=60  # More corrective epochs
)


# Get predictions
test_accuracy, test_classification_metrics, test_confusion_metrics = trainer.predict(val_loader)
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"\nTest Classification Report: \n{test_classification_metrics}")
print(f"\nTest Confusion Metrics: \n{test_confusion_metrics}")


