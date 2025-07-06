# Import Libraries
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import time

import warnings
warnings.filterwarnings('ignore')

# Parameters
params = {
    "feat_d": 200,
    "hidden_size": 512,
    "n_classes": 7,
    "num_nets": 40,
    "boost_rate": 0.05,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "epochs_per_stage": 20,
    "correct_epoch": 10,
    "early_stopping_steps": 20,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)

# Process data 
df = pd.read_csv("/home/chandru/binp37/results/metasub/metasub_training_testing_data.csv")
df = pd.concat([df.iloc[:,:-4],df['continent']],axis=1)
x_data = df[df.columns[:-1]][:].to_numpy()
print(x_data.shape)
y_data = df[df.columns[-1]][:].to_numpy()
le = LabelEncoder()
y_data = le.fit_transform(y_data)
print(le.classes_)

continent_encoding_map = dict(zip(le.transform(le.classes_), le.classes_))
print(continent_encoding_map)


# Dataset class
class TrainDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self,idx):
        dct = {
            'x': torch.tensor(self.features[idx,:],dtype=torch.float),
            'y': torch.tensor(self.targets[idx,:],dtype=torch.float)
        }
        return dct

class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
    


class DynamicNet(object):
    def __init__(self, c0, lr):
        self.models = []
        self.c0 = c0
        self.lr = lr
        self.boost_rate  = nn.Parameter(torch.tensor(lr, requires_grad=True,device=device))

    def to(self, device):
        self.c0 = self.c0.to(device)
        self.boost_rate = self.boost_rate.to(device)
        for m in self.models:
            m.to(device)

    def add(self, model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        params.append(self.boost_rate)
        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self, x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0 = np.repeat(self.c0.detach().cpu().numpy().reshape(1,-1), batch, axis=0)
            return None, torch.Tensor(c0).to(device=device)
        middle_feat_cum = None
        prediction = None
        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, prediction = m(x, middle_feat_cum)
                else:
                    middle_feat_cum, pred = m(x, middle_feat_cum)
                    prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    def forward_grad(self, x):
        if len(self.models) == 0:
            batch = x.shape[0]
            c0 = np.repeat(self.c0.detach().cpu().numpy().reshape(1, -1), batch, axis=0)
            return None, torch.Tensor(c0).cuda()
        # at least one model
        middle_feat_cum = None
        prediction = None
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, prediction = m(x, middle_feat_cum)
            else:
                middle_feat_cum, pred = m(x, middle_feat_cum)
                prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    @classmethod
    def from_file(cls, path, builder):
        d = torch.load(path)
        net = DynamicNet(d['c0'], d['lr'])
        net.boost_rate = d['boost_rate']
        for stage, m in enumerate(d['models']):
            submod = builder(stage)
            submod.load_state_dict(m)
            net.add(submod)
        return net

    def to_file(self, path):
        models = [m.state_dict() for m in self.models]
        d = {'models': models, 'c0': self.c0, 'lr': self.lr, 'boost_rate': self.boost_rate}
        torch.save(d, path)    



class MLP_2HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2,dim_out):
        super(MLP_2HL, self).__init__()
        self.bn2 = nn.BatchNorm1d(dim_in)

        self.layer1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim_in, dim_hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden1),
            nn.Dropout(0.4),
            nn.Linear(dim_hidden1, dim_hidden2)
        )
        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_hidden2, dim_out)
        )

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)

        middle_feat = self.layer1(x)
        out = self.layer2(middle_feat)
        return middle_feat, out

    @classmethod
    def get_model(cls, stage, params):
        if stage == 0:
            dim_in = params["feat_d"]
        else:
            dim_in = params["feat_d"] + params["hidden_size"]
        model = MLP_2HL(dim_in, params["hidden_size"], params["hidden_size"], params["n_classes"])
        return model



def get_optim(params, lr, weight_decay):
    optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
    #optimizer = SGD(params, lr, weight_decay=weight_decay)
    return optimizer

def logloss(net_ensemble, test_loader):
    loss = 0
    total = 0
    loss_f = nn.CrossEntropyLoss() # Cross entropy loss function
    for data in test_loader:
        x = data["x"].to(device)
        y = data["y"].to(device)
        # y = (y + 1) / 2
        with torch.no_grad():
            _, out = net_ensemble.forward(x)
        y_labels = torch.argmax(y,dim=1)
        loss += loss_f(out,y_labels)
        total += 1

    return loss / total



y_onehot = np.eye(params["n_classes"])[y_data]


# Split intp train and validation sets
X_train, X_test, y_train, y_test = train_test_split(x_data, y_onehot, test_size=0.2, random_state=42, stratify=y_onehot)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

train_ds = TrainDataset(X_train, y_train)
val_ds = TrainDataset(X_val, y_val)
test_ds = TrainDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=params["batch_size"],shuffle=False)
test_loader = DataLoader(test_ds, batch_size=params["batch_size"],shuffle=False)


print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")


# Initialiaze GrowNet
c0 = torch.tensor(np.log(np.mean(y_train, axis=0)), dtype=torch.float).unsqueeze(0).to(device)

net_ensemble = DynamicNet(c0, params["boost_rate"])
net_ensemble.to(device)

loss_stagewise = nn.MSELoss(reduction="none")
loss_corrective = nn.CrossEntropyLoss()

best_val_loss = float("inf")
best_stage = 0
early_stop = 0
lr = params["lr"]

print("Initial Logloss:", logloss(net_ensemble,val_loader).item())




for stage in range(params["num_nets"]):
    t0 = time.time()

    print(f"\n Training weak learner {stage+1}/{params["num_nets"]}")

    model = MLP_2HL.get_model(stage,params).to(device)
    optimizer = get_optim(model.parameters(), lr, params["weight_decay"])
    net_ensemble.to_train()

    stage_train_losses = []

    for epoch in range(params["epochs_per_stage"]):
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            with torch.no_grad():
                _, out_prev = net_ensemble.forward(x)
                # Cross entropy gradients for muli-class
                p = torch.softmax(out_prev,dim=1)
                grad_direction = y- p
                # Hessian approximation for multi-class
                h = p * (1-p)
                h = h.sum(dim=1, keepdim=True)
            
            middle_feat, out = model(x, None if stage == 0 else net_ensemble.forward_grad(x)[0])
            loss = loss_stagewise(net_ensemble.boost_rate*out, grad_direction)
            loss = (loss*h).mean()

            model.zero_grad()
            loss.backward()
            optimizer.step()
            stage_train_losses.append(loss.item())
        
    net_ensemble.add(model)
    avg_stage_loss = np.mean(stage_train_losses)
    print(f"Stage {stage+1} finished | Avg Train Loss: {avg_stage_loss:.5f} | Time: {time.time() - t0:.1f}s")


    # Corrective step
    if stage > 0:
        if stage % 3 == 0:
            lr /= 2
        corrective_optimizer = get_optim(net_ensemble.parameters(), lr/2, params["weight_decay"])
        corrective_losses = []

        for _ in range(params["correct_epoch"]):
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)

                _, out = net_ensemble.forward_grad(x)
                loss = loss_corrective(out,y).mean()
                corrective_optimizer.zero_grad()
                loss.backward()
                corrective_optimizer.step()
                corrective_losses.append(loss.item())
        print(f"Fully corrective step avg losse: {np.mean(corrective_losses):.3f}")

    # Validation
    val_loss = logloss(net_ensemble, val_loader).item()
    print(f"Validation LogLoss: {val_loss:.5f} | Boost rate: {net_ensemble.boost_rate.item():.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_stage = stage
        early_stop = 0
    else:
        early_stop += 1
        if early_stop > params["early_stopping_steps"]:
            print("Early stopping!")
            break

print(f"\nBest model was at stage {best_stage+1} with Val LogLoss: {best_val_loss:.5f}")


# --- Predict on test set ---
y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        _, logits = net_ensemble.forward(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_true.extend(np.argmax(y, axis=1))  # Convert one-hot to integer
        y_pred.extend(preds)

# --- Classification Metrics ---
print("\nClassification Report on Test Set")
print(classification_report(y_true, y_pred, target_names=le.classes_))

print("\nAccuracy Score:", accuracy_score(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))