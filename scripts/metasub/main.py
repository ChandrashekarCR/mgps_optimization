
# Import libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
import argparse
import os
import csv
import sys

# --- Import shared scripts ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from scripts.metasub.process_transfrom_data import process_transform_data
from scripts.metasub.nn_architecture.nn_continent.nn_continent_model import CombinedNeuralNetXYZModel
from scripts.metasub.accuracy_metrics.calculate_model_accuracy import check_combined_accuracy
from scripts.metasub.helper_plots.plotting import PredictionMetrics
#from scripts.metasub.nn_architecture.nn_latlong.nn_latlong_model import CombinedNeuralNetCNNXYZModel, augment_data_with_lat_lon_noise


# --- Custom Dataset ---
class CustDat(Dataset):
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        dp = torch.tensor(self.df[idx], dtype=torch.float32)
        targ = torch.tensor(self.target[idx], dtype=torch.float32)
        continent_city = targ[:2].long()
        lat_lon = targ[2:]
        return dp, continent_city, lat_lon


# --- Training Loop ---
def training_loop(train_dl,val_dl,model,optimizer,criterion_continent=None,criterion_city=None,criterion_xyz=None,device="cpu",
    num_epochs=100,patience=10,has_continent=True,has_city=True,has_xyz=True):
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    # Per-task loss histories
    # Explicit task flag dictionary
    task_flags = {
    "continent": has_continent,
    "city": has_city,
    "xyz": has_xyz
    }

    # Initialize history dicts only for enabled tasks
    train_hist = {k: [] for k, v in task_flags.items() if v}
    val_hist   = {k: [] for k, v in task_flags.items() if v}


    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()

        running = {"continent": 0.0, "city": 0.0, "xyz": 0.0}
        drop_rate = getattr(model, "initial_dropout_rate", 0.0) + \
                    (getattr(model, "max_dropout_rate", 0.0) - getattr(model, "initial_dropout_rate", 0.0)) * (epoch / num_epochs)

        for X, cont_city, xyz in train_dl:
            X, cont_city, xyz = X.to(device), cont_city.to(device), xyz.to(device).float()

            optimizer.zero_grad()
            outs = model(X, drop_rate) if hasattr(model, "initial_dropout_rate") else model(X)

            idx = 0
            losses = []

            if has_continent:
                cont_logits = outs[idx]; idx += 1
                tgt_cont = cont_city[:, 0]
                l_cont = criterion_continent(cont_logits, tgt_cont)
                running["continent"] += l_cont.item()
                losses.append(l_cont)

            if has_city:
                city_logits = outs[idx]; idx += 1
                tgt_city = cont_city[:, 1]
                l_city = criterion_city(city_logits, tgt_city)
                running["city"] += l_city.item()
                losses.append(l_city)

            if has_xyz:
                xyz_pred = outs[idx]; idx += 1
                l_xyz = criterion_xyz(xyz_pred, xyz)
                running["xyz"] += l_xyz.item()
                losses.append(l_xyz)

            total_loss = sum(losses)
            total_loss.backward()
            optimizer.step()

        for k in running:
            if k in train_hist:
                train_hist[k].append(running[k] / len(train_dl))

        # Validation
        model.eval()
        with torch.no_grad():
            running_val = {k: 0.0 for k in running}
            for Xv, ccv, xyzv in val_dl:
                Xv, ccv, xyzv = Xv.to(device), ccv.to(device), xyzv.to(device).float()
                outs_val = model(Xv, drop_rate) if hasattr(model, "initial_dropout_rate") else model(Xv)
                idx = 0

                if has_continent:
                    cl = outs_val[idx]; idx += 1
                    running_val["continent"] += criterion_continent(cl, ccv[:, 0]).item()
                if has_city:
                    cl2 = outs_val[idx]; idx += 1
                    running_val["city"] += criterion_city(cl2, ccv[:, 1]).item()
                if has_xyz:
                    xp = outs_val[idx]; idx += 1
                    running_val["xyz"] += criterion_xyz(xp, xyzv).item()

        val_tot = 0.0
        for k in running_val:
            if k in val_hist:
                avg = running_val[k] / len(val_dl)
                val_hist[k].append(avg)
                val_tot += avg

        if val_tot < best_val_loss:
            best_val_loss = val_tot
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping @ epoch {epoch}")
                break

        if epoch % 10 == 0:
            msg = f"Epoch {epoch}/{num_epochs} | " + " | ".join(
                f"{k} train {train_hist[k][-1]:.4f} val {val_hist[k][-1]:.4f}" for k in train_hist
            ) + f" | dropout {drop_rate:.3f} | time {time.time()-epoch_start:.1f}s"
            print(msg)

    if best_state:
        model.load_state_dict(best_state)
        print(f"Restored best model (val loss {best_val_loss:.4f})")

    return train_hist, val_hist


# --- Main Code (with Argument Parsing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-task NN for continent/city/xyz.")
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
    
    args = parser.parse_args()

    print(f"Task settings — Continent: {args.continent}, City: {args.city}, XYZ: {args.xyz}")

    #Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    hidden_dim = 64
    initial_dropout = args.initial_dropout
    max_dropout = args.max_dropout
    patience = args.patience
    input_size = 200




    # Load and preprocess data
    in_data, X, y, le_continent, le_city, coordinate_scaler, \
        continent_encoding_map, city_encoding_map = process_transform_data.process_data(args.data_path)

    X_train, X_test, y_train, y_test = process_transform_data.split_data(X, y, test_size=0.2,random_state=123)
    X_train, X_val, y_train, y_val = process_transform_data.split_data(X_train, y_train, test_size=0.2,random_state=123)

    hyperparams = {"data":args.data_path,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "epochs": args.epochs,
    "hidden_dim": 128,
    "initial_dropout": args.initial_dropout,
    "max_dropout": args.max_dropout,
    "patience": args.patience,
    "input_size": X_train.shape[1]}

    # Augment Data
    #X_train_augmented, y_train_augmented = augment_data_with_lat_lon_noise(X_train, y_train)

    # Create DataLoaders - Train, Validate and Test
    train_dl = DataLoader(CustDat(X_train, y_train),
                              batch_size=hyperparams['batch_size'], shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory)

    val_dl = DataLoader(CustDat(X_val, y_val),
                        batch_size=hyperparams['batch_size'],
                        shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.pin_memory)

    test_dl = DataLoader(CustDat(X_test, y_test),
                             batch_size=hyperparams['batch_size'], shuffle=False,
                             num_workers=args.num_workers, pin_memory=args.pin_memory)
    
    #train_dl_augmented = DataLoader(CustDat(X_train_augmented, y_train_augmented),
    #                             batch_size=hyperparams['batch_size'], shuffle=True,
    #                             num_workers=args.num_workers, pin_memory=args.pin_memory)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model
    num_continent = len(in_data['continent_encoding'].unique())
    model = CombinedNeuralNetXYZModel(input_size=hyperparams['input_size'],hidden_dim=hyperparams['hidden_dim'],\
                                      initial_dropout_rate=hyperparams['initial_dropout'],max_dropout_rate=hyperparams['max_dropout'],num_continent=num_continent).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'],weight_decay=0.0001)

    # Loss functions
    criterion_cont = nn.CrossEntropyLoss(weight=(1 / torch.tensor(in_data['continent_encoding'].value_counts().sort_index().tolist(), dtype=torch.float32)
    ).to(device))
    criterion_city = nn.CrossEntropyLoss() if args.city else None
    criterion_xyz = nn.MSELoss() if args.xyz else None

    # Run training and validation loops
    train_loss,val_loss=training_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        optimizer=optimizer,
        criterion_continent=criterion_cont if args.continent else None,
        criterion_city=criterion_city,
        criterion_xyz=criterion_xyz,
        device=device,
        num_epochs=hyperparams["epochs"],
        patience=hyperparams['patience'],
        has_continent=args.continent,
        has_city=args.city,
        has_xyz=args.xyz)
    
    # Plot losses
    PredictionMetrics.plot_losses(train_losses=train_loss,val_losses=val_loss,filename='main_losses.png')

    # Run testing loop
    test_results = check_combined_accuracy(
    loader=test_dl,
    model=model,
    coordinate_scaler=coordinate_scaler,
    device="cpu",
    has_continent=args.continent,
    has_city=args.city,
    has_xyz=args.xyz)


    filtered_results = {k: v for k, v in test_results.items() if 'predicted' not in k and 'target' not in k}


    # Combine everything to write
    record = {**hyperparams, **filtered_results}

    # Save hyperparameters and stats to CSV
    def save_params_to_csv(record, csv_file="model_run_log.csv"):
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())

            if not file_exists:
                writer.writeheader()  # Write header only once

            writer.writerow(record)
        print(f"Model config written to {csv_file}")

    save_params_to_csv(record=record)

# python main.py --continent -d ../../results/metasub_training_testing_data.csv -b 32 -lr 0.001 -n 1 -e 200 -c True 

# srun --partition=gpua40i --gres=gpu:1 --time=01:00:00 --nodes=1 --ntasks-per-node=4 --cpus-per-task=1 --mem=16GB --pty bash -i
