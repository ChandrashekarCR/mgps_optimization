import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
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
import json
from pathlib import Path
import uuid  # Add this import for generating unique IDs

# --- Import shared scripts ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from scripts.metasub.process_transfrom_data import process_transform_data
from scripts.metasub.nn_architecture.nn_continent.nn_continent_model import CombinedNeuralNetXYZModel
from scripts.metasub.accuracy_metrics.calculate_model_accuracy import check_combined_accuracy

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

        # Reporting for Ray Tune (adjust to use the correct metrics)
        tune.report({
            "epoch": epoch,
            "train_continent_loss": train_hist["continent"][-1] if "continent" in train_hist else 0.0,
            "val_continent_loss": val_hist["continent"][-1] if "continent" in val_hist else 0.0,
            "train_cities_loss": train_hist["city"][-1] if "city" in train_hist else 0.0,
            "val_cities_loss": val_hist["city"][-1] if "city" in val_hist else 0.0,
            "train_xyz_loss": train_hist["xyz"][-1] if "xyz" in train_hist else 0.0,
            "val_xyz_loss": val_hist["xyz"][-1] if "xyz" in val_hist else 0.0,
            "val_loss": val_tot # The overall validation loss for metric tracking
        })

        if epoch % 10 == 0:
            msg = f"Epoch {epoch}/{num_epochs} | " + " | ".join(
                f"{k} train {train_hist[k][-1]:.4f} val {val_hist[k][-1]:.4f}" for k in train_hist
            ) + f" | dropout {drop_rate:.3f} | time {time.time()-epoch_start:.1f}s"
            print(msg)

    if best_state:
        model.load_state_dict(best_state)
        print(f"Restored best model (val loss {best_val_loss:.4f})")

    return train_hist, val_hist

# Function to run all the hyperparameters using ray
def hyperparameter_train_model(config):

    # Load and preprocess data
    in_data, X, y, le_continent, le_city, coordinate_scaler, \
        continent_encoding_map, city_encoding_map = process_transform_data.process_data(args.data_path)

    X_train, X_test, y_train, y_test = process_transform_data.split_data(X, y, test_size=0.2,random_state=123)
    X_train, X_val, y_train, y_val = process_transform_data.split_data(X_train, y_train, test_size=0.2,random_state=123)

    # Create DataLoaders - Train, Validate and Test
    train_dl = DataLoader(CustDat(X_train, y_train),
                              batch_size=config['batch_size'], shuffle=True)

    val_dl = DataLoader(CustDat(X_val, y_val),
                        batch_size=config['batch_size'],
                        shuffle=False)

    test_dl = DataLoader(CustDat(X_test, y_test),
                             batch_size=config['batch_size'], shuffle=False)
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model
    num_continent = len(in_data['continent_encoding'].unique())
    model = CombinedNeuralNetXYZModel(input_size=200,hidden_dim=config['layer_size'],initial_dropout_rate=config['initial_dropout_rate'],max_dropout_rate=config['max_dropout_rate'],\
                                      num_continent=num_continent).to(device)


    # Optimizer
    if config['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
    else:  # Adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

    # Loss functions
    criterion_cont = nn.CrossEntropyLoss(weight=(1 / torch.tensor(in_data['continent_encoding'].value_counts().sort_index().tolist(), dtype=torch.float32)
    ).to(device))
    criterion_city = nn.CrossEntropyLoss() if args.city else None
    criterion_xyz = nn.MSELoss() if args.xyz else None

    # Training (calling the new generalized training_loop)
    train_hist, val_hist = training_loop(
        train_dl=train_dl,
        val_dl=val_dl,
        model=model,
        optimizer=optimizer,
        criterion_continent=criterion_cont if args.continent else None,
        criterion_city=criterion_city,
        criterion_xyz=criterion_xyz,
        device=device,
        num_epochs=config['epochs'],
        patience=config['lr_patience'], # Re-using this as patience for early stopping
        has_continent=args.continent,
        has_city=args.city,
        has_xyz=args.xyz
    )

    # Return final metrics from the histories
    # The `val_loss` for Ray Tune should be the overall combined validation loss.
    # We will compute it from the individual task losses.
    final_val_continent_loss = val_hist['continent'][-1] if 'continent' in val_hist else 0.0
    final_val_cities_loss = val_hist['city'][-1] if 'city' in val_hist else 0.0
    final_val_xyz_loss = val_hist['xyz'][-1] if 'xyz' in val_hist else 0.0

    # Calculate the combined validation loss for Ray Tune's 'val_loss' metric
    # This assumes all tasks are always present. Adjust if tasks can be optional.
    final_val_total_loss = final_val_continent_loss + final_val_cities_loss + final_val_xyz_loss

    # After training, evaluate on test set
    test_results = check_combined_accuracy(
            loader=test_dl,
            model=model,
            coordinate_scaler=coordinate_scaler,
            device=device,
            has_continent=args.continent,
            has_city=args.city,
            has_xyz=args.xyz
        )
        
    # Prepare metrics
    final_metrics = {
            "val_loss": final_val_total_loss,
            **{k: v for k, v in test_results.items() 
               if 'predicted' not in k and 'target' not in k}
        }
        
    # Save trial results
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.mkdir(exist_ok=True)
        
        # Generate a unique trial ID using UUID instead of tune.get_trial_id()
        trial_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        
        # Save config and metrics
        trial_results = {
            "trial_id": trial_id,
            "config": config,
            "metrics": final_metrics
        }
        
        # Also save to CSV
        save_trial_results(config, final_metrics, args.save_path)
    
    return final_metrics

# Then modify your save_trial_results function to handle the additional metrics:
def save_trial_results(config, metrics, save_path="ray_tune_results"):
    os.makedirs(save_path, exist_ok=True)
    csv_file = os.path.join(save_path, "hyperparameter_results.csv")
    
    # Create complete record
    record = {
        **config,  # All hyperparameters
        **metrics  # All validation and test metrics
    }
    
    # Write to CSV
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def trainable(config):
    # This wrapper is needed for Ray Tune
    return hyperparameter_train_model(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hierarchical neural network for location prediction.")
    parser.add_argument("--continent", action="store_true", help="Enable continent prediction")
    parser.add_argument("--city", action="store_true", help="Enable city prediction")
    parser.add_argument("--xyz", action="store_true", help="Enable XYZ prediction")
    parser.add_argument('-d',"--data_path", type=str, required=True, help="Path to the input CSV data file.")
    parser.add_argument('-t',"--test_size", type=float, default=0.2, help="Fraction of data to use for testing.")
    parser.add_argument('-r',"--random_state", type=int, default=123, help="Random seed for data splitting.")
    parser.add_argument('-n',"--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument('-p',"--pin_memory", type=bool, default=False, help="Pin memory for DataLoader (improves performance on CUDA).")
    parser.add_argument('-c',"--use_cuda", type=bool, default=False, help="Enable CUDA if available.")
    parser.add_argument('-s',"--save_path", type=str, default=None, help="Path to save the trained models.")

    args = parser.parse_args()


    # Initialize Ray with proper resources
    ray.init(num_cpus=20,num_gpus=1 if args.use_cuda else 0)

    config = {
        # Learning parameters
        'lr': tune.loguniform(1e-5, 1e-2),
        'optimizer': tune.choice(['adam', 'sgd','adamw']),
        'momentum': tune.uniform(0.85, 0.99),  # Only used for SGD
        'weight_decay': tune.loguniform(1e-6, 1e-3),  # L2 regularization,

        # Architecture parameters
        'layer_size': tune.qrandint(128, 1024, 32),  # Quantized to multiples of 32
        'initial_dropout_rate': tune.uniform(0.0, 0.5),  # Wider dropout range
        'max_dropout_rate': tune.uniform(0.5,0.7),

        # Training parameters
        'batch_size': tune.choice([32, 64, 128, 256]),
        'epochs': 200,

        # Learning rate scheduling (will be used for patience in early stopping)
        'lr_patience': tune.randint(3, 10),  # For ReduceLROnPlateau (re-used for early stopping patience)
        'lr_factor': tune.uniform(0.1, 0.5)  # LR reduction factor (not directly used by new training_loop)
    }

    # Configure scheduler
    scheduler = ASHAScheduler(
        metric='val_loss',
        mode='min',
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )


    # Calculate maximum concurrent trials based on resources
    max_concurrent = 8  # Adjust based on your GPU memory (e.g., 4-8 trials per GPU)
    resources_per_trial = {"cpu": 4, "gpu": 0.125}  # 1/5 of GPU per trial if using 5 concurrent

    # Run the tuning with parallel execution
    tuner = tune.Tuner(
        tune.with_resources(
            trainable,
            resources=resources_per_trial
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=100,
            max_concurrent_trials=max_concurrent
        ),
        param_space=config
    )

    results = tuner.fit()

    # After getting best result:
    best_result = results.get_best_result(metric="val_loss", mode="min")
    print("\nBest trial config:", best_result.config)
    print("Best validation loss:", best_result.metrics["val_loss"])
    
    if args.save_path:
        save_path = Path(args.save_path)
        # Save best config
        with open(save_path / "best_config.json", "w") as f:
            json.dump(best_result.config, f, indent=2)
        
        # Save all results summary
        results_df = pd.DataFrame([trial.metrics for trial in results])
        results_df.to_csv(save_path / "all_trials_summary.csv", index=False)
    
    ray.shutdown()


# python scripts/metasub/ray_main.py --continent -d /home/chandru/binp37/results/metasub_training_testing_data.csv -c True -s /home/chandru/binp37/scripts/metasub/

