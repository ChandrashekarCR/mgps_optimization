# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time
from tqdm import tqdm
import os
import argparse
import sys
import gc
import psutil
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle


# Global variables
DEFAULT_OUTPUT_FILE = os.getcwd()

def evaluate_rfe_single(args_tuple):
    """Single RFE evaluation function for multiprocessing"""
    n_features, fold, X_data, y_data, random_state_base, cv = args_tuple
    
    print(f"[TASK START] Processing {n_features} features, fold {fold} (PID: {os.getpid()})")
    start_time = time.time()
    
    try:
        print(f"[TASK {n_features}-{fold}] Creating RFE pipeline...")
        # Create pipeline with memory-efficient settings
        rfe = RFE(
            estimator=RandomForestClassifier(
                n_jobs=1,  # Single job per worker
                random_state=random_state_base + fold,
                n_estimators=50,
                max_depth=10
            ), 
            n_features_to_select=n_features, 
            step=1
        )
        pipe = make_pipeline(rfe)
        print(f"[TASK {n_features}-{fold}] RFE pipeline created successfully")

        print(f"[TASK {n_features}-{fold}] Creating cross-validation splits...")
        # Create cross-validation split
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state_base)
        splits = list(skf.split(X_data, y_data))
        print(f"[TASK {n_features}-{fold}] Created {len(splits)} CV splits")
        
        if fold >= len(splits):
            print(f"[TASK {n_features}-{fold}] ERROR: Fold {fold} >= number of splits {len(splits)}")
            return n_features, fold, 0.0, np.array([])
            
        train_index, test_index = splits[fold]
        print(f"[TASK {n_features}-{fold}] Using fold {fold}: train_size={len(train_index)}, test_size={len(test_index)}")

        print(f"[TASK {n_features}-{fold}] Extracting training and testing data...")
        # Extract training and testing data
        X_train = X_data.iloc[train_index, :].copy()
        X_test = X_data.iloc[test_index, :].copy()
        y_train = y_data.iloc[train_index].copy()
        y_test = y_data.iloc[test_index].copy()
        print(f"[TASK {n_features}-{fold}] Data extracted: X_train={X_train.shape}, X_test={X_test.shape}")

        print(f"[TASK {n_features}-{fold}] Starting model fitting (this may take several minutes)...")
        fit_start = time.time()
        # Fit the model
        pipe.fit(X_train, y_train)
        fit_time = time.time() - fit_start
        print(f"[TASK {n_features}-{fold}] Model fitting completed in {fit_time:.2f} seconds")
        
        print(f"[TASK {n_features}-{fold}] Calculating score...")
        score = pipe.score(X_test, y_test)
        support = pipe[0].support_.copy()
        selected_features = np.sum(support)
        print(f"[TASK {n_features}-{fold}] Score calculated: {score:.4f}, Selected features: {selected_features}")

        # Clean up
        print(f"[TASK {n_features}-{fold}] Cleaning up memory...")
        del X_train, X_test, y_train, y_test, pipe
        gc.collect()

        elapsed_time = time.time() - start_time
        print(f"[TASK COMPLETE] {n_features} features, fold {fold}: score={score:.4f}, time={elapsed_time:.2f}s")
        return n_features, fold, score, support
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[TASK ERROR] {n_features} features, fold {fold}: {str(e)} (after {elapsed_time:.2f}s)")
        return n_features, fold, 0.0, np.array([])

def parallel_rfe_feature_selection(X: pd.DataFrame, y: pd.Series, n_jobs: int = 1, random_state: int = 123,
                                   cv: int = 10, subsets: list = None, remove_correlated: bool = True,
                                   correlation_threshold: float = 0.98, num_cpus: int = None):
    """
    Performs parallel Recursive Feature Elimination (RFE) with cross-validation to select the best feature subset.
    """
    print("="*80)
    print("STARTING PARALLEL RFE FEATURE SELECTION")
    print("="*80)
    
    # Check available memory
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"[MEMORY] Available memory: {available_memory:.2f} GB")
    
    # Calculate number of CPUs to use
    if num_cpus is None:
        num_cpus = min(cpu_count(), 32)  # Cap at 32 for stability
    
    print(f"[CPU] Using {num_cpus} CPUs for multiprocessing (total available: {cpu_count()})")
    print(f"[DATA] Input data shape: {X.shape}")
    print(f"[DATA] Target variable shape: {y.shape}")
    print(f"[DATA] Target classes: {sorted(y.unique())}")

    # Calculate correlation matrix and remove correlated features
    if remove_correlated:
        print(f"\n[CORRELATION] Starting correlation analysis...")
        print(f"[CORRELATION] Threshold: {correlation_threshold}")
        corr_start = time.time()
        
        corr_matrix = X.corr()
        print(f"[CORRELATION] Correlation matrix computed in {time.time() - corr_start:.2f} seconds")
        
        # Create upper triangle mask
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find correlated features
        correlated_features = []
        for column in upper_tri.columns:
            if any(upper_tri[column].abs() > correlation_threshold):
                correlated_features.append(column)
        
        print(f"[CORRELATION] Found {len(correlated_features)} correlated features to remove")
        
        # Clean up memory
        del corr_matrix, upper_tri
        gc.collect()

        # Drop correlated features
        if correlated_features:
            X = X.drop(columns=correlated_features)
            print(f"[CORRELATION] After removal, data shape: {X.shape}")
        else:
            print(f"[CORRELATION] No features removed")

    # Determine default subset sizes if not provided
    num_features = X.shape[1]
    print(f"\n[SUBSETS] Total features available: {num_features}")
    
    if subsets is None:
        subsets = [num_features // 2, num_features // 4, num_features // 8, 
                   num_features // 16, num_features // 32, num_features // 64]
        subsets = [s for s in subsets if s > 0]

    n_features_options = sorted(list(set(subsets)))
    total_iterations = len(n_features_options) * cv

    print(f"[SUBSETS] Feature subsets to test: {n_features_options}")
    print(f"[SUBSETS] Cross-validation folds: {cv}")
    print(f"[SUBSETS] Total iterations: {total_iterations}")

    start_time = time.time()
    
    # Prepare all tasks
    print(f"\n[TASKS] Preparing {total_iterations} tasks...")
    all_tasks = []
    task_count = 0
    for n_features in n_features_options:
        for fold in range(cv):
            all_tasks.append((n_features, fold, X, y, random_state, cv))
            task_count += 1
            print(f"[TASKS] Task {task_count}: {n_features} features, fold {fold}")
    
    results = []
    all_supports = {}
    
    # Use multiprocessing instead of Ray
    print(f"\n[MULTIPROCESSING] Starting pool with {num_cpus} processes...")
    print(f"[MULTIPROCESSING] Each task will be processed independently")
    print(f"[MULTIPROCESSING] Progress will be shown below:")
    print("-" * 80)
    
    with Pool(processes=num_cpus) as pool:
        # Process all tasks in parallel
        pool_results = list(tqdm(
            pool.imap(evaluate_rfe_single, all_tasks),
            total=len(all_tasks),
            desc='Parallel RFE + Cross-validation'
        ))
        
        print("-" * 80)
        print(f"[MULTIPROCESSING] All tasks completed, processing results...")
        
        # Collect results
        valid_results = 0
        invalid_results = 0
        for n_features_res, fold_res, score_res, support_res in pool_results:
            if score_res > 0:  # Only store valid results
                results.append((n_features_res, score_res))
                all_supports[(n_features_res, fold_res)] = support_res
                valid_results += 1
                print(f"[RESULTS] Valid result: {n_features_res} features, fold {fold_res}, accuracy: {score_res:.4f}")
            else:
                invalid_results += 1
                print(f"[RESULTS] Invalid result: {n_features_res} features, fold {fold_res}")
        
        print(f"\n[RESULTS SUMMARY] Valid results: {valid_results}, Invalid results: {invalid_results}")

    # Aggregate results
    if not results:
        print("[ERROR] No valid results obtained. Check your data and parameters.")
        raise ValueError("No valid results obtained. Check your data and parameters.")
    
    print(f"\n[AGGREGATION] Processing {len(results)} valid results...")
    results_df = pd.DataFrame(results, columns=["n_features", "accuracy"])
    print(f"[AGGREGATION] Results DataFrame shape: {results_df.shape}")
    
    results_df = results_df.groupby("n_features").mean().reset_index()
    print(f"[AGGREGATION] Grouped results:")
    for _, row in results_df.iterrows():
        print(f"[AGGREGATION] {int(row['n_features'])} features: {row['accuracy']:.6f} mean accuracy")

    # Find best feature subset
    best_row = results_df.loc[results_df["accuracy"].idxmax()]
    best_n_features = int(best_row["n_features"])
    best_accuracy = best_row["accuracy"]

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n[FINAL RESULTS]")
    print(f"[FINAL RESULTS] Best number of features: {best_n_features}")
    print(f"[FINAL RESULTS] Best accuracy: {best_accuracy:.6f}")
    print(f"[FINAL RESULTS] Total time: {elapsed_time:.2f} seconds")

    best_params = {"rfe__n_features_to_select": best_n_features}

    print(f"[CLEANUP] Running garbage collection...")
    gc.collect()
    print("="*80)
    print("PARALLEL RFE FEATURE SELECTION COMPLETED")
    print("="*80)

    return best_params, best_accuracy, results_df, elapsed_time, all_supports, X

def check_output_path(output_file_path: str) -> str:
    """
    Validates the output file path and handles the following scenarios:
    
    1. If the provided path is a directory, the output will be saved as 'default_output.txt' in that directory.
    2. If the path includes a valid directory and filename, use the specified filename.
    3. If the user provides only a filename without a directory, use the current working directory.
    4. If the directory is invalid, raise an error and exit.

    Args:
        output_file_path (str): The path where the output file should be saved.

    Returns:
        str: The final output file path.

    Raises:
        FileNotFoundError: If the provided directory does not exist.
    """

    # Normalize the path and handle relative paths
    output_file_path = os.path.normpath(output_file_path)
    print(f"Normalized path: {output_file_path}")

    # Get the current working directory
    wd = os.getcwd()

    # Case 1: If the provided path is a directory, use 'default_output.txt'
    if os.path.isdir(output_file_path):
        print(f"'{output_file_path}' is a directory. Using 'default_output.txt' as the filename.")
        return os.path.join(output_file_path, 'default_output.txt')

    # Extract the directory and filename from the provided path
    directory = os.path.dirname(output_file_path)
    filename = os.path.basename(output_file_path)

    # Case 2: If only a filename is provided without a directory, use the current working directory
    if not directory:
        print(f"No directory provided. Using the current directory with filename '{filename}'.")
        return os.path.join(wd, filename)

    # Case 3: If the directory exists, use the provided filename
    if os.path.isdir(directory):
        print(f"Valid directory found. Using '{filename}' as the output file.")
        return output_file_path

    # Case 4: If the directory is not valid, raise an error
    try:
        raise FileNotFoundError(f"The directory '{directory}' is invalid.")
    except FileNotFoundError as e:
        print(f"{e} Please try again...")
        sys.exit(1)


if __name__ == "__main__":

    # Parse command line arguements
    parser = argparse.ArgumentParser(
        prog='rfe_feature_selection.py',
        usage='python3 rfe_feature_selection.py -i <input processed metadata file> -o <output_file<optional>>',
        description='This program is used to calculate the best features that can be used to train the neural network model.'        
    )
    parser.add_argument('-i','--input_csv_file',dest='csv_file',help='Enter the processed metadata csv file.')
    parser.add_argument('-s','--start_taxa_column',dest='start_index',help='Enter the starting column number of the taxa. (python indexing)',type=int)
    parser.add_argument('-p','--prediction_column',dest='prediction_column',help='Enter the column to be predicted on', type=str)
    parser.add_argument('-o','--output_file',dest='output_file',help='Enter the path of the output file and the name.',default=DEFAULT_OUTPUT_FILE,nargs='?')
    parser.add_argument('--num_cpus', dest='num_cpus', help='Number of CPUs for multiprocessing',type=int, default=None)

    # Parse arguements
    args= parser.parse_args()

    print(f"[MAIN] Loading data from: {args.csv_file}")
    # Fix the dtype warning by specifying low_memory=False
    data = pd.read_csv(args.csv_file, low_memory=False)
    print(f"[MAIN] Data loaded successfully. Shape: {data.shape}")
    
    # Check memory requirements
    data_memory = data.memory_usage(deep=True).sum() / (1024**3)
    print(f"[MAIN] Dataset memory usage: {data_memory:.2f} GB")

    print(f"\n[MAIN] Starting RFE feature selection...")
    best_parameters, best_score, all_results, time_taken, all_supports, return_data = parallel_rfe_feature_selection(
            X=data.iloc[:,args.start_index:],
            y=data[args.prediction_column],
            n_jobs=1,  # Reduced to prevent memory issues
            random_state=123,
            cv=3,
            subsets=[50,100,200,300,400],
            remove_correlated=True,
            correlation_threshold=0.95,
            num_cpus=args.num_cpus
        )

    print(f'\n[MAIN] Best params: {best_parameters}')
    print(f'[MAIN] Best accuracy: {best_score:.6f}')
    print(f'[MAIN] Mean accuracy for all tested feature subsets:\n{all_results}')
    print(f'[MAIN] Total time taken: {time_taken:.2f} seconds')

    # Get the support_ for the best performing number of features (across all folds)
    best_n_features_from_params = best_parameters['rfe__n_features_to_select']
    best_supports = {k: v for k, v in all_supports.items() if k[0] == best_n_features_from_params}

    print(f"\n[OUTPUT] Processing output for {best_n_features_from_params} features...")
    print(f"[OUTPUT] Found {len(best_supports)} support arrays for best model")

    # Store the support arrays for the best models in this variable
    support_for_best_models = best_supports

    if support_for_best_models:
        first_support = list(support_for_best_models.values())[0]
        print(f"[OUTPUT] Using support array with {np.sum(first_support)} selected features")
        
        nn_data = pd.concat([
            data[return_data.columns[first_support]],
            data[[args.prediction_column,'continent','latitude','longitude']]
        ], axis=1)
        
        print(f"[OUTPUT] Final dataset shape: {nn_data.shape}")
        
        # Check if the output file path is valid
        validated_output_file_path = check_output_path(args.output_file)
        print(f"[OUTPUT] Saving results to: {validated_output_file_path}")
        
        # Save the data into csv format
        nn_data.to_csv(path_or_buf=validated_output_file_path, index=False)
        print("[OUTPUT] Results saved successfully!")
    else:
        print("[OUTPUT] No valid feature selection results obtained.")