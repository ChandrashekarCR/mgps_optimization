# Import libraries
import ray
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


# Global variables
DEFAULT_OUTPUT_FILE = os.getcwd()

def parallel_rfe_feature_selection(X: pd.DataFrame, y: pd.Series, n_jobs: int = 1, random_state: int = 123,
                                   cv: int = 10, subsets: list = None, remove_correlated: bool = True,
                                   correlation_threshold: float = 0.98, num_cpus: int = None):
    """
    Performs parallel Recursive Feature Elimination (RFE) with cross-validation to select the best feature subset.
    """
    # Check available memory
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB
    print(f"Available memory: {available_memory:.2f} GB")
    
    # Calculate memory per Ray worker (leave some headroom)
    if num_cpus is None:
        num_cpus = min(os.cpu_count(), 16)  # Limit to 16 cores max
    
    memory_per_worker = max(1.0, (available_memory * 0.8) / num_cpus)  # 80% of available memory
    print(f"Using {num_cpus} CPUs with ~{memory_per_worker:.2f} GB per worker")
    
    # Initialize Ray with memory limits
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        ignore_reinit_error=True, 
        num_cpus=num_cpus,
        object_store_memory=int(available_memory * 0.3 * 1024**3),  # 30% for object store
        _memory=int(available_memory * 0.7 * 1024**3),  # 70% for workers
        _temp_dir='/tmp/ray'
    )

    # Use fewer jobs for RandomForest to reduce memory pressure
    rf_n_jobs = max(1, min(4, n_jobs))  # Limit RF parallelization
    model = RandomForestClassifier(
        n_jobs=rf_n_jobs, 
        random_state=random_state,
        n_estimators=50,  # Reduce trees to save memory
        max_depth=10      # Limit depth to save memory
    )

    if remove_correlated:
        print("Calculating correlation matrix...")
        # Simplified correlation removal to avoid indexing errors
        print("Computing correlation matrix (this may take a while)...")
        corr_matrix = X.corr()
        
        # Create upper triangle mask
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find correlated features
        correlated_features = []
        for column in upper_tri.columns:
            if any(upper_tri[column].abs() > correlation_threshold):
                correlated_features.append(column)
        
        # Clean up memory
        del corr_matrix, upper_tri
        gc.collect()

        # Drop correlated features
        if correlated_features:
            X = X.drop(columns=correlated_features)
            print(f"Correlated features removed: {len(correlated_features)}")

    # Determine default subset sizes if not provided
    num_features = X.shape[1]
    if subsets is None:
        subsets = [num_features // 2, num_features // 4, num_features // 8, 
                   num_features // 16, num_features // 32, num_features // 64]
        subsets = [s for s in subsets if s > 0]

    n_features_options = sorted(list(set(subsets)))
    total_iterations = len(n_features_options) * cv

    print(f"\nStarting RFE with subsets of features: {n_features_options}")
    print(f"Total iterations: {total_iterations}")

    # Define remote function with better error handling and memory management
    @ray.remote(memory=int(memory_per_worker * 1024**3))
    def evaluate_rfe_remote(n_features, fold, X_remote, y_remote, random_state_base):
        """Performs RFE feature selection and evaluates performance for a given fold."""
        try:
            # Create pipeline with memory-efficient settings
            rfe = RFE(
                estimator=RandomForestClassifier(
                    n_jobs=1,  # Single job per worker
                    random_state=random_state_base + fold,
                    n_estimators=50,
                    max_depth=10
                ), 
                n_features_to_select=n_features, 
                step=1    #max(1, min(10, X_remote.shape[1] // 100))  # Adaptive step size
            )
            pipe = make_pipeline(rfe)

            # Create cross-validation split
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state_base)
            splits = list(skf.split(X_remote, y_remote))
            
            if fold >= len(splits):
                return n_features, fold, 0.0, np.array([])
                
            train_index, test_index = splits[fold]

            # Extract training and testing data
            X_train = X_remote.iloc[train_index, :].copy()
            X_test = X_remote.iloc[test_index, :].copy()
            y_train = y_remote.iloc[train_index].copy()
            y_test = y_remote.iloc[test_index].copy()

            # Fit the model
            pipe.fit(X_train, y_train)
            score = pipe.score(X_test, y_test)
            support = pipe[0].support_.copy()

            # Clean up
            del X_train, X_test, y_train, y_test, pipe
            gc.collect()

            return n_features, fold, score, support
            
        except Exception as e:
            print(f"Error in fold {fold} with {n_features} features: {str(e)}")
            return n_features, fold, 0.0, np.array([])

    start_time = time.time()
    
    # Store data in Ray object store
    print("Storing data in Ray object store...")
    X_ray = ray.put(X)
    y_ray = ray.put(y)
    
    # Process in smaller batches to avoid overwhelming the system
    batch_size = min(num_cpus * 2, 20)  # Process 2x CPUs worth at a time
    all_tasks = [(n_features, fold) for n_features in n_features_options for fold in range(cv)]
    
    results = []
    all_supports = {}
    
    with tqdm(total=total_iterations, desc='Parallel RFE + Cross-validation') as pbar:
        for i in range(0, len(all_tasks), batch_size):
            batch_tasks = all_tasks[i:i+batch_size]
            
            # Submit batch of tasks
            ray_tasks = [
                evaluate_rfe_remote.remote(n_features, fold, X_ray, y_ray, random_state)
                for n_features, fold in batch_tasks
            ]
            
            # Wait for batch completion
            while ray_tasks:
                done, ray_tasks = ray.wait(ray_tasks, num_returns=min(len(ray_tasks), num_cpus))
                batch_results = ray.get(done)
                
                for n_features_res, fold_res, score_res, support_res in batch_results:
                    if score_res > 0:  # Only store valid results
                        results.append((n_features_res, score_res))
                        all_supports[(n_features_res, fold_res)] = support_res
                    pbar.update(1)
                
                # Force garbage collection between batches
                gc.collect()

    # Aggregate results
    if not results:
        raise ValueError("No valid results obtained. Check your data and parameters.")
        
    results_df = pd.DataFrame(results, columns=["n_features", "accuracy"])
    results_df = results_df.groupby("n_features").mean().reset_index()

    # Find best feature subset
    best_row = results_df.loc[results_df["accuracy"].idxmax()]
    best_n_features = int(best_row["n_features"])
    best_accuracy = best_row["accuracy"]

    end_time = time.time()
    elapsed_time = end_time - start_time

    best_params = {"rfe__n_features_to_select": best_n_features}

    ray.shutdown()
    gc.collect()

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
        usage='python rfe_feature_selection.py -i <input processed metadata file> -o <output_file<optional>>',
        description='This program is used to calculate the best features that can be used to train the neural network model.'        
    )
    parser.add_argument('-i','--input_csv_file',dest='csv_file',help='Enter the processed metadata csv file.')
    parser.add_argument('-s','--start_taxa_column',dest='start_index',help='Enter the starting column number of the taxa. (python indexing)',type=int)
    parser.add_argument('-p','--prediction_column',dest='prediction_column',help='Enter the column to be predicted on', type=str)
    parser.add_argument('-o','--output_file',dest='output_file',help='Enter the path of the output file and the name.',default=DEFAULT_OUTPUT_FILE,nargs='?')
    parser.add_argument('--num_cpus', dest='num_cpus', help='Number of CPUs for Ray',type=int, default=None)

    # Parse arguements
    args= parser.parse_args()

    print(f"Loading data from: {args.csv_file}")
    # Fix the dtype warning by specifying low_memory=False
    data = pd.read_csv(args.csv_file, low_memory=False)
    print(f"Data loaded successfully. Shape: {data.shape}")
    
    # Check memory requirements
    data_memory = data.memory_usage(deep=True).sum() / (1024**3)
    print(f"Dataset memory usage: {data_memory:.2f} GB")

    best_parameters, best_score, all_results, time_taken, all_supports, return_data = parallel_rfe_feature_selection(
            X=data.iloc[:,args.start_index:],
            y=data[args.prediction_column],
            n_jobs=1,  # Reduced to prevent memory issues
            random_state=123,
            cv=5,
            subsets=[200,300,400,500,1000,1500,2000],
            remove_correlated=True,
            correlation_threshold=0.95,
            num_cpus=args.num_cpus
        )

    print(f'\nBest params: {best_parameters}')
    print(f'Best accuracy: {best_score:.6f}')
    print(f'Mean accuracy for all tested feature subsets:\n{all_results}')
    print(f'Total time taken: {time_taken:.2f} seconds')

    # Get the support_ for the best performing number of features (across all folds)
    best_n_features_from_params = best_parameters['rfe__n_features_to_select']
    best_supports = {k: v for k, v in all_supports.items() if k[0] == best_n_features_from_params}

    # Store the support arrays for the best models in this variable
    support_for_best_models = best_supports

    if support_for_best_models:
        first_support = list(support_for_best_models.values())[0]
        nn_data = pd.concat([
            data[return_data.columns[first_support]],
            data[[args.prediction_column,'continent','latitude','longitude']]
        ], axis=1)
        
        # Check if the output file path is valid
        validated_output_file_path = check_output_path(args.output_file)
        print(f"Saving results to: {validated_output_file_path}")
        
        # Save the data into csv format
        nn_data.to_csv(path_or_buf=validated_output_file_path, index=False)
        print("Results saved successfully!")
    else:
        print("No valid feature selection results obtained.")